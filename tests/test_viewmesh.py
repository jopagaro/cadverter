"""The 3D view's mesh: every triangle traces back to its B-REP face, and a click reads sensibly."""
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")
import cadvert  # noqa: E402
from cadvert.document import assign_feature_ids  # noqa: E402
from cadvert.viewmesh import build_view_mesh, face_summary, read_header  # noqa: E402

SAMPLE = Path(__file__).parent.parent / "samples" / "test_block_with_holes.step"


@pytest.fixture(scope="module")
def part():
    if not SAMPLE.exists():
        pytest.skip("sample STEP missing")
    return cadvert.analyze(SAMPLE)


def unpack(data: bytes) -> dict:
    """Read the packed mesh back exactly as the app does."""
    h = read_header(data)
    (n,) = struct.unpack_from("<I", data, 4)
    off = 8 + n
    out = {"header": h}
    for name, dtype, count in [
        ("positions", np.float32, 3 * h["vertices"]),
        ("normals", np.float32, 3 * h["vertices"]),
        ("indices", np.uint32, 3 * h["triangles"]),
        ("ranges", np.uint32, 2 * h["faces"]),
        ("edge_points", np.float32, 3 * h["edge_points"]),
        ("edge_segments", np.uint32, 2 * h["edge_segments"]),
    ]:
        out[name] = np.frombuffer(data, dtype=dtype, count=count, offset=off)
        off += out[name].nbytes
    assert off == len(data), "trailing or missing bytes"
    return out


def test_mesh_layout_round_trips(part):
    m = unpack(build_view_mesh(part.shape, part.graph))
    h = m["header"]
    assert h["faces"] == len(part.graph.faces)
    assert h["triangles"] > 0 and h["edge_segments"] > 0
    assert m["indices"].max() < h["vertices"]
    assert m["edge_segments"].max() < h["edge_points"]
    norms = np.linalg.norm(m["normals"].reshape(-1, 3), axis=1)
    assert np.allclose(norms, 1, atol=1e-3)

    # Each face owns one contiguous run of triangles, in face-ID order, covering them all.
    ranges = m["ranges"].reshape(-1, 2)
    assert (ranges[:, 1] > 0).all()
    assert ranges[0, 0] == 0
    assert (ranges[1:, 0] == ranges[:-1, 0] + ranges[:-1, 1]).all()
    assert ranges[-1, 0] + ranges[-1, 1] == h["triangles"]

    # The mesh fills the part's bounding box.
    pts = m["positions"].reshape(-1, 3)
    lo, hi = np.array(h["bbox"][:3]), np.array(h["bbox"][3:])
    assert np.allclose(pts.min(axis=0), lo, atol=0.01) and np.allclose(pts.max(axis=0), hi, atol=0.01)


def test_triangles_face_outward(part):
    """Summed (area-weighted) outward normals of a closed solid point nowhere: they cancel.
    A face wound the wrong way would leave a residue the size of its area."""
    m = unpack(build_view_mesh(part.shape, part.graph))
    pts = m["positions"].reshape(-1, 3).astype(np.float64)
    tri = m["indices"].reshape(-1, 3)
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cross = np.cross(b - a, c - a)
    total_area = np.linalg.norm(cross, axis=1).sum() / 2
    assert np.linalg.norm(cross.sum(axis=0)) / 2 < total_area * 1e-3
    # Outward means the signed volume is positive and matches the solid's.
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert volume == pytest.approx(part.graph.volume, rel=0.02)


def test_seam_edges_are_left_out(part):
    with_graph = read_header(build_view_mesh(part.shape, part.graph))
    without = read_header(build_view_mesh(part.shape, None))
    assert with_graph["edge_segments"] < without["edge_segments"]


def test_clicking_a_hole_names_the_hole(part):
    ids = assign_feature_ids(part.features)
    hole_i = next(i for i, f in enumerate(part.features) if "HOLE" in f.feature_type)
    hole = part.features[hole_i]
    info = face_summary(hole.face_ids[0], part.graph, part.features, ids, units="mm")
    assert info["id"] == f"F{hole.face_ids[0]}"
    assert info["feature"] == ids[hole_i]
    assert info["feature_detail"].startswith(("Through hole ∅", "Blind hole ∅"))
    assert set(info["feature_faces"]) == set(hole.face_ids)

    plane = next(f for f in part.graph.faces if f.geometry.get("type") == "PLANE")
    flat = face_summary(plane.id, part.graph, part.features, ids, units="mm")
    assert flat["kind"] == "Flat face" and flat["detail"].endswith("mm²")
    assert face_summary(10_000, part.graph) is None


def test_mesh_and_face_endpoints(part, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from cadvert import server

    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    ids = assign_feature_ids(part.features)
    server._sessions["s1"] = {"graph": part.graph, "features": part.features, "feature_ids": ids,
                              "shape": part.shape, "units": "mm", "assembly": None}
    (tmp_path / "s1").mkdir()
    try:
        client = TestClient(server.app)
        r = client.get("/mesh/s1")
        assert r.status_code == 200 and r.content[:4] == b"CVM1"
        assert (tmp_path / "s1" / "view.mesh").exists()          # cached for next time
        assert client.get("/mesh/s1").content == r.content

        assert client.get("/face/s1/1").json()["id"] == "F1"
        assert client.get("/face/s1/99999").status_code == 404
        assert client.get("/mesh/nope").status_code == 404
    finally:
        server._sessions.pop("s1", None)
