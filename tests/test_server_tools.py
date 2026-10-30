"""The six geometry tools behind /chat and /tool, run against the sample part."""
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
import cadvert  # noqa: E402
from cadvert import server  # noqa: E402
from cadvert.document import assign_feature_ids  # noqa: E402

SAMPLE = Path(__file__).parent.parent / "samples" / "test_block_with_holes.step"


@pytest.fixture(scope="module")
def session():
    if not SAMPLE.exists():
        pytest.skip("sample STEP missing")
    result = cadvert.analyze(SAMPLE)
    return {
        "graph":          result.graph,
        "features":       result.features,
        "feature_ids":    assign_feature_ids(result.features),
        "shape":          result.shape,
        "face_shape_map": server._build_face_shape_map(result.shape),
        "units":          result.units,
        "is_mesh":        False,
    }


@pytest.mark.parametrize("name,args", [
    ("get_feature",      {"feature_id": "hole_1"}),
    ("get_face",         {"face_id": "F7"}),
    ("get_edge",         {"edge_id": "E10"}),
    ("measure_distance", {"entity_a": "hole_1", "entity_b": "F1"}),
    ("get_neighbors",    {"face_id": "F3", "depth": 1}),
    ("search_faces",     {"surface_type": "cylinder"}),
])
def test_every_tool_runs_without_error(session, name, args):
    out = server._execute_tool(session, name, args)
    assert isinstance(out, dict)
    assert "error" not in out, out


def test_tool_details(session):
    feat = server._execute_tool(session, "get_feature", {"feature_id": "hole_1"})
    assert feat["type"] == "THROUGH_HOLE"
    assert feat["parameters"]["diameter"] == pytest.approx(8.0)
    assert feat["faces"] and "edge_convexity" in feat["faces"][0]
    assert any(e["length"] for e in feat["edges"])          # circle arc lengths now populated

    edge = server._execute_tool(session, "get_edge", {"edge_id": "E10"})
    assert edge["length"] == pytest.approx(25.1327, abs=1e-3)

    dist = server._execute_tool(session, "measure_distance", {"entity_a": "hole_1", "entity_b": "hole_2"})
    assert dist["distance"] == pytest.approx(22.0)

    nb = server._execute_tool(session, "get_neighbors", {"face_id": "F3", "depth": 1})
    assert nb["count"] == 6 and all("convexity" in n for n in nb["neighbors"])

    missing = server._execute_tool(session, "get_feature", {"feature_id": "hole_99"})
    assert "error" in missing and "hole_1" in missing["available_ids"]


def test_rendered_views_served_without_system_mime_tables():
    """The App Sandbox denies /etc/apache2/mime.types; serving a PNG must not read it."""
    import mimetypes

    assert mimetypes.knownfiles == []
    assert mimetypes.guess_type("part_front.png")[0] == "image/png"


def test_export_endpoint_serves_files_to_save(session):
    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    server._sessions["x1"] = {**session, "name": "Block", "assembly": None}
    try:
        stl = client.get("/export/x1/stl")
        assert stl.status_code == 200 and stl.headers["content-type"] == "model/stl"
        assert "filename*=UTF-8''Block.stl" in stl.headers["content-disposition"]
        assert len(stl.content) > 84

        table = client.get("/export/x1/features.csv")
        assert table.status_code == 200 and "hole_1" in table.content.decode("utf-8-sig")
        assert "Block%20features.csv" in table.headers["content-disposition"]

        assert client.get("/export/x1/bom.csv").status_code == 422    # a single part
        assert client.get("/export/x1/dwg").status_code == 400
        assert client.get("/export/nope/stl").status_code == 404
    finally:
        server._sessions.pop("x1", None)
