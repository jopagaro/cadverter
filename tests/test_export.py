"""Export: meshes a slicer or web viewer can open, STEP, and the part's tables as CSV.

The checks are the ones a user would notice: a printed part must be closed and the right
size, a 3MF must open in a slicer, a GLB must land in metres with Y up, and the tables must
read correctly in a spreadsheet.
"""
import csv
import io
import json
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

import cadvert
from cadvert.assembly import AssemblyInfo, ComponentInstance
from cadvert.document import assign_feature_ids
from cadvert.export import ExportError, export_part, open_edges, tessellate, weld

SAMPLE = Path(__file__).parent.parent / "samples" / "test_block_with_holes.step"
NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


@pytest.fixture(scope="module")
def part():
    if not SAMPLE.exists():
        pytest.skip("sample STEP missing")
    return cadvert.analyze(SAMPLE)


def _export(kind, part, **kw):
    return export_part(kind, part.shape, name="Block", graph=part.graph, features=part.features,
                       feature_ids=assign_feature_ids(part.features), **kw)


def _signed_volume(pts, tris):
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _read_stl(data):
    (n,) = struct.unpack_from("<I", data, 80)
    rows = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                         count=n, offset=84)
    return rows["v"].astype(np.float64)


def _read_3mf(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= set(z.namelist())
        root = ET.fromstring(z.read("3D/3dmodel.model"))
    assert root.get("unit") == "millimeter"
    objects = []
    for obj in root.findall("m:resources/m:object", NS):
        pts = np.array([[float(v.get(k)) for k in "xyz"] for v in obj.iter(f"{{{NS['m']}}}vertex")])
        tris = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in obj.iter(f"{{{NS['m']}}}triangle")])
        objects.append((obj.get("name"), pts, tris))
    assert len(root.findall("m:build/m:item", NS)) == len(objects)
    return objects


def test_a_printed_part_is_closed_and_true_to_size(part):
    body = weld(tessellate(part.shape, name="Block")[0])
    assert open_edges(body.triangles) == 0
    assert _signed_volume(body.positions, body.triangles) == pytest.approx(part.graph.volume, rel=1e-3)

    tri = _read_stl(_export("stl", part))
    assert len(tri) == len(body.triangles)
    lo, hi = tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)
    assert hi - lo == pytest.approx([60, 40, 20], abs=1e-3)
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    assert volume == pytest.approx(part.graph.volume, rel=1e-3)      # wound outward


def test_3mf_opens_as_one_closed_object(part):
    [(name, pts, tris)] = _read_3mf(_export("3mf", part))
    assert name == "Block"
    assert tris.max() < len(pts)
    assert open_edges(tris) == 0
    assert _signed_volume(pts, tris) == pytest.approx(part.graph.volume, rel=1e-3)


def test_obj_lists_vertices_then_faces(part):
    text = _export("obj", part).decode()
    assert text.startswith("# CADVERT export (millimetres)")
    assert "\no Block\n" in text
    v = [ln for ln in text.splitlines() if ln.startswith("v ")]
    f = [ln for ln in text.splitlines() if ln.startswith("f ")]
    assert max(int(i) for ln in f for i in ln.split()[1:]) == len(v)   # 1-based, all used


def test_glb_is_in_metres_with_y_up(part):
    data = _export("glb", part)
    magic, version, total = struct.unpack_from("<4sII", data)
    assert (magic, version, total) == (b"glTF", 2, len(data))
    (n, kind) = struct.unpack_from("<I4s", data, 12)
    assert kind == b"JSON"
    gltf = json.loads(data[20:20 + n])
    root = gltf["nodes"][gltf["scenes"][0]["nodes"][0]]
    assert root["scale"] == [0.001] * 3
    assert root["rotation"] == pytest.approx([-0.7071068, 0, 0, 0.7071068])
    pos = gltf["accessors"][0]
    assert np.subtract(pos["max"], pos["min"]) == pytest.approx([60, 40, 20], abs=1e-3)
    (m, kind) = struct.unpack_from("<I4s", data, 20 + n)
    assert kind == b"BIN\0" and m == gltf["buffers"][0]["byteLength"]
    assert all(v["byteOffset"] % 4 == 0 for v in gltf["bufferViews"])


def test_step_reads_back_as_the_same_solid(part, tmp_path):
    path = tmp_path / "out.step"
    path.write_bytes(_export("step", part))
    again = cadvert.analyze(path)
    assert again.graph.volume == pytest.approx(part.graph.volume, rel=1e-6)
    assert len(again.graph.faces) == len(part.graph.faces)


def test_feature_table_gives_each_hole_its_opening(part):
    rows = list(csv.DictReader(io.StringIO(_export("features.csv", part).decode("utf-8-sig"))))
    holes = [r for r in rows if r["Type"] == "Through Hole"]
    assert len(holes) == 2
    assert {(r["X (mm)"], r["Y (mm)"]) for r in holes} == {("15", "20"), ("45", "20")}
    assert all(r["Diameter (mm)"] == "8" and r["Depth (mm)"] == "20" and r["Axis"] == "+Z" for r in holes)
    assert rows[0]["Description"].startswith("Through hole ∅8")


def test_what_a_part_cannot_be_exported_as(part):
    with pytest.raises(ExportError, match="single part"):
        _export("bom.csv", part)
    with pytest.raises(ExportError, match="mesh"):
        _export("step", part, is_mesh=True)
    with pytest.raises(ExportError, match="Unknown"):
        _export("dwg", part)
    with pytest.raises(ExportError, match="No features"):
        export_part("features.csv", part.shape, features=[], feature_ids=[])


def _three_blocks():
    """A plate and two identical blocks, side by side — the smallest useful assembly."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for x, size in ((0, (40, 40, 5)), (50, (10, 10, 10)), (70, (10, 10, 10))):
        builder.Add(compound, BRepPrimAPI_MakeBox(gp_Pnt(x, 0, 0), *size).Shape())
    instances = [ComponentInstance(0, "Plate", face_ids=list(range(1, 7)), volume=8000),
                 ComponentInstance(1, "Block", path=("Fixture",), face_ids=list(range(7, 13)), volume=1000),
                 ComponentInstance(2, "Block", path=("Fixture",), face_ids=list(range(13, 19)), volume=1000)]
    owner = {f: inst.index for inst in instances for f in inst.face_ids}
    return compound, AssemblyInfo(instances=instances, face_owner=owner, root_name="Fixture")


def test_an_assembly_exports_one_object_per_component():
    shape, assembly = _three_blocks()
    objects = _read_3mf(export_part("3mf", shape, name="Fixture", assembly=assembly))
    assert [o[0] for o in objects] == ["Plate", "Block 1", "Block 2"]
    assert all(open_edges(tris) == 0 for _, _, tris in objects)
    assert [round(_signed_volume(p, t)) for _, p, t in objects] == [8000, 1000, 1000]

    text = export_part("obj", shape, assembly=assembly).decode()
    assert "\no Block_2\n" in text

    rows = list(csv.reader(io.StringIO(export_part("bom.csv", shape, assembly=assembly).decode("utf-8-sig"))))
    assert rows[0][:3] == ["Item", "Part", "Quantity"]
    assert rows[1][1:5] == ["Block", "2", "1000", "2000"]
    assert rows[2][1:3] == ["Plate", "1"]


def test_exporting_leaves_the_view_mesh_alone(part):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    def triangles():
        total, it = 0, TopExp_Explorer(part.shape, TopAbs_FACE)
        while it.More():
            t = BRep_Tool.Triangulation_s(TopoDS.Face_s(it.Current()), TopLoc_Location())
            total += t.NbTriangles() if t is not None else 0
            it.Next()
        return total

    before = triangles()
    _export("stl", part)
    assert triangles() == before
