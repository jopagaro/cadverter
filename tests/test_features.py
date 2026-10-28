"""Feature classification on parts built from scratch, one tricky case per test.

Each part is modelled here with OpenCASCADE, so the expected answer is known exactly.
These pin down three mistakes the detector used to make on ordinary machined parts:
through holes called "blind" when anything stood above the surface, counterbores split
into two unrelated holes, and bosses with a flat top called holes.
"""
from collections import Counter

import pytest

pytest.importorskip("OCP")

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from cadvert import analyze

Z = gp_Dir(0, 0, 1)


def box(dx, dy, dz, x=0, y=0, z=0):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def cyl(x, y, z, d, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), Z), d / 2, h).Shape()


def cone(x, y, z, r_bottom, r_top, h):
    return BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, y, z), Z), r_bottom, r_top, h).Shape()


def cut(a, b):
    return BRepAlgoAPI_Cut(a, b).Shape()


def fuse(a, b):
    return BRepAlgoAPI_Fuse(a, b).Shape()


def features_of(shape, tmp_path):
    path = tmp_path / "part.step"
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    assert w.Write(str(path)) == IFSelect_RetDone
    return analyze(str(path)).features


def of_type(features, ftype):
    return [f for f in features if f.feature_type == ftype]


def test_through_hole_stays_through_when_a_boss_rises_above(tmp_path):
    # The boss makes the plate's top face no longer the part's highest point — the old
    # bounding-box test then called every hole opening onto it "blind".
    plate = fuse(box(80, 50, 10), cyl(60, 25, 10, 16, 8))
    part = cut(plate, cyl(20, 25, -1, 6, 12))
    feats = features_of(part, tmp_path)
    holes = [f for f in feats if f.feature_type in ("THROUGH_HOLE", "BLIND_HOLE")]
    assert [h.feature_type for h in holes] == ["THROUGH_HOLE"]
    assert holes[0].parameters["depth"] == pytest.approx(10.0)


def test_flat_bottom_blind_hole(tmp_path):
    part = cut(box(40, 40, 20), cyl(20, 20, 14, 8, 10))      # 6 mm deep from the top
    [hole] = of_type(features_of(part, tmp_path), "BLIND_HOLE")
    assert hole.parameters["diameter"] == pytest.approx(8.0)
    assert hole.parameters["depth"] == pytest.approx(6.0)


def test_drill_point_blind_hole(tmp_path):
    drill = fuse(cyl(20, 20, 10, 8, 20), cone(20, 20, 10 - 2.4, 0.0, 4.0, 2.4))
    feats = features_of(cut(box(40, 40, 20), drill), tmp_path)
    assert len(of_type(feats, "BLIND_HOLE")) == 1
    assert not of_type(feats, "THROUGH_HOLE")


def test_countersunk_hole_is_one_through_feature(tmp_path):
    # Like a counterbore, a countersunk hole is reported as one feature covering the bore.
    sink = fuse(cyl(20, 20, -1, 6, 22), cone(20, 20, 17, 3.0, 6.0, 3.0))   # widens to the top
    feats = features_of(cut(box(40, 40, 20), sink), tmp_path)
    [cs] = of_type(feats, "COUNTERSINK")
    assert cs.parameters["bore_diameter"] == pytest.approx(6.0)
    assert cs.parameters["through"] is True
    assert not of_type(feats, "BLIND_HOLE")


def test_drill_point_is_not_a_countersink(tmp_path):
    drill = fuse(cyl(20, 20, 10, 8, 20), cone(20, 20, 10 - 2.4, 0.0, 4.0, 2.4))
    assert not of_type(features_of(cut(box(40, 40, 20), drill), tmp_path), "COUNTERSINK")


def test_through_counterbore_is_one_feature(tmp_path):
    part = cut(cut(box(40, 40, 14), cyl(20, 20, -1, 6.6, 16)), cyl(20, 20, 7.5, 11, 10))
    feats = features_of(part, tmp_path)
    [cb] = of_type(feats, "COUNTERBORE")
    p = cb.parameters
    assert p["bore_diameter"] == pytest.approx(6.6)
    assert p["cbore_diameter"] == pytest.approx(11.0)
    assert p["cbore_depth"] == pytest.approx(6.5)
    assert p["total_depth"] == pytest.approx(14.0)
    assert p["through"] is True
    assert not of_type(feats, "BLIND_HOLE") and not of_type(feats, "THROUGH_HOLE")


def test_blind_counterbore(tmp_path):
    part = cut(cut(box(40, 40, 20), cyl(20, 20, 8, 5, 20)), cyl(20, 20, 15, 9, 10))
    [cb] = of_type(features_of(part, tmp_path), "COUNTERBORE")
    assert cb.parameters["through"] is False
    assert cb.parameters["total_depth"] == pytest.approx(12.0)


def test_boss_with_a_flat_top_is_not_a_hole(tmp_path):
    part = fuse(box(60, 60, 8), cyl(30, 30, 8, 20, 12))
    feats = features_of(part, tmp_path)
    [boss] = of_type(feats, "BOSS")
    assert boss.parameters["diameter"] == pytest.approx(20.0)
    assert boss.parameters["height"] == pytest.approx(12.0)
    assert not of_type(feats, "BLIND_HOLE") and not of_type(feats, "THROUGH_HOLE")


def test_machined_plate_reads_as_designed(tmp_path):
    # Everything at once: counterbored corners, a hole row, a bored boss in a pocket.
    plate = box(140, 90, 14)
    plate = cut(plate, box(60, 46, 20, 40, 22, 6))
    plate = fuse(plate, cyl(70, 45, 6, 22, 12))
    plate = cut(plate, cyl(70, 45, -1, 8.5, 30))
    for x, y in ((12, 12), (128, 12), (12, 78), (128, 78)):
        plate = cut(cut(plate, cyl(x, y, -1, 6.6, 16)), cyl(x, y, 7.5, 11, 10))
    for x in (35, 52.5, 70, 87.5, 105):
        plate = cut(plate, cyl(x, 82, -1, 4.2, 16))
    feats = features_of(plate, tmp_path)
    counts = Counter(f.feature_type for f in feats if f.feature_type != "PATTERN")
    assert counts["COUNTERBORE"] == 4
    assert counts["THROUGH_HOLE"] == 6        # five in the row + the bore through the boss
    assert counts["BOSS"] == 1
    assert counts["BLIND_HOLE"] == 0
    patterns = {(p.parameters["child_type"], p.parameters["count"]) for p in of_type(feats, "PATTERN")}
    assert ("COUNTERBORE", 4) in patterns and ("THROUGH_HOLE", 5) in patterns


# ── Pockets, fillets, and the look-alikes that must not trigger them ────────────

def _rounded_vertical_edges(shape, r):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.GeomAbs import GeomAbs_Line
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    f = BRepFilletAPI_MakeFillet(shape)
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        e = TopoDS.Edge_s(ex.Current())
        c = BRepAdaptor_Curve(e)
        if c.GetType() == GeomAbs_Line and abs(abs(c.Line().Direction().Z()) - 1) < 1e-9:
            f.Add(r, e)
        ex.Next()
    return f.Shape()


def test_milled_pocket_with_depth(tmp_path):
    part = cut(box(80, 60, 20), box(40, 30, 10, 20, 15, 10))
    [pocket] = of_type(features_of(part, tmp_path), "POCKET")
    assert len(pocket.parameters["wall_face_ids"]) == 4
    assert pocket.parameters["depth"] == pytest.approx(10.0)


def test_hole_through_a_pocket_floor_keeps_the_pocket(tmp_path):
    part = cut(cut(box(80, 60, 20), box(40, 30, 10, 20, 15, 10)), cyl(40, 30, -1, 6, 30))
    feats = features_of(part, tmp_path)
    assert len(of_type(feats, "POCKET")) == 1
    assert len(of_type(feats, "THROUGH_HOLE")) == 1


@pytest.mark.parametrize("name,shape", [
    ("blind hole", lambda: cut(box(40, 40, 20), cyl(20, 20, 14, 8, 10))),
    ("raised block", lambda: fuse(box(80, 60, 10), box(20, 20, 10, 30, 20, 10))),
    ("plain box", lambda: box(40, 30, 20)),
])
def test_no_phantom_pockets(tmp_path, name, shape):
    # Each of these used to report a pocket — sometimes the whole part as one.
    assert not of_type(features_of(shape(), tmp_path), "POCKET"), name


def test_rounded_outside_edges_are_fillets(tmp_path):
    fillets = of_type(features_of(_rounded_vertical_edges(box(60, 40, 10), 5.0), tmp_path), "FILLET")
    assert len(fillets) == 4
    assert all(f.parameters["radius"] == pytest.approx(5.0) and f.parameters["convex"] for f in fillets)


def test_rounded_pocket_corners_are_inside_fillets(tmp_path):
    pocket = _rounded_vertical_edges(box(40, 30, 20, 20, 15, 10), 4.0)
    feats = features_of(cut(box(80, 60, 20), pocket), tmp_path)
    fillets = of_type(feats, "FILLET")
    assert len(fillets) == 4
    assert all(f.parameters["radius"] == pytest.approx(4.0) and not f.parameters["convex"] for f in fillets)
    [p] = of_type(feats, "POCKET")
    assert p.parameters["depth"] == pytest.approx(10.0)


def test_slot_ends_are_not_fillets(tmp_path):
    # A slot's round ends sweep a half-turn; only quarter-turn-ish blends are fillets.
    stadium = fuse(fuse(box(30, 10, 30, 25, 25, -5), cyl(25, 30, -5, 10, 30)), cyl(55, 30, -5, 10, 30))
    feats = features_of(cut(box(80, 60, 15), stadium), tmp_path)
    assert not of_type(feats, "FILLET")


def test_chamfered_edge_is_a_chamfer(tmp_path):
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    b = box(40, 30, 20)
    ch = BRepFilletAPI_MakeChamfer(b)
    ex = TopExp_Explorer(b, TopAbs_EDGE)
    ch.Add(2.0, TopoDS.Edge_s(ex.Current()))
    feats = features_of(ch.Shape(), tmp_path)
    assert len(of_type(feats, "CHAMFER")) == 1
    assert not of_type(feats, "POCKET")
