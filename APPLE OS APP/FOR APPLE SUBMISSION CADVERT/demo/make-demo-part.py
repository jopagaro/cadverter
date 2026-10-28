"""Build the demo part used for App Store screenshots: a machined mounting plate.

Designed here from scratch, so it is ours to show anywhere — never use a downloaded model
in marketing. Run with any Python that has cadquery-ocp (the repo's .venv does):

    .venv/bin/python "APPLE OS APP/FOR APPLE SUBMISSION CADVERT/demo/make-demo-part.py"

Writes mounting-plate.step next to this script.
"""
from pathlib import Path

from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.GeomAbs import GeomAbs_Line
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

L, W, T = 140.0, 90.0, 14.0           # plate
Z = gp_Dir(0, 0, 1)


def box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def cyl(x, y, z, d, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), Z), d / 2, h).Shape()


def cut(a, b):
    return BRepAlgoAPI_Cut(a, b).Shape()


def fuse(a, b):
    return BRepAlgoAPI_Fuse(a, b).Shape()


def fillet_vertical_edges(shape, r):
    f = BRepFilletAPI_MakeFillet(shape)
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        e = TopoDS.Edge_s(ex.Current())
        c = BRepAdaptor_Curve(e)
        if c.GetType() == GeomAbs_Line and abs(abs(c.Line().Direction().Z()) - 1) < 1e-9:
            f.Add(r, e)
        ex.Next()
    return f.Shape()


part = fillet_vertical_edges(box(0, 0, 0, L, W, T), 8.0)                 # rounded plate corners

pocket = fillet_vertical_edges(box(40, 22, 6, 60, 46, 20), 6.0)          # 8 mm deep, R6 corners
part = cut(part, pocket)

part = fuse(part, cyl(70, 45, 6, 22, 12))                                # boss rising from the pocket floor
part = cut(part, cyl(70, 45, -1, 8.5, 30))                               # bore through the boss

for x, y in ((12, 12), (128, 12), (12, 78), (128, 78)):                  # M6 counterbored corners
    part = cut(part, cyl(x, y, -1, 6.6, T + 2))
    part = cut(part, cyl(x, y, T - 6.5, 11.0, 10))

for x in (35, 52.5, 70, 87.5, 105):                                      # row of M5 tapping holes
    part = cut(part, cyl(x, 82, -1, 4.2, T + 2))

out = Path(__file__).with_name("mounting-plate.step")
w = STEPControl_Writer()
w.Transfer(part, STEPControl_AsIs)
assert w.Write(str(out)) == IFSelect_RetDone
print("wrote", out)
