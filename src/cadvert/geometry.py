"""Extract exact analytical geometry from OCC face and edge adapters.

Returns plain Python dicts so the rest of the pipeline has no OCC dependency.
All coordinates are in the STEP file's native units (typically mm).
"""

from __future__ import annotations
import math
from typing import Any

from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_BSplineCurve,
    GeomAbs_BezierCurve,
)
from OCP.TopoDS import TopoDS_Face, TopoDS_Edge
from OCP.BRep import BRep_Tool
from OCP.gp import gp_Pnt


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

def extract_face_geometry(face: TopoDS_Face) -> dict[str, Any]:
    """Return a dict describing the analytical geometry of *face*."""
    adaptor = BRepAdaptor_Surface(face, True)
    stype = adaptor.GetType()

    if stype == GeomAbs_Plane:
        return _plane(adaptor)
    elif stype == GeomAbs_Cylinder:
        return _cylinder(adaptor)
    elif stype == GeomAbs_Cone:
        return _cone(adaptor)
    elif stype == GeomAbs_Sphere:
        return _sphere(adaptor)
    elif stype == GeomAbs_Torus:
        return _torus(adaptor)
    elif stype in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
        return _bspline_surface(adaptor, stype)
    else:
        return {"type": "OTHER", "raw_type": str(stype)}


def _plane(adaptor: BRepAdaptor_Surface) -> dict:
    pln = adaptor.Plane()
    ax = pln.Axis()
    loc = ax.Location()
    direction = ax.Direction()
    normal = (direction.X(), direction.Y(), direction.Z())
    origin = (loc.X(), loc.Y(), loc.Z())
    # Plane equation: n·p = d  where d = n·origin
    d = sum(normal[i] * origin[i] for i in range(3))
    return {
        "type": "PLANE",
        "normal": normal,
        "origin": origin,
        "d": d,  # offset: n·x = d
    }


def _cylinder(adaptor: BRepAdaptor_Surface) -> dict:
    cyl = adaptor.Cylinder()
    ax = cyl.Axis()
    loc = ax.Location()
    direction = ax.Direction()
    return {
        "type": "CYLINDER",
        "axis_origin": (loc.X(), loc.Y(), loc.Z()),
        "axis_dir": (direction.X(), direction.Y(), direction.Z()),
        "radius": cyl.Radius(),
    }


def _cone(adaptor: BRepAdaptor_Surface) -> dict:
    cone = adaptor.Cone()
    ax = cone.Axis()
    loc = ax.Location()
    direction = ax.Direction()
    apex = cone.Apex()
    return {
        "type": "CONE",
        "axis_origin": (loc.X(), loc.Y(), loc.Z()),
        "axis_dir": (direction.X(), direction.Y(), direction.Z()),
        "apex": (apex.X(), apex.Y(), apex.Z()),
        "half_angle": math.degrees(cone.SemiAngle()),
        "radius_at_origin": cone.RefRadius(),
    }


def _sphere(adaptor: BRepAdaptor_Surface) -> dict:
    sph = adaptor.Sphere()
    center = sph.Location()
    return {
        "type": "SPHERE",
        "center": (center.X(), center.Y(), center.Z()),
        "radius": sph.Radius(),
    }


def _torus(adaptor: BRepAdaptor_Surface) -> dict:
    tor = adaptor.Torus()
    ax = tor.Axis()
    loc = ax.Location()
    direction = ax.Direction()
    return {
        "type": "TORUS",
        "center": (loc.X(), loc.Y(), loc.Z()),
        "axis_dir": (direction.X(), direction.Y(), direction.Z()),
        "major_radius": tor.MajorRadius(),
        "minor_radius": tor.MinorRadius(),
    }


def _bspline_surface(adaptor: BRepAdaptor_Surface, stype) -> dict:
    """Extract NURBS/BSpline surface control points and knots."""
    try:
        surf = adaptor.BSpline()
        nu = surf.NbUPoles()
        nv = surf.NbVPoles()
        control_points = []
        for i in range(1, nu + 1):
            row = []
            for j in range(1, nv + 1):
                p = surf.Pole(i, j)
                w = surf.Weight(i, j)
                row.append((p.X(), p.Y(), p.Z(), w))
            control_points.append(row)

        u_knots = [surf.UKnot(i) for i in range(1, surf.NbUKnots() + 1)]
        v_knots = [surf.VKnot(i) for i in range(1, surf.NbVKnots() + 1)]
        u_mults = [surf.UMultiplicity(i) for i in range(1, surf.NbUKnots() + 1)]
        v_mults = [surf.VMultiplicity(i) for i in range(1, surf.NbVKnots() + 1)]

        # Expand knot vector (repeat by multiplicity)
        u_knot_vec = _expand_knots(u_knots, u_mults)
        v_knot_vec = _expand_knots(v_knots, v_mults)

        return {
            "type": "NURBS_SURFACE",
            "degree_u": surf.UDegree(),
            "degree_v": surf.VDegree(),
            "n_poles_u": nu,
            "n_poles_v": nv,
            "control_points": control_points,  # [u][v] = (x,y,z,w)
            "knot_vector_u": u_knot_vec,
            "knot_vector_v": v_knot_vec,
            "is_rational": surf.IsURational() or surf.IsVRational(),
        }
    except Exception as e:
        return {"type": "NURBS_SURFACE", "error": str(e)}


def _expand_knots(knots: list[float], mults: list[int]) -> list[float]:
    result = []
    for k, m in zip(knots, mults):
        result.extend([k] * m)
    return result


# ---------------------------------------------------------------------------
# Curve / edge extraction
# ---------------------------------------------------------------------------

def extract_edge_geometry(edge: TopoDS_Edge) -> dict[str, Any]:
    """Return a dict describing the edge curve geometry."""
    adaptor = BRepAdaptor_Curve(edge)
    ctype = adaptor.GetType()

    if ctype == GeomAbs_Line:
        return _line(adaptor, edge)
    elif ctype == GeomAbs_Circle:
        return _circle(adaptor)
    elif ctype == GeomAbs_Ellipse:
        return _ellipse(adaptor)
    elif ctype in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
        return _bspline_curve(adaptor)
    else:
        return {"type": "OTHER_CURVE", "raw_type": str(ctype)}


def _line(adaptor: BRepAdaptor_Curve, edge: TopoDS_Edge) -> dict:
    line = adaptor.Line()
    loc = line.Location()
    direction = line.Direction()

    f = adaptor.FirstParameter()
    t = adaptor.LastParameter()
    length = abs(t - f)  # for a line adaptor, param = arc length

    p_start = gp_Pnt()
    p_end = gp_Pnt()
    adaptor.D0(f, p_start)
    adaptor.D0(t, p_end)

    return {
        "type": "LINE",
        "start": (p_start.X(), p_start.Y(), p_start.Z()),
        "end": (p_end.X(), p_end.Y(), p_end.Z()),
        "direction": (direction.X(), direction.Y(), direction.Z()),
        "length": length,
    }


def _circle(adaptor: BRepAdaptor_Curve) -> dict:
    circ = adaptor.Circle()
    center = circ.Location()
    normal = circ.Axis().Direction()

    f = adaptor.FirstParameter()
    t = adaptor.LastParameter()
    arc_length = circ.Radius() * abs(t - f)
    is_full = abs(abs(t - f) - 2 * math.pi) < 1e-6

    return {
        "type": "CIRCLE",
        "center": (center.X(), center.Y(), center.Z()),
        "normal": (normal.X(), normal.Y(), normal.Z()),
        "radius": circ.Radius(),
        "arc_length": arc_length,
        "is_full_circle": is_full,
    }


def _ellipse(adaptor: BRepAdaptor_Curve) -> dict:
    ell = adaptor.Ellipse()
    center = ell.Location()
    normal = ell.Axis().Direction()
    return {
        "type": "ELLIPSE",
        "center": (center.X(), center.Y(), center.Z()),
        "normal": (normal.X(), normal.Y(), normal.Z()),
        "major_radius": ell.MajorRadius(),
        "minor_radius": ell.MinorRadius(),
    }


def _bspline_curve(adaptor: BRepAdaptor_Curve) -> dict:
    try:
        curve = adaptor.BSpline()
        poles = [(curve.Pole(i).X(), curve.Pole(i).Y(), curve.Pole(i).Z())
                 for i in range(1, curve.NbPoles() + 1)]
        weights = [curve.Weight(i) for i in range(1, curve.NbPoles() + 1)]
        knots = [curve.Knot(i) for i in range(1, curve.NbKnots() + 1)]
        mults = [curve.Multiplicity(i) for i in range(1, curve.NbKnots() + 1)]
        knot_vec = _expand_knots(knots, mults)

        return {
            "type": "NURBS_CURVE",
            "degree": curve.Degree(),
            "control_points": poles,
            "weights": weights,
            "knot_vector": knot_vec,
            "is_rational": curve.IsRational(),
        }
    except Exception as e:
        return {"type": "NURBS_CURVE", "error": str(e)}
