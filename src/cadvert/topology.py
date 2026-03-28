"""B-REP topology walker.

Enumerates all faces, edges, and vertices from an OCC shape using OCC's
own IndexedMapOfShape for correct deduplication.

Builds a connectivity graph: which edges bound which faces, and for each
edge which two faces it separates (with dihedral angle + convexity).

Returns plain dataclasses — no OCC objects escape this module after build().
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopAbs import (
    TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
    TopAbs_REVERSED,
)
from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex, TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.gp import gp_Pnt
from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_IndexedDataMapOfShapeListOfShape

from .geometry import extract_face_geometry, extract_edge_geometry


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VertexInfo:
    id: int
    point: tuple[float, float, float]


@dataclass
class EdgeInfo:
    id: int
    geometry: dict[str, Any]
    vertex_ids: list[int]
    face_ids: list[int]          # exactly 2 for manifold solids
    dihedral_angle: float | None = None
    convexity: str = "unknown"   # "convex", "concave", "tangent"


@dataclass
class FaceInfo:
    id: int
    geometry: dict[str, Any]
    edge_ids: list[int]
    area: float


@dataclass
class TopologyGraph:
    faces: list[FaceInfo]
    edges: list[EdgeInfo]
    vertices: list[VertexInfo]

    bounding_box: dict[str, tuple[float, float]]
    volume: float
    surface_area: float
    center_of_mass: tuple[float, float, float]
    body_count: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_topology(shape: TopoDS_Shape, body_count: int) -> TopologyGraph:
    """Walk the B-REP and return a fully populated TopologyGraph."""

    # Build indexed maps (OCC handles identity/deduplication correctly)
    face_imap = TopTools_IndexedMapOfShape()
    edge_imap = TopTools_IndexedMapOfShape()
    vtx_imap = TopTools_IndexedMapOfShape()

    TopExp.MapShapes_s(shape, TopAbs_FACE, face_imap)
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edge_imap)
    TopExp.MapShapes_s(shape, TopAbs_VERTEX, vtx_imap)

    # Build face→edges and edge→faces adjacency using OCC's data map
    # TopExp.MapShapesAndAncestors gives us edge → [faces] map
    edge_to_faces_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces_map)

    vtx_to_edges_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_VERTEX, TopAbs_EDGE, vtx_to_edges_map)

    # face_id → [edge_id]  (built by inverting edge_imap → face_imap)
    face_to_edge_ids: dict[int, list[int]] = {i: [] for i in range(1, face_imap.Extent() + 1)}
    edge_to_face_ids: dict[int, list[int]] = {}
    edge_to_vertex_ids: dict[int, list[int]] = {}

    for eid in range(1, edge_imap.Extent() + 1):
        occ_edge = TopoDS.Edge_s(edge_imap.FindKey(eid))

        # Faces adjacent to this edge
        if edge_to_faces_map.Contains(occ_edge):
            face_list = edge_to_faces_map.FindFromKey(occ_edge)
            adjacent_fids = []
            for adj_face in face_list:
                fid = face_imap.FindIndex(adj_face)
                if fid > 0:
                    adjacent_fids.append(fid)
                    if eid not in face_to_edge_ids[fid]:
                        face_to_edge_ids[fid].append(eid)
            edge_to_face_ids[eid] = adjacent_fids
        else:
            edge_to_face_ids[eid] = []

        # Vertices of this edge
        v_imap_local = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(occ_edge, TopAbs_VERTEX, v_imap_local)
        vids = []
        for vi in range(1, v_imap_local.Extent() + 1):
            vtx = v_imap_local.FindKey(vi)
            vid = vtx_imap.FindIndex(vtx)
            if vid > 0:
                vids.append(vid)
        edge_to_vertex_ids[eid] = vids

    # Build vertex info
    vertex_infos = []
    for vid in range(1, vtx_imap.Extent() + 1):
        occ_vtx = TopoDS.Vertex_s(vtx_imap.FindKey(vid))
        pt = BRep_Tool.Pnt_s(occ_vtx)
        vertex_infos.append(VertexInfo(id=vid, point=(pt.X(), pt.Y(), pt.Z())))

    # Build face info
    face_infos = []
    for fid in range(1, face_imap.Extent() + 1):
        occ_face = TopoDS.Face_s(face_imap.FindKey(fid))
        geom = extract_face_geometry(occ_face)
        area = _face_area(occ_face)
        face_infos.append(FaceInfo(
            id=fid,
            geometry=geom,
            edge_ids=sorted(face_to_edge_ids[fid]),
            area=area,
        ))

    # Build edge info
    edge_infos = []
    for eid in range(1, edge_imap.Extent() + 1):
        occ_edge = TopoDS.Edge_s(edge_imap.FindKey(eid))
        geom = extract_edge_geometry(occ_edge)
        face_ids = edge_to_face_ids[eid]

        dihedral, convexity = None, "unknown"
        if len(face_ids) == 2:
            f1 = TopoDS.Face_s(face_imap.FindKey(face_ids[0]))
            f2 = TopoDS.Face_s(face_imap.FindKey(face_ids[1]))
            dihedral, convexity = _dihedral_angle(occ_edge, f1, f2)

        edge_infos.append(EdgeInfo(
            id=eid,
            geometry=geom,
            vertex_ids=edge_to_vertex_ids[eid],
            face_ids=face_ids,
            dihedral_angle=dihedral,
            convexity=convexity,
        ))

    bbox = _bounding_box(shape)
    volume, surface_area, com = _mass_properties(shape)

    return TopologyGraph(
        faces=face_infos,
        edges=edge_infos,
        vertices=vertex_infos,
        bounding_box=bbox,
        volume=volume,
        surface_area=surface_area,
        center_of_mass=com,
        body_count=body_count,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _face_area(face: TopoDS_Face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def _bounding_box(shape: TopoDS_Shape) -> dict[str, tuple[float, float]]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return {"X": (xmin, xmax), "Y": (ymin, ymax), "Z": (zmin, zmax)}


def _mass_properties(shape: TopoDS_Shape) -> tuple[float, float, tuple]:
    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol_props)
    volume = vol_props.Mass()
    com = vol_props.CentreOfMass()

    surf_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf_props)
    surface_area = surf_props.Mass()

    return volume, surface_area, (com.X(), com.Y(), com.Z())


def _dihedral_angle(
    edge: TopoDS_Edge,
    face1: TopoDS_Face,
    face2: TopoDS_Face,
) -> tuple[float | None, str]:
    """Compute the dihedral angle between two faces at their shared edge."""
    try:
        adaptor = BRepAdaptor_Curve(edge)
        f_param = adaptor.FirstParameter()
        t_param = adaptor.LastParameter()
        mid_param = (f_param + t_param) * 0.5

        mid_pt = gp_Pnt()
        adaptor.D0(mid_param, mid_pt)

        n1 = _face_normal_at(face1, mid_pt)
        n2 = _face_normal_at(face2, mid_pt)

        if n1 is None or n2 is None:
            return None, "unknown"

        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(n1, n2))))
        angle = math.degrees(math.acos(dot))

        if angle < 1.0:
            return round(angle, 3), "tangent"

        # Convexity: cross n1 with edge tangent; dot with n2
        d1 = adaptor.DN(mid_param, 1)
        tang = (d1.X(), d1.Y(), d1.Z())
        cross = (
            n1[1] * tang[2] - n1[2] * tang[1],
            n1[2] * tang[0] - n1[0] * tang[2],
            n1[0] * tang[1] - n1[1] * tang[0],
        )
        dot_check = sum(cross[i] * n2[i] for i in range(3))
        convexity = "concave" if dot_check > 0 else "convex"

        return round(angle, 3), convexity
    except Exception:
        return None, "unknown"


def _face_normal_at(face: TopoDS_Face, pt: gp_Pnt) -> tuple | None:
    """Evaluate the outward surface normal at the point on *face* nearest to *pt*."""
    try:
        from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCP.BRepLProp import BRepLProp_SLProps

        surf_handle = BRep_Tool.Surface_s(face)
        proj = GeomAPI_ProjectPointOnSurf(pt, surf_handle)
        if proj.NbPoints() == 0:
            return None

        u, v = proj.LowerDistanceParameters()
        adaptor = BRepAdaptor_Surface(face, True)
        props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
        if not props.IsNormalDefined():
            return None

        n = props.Normal()
        if face.Orientation() == TopAbs_REVERSED:
            return (-n.X(), -n.Y(), -n.Z())
        return (n.X(), n.Y(), n.Z())
    except Exception:
        return None
