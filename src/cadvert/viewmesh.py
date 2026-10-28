"""Viewer mesh: the part as triangles an app can draw, each one tagged with its B-REP face.

An interactive 3D view needs triangles, but CADVERT's view also needs to know which face a
triangle came from, so a click lands on "F212 · hole_7" rather than an anonymous polygon.
Faces are numbered exactly as topology.py numbers them (TopExp map order, 1-based), and each
face's triangles are contiguous, so a picked triangle maps back to its face with a binary
search over the face ranges.

Binary layout (little-endian), version 1:

    b"CVM1"                 magic
    uint32                  header length H
    H bytes                 JSON header, space-padded to a multiple of 4
    float32[3V]             vertex positions
    float32[3V]             vertex normals (smooth within a face, sharp across edges)
    uint32[3T]              triangle vertex indices
    uint32[2F]              per face: first triangle, triangle count (face ID = index + 1)
    float32[3P]             edge points
    uint32[2S]              edge segments, as pairs of edge-point indices

The header carries the counts (V, T, F, P, S), the bounding box and the units.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"CVM1"


def build_view_mesh(shape, graph=None, units: str = "mm") -> bytes:
    """Tessellate *shape* for display and pack it in the format described above.

    *graph* (a TopologyGraph) is optional; with it, seam edges — the line where a cylinder's
    surface wraps onto itself — are left out, as every CAD viewer does.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS

    lo, hi = _bounds(shape)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    # Fine enough that small holes read as round on a large assembly, coarse enough to stay
    # quick: the angular limit does most of the work on curved faces.
    linear = max(diag * 0.0008, 1e-4)
    BRepMesh_IncrementalMesh(shape, linear, False, 0.3, True)

    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, faces)

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    ranges = np.zeros((faces.Extent(), 2), dtype=np.uint32)
    vertex_base = tri_base = 0

    for i in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(i))
        pts, tris = _face_triangles(face, reversed_=face.Orientation() == TopAbs_REVERSED)
        ranges[i - 1] = (tri_base, len(tris))
        if not len(tris):
            continue
        positions.append(pts)
        normals.append(_vertex_normals(pts, tris))
        triangles.append(tris + vertex_base)
        vertex_base += len(pts)
        tri_base += len(tris)

    edge_points, edge_segments = _edges(shape, graph, linear)

    pos = np.concatenate(positions).astype(np.float32) if positions else np.zeros((0, 3), np.float32)
    nrm = np.concatenate(normals).astype(np.float32) if normals else np.zeros((0, 3), np.float32)
    idx = np.concatenate(triangles).astype(np.uint32) if triangles else np.zeros((0, 3), np.uint32)

    header = {
        "version": 1,
        "units": units,
        "vertices": int(len(pos)),
        "triangles": int(len(idx)),
        "faces": int(len(ranges)),
        "edge_points": int(len(edge_points)),
        "edge_segments": int(len(edge_segments)),
        "bbox": [float(v) for v in (*lo, *hi)],
    }
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 4)
    return b"".join([
        MAGIC, struct.pack("<I", len(head)), head,
        pos.tobytes(), nrm.tobytes(), idx.tobytes(), ranges.tobytes(),
        edge_points.astype(np.float32).tobytes(), edge_segments.astype(np.uint32).tobytes(),
    ])


def read_header(data: bytes) -> dict:
    """The JSON header of a packed mesh (for tests and debugging)."""
    if data[:4] != MAGIC:
        raise ValueError("not a CADVERT view mesh")
    (n,) = struct.unpack_from("<I", data, 4)
    return json.loads(data[8:8 + n])


# ── Faces ────────────────────────────────────────────────────────────────────

def _face_triangles(face, reversed_: bool) -> tuple[np.ndarray, np.ndarray]:
    """World-space nodes and 0-based triangles of one meshed face, wound outward."""
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbTriangles() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    pts = np.array([tri.Node(k).Coord() for k in range(1, tri.NbNodes() + 1)], dtype=np.float64)
    tris = np.array([tri.Triangle(k).Get() for k in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
    if reversed_:
        tris = tris[:, [0, 2, 1]]
    if not loc.IsIdentity():
        pts = _apply(loc.Transformation(), pts)
    return pts, tris


def _vertex_normals(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals within one face, so curved faces shade smoothly."""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    length = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.divide(vn, length, out=np.zeros_like(vn), where=length > 0)


def _apply(trsf, pts: np.ndarray) -> np.ndarray:
    m = np.array([[trsf.Value(r, c) for c in range(1, 5)] for r in range(1, 4)])
    return pts @ m[:, :3].T + m[:, 3]


def _bounds(shape) -> tuple[np.ndarray, np.ndarray]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return np.zeros(3), np.ones(3)
    x0, y0, z0, x1, y1, z1 = box.Get()
    return np.array([x0, y0, z0]), np.array([x1, y1, z1])


# ── Edges ────────────────────────────────────────────────────────────────────

def _edges(shape, graph, deflection: float) -> tuple[np.ndarray, np.ndarray]:
    """Polylines for the part's visible edges, as a point list and index pairs."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_TangentialDeflection
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS

    seams: set[int] = set()
    if graph is not None:
        seams = {e.id for e in graph.edges if len(set(e.face_ids)) < 2}

    edges = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edges)

    points: list[tuple[float, float, float]] = []
    segments: list[tuple[int, int]] = []
    for i in range(1, edges.Extent() + 1):
        if i in seams:
            continue
        edge = TopoDS.Edge_s(edges.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        try:
            curve = BRepAdaptor_Curve(edge)
            sampler = GCPnts_TangentialDeflection(curve, 0.3, deflection, 2)
        except Exception:
            continue
        n = sampler.NbPoints()
        if n < 2:
            continue
        base = len(points)
        points.extend(sampler.Value(k).Coord() for k in range(1, n + 1))
        segments.extend((base + k, base + k + 1) for k in range(n - 1))

    if not points:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.uint32)
    return np.array(points, dtype=np.float64), np.array(segments, dtype=np.uint32)


# ── Face info for a click ────────────────────────────────────────────────────

_KIND = {
    "PLANE": "Flat face", "CYLINDER": "Cylindrical face", "CONE": "Conical face",
    "SPHERE": "Spherical face", "TORUS": "Toroidal face", "NURBS_SURFACE": "Freeform face",
    "BSPLINE_SURFACE": "Freeform face", "BEZIER_SURFACE": "Freeform face",
}


def face_summary(face_id: int, graph, features=None, feature_ids=None, assembly=None,
                 units: str = "mm") -> dict | None:
    """A short, human description of one face and what it belongs to — what the app shows
    when someone clicks it. None if the face doesn't exist."""
    from .utils import fmt

    face = next((f for f in graph.faces if f.id == face_id), None) if graph else None
    if face is None:
        return None
    g = face.geometry or {}
    kind = g.get("type", "")
    out: dict = {"id": f"F{face_id}", "kind": _KIND.get(kind, "Face")}

    if kind == "CYLINDER" and g.get("radius"):
        out["detail"] = f"∅{fmt(2 * g['radius'])} {units}"
    elif kind == "CONE" and g.get("half_angle") is not None:
        out["detail"] = f"{fmt(2 * g['half_angle'])}° included"
    elif kind == "SPHERE" and g.get("radius"):
        out["detail"] = f"R{fmt(g['radius'])} {units}"
    elif kind == "TORUS" and g.get("minor_radius"):
        out["detail"] = f"R{fmt(g['minor_radius'])} {units}"
    elif face.area:
        out["detail"] = f"{fmt(face.area)} {units}²"

    for feat, fid in zip(features or [], feature_ids or []):
        if face_id in feat.face_ids:
            out["feature"] = fid
            out["feature_detail"] = _feature_detail(feat, units)
            out["feature_faces"] = sorted(set(feat.face_ids))
            break

    if assembly is not None:
        owner = assembly.owner_of_faces([face_id])
        if owner is not None:
            out["component"] = owner.name
    return out


def _feature_detail(feat, units: str) -> str:
    from .utils import fmt

    p, t = feat.parameters, feat.feature_type
    if t in ("THROUGH_HOLE", "BLIND_HOLE"):
        kind = "Through hole" if t == "THROUGH_HOLE" else "Blind hole"
        depth = f", {fmt(p['depth'])} {units} deep" if p.get("depth") else ""
        std = f" ({feat.standard_match})" if getattr(feat, "standard_match", None) else ""
        return f"{kind} ∅{fmt(p['diameter'])} {units}{depth}{std}"
    if t == "COUNTERBORE":
        return f"Counterbore ∅{fmt(p['bore_diameter'])} / ∅{fmt(p['cbore_diameter'])} {units}"
    if t == "COUNTERSINK":
        return f"Countersink {fmt(p['cone_half_angle'] * 2)}°, ∅{fmt(p['bore_diameter'])} {units}"
    if t == "BOSS":
        return f"Boss ∅{fmt(p['diameter'])} {units}, {fmt(p['height'])} {units} high"
    if t == "FILLET":
        return f"Fillet R{fmt(p['radius'])} {units}"
    if t == "CHAMFER":
        return f"Chamfer {fmt(p['width'])} {units}" if p.get("width") else "Chamfer"
    if t == "POCKET":
        return f"Pocket, {fmt(p['depth'])} {units} deep" if p.get("depth") is not None else "Pocket"
    if t == "SLOT":
        return f"Slot {fmt(p['width'])} × {fmt(p['length'])} {units}"
    if t == "PATTERN":
        return f"Pattern of {p['count']}"
    return t.replace("_", " ").capitalize()
