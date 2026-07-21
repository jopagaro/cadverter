"""Tessellation and point-cloud sampling.

Turns any loaded shape — B-REP *or* mesh — into a triangle mesh and, from
that, a sampled point cloud. This is the representation that neural pipelines
(PointNet, 3D CNNs) consume, complementing the analytical text/graph outputs.

B-REP shapes are tessellated on demand via OpenCASCADE; mesh files (STL/OBJ)
already carry a triangulation and are read directly.
"""

from __future__ import annotations

import numpy as np


def tessellate(shape, deflection: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate a shape into (vertices Nx3, triangles Mx3).

    ``deflection`` is the max chord deviation (mm) — smaller means a finer,
    more faithful mesh. Safe to call on already-triangulated mesh shapes.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.TopExp import TopExp_Explorer

    BRepMesh_IncrementalMesh(shape, deflection)

    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = face.Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                a, b, c = t.Get()
                # OCC triangle node indices are 1-based within this face
                tris.append((base + a - 1, base + b - 1, base + c - 1))
        exp.Next()

    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.asarray(verts, dtype=float), np.asarray(tris, dtype=int)


def sample_points(
    shape,
    count: int = 2048,
    *,
    deflection: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Sample ``count`` points over the surface, returned as an (N, 3) array.

    Uses area-weighted surface sampling via trimesh when available (the
    correct choice for ML point clouds); otherwise falls back to the raw
    tessellation vertices.
    """
    verts, tris = tessellate(shape, deflection=deflection)
    if len(verts) == 0:
        return np.zeros((0, 3))

    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
        return np.asarray(pts, dtype=float)
    except ImportError:
        # Fallback: deterministic subsample/pad of tessellation vertices.
        rng = np.random.default_rng(seed)
        if len(verts) >= count:
            idx = rng.choice(len(verts), size=count, replace=False)
        else:
            idx = rng.choice(len(verts), size=count, replace=True)
        return verts[idx]
