"""Export: the part as a file other programs open.

Meshes for 3D printing and the web (STL, 3MF, OBJ, GLB), a STEP file for solids that came
in as IGES or BREP, and the part's tables as CSV (the bill of materials of an assembly, and
every detected feature).

The mesh writers need nothing beyond numpy, so the core library stays light. Coordinates
are millimetres throughout: OpenCascade converts every file to millimetres when it reads it,
which is also what slicers expect. GLB is the one exception — glTF is defined in metres
with Y up, so that file is scaled and turned to match.

STL, 3MF and OBJ are welded: a vertex shared by two faces is written once, so a slicer sees
one closed surface rather than a pile of loose patches. GLB keeps each face's own vertices,
so the edges between faces stay sharp when it is lit.
"""

from __future__ import annotations

import csv
import io
import json
import struct
import zipfile
from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

import numpy as np

# kind → (file extension, media type)
FORMATS = {
    "stl":          (".stl", "model/stl"),
    "3mf":          (".3mf", "model/3mf"),
    "obj":          (".obj", "model/obj"),
    "glb":          (".glb", "model/gltf-binary"),
    "step":         (".step", "model/step"),
    "bom.csv":      (".csv", "text/csv"),
    "features.csv": (".csv", "text/csv"),
}

MESH_KINDS = ("stl", "3mf", "obj", "glb")


class ExportError(ValueError):
    """The part can't be written in the format asked for (e.g. a BOM of a single part)."""


@dataclass
class Body:
    """One piece of the export — the whole part, or one component of an assembly."""

    name: str
    positions: np.ndarray       # (V, 3) float64, millimetres
    triangles: np.ndarray       # (T, 3) int64, wound outward
    normals: np.ndarray         # (V, 3) float64, smooth within a face


# ── Entry point ──────────────────────────────────────────────────────────────

def export_part(kind: str, shape, *, name: str = "Part", graph=None, features=None,
                feature_ids=None, assembly=None, is_mesh: bool = False) -> bytes:
    """The part written as *kind* (a key of FORMATS)."""
    if kind not in FORMATS:
        raise ExportError(f"Unknown export format {kind!r}")
    if kind == "bom.csv":
        return bom_csv(assembly)
    if kind == "features.csv":
        return features_csv(features, feature_ids, graph, assembly)
    if kind == "step":
        if is_mesh:
            raise ExportError("A mesh has no solid geometry to write as STEP")
        return step_bytes(shape)

    bodies = tessellate(shape, name=name, assembly=assembly)
    if not bodies:
        raise ExportError("The part has no surfaces to export")
    if kind == "glb":
        return glb_bytes(bodies)
    welded = [weld(b) for b in bodies]
    if kind == "stl":
        return stl_bytes(welded, name)
    if kind == "3mf":
        return threemf_bytes(welded)
    return obj_bytes(welded)


# ── Tessellation ─────────────────────────────────────────────────────────────

def tessellate(shape, *, name: str = "Part", assembly=None) -> list[Body]:
    """Triangles fine enough to print, one Body per component (or one for a single part).

    Meshes a copy, so the session's own shape keeps the coarser triangles the 3D view uses.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS

    from .viewmesh import _bounds, _face_triangles, _vertex_normals

    copy = BRepBuilderAPI_Copy(shape, True, False).Shape()
    lo, hi = _bounds(copy)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    # About 0.015 mm off the true surface on a hand-sized part and never coarser than 0.1 mm
    # on a large assembly, with 0.2 rad (about 31 sides to a hole) so small holes print round.
    linear = min(max(diag * 0.0002, 0.002), 0.1)
    BRepMesh_IncrementalMesh(copy, linear, False, 0.2, True)

    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(copy, TopAbs_FACE, faces)

    owner = assembly.face_owner if assembly else {}
    groups: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for i in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(i))
        pts, tris = _face_triangles(face, reversed_=face.Orientation() == TopAbs_REVERSED)
        if len(tris):
            groups.setdefault(owner.get(i, -1), []).append((pts, tris))

    names = _body_names(assembly, name)
    bodies = []
    for key in sorted(groups):
        positions, triangles, normals, base = [], [], [], 0
        for pts, tris in groups[key]:
            positions.append(pts)
            normals.append(_vertex_normals(pts, tris))
            triangles.append(tris + base)
            base += len(pts)
        bodies.append(Body(names.get(key, name), np.concatenate(positions),
                           np.concatenate(triangles), np.concatenate(normals)))
    return bodies


def _body_names(assembly, fallback: str) -> dict[int, str]:
    """Component names, numbered where a part appears more than once ("Bolt 2")."""
    if not assembly:
        return {}
    from collections import Counter
    totals = Counter(inst.name for inst in assembly.instances)
    seen: Counter = Counter()
    out = {}
    for inst in assembly.instances:
        seen[inst.name] += 1
        out[inst.index] = inst.name if totals[inst.name] == 1 else f"{inst.name} {seen[inst.name]}"
    out[-1] = fallback
    return out


def weld(body: Body, tolerance: float = 1e-5) -> Body:
    """Merge vertices that sit at the same point and drop triangles that collapse.

    Neighbouring faces are meshed along the same edge points, so after welding a closed
    solid has every edge shared by exactly two triangles.
    """
    key = np.round(body.positions / tolerance).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    tris = inverse[body.triangles]
    keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return Body(body.name, body.positions[first], tris[keep], np.zeros((len(first), 3)))


def open_edges(triangles: np.ndarray) -> int:
    """How many edges are used by other than two triangles — 0 for a watertight mesh."""
    if not len(triangles):
        return 0
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts != 2))


# ── Mesh writers ─────────────────────────────────────────────────────────────

def stl_bytes(bodies: list[Body], name: str = "Part") -> bytes:
    """Binary STL — every body in one file, as slicers expect."""
    pts = np.concatenate([b.positions[b.triangles] for b in bodies])        # (T, 3, 3)
    normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)

    record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rows = np.zeros(len(pts), dtype=record)
    rows["n"] = normal
    rows["v"] = pts
    header = f"CADVERT export: {name} (millimetres)".encode("ascii", "replace")[:80].ljust(80, b" ")
    return header + struct.pack("<I", len(rows)) + rows.tobytes()


def threemf_bytes(bodies: list[Body]) -> bytes:
    """3MF, one object per body, so a slicer can place or skip components separately."""
    objects, items = [], []
    for n, body in enumerate(bodies, start=1):
        vertices = _repeat('<vertex x="%.5f" y="%.5f" z="%.5f"/>', body.positions)
        triangles = _repeat('<triangle v1="%d" v2="%d" v3="%d"/>', body.triangles)
        objects.append(f'<object id="{n}" type="model" name={quoteattr(body.name)}><mesh>'
                       f"<vertices>{vertices}</vertices><triangles>{triangles}</triangles>"
                       "</mesh></object>")
        items.append(f'<item objectid="{n}"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             '<metadata name="Application">CADVERT</metadata>'
             f'<resources>{"".join(objects)}</resources><build>{"".join(items)}</build></model>')

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                   "</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
                   'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
                   "</Relationships>")
        z.writestr("3D/3dmodel.model", model)
    return out.getvalue()


def obj_bytes(bodies: list[Body]) -> bytes:
    """Wavefront OBJ, one named object per body."""
    lines = ["# CADVERT export (millimetres)\n"]
    base = 1
    for body in bodies:
        lines.append(f"o {_obj_name(body.name)}\n")
        lines.append(_repeat("v %.5f %.5f %.5f\n", body.positions))
        lines.append(_repeat("f %d %d %d\n", body.triangles + base))
        base += len(body.positions)
    return "".join(lines).encode()


def _obj_name(name: str) -> str:
    return "_".join(name.split()) or "Part"


def _repeat(fmt: str, rows: np.ndarray) -> str:
    """*fmt* filled in once per row — one C-level format call, far quicker than a loop."""
    return (fmt * len(rows)) % tuple(rows.ravel().tolist())


def glb_bytes(bodies: list[Body]) -> bytes:
    """Binary glTF 2.0 for the web: metres, Y up, one mesh node per body."""
    blob = bytearray()
    views, accessors, meshes, nodes = [], [], [], []

    def add(array: np.ndarray, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes, "target": target})
        blob.extend(array.tobytes())
        return len(views) - 1

    for body in bodies:
        pos = body.positions.astype(np.float32)
        nrm = body.normals.astype(np.float32)
        idx = body.triangles.astype(np.uint32).reshape(-1)
        p = len(accessors)
        accessors += [
            {"bufferView": add(pos, 34962), "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()},
            {"bufferView": add(nrm, 34962), "componentType": 5126, "count": len(nrm), "type": "VEC3"},
            {"bufferView": add(idx, 34963), "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ]
        meshes.append({"name": body.name, "primitives": [
            {"attributes": {"POSITION": p, "NORMAL": p + 1}, "indices": p + 2, "material": 0}]})
        nodes.append({"name": body.name, "mesh": len(meshes) - 1})

    # The root turns CAD's Z-up millimetres into glTF's Y-up metres.
    nodes.append({"name": "CADVERT export", "children": list(range(len(bodies))),
                  "rotation": [-0.70710678, 0.0, 0.0, 0.70710678], "scale": [0.001, 0.001, 0.001]})
    gltf = {
        "asset": {"version": "2.0", "generator": "CADVERT"},
        "scene": 0,
        "scenes": [{"nodes": [len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [{"name": "Part", "doubleSided": False, "pbrMetallicRoughness": {
            "baseColorFactor": [0.72, 0.74, 0.77, 1.0], "metallicFactor": 0.3, "roughnessFactor": 0.5}}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    while len(blob) % 4:
        blob.append(0)
    head = json.dumps(gltf, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 4)
    total = 12 + 8 + len(head) + 8 + len(blob)
    return b"".join([
        struct.pack("<4sII", b"glTF", 2, total),
        struct.pack("<I4s", len(head), b"JSON"), head,
        struct.pack("<I4s", len(blob), b"BIN\0"), bytes(blob),
    ])


# ── STEP ─────────────────────────────────────────────────────────────────────

def step_bytes(shape) -> bytes:
    """The solid as an AP214 STEP file, in millimetres."""
    import os
    import tempfile
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise ExportError("OpenCascade couldn't convert the part to STEP")
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    try:
        if writer.Write(path) != IFSelect_RetDone:
            raise ExportError("OpenCascade couldn't write the STEP file")
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ── Tables ───────────────────────────────────────────────────────────────────

def bom_csv(assembly) -> bytes:
    """The bill of materials: one row per distinct part, most used first."""
    if not assembly or assembly.instance_count < 2:
        raise ExportError("This file is a single part, so it has no bill of materials")
    rows = [["Item", "Part", "Quantity", "Volume each (mm³)", "Volume total (mm³)", "Where"]]
    for n, r in enumerate(assembly.bill_of_materials(), start=1):
        rows.append([n, r["name"], r["quantity"], _num(r["volume_each"]), _num(r["volume_total"]),
                     "; ".join(r["locations"])])
    return _csv(rows)


def features_csv(features, feature_ids, graph=None, assembly=None) -> bytes:
    """Every detected feature: what it is, its key sizes, and where it is.

    For a hole the position is the centre of its opening; for other features it is the
    centre the detector recorded, if any.
    """
    from .viewmesh import _feature_detail

    if not features:
        raise ExportError("No features were found on this part")
    edges = {e.id: e for e in graph.edges} if graph is not None else {}
    rows = [["ID", "Type", "Description", "Diameter (mm)", "Depth (mm)", "Radius (mm)",
             "Standard", "X (mm)", "Y (mm)", "Z (mm)", "Axis", "Component", "Faces"]]
    for feat, fid in zip(features, feature_ids or [""] * len(features)):
        p = feat.parameters
        diameter = p.get("diameter") or p.get("bore_diameter")
        point = _feature_point(feat, edges)
        axis = p.get("axis") or p.get("axis_dir")
        owner = assembly.owner_of_faces(feat.face_ids) if assembly else None
        rows.append([
            fid, feat.feature_type.replace("_", " ").title(), _feature_detail(feat, "mm"),
            _num(diameter), _num(p.get("depth")), _num(p.get("radius") if feat.feature_type == "FILLET" else None),
            feat.standard_match or "",
            *(_num(v) for v in (point if point is not None else (None, None, None))),
            _axis_label(axis), owner.name if owner else "",
            " ".join(f"F{f}" for f in sorted(set(feat.face_ids))),
        ])
    return _csv(rows)


def _feature_point(feat, edges: dict):
    p = feat.parameters
    entry = p.get("entry_face_id")
    if entry is not None:
        for eid in feat.edge_ids:
            e = edges.get(eid)
            if e is not None and entry in e.face_ids and (e.geometry or {}).get("center"):
                return e.geometry["center"]
    return p.get("center") or p.get("axis_origin")


def _axis_label(axis) -> str:
    if not axis:
        return ""
    v = np.asarray(axis, dtype=float)
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) > 0.9999:
        return ("+" if v[k] > 0 else "-") + "XYZ"[k]
    return "(" + ", ".join(f"{c:.3f}" for c in v) + ")"


def _num(v) -> str:
    if v is None:
        return ""
    v = round(float(v), 4)
    return f"{v:.4f}".rstrip("0").rstrip(".") if v != 0 else "0"


def _csv(rows) -> bytes:
    out = io.StringIO()
    csv.writer(out, lineterminator="\r\n").writerows(rows)
    # The byte-order mark makes Excel read the file as UTF-8 (the ∅ and ³ signs).
    return ("﻿" + out.getvalue()).encode("utf-8")
