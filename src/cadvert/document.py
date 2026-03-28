"""Assemble the Hierarchical Spatial Document (HSD) from topology data.

Complexity budgeting
────────────────────
Parts with ≤ INLINE_TOPOLOGY_THRESHOLD faces get the full topology map inline
(current behaviour). Larger parts produce two documents:

  <stem>.hsd.txt         — SUMMARY: global props + features + spatial + views.
                           The topology map is replaced by a compact overview
                           (face-type histogram, connectivity stats). Self-
                           contained for ~90% of engineering questions.

  <stem>.full_topology.txt — complete face/edge dump, same format as the inline
                             topology map, referenced from the summary.

Both documents are written by render_document() when the part is large.
The caller receives the summary text; the full topology is written to disk
and its path reported in the summary header.
"""

from __future__ import annotations
import math
from pathlib import Path

from .topology import TopologyGraph, FaceInfo, EdgeInfo
from .features import DetectedFeature
from .spatial import SpatialRelationship
from .ingest import GDTAnnotation
from .utils import fmt, fmt_pt, fmt_vec, file_sha256, DISPLAY_PRECISION


HEADER_WIDTH = 70
INLINE_TOPOLOGY_THRESHOLD = 100   # faces — above this, split the document


def render_document(
    graph: TopologyGraph | None,
    source_path: str | Path,
    part_name: str | None = None,
    features: list[DetectedFeature] | None = None,
    spatial: list[SpatialRelationship] | None = None,
    rendered_views: list[Path] | None = None,
    validation_report: str | None = None,
    units: str = "mm",
    gdt_annotations: list[GDTAnnotation] | None = None,
    mesh_info: dict | None = None,
) -> str:
    """Render the HSD document (sections 1–5).

    For parts with > INLINE_TOPOLOGY_THRESHOLD faces, the topology map is
    written to a separate <stem>.full_topology.txt file and only a compact
    topology overview is inlined in the returned summary.
    """
    source_path = Path(source_path)
    part_name = part_name or source_path.stem
    sha = file_sha256(source_path)
    is_mesh = mesh_info is not None
    large_part = (not is_mesh) and graph is not None and len(graph.faces) > INLINE_TOPOLOGY_THRESHOLD

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    precision_note = (
        "PRECISION: approximate — mesh format, no analytical geometry"
        if is_mesh else
        "PRECISION: all values exact from B-REP analytical definitions"
    )
    lines += [
        f"PART: {part_name}",
        f"SOURCE: {source_path.name} (SHA256: {sha})",
        f"UNITS: {units}",
        precision_note,
    ]
    if is_mesh:
        fmt = mesh_info.get("format", "MESH")
        tri = mesh_info.get("triangle_count", 0)
        lines.append(
            f"FORMAT: {fmt} (mesh — {tri:,} triangles)\n"
            "NOTE: Topology map, feature detection, and spatial analysis are not\n"
            "available for mesh formats. Export as STEP from your CAD tool for\n"
            "full analysis with exact geometry and manufacturing features."
        )
    if large_part:
        topo_path = source_path.with_suffix(".full_topology.txt")
        lines.append(
            f"DOCUMENT MODE: SUMMARY ({len(graph.faces)} faces exceeds inline threshold "
            f"of {INLINE_TOPOLOGY_THRESHOLD}). "
            f"Full topology: {topo_path.name}"
        )
    if validation_report:
        lines.append("")
        lines.extend(validation_report.splitlines())
    lines.append("")

    # ── Section 1: Global Properties ─────────────────────────────────────────
    lines += _section_header("GLOBAL PROPERTIES")
    if is_mesh:
        lines += _global_properties_mesh(mesh_info, units)
    else:
        lines += _global_properties(graph, units)
    lines.append("")

    # ── Section 2: Topology Map (B-REP only) ──────────────────────────────────
    if not is_mesh and graph is not None:
        if large_part:
            lines += _section_header("TOPOLOGY OVERVIEW")
            lines += _topology_overview(graph)
            lines.append(
                f"\nFull per-face topology is in: {topo_path.name}\n"
                "Reference it when you need exact geometry for a specific face or edge."
            )
            lines.append("")
            full_topo = _render_full_topology(graph, part_name)
            topo_path.write_text(full_topo, encoding="utf-8")
        else:
            lines += _section_header("TOPOLOGY MAP")
            lines.append(
                "Every face is listed with its exact analytical surface definition.\n"
                "Every edge shows which two faces it connects, the curve geometry,\n"
                "and the dihedral angle (convex = outside corner, concave = inside corner).\n"
            )
            edge_by_id = {e.id: e for e in graph.edges}
            for face in graph.faces:
                lines += _face_block(face, edge_by_id, graph)
                lines.append("")

    # ── Section 3: Identified Features ───────────────────────────────────────
    if features is not None:
        lines += _section_header("IDENTIFIED FEATURES")
        lines += _features_section(features) if features else ["No manufacturing features detected."]
        lines.append("")

    # ── Section 4: Spatial Relationships ─────────────────────────────────────
    if spatial is not None:
        lines += _section_header("SPATIAL RELATIONSHIPS")
        lines += _spatial_section(spatial) if spatial else ["No spatial relationships computed."]
        lines.append("")

    # ── Section 5: GD&T Annotations (AP242 only) ─────────────────────────────
    if gdt_annotations:
        lines += _section_header("GD&T ANNOTATIONS (AP242 PMI)")
        lines += _gdt_section(gdt_annotations, units)
        lines.append("")

    # ── Section 6: Rendered Views ─────────────────────────────────────────────
    if rendered_views is not None:
        lines += _section_header("RENDERED VIEWS")
        if rendered_views:
            lines += _views_section(rendered_views)
        else:
            lines.append("No rendered views available.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Topology overview (large-part summary replacement for section 2)
# ---------------------------------------------------------------------------

def _topology_overview(graph: TopologyGraph) -> list[str]:
    from collections import Counter
    type_counts = Counter(f.geometry.get("type", "OTHER") for f in graph.faces)
    edge_type_counts = Counter(e.geometry.get("type", "OTHER") for e in graph.edges)

    # Connectivity stats
    edge_face_counts = [len(e.face_ids) for e in graph.edges]
    manifold = sum(1 for c in edge_face_counts if c == 2)
    boundary = sum(1 for c in edge_face_counts if c == 1)
    non_manifold = sum(1 for c in edge_face_counts if c > 2)

    lines = ["Face type distribution:"]
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {t:<20} {c:>4} faces")
    lines.append("")
    lines.append("Edge type distribution:")
    for t, c in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {t:<20} {c:>4} edges")
    lines.append("")
    lines.append("Connectivity:")
    lines.append(f"  Manifold edges (2 faces):     {manifold}")
    lines.append(f"  Boundary edges (1 face):      {boundary}")
    lines.append(f"  Non-manifold edges (>2 faces):{non_manifold}")

    # List the analytically distinct cylinders (holes/bores) — highest LLM value
    cyls = [f for f in graph.faces if f.geometry.get("type") == "CYLINDER"]
    if cyls:
        lines.append("")
        lines.append(f"Cylindrical features ({len(cyls)} faces):")
        for f in sorted(cyls, key=lambda x: x.geometry["radius"]):
            g = f.geometry
            lines.append(
                f"  [F{f.id}]  R={fmt(g['radius'])}  axis={fmt_vec(g['axis_dir'])}"
                f"  origin=({fmt(g['axis_origin'][0])},{fmt(g['axis_origin'][1])},{fmt(g['axis_origin'][2])})"
            )

    return lines


def _render_full_topology(graph: TopologyGraph, part_name: str) -> str:
    """Render the complete face/edge topology map as a standalone document."""
    lines = [
        f"FULL TOPOLOGY: {part_name}",
        f"Faces: {len(graph.faces)}  Edges: {len(graph.edges)}  Vertices: {len(graph.vertices)}",
        "",
    ]
    lines += _section_header("TOPOLOGY MAP")
    lines.append(
        "Every face with its exact analytical surface definition.\n"
        "Every edge showing which two faces it connects, curve geometry,\n"
        "and dihedral angle (convex = outside corner, concave = inside corner).\n"
    )
    edge_by_id = {e.id: e for e in graph.edges}
    for face in graph.faces:
        lines += _face_block(face, edge_by_id, graph)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 1
# ---------------------------------------------------------------------------

def _global_properties_mesh(mesh_info: dict, units: str = "mm") -> list[str]:
    """Render global properties for a mesh-format file."""
    lines = [f"Format:        {mesh_info.get('format', 'MESH')} (triangulated mesh)"]
    lines.append(f"Triangles:     {mesh_info.get('triangle_count', 0):,}")

    if "bbox" in mesh_info:
        bb = mesh_info["bbox"]
        lines.append(
            f"Bounding box:  X[{fmt(bb['X'][0])}, {fmt(bb['X'][1])}]"
            f"  Y[{fmt(bb['Y'][0])}, {fmt(bb['Y'][1])}]"
            f"  Z[{fmt(bb['Z'][0])}, {fmt(bb['Z'][1])}]"
        )
    if "volume" in mesh_info:
        lines.append(f"Volume:        ~{mesh_info['volume']:,.3f} {units}³  (approximation from mesh)")
    if "surface_area" in mesh_info:
        lines.append(f"Surface area:  ~{mesh_info['surface_area']:,.3f} {units}²  (approximation from mesh)")

    lines += [
        "",
        "NOTE: Values marked ~ are approximations from the triangle mesh.",
        "Feature detection, exact hole diameters, and surface types are not",
        "available. Export as STEP from your CAD tool for full analysis.",
    ]
    return lines


def _global_properties(g: TopologyGraph, units: str = "mm") -> list[str]:
    bb = g.bounding_box
    lines = [
        f"Bounding box:  X[{fmt(bb['X'][0])}, {fmt(bb['X'][1])}]"
        f"  Y[{fmt(bb['Y'][0])}, {fmt(bb['Y'][1])}]"
        f"  Z[{fmt(bb['Z'][0])}, {fmt(bb['Z'][1])}]",
        f"Volume:        {g.volume:,.3f} {units}³",
        f"Surface area:  {g.surface_area:,.3f} {units}²",
        f"Center of mass: {fmt_pt(g.center_of_mass)}",
        f"Total faces:   {len(g.faces)}",
        f"Total edges:   {len(g.edges)}",
        f"Total vertices:{len(g.vertices)}",
        f"Unique bodies: {g.body_count}",
    ]
    return lines


# ---------------------------------------------------------------------------
# Section 2
# ---------------------------------------------------------------------------

def _face_block(face: FaceInfo, edge_by_id: dict, graph: TopologyGraph) -> list[str]:
    lines = []
    geom = face.geometry
    face_label = _face_label(face, geom)
    lines.append(f"[F{face.id}] {face_label}  (area={fmt(face.area)} mm²)")

    for eid in face.edge_ids:
        edge = edge_by_id.get(eid)
        if edge is None:
            continue
        lines.append(_edge_line(edge, face.id))

    return lines


def _face_label(face: FaceInfo, geom: dict) -> str:
    t = geom.get("type", "UNKNOWN")
    p = DISPLAY_PRECISION

    if t == "PLANE":
        n = geom["normal"]
        d = geom["d"]
        # Determine dominant axis for a readable label
        dominant = _dominant_axis(n)
        if dominant:
            return f"PLANE {dominant}={fmt(d / _axis_component(n, dominant))}"
        return (
            f"PLANE  normal={fmt_vec(n)}  offset={fmt(d)}"
        )

    elif t == "CYLINDER":
        ax = geom["axis_dir"]
        orig = geom["axis_origin"]
        r = geom["radius"]
        return (
            f"CYLINDER  axis={fmt_vec(ax)}"
            f"  center=({fmt(orig[0])},{fmt(orig[1])})"
            f"  R={fmt(r)}"
        )

    elif t == "CONE":
        apex = geom["apex"]
        ax = geom["axis_dir"]
        return (
            f"CONE  apex={fmt_pt(apex)}  axis={fmt_vec(ax)}"
            f"  half_angle={fmt(geom['half_angle'])}°"
        )

    elif t == "SPHERE":
        c = geom["center"]
        return f"SPHERE  center={fmt_pt(c)}  R={fmt(geom['radius'])}"

    elif t == "TORUS":
        c = geom["center"]
        return (
            f"TORUS  center={fmt_pt(c)}"
            f"  R_major={fmt(geom['major_radius'])}"
            f"  R_minor={fmt(geom['minor_radius'])}"
        )

    elif t == "NURBS_SURFACE":
        if "error" in geom:
            return f"NURBS_SURFACE  (extraction error: {geom['error']})"
        return (
            f"NURBS_SURFACE  degree=({geom['degree_u']},{geom['degree_v']})"
            f"  poles={geom['n_poles_u']}×{geom['n_poles_v']}"
            + ("  rational" if geom.get("is_rational") else "")
        )

    return f"{t}"


def _edge_line(edge: EdgeInfo, current_face_id: int) -> str:
    geom = edge.geometry
    # seam edge: same face on both sides (cylinder/torus seam)
    if len(edge.face_ids) == 2 and edge.face_ids[0] == edge.face_ids[1]:
        other_str = f"[F{edge.face_ids[0]}] (seam)"
    else:
        other_faces = [fid for fid in edge.face_ids if fid != current_face_id]
        other_str = ", ".join(f"[F{fid}]" for fid in other_faces) if other_faces else "boundary"

    angle_str = ""
    if edge.dihedral_angle is not None:
        angle_str = f" — {fmt(edge.dihedral_angle)}° {edge.convexity}"

    curve_desc = _curve_desc(geom)
    return f"  ├─ E{edge.id} ({curve_desc}) → {other_str}{angle_str}"


def _curve_desc(geom: dict) -> str:
    t = geom.get("type", "UNKNOWN")
    p = DISPLAY_PRECISION

    if t == "LINE":
        return f"line, L={fmt(geom['length'])}"

    elif t == "CIRCLE":
        c = geom["center"]
        r = geom["radius"]
        full = " full" if geom.get("is_full_circle") else ""
        return (
            f"circle{full}, R={fmt(r)}"
            f", center=({fmt(c[0])},{fmt(c[1])},{fmt(c[2])})"
        )

    elif t == "ELLIPSE":
        return (
            f"ellipse, R_maj={fmt(geom['major_radius'])}"
            f", R_min={fmt(geom['minor_radius'])}"
        )

    elif t == "NURBS_CURVE":
        if "error" in geom:
            return f"nurbs_curve (error)"
        return f"nurbs_curve, deg={geom['degree']}, {len(geom['control_points'])} poles"

    return t.lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_header(title: str) -> list[str]:
    bar = "═" * HEADER_WIDTH
    return [bar, f"{'═══ ' + title + ' ═══':^{HEADER_WIDTH}}", bar, ""]


def _dominant_axis(normal: tuple) -> str | None:
    """If normal is essentially aligned to one axis, return 'x', 'y', or 'z'."""
    tol = 1e-6
    nx, ny, nz = (abs(v) for v in normal)
    if nx > 1 - tol and ny < tol and nz < tol:
        return "x"
    if ny > 1 - tol and nx < tol and nz < tol:
        return "y"
    if nz > 1 - tol and nx < tol and ny < tol:
        return "z"
    return None


def _axis_component(normal: tuple, axis: str) -> float:
    return {"x": normal[0], "y": normal[1], "z": normal[2]}[axis]


# ---------------------------------------------------------------------------
# Section 3: Identified Features
# ---------------------------------------------------------------------------

def _features_section(features: list[DetectedFeature]) -> list[str]:
    lines = []
    # Group patterns separately so they appear last
    regular = [f for f in features if f.feature_type != "PATTERN"]
    patterns = [f for f in features if f.feature_type == "PATTERN"]

    for feat in regular:
        lines += _feature_block(feat)
        lines.append("")

    for feat in patterns:
        lines += _feature_block(feat)
        lines.append("")

    return lines


def _feature_block(feat: DetectedFeature) -> list[str]:
    p = feat.parameters
    ftype = feat.feature_type
    conf_str = f"  (confidence: {feat.confidence:.0%})" if feat.confidence < 1.0 else ""
    lines = [f"FEATURE: {ftype.replace('_', ' ').title()}{conf_str}"]

    face_str = ", ".join(f"[F{fid}]" for fid in feat.face_ids)
    edge_str = ", ".join(f"E{eid}" for eid in feat.edge_ids)
    lines.append(f"  Faces: {face_str}")
    if feat.edge_ids:
        lines.append(f"  Edges: {edge_str}")

    if ftype in ("THROUGH_HOLE", "BLIND_HOLE"):
        lines.append(f"  Diameter:   {fmt(p['diameter'])} mm")
        lines.append(f"  Radius:     {fmt(p['radius'])} mm")
        lines.append(f"  Depth:      {fmt(p['depth'])} mm"
                     + (" (through)" if ftype == "THROUGH_HOLE" else " (blind)"))
        ax = p["axis"]
        ao = p["axis_origin"]
        lines.append(f"  Axis:       {fmt_vec(ax)}")
        lines.append(f"  Axis point: {fmt_pt(ao)}")
        if ftype == "THROUGH_HOLE":
            lines.append(f"  Entry face: [F{p['entry_face_id']}]  Exit face: [F{p['exit_face_id']}]")
        else:
            if p.get("entry_face_id"):
                lines.append(f"  Entry face: [F{p['entry_face_id']}]")
            lines.append(f"  Bottom face:[F{p['bottom_face_id']}]")
        if feat.standard_match:
            lines.append(f"  Standard:   {feat.standard_match}")

    elif ftype == "BOSS":
        lines.append(f"  Diameter: {fmt(p['diameter'])} mm")
        lines.append(f"  Height:   {fmt(p['height'])} mm")
        lines.append(f"  Axis:     {fmt_vec(p['axis'])}")
        if feat.standard_match:
            lines.append(f"  Standard: {feat.standard_match}")

    elif ftype == "FILLET":
        lines.append(f"  Radius: {fmt(p['radius'])} mm")
        adj = ", ".join(f"[F{fid}]" for fid in p.get("adjacent_face_ids", []))
        lines.append(f"  Blends: {adj}")

    elif ftype == "CHAMFER":
        if p.get("width") is not None:
            lines.append(f"  Width: {fmt(p['width'])} mm")
        adj = ", ".join(f"[F{fid}]" for fid in p.get("adjacent_face_ids", []))
        lines.append(f"  Between: {adj}")

    elif ftype == "POCKET":
        lines.append(f"  Floor face:  [F{p['floor_face_id']}]")
        wall_str = ", ".join(f"[F{fid}]" for fid in p.get("wall_face_ids", []))
        if wall_str:
            lines.append(f"  Wall faces:  {wall_str}")

    elif ftype == "COUNTERBORE":
        lines.append(f"  Bore diameter:   {fmt(p['bore_diameter'])} mm")
        lines.append(f"  Cbore diameter:  {fmt(p['cbore_diameter'])} mm")
        lines.append(f"  Total depth:     {fmt(p['total_depth'])} mm")
        lines.append(f"  Cbore depth:     {fmt(p['cbore_depth'])} mm")
        lines.append(f"  Axis:            {fmt_vec(p['axis'])}")
        lines.append(f"  Bore face [F{p['bore_face_id']}]  Cbore face [F{p['cbore_face_id']}]")
        if feat.standard_match:
            lines.append(f"  Standard:        {feat.standard_match}")

    elif ftype == "COUNTERSINK":
        lines.append(f"  Included angle:  {fmt(p['cone_half_angle'] * 2)}°")
        lines.append(f"  Bore diameter:   {fmt(p['bore_diameter'])} mm")
        lines.append(f"  Axis:            {fmt_vec(p['axis'])}")
        lines.append(f"  Cone face [F{p['cone_face_id']}]  Bore face [F{p['bore_face_id']}]")

    elif ftype == "SLOT":
        lines.append(f"  Width:   {fmt(p['width'])} mm  (radius={fmt(p['radius'])} mm)")
        lines.append(f"  Length:  {fmt(p['length'])} mm")
        lines.append(f"  Axis:    {fmt_vec(p['axis'])}")
        end_str = f"[F{p['end_a_face_id']}], [F{p['end_b_face_id']}]"
        lines.append(f"  End caps: {end_str}")
        wall_str = ", ".join(f"[F{fid}]" for fid in p.get("wall_face_ids", []))
        if wall_str:
            lines.append(f"  Walls:   {wall_str}")

    elif ftype == "PATTERN":
        lines.append(f"  Count:    {p['count']}× {p['child_type'].replace('_',' ').title()}")
        lines.append(f"  Diameter: {fmt(p['diameter'])} mm each")
        child_str = ", ".join(f"[F{fid}]" for fid in p.get("child_face_ids", []))
        lines.append(f"  Members:  {child_str}")

    if feat.notes:
        lines.append(f"  Note: {feat.notes}")

    return lines


# ---------------------------------------------------------------------------
# Section 4: Spatial Relationships
# ---------------------------------------------------------------------------

def _spatial_section(rels: list[SpatialRelationship]) -> list[str]:
    lines = []
    # Group by description type
    groups: dict[str, list[SpatialRelationship]] = {}
    for r in rels:
        groups.setdefault(r.description, []).append(r)

    for desc, group in groups.items():
        lines.append(f"{desc}:")
        for r in group:
            note = f"  [{r.notes}]" if r.notes else ""
            lines.append(f"  {r.from_ref} → {r.to_ref}:  {fmt(r.value)} mm{note}")
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Section 5: GD&T Annotations
# ---------------------------------------------------------------------------

def _gdt_section(annotations: list[GDTAnnotation], units: str) -> list[str]:
    """Render AP242 GD&T annotations as a compact block."""
    lines = [
        f"Extracted from AP242 PMI data ({len(annotations)} annotations).",
        f"All tolerance values in {units}.",
        "",
    ]
    dim = [a for a in annotations if a.annotation_type == "DIMENSIONAL"]
    geo = [a for a in annotations if a.annotation_type == "GEOMETRIC"]

    if dim:
        lines.append("Dimensional tolerances:")
        for a in dim:
            face_note = (
                f" → STEP faces {a.face_step_ids}" if a.face_step_ids else ""
            )
            lines.append(f"  {a.symbol}{face_note}")
        lines.append("")

    if geo:
        lines.append("Geometric tolerances:")
        for a in geo:
            face_note = (
                f" → STEP faces {a.face_step_ids}" if a.face_step_ids else ""
            )
            datum_note = (
                f"  datums: {', '.join(a.datum_refs)}" if a.datum_refs else ""
            )
            lines.append(f"  {a.symbol:<30}{datum_note}{face_note}")

    return lines


# ---------------------------------------------------------------------------
# Section 6: Rendered Views
# ---------------------------------------------------------------------------

def _views_section(paths: list[Path]) -> list[str]:
    lines = ["The following PNG files have been written alongside this document:"]
    for p in paths:
        lines.append(f"  {p.name}")
    return lines


# ---------------------------------------------------------------------------
# NURBS detail block (appended to NURBS faces for full lossless dump)
# ---------------------------------------------------------------------------

def render_nurbs_detail(face_id: int, geom: dict) -> str:
    """Return the full control-point dump for a NURBS face."""
    lines = [f"[F{face_id}] NURBS_SURFACE — full definition:"]
    lines.append(f"  Degree: ({geom['degree_u']}, {geom['degree_v']})")
    lines.append(f"  Control point grid: {geom['n_poles_u']} × {geom['n_poles_v']}")
    for i, row in enumerate(geom["control_points"]):
        for j, (x, y, z, w) in enumerate(row):
            wstr = f", w={fmt(w)}" if abs(w - 1.0) > 1e-6 else ""
            lines.append(f"    CP[{i}][{j}] = ({fmt(x)}, {fmt(y)}, {fmt(z)}){wstr}")
    lines.append(f"  Knot vector U: {geom['knot_vector_u']}")
    lines.append(f"  Knot vector V: {geom['knot_vector_v']}")
    return "\n".join(lines)
