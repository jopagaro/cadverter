"""Spatial relationship computations.

Sections:
  1. Feature-to-feature distances (all types, not just holes)
  2. Wall thicknesses (hole / boss to outer surface)
  3. Minimum surface clearance between non-adjacent features (OCC BRepExtrema)
  4. Draft angle analysis (face normals vs. a pull direction)
  5. Undercut detection (faces that can't be reached from the pull direction)
  6. Symmetry detection (reflection planes through center of mass)
  7. Overall dimensions
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from .topology import TopologyGraph, FaceInfo
from .features import DetectedFeature
from .utils import fmt, normalize


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SpatialRelationship:
    description: str
    value: float
    from_ref: str
    to_ref: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_spatial_relationships(
    graph: TopologyGraph,
    features: list[DetectedFeature],
    shape=None,                        # OCC TopoDS_Shape, optional (for BRepExtrema)
    pull_direction: tuple = (0, 0, 1), # for draft angle / undercut analysis
) -> list[SpatialRelationship]:
    rels: list[SpatialRelationship] = []
    face_by_id = {f.id: f for f in graph.faces}
    outer_planes = _outer_plane_faces(graph)

    holes  = [f for f in features if f.feature_type in ("THROUGH_HOLE", "BLIND_HOLE")]
    bosses = [f for f in features if f.feature_type == "BOSS"]
    cbores = [f for f in features if f.feature_type == "COUNTERBORE"]
    all_locating = holes + bosses + cbores

    # ── 1. Feature-to-feature distances (all types) ───────────────────────────
    rels += _feature_to_feature(all_locating)

    # ── 2. Wall thicknesses ───────────────────────────────────────────────────
    rels += _wall_thicknesses_all(all_locating, outer_planes, face_by_id)

    # ── 3. Min surface clearance (BRepExtrema, needs OCC shape) ───────────────
    if shape is not None:
        rels += _min_surface_clearances(features, graph, shape)

    # ── 4. Draft angle analysis ───────────────────────────────────────────────
    rels += _draft_angle_analysis(graph, normalize(pull_direction))

    # ── 5. Undercut detection ─────────────────────────────────────────────────
    rels += _undercut_detection(graph, normalize(pull_direction))

    # ── 6. Symmetry detection ─────────────────────────────────────────────────
    rels += _symmetry_detection(graph)

    # ── 7. Overall dimensions ─────────────────────────────────────────────────
    bb = graph.bounding_box
    for ax, key in (("X", "X"), ("Y", "Y"), ("Z", "Z")):
        span = bb[key][1] - bb[key][0]
        rels.append(SpatialRelationship(
            description=f"Overall {ax} dimension",
            value=span,
            from_ref=f"{ax}min={fmt(bb[key][0])}",
            to_ref=f"{ax}max={fmt(bb[key][1])}",
        ))

    return rels


# ---------------------------------------------------------------------------
# 1. Feature-to-feature distances
# ---------------------------------------------------------------------------

def _feature_to_feature(
    features: list[DetectedFeature],
) -> list[SpatialRelationship]:
    rels = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            fi, fj = features[i], features[j]
            ci = _feature_axis_origin(fi)
            cj = _feature_axis_origin(fj)
            if ci is None or cj is None:
                continue

            dist_cc = _pt_dist(ci, cj)
            ri = fi.parameters.get("radius", fi.parameters.get("bore_diameter", 0) / 2)
            rj = fj.parameters.get("radius", fj.parameters.get("bore_diameter", 0) / 2)
            clearance = dist_cc - ri - rj

            li = _feature_short_label(fi, i)
            lj = _feature_short_label(fj, j)

            rels.append(SpatialRelationship(
                description="Feature center to feature center",
                value=round(dist_cc, 3),
                from_ref=f"{li} center",
                to_ref=f"{lj} center",
            ))
            if clearance > 0:
                rels.append(SpatialRelationship(
                    description="Feature edge clearance",
                    value=round(clearance, 3),
                    from_ref=f"{li} edge",
                    to_ref=f"{lj} edge",
                ))
    return rels


# ---------------------------------------------------------------------------
# 2. Wall thicknesses
# ---------------------------------------------------------------------------

def _wall_thicknesses_all(
    features: list[DetectedFeature],
    outer_planes: list[FaceInfo],
    face_by_id: dict,
) -> list[SpatialRelationship]:
    rels = []
    for i, feat in enumerate(features):
        ao = _feature_axis_origin(feat)
        r  = feat.parameters.get("radius", feat.parameters.get("bore_diameter", 0) / 2)
        ax = feat.parameters.get("axis")
        if ao is None or ax is None:
            continue
        label = _feature_short_label(feat, i)
        for face_label, thickness in _wall_thickness_to_planes(ao, r, ax, outer_planes):
            rels.append(SpatialRelationship(
                description="Wall thickness",
                value=thickness,
                from_ref=f"{label} bore edge",
                to_ref=f"outer surface {face_label}",
                notes="Minimum material between feature bore and outer surface",
            ))
    return rels


def _wall_thickness_to_planes(
    ao: tuple, radius: float, axis: tuple, outer_planes: list[FaceInfo],
) -> list[tuple[str, float]]:
    results = []
    for face in outer_planes:
        g = face.geometry
        n, d = g["normal"], g["d"]
        dot_axis = abs(sum(n[i] * axis[i] for i in range(3)))
        if dot_axis > 0.9:
            continue  # entry/exit face, not a wall
        dist_center = abs(sum(n[i] * ao[i] for i in range(3)) - d)
        wall = dist_center - radius
        if wall >= 0:
            results.append((f"[F{face.id}] {_plane_label(face)}", round(wall, 3)))
    return sorted(results, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# 3. Minimum surface clearance via BRepExtrema
# ---------------------------------------------------------------------------

def _min_surface_clearances(
    features: list[DetectedFeature],
    graph: TopologyGraph,
    shape,
) -> list[SpatialRelationship]:
    """Use OCC BRepExtrema to compute actual minimum surface-to-surface
    distances between non-adjacent feature faces."""
    try:
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.TopExp import TopExp
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopoDS import TopoDS
        from OCP.BRep import BRep_Tool
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    except ImportError:
        return []

    face_imap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_imap)

    # Build face pairs: holes vs holes, limiting to features that have face_ids
    hole_feats = [f for f in features if f.feature_type in ("THROUGH_HOLE", "BLIND_HOLE")]
    rels = []

    for i in range(len(hole_feats)):
        for j in range(i + 1, len(hole_feats)):
            fi, fj = hole_feats[i], hole_feats[j]
            fid_i = fi.face_ids[0] if fi.face_ids else None
            fid_j = fj.face_ids[0] if fj.face_ids else None
            if fid_i is None or fid_j is None:
                continue
            if fid_i > face_imap.Extent() or fid_j > face_imap.Extent():
                continue

            occ_fi = TopoDS.Face_s(face_imap.FindKey(fid_i))
            occ_fj = TopoDS.Face_s(face_imap.FindKey(fid_j))

            try:
                dist_calc = BRepExtrema_DistShapeShape(occ_fi, occ_fj)
                dist_calc.Perform()
                if dist_calc.IsDone():
                    min_dist = dist_calc.Value()
                    li = _feature_short_label(fi, i)
                    lj = _feature_short_label(fj, j)
                    rels.append(SpatialRelationship(
                        description="Minimum surface-to-surface clearance",
                        value=round(min_dist, 3),
                        from_ref=f"{li} surface",
                        to_ref=f"{lj} surface",
                        notes="Exact OCC BRepExtrema distance",
                    ))
            except Exception:
                pass

    return rels


# ---------------------------------------------------------------------------
# 4. Draft angle analysis
# ---------------------------------------------------------------------------

def _draft_angle_analysis(
    graph: TopologyGraph,
    pull_dir: tuple,
) -> list[SpatialRelationship]:
    """Compute draft angle for every planar face relative to pull_dir.
    Draft angle = complement of angle between face normal and pull direction.
    A face at 0° draft is perfectly parallel to pull (a side wall, easy to pull).
    A face at 90° is perpendicular to pull (a top/bottom face).
    Faces below ~1° draft on non-trivial parts may cause ejection problems.
    """
    rels = []
    for face in graph.faces:
        g = face.geometry
        if g.get("type") != "PLANE":
            continue
        n = normalize(g["normal"])
        # Draft angle = angle between face normal and pull direction - 90°
        # (0° = parallel to pull = vertical wall, ideal; negative = undercut)
        dot = sum(n[i] * pull_dir[i] for i in range(3))
        angle_from_pull = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        draft = 90.0 - angle_from_pull   # positive = drafted away from pull
        if abs(draft) < 89.0:  # skip faces nearly perpendicular to pull (top/bottom)
            rels.append(SpatialRelationship(
                description="Draft angle",
                value=round(draft, 2),
                from_ref=f"[F{face.id}] normal={tuple(round(x,3) for x in n)}",
                to_ref=f"pull direction {pull_dir}",
                notes=(
                    "negative = undercut"
                    if draft < 0
                    else ("< 1° may cause ejection issues" if draft < 1.0 else "")
                ),
            ))
    return rels


# ---------------------------------------------------------------------------
# 5. Undercut detection
# ---------------------------------------------------------------------------

def _undercut_detection(
    graph: TopologyGraph,
    pull_dir: tuple,
) -> list[SpatialRelationship]:
    """Report faces whose outward normal has a component opposing the pull direction.
    These are regions a straight mold pull can't reach — undercuts.
    """
    rels = []
    undercut_faces = []
    for face in graph.faces:
        g = face.geometry
        if g.get("type") not in ("PLANE", "CYLINDER", "CONE"):
            continue
        n = _representative_normal(face, pull_dir)
        if n is None:
            continue
        dot = sum(n[i] * pull_dir[i] for i in range(3))
        if dot < -0.05:   # normal pointing against pull by > ~3°
            undercut_faces.append(face.id)

    if undercut_faces:
        rels.append(SpatialRelationship(
            description="Undercut faces (cannot be reached by straight pull)",
            value=float(len(undercut_faces)),
            from_ref=f"pull direction {pull_dir}",
            to_ref=f"faces {undercut_faces[:10]}" + (" ..." if len(undercut_faces) > 10 else ""),
            notes=f"{len(undercut_faces)} face(s) are undercuts relative to pull {pull_dir}",
        ))
    else:
        rels.append(SpatialRelationship(
            description="Undercut faces (cannot be reached by straight pull)",
            value=0.0,
            from_ref=f"pull direction {pull_dir}",
            to_ref="all faces",
            notes="No undercuts detected — part can be straight-pulled in this direction",
        ))
    return rels


def _representative_normal(face: FaceInfo, pull_dir: tuple) -> tuple | None:
    g = face.geometry
    t = g.get("type")
    if t == "PLANE":
        return normalize(g["normal"])
    elif t == "CYLINDER":
        # For a cylinder, the representative normal for undercut check is the
        # radial direction most opposing pull. If the axis is parallel to pull,
        # no part of the cylinder is an undercut.
        ax = normalize(g["axis_dir"])
        dot_ax_pull = abs(sum(ax[i] * pull_dir[i] for i in range(3)))
        if dot_ax_pull > 0.99:
            return None  # axis-aligned cylinder, no undercut
        # Worst-case radial direction: component of (-pull_dir) perpendicular to axis
        neg_pull = tuple(-pull_dir[i] for i in range(3))
        proj = sum(neg_pull[i] * ax[i] for i in range(3))
        radial = tuple(neg_pull[i] - proj * ax[i] for i in range(3))
        return normalize(radial)
    elif t == "CONE":
        return normalize(g["axis_dir"])
    return None


# ---------------------------------------------------------------------------
# 6. Symmetry detection
# ---------------------------------------------------------------------------

def _symmetry_detection(graph: TopologyGraph) -> list[SpatialRelationship]:
    """Check whether the part has reflection symmetry about any of the three
    principal planes through the center of mass.

    Method: for each plane, compute the histogram of signed distances from
    all face centroids to the plane. If the histogram is symmetric
    (each distance d has a corresponding -d within tolerance), report symmetry.
    """
    rels = []
    com = graph.center_of_mass
    bb  = graph.bounding_box

    # Candidate planes: XY (normal Z), XZ (normal Y), YZ (normal X)
    candidate_planes = [
        ((1, 0, 0), "YZ plane (X symmetry)", com[0]),
        ((0, 1, 0), "XZ plane (Y symmetry)", com[1]),
        ((0, 0, 1), "XY plane (Z symmetry)", com[2]),
    ]

    # Collect face "centroids" (bounding-box center is a proxy; use face area for weighting)
    # We use face geometry anchor points
    anchors = _face_anchor_points(graph)

    for normal, label, offset in candidate_planes:
        # Signed distance of each anchor to the plane
        dists = sorted([
            sum(normal[i] * pt[i] for i in range(3)) - offset
            for pt in anchors
        ])
        is_sym = _check_symmetric_distances(dists, tol=1.0)
        rels.append(SpatialRelationship(
            description="Symmetry",
            value=1.0 if is_sym else 0.0,
            from_ref=label,
            to_ref="part geometry",
            notes="symmetric" if is_sym else "not symmetric",
        ))

    return rels


def _face_anchor_points(graph: TopologyGraph) -> list[tuple]:
    """Return one representative point per face."""
    pts = []
    for face in graph.faces:
        g = face.geometry
        t = g.get("type")
        if t == "PLANE":
            pts.append(tuple(g["origin"]))
        elif t == "CYLINDER":
            pts.append(tuple(g["axis_origin"]))
        elif t == "SPHERE":
            pts.append(tuple(g["center"]))
        elif t == "TORUS":
            pts.append(tuple(g["center"]))
        elif t == "CONE":
            pts.append(tuple(g["apex"]))
    return pts


def _check_symmetric_distances(dists: list[float], tol: float) -> bool:
    """True if for every d in dists, there is a -d within tol."""
    if not dists:
        return False
    remaining = list(dists)
    for d in list(dists):
        mirror = -d
        best = min(remaining, key=lambda x: abs(x - mirror))
        if abs(best - mirror) <= tol:
            remaining.remove(best)
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_axis_origin(feat: DetectedFeature) -> tuple | None:
    ao = feat.parameters.get("axis_origin")
    return tuple(ao) if ao else None


def _feature_short_label(feat: DetectedFeature, idx: int) -> str:
    short = feat.feature_type.replace("_", " ").title()
    d = feat.parameters.get("diameter",
        feat.parameters.get("bore_diameter",
        feat.parameters.get("width", 0)))
    return f"{short}-{idx+1} (∅{fmt(d)})"


def _pt_dist(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _outer_plane_faces(graph: TopologyGraph) -> list[FaceInfo]:
    bb = graph.bounding_box
    tol = 0.5
    outer = []
    for face in graph.faces:
        g = face.geometry
        if g.get("type") != "PLANE":
            continue
        n, d = g["normal"], g["d"]
        for i, key in enumerate(("X", "Y", "Z")):
            if abs(abs(n[i]) - 1.0) < 0.05:
                plane_pos = d / (n[i] if abs(n[i]) > 1e-9 else 1.0)
                lo, hi = bb[key]
                if abs(plane_pos - lo) < tol or abs(plane_pos - hi) < tol:
                    outer.append(face)
                    break
    return outer


def _plane_label(face: FaceInfo) -> str:
    g = face.geometry
    n, d = g["normal"], g["d"]
    for i, ax in enumerate("XYZ"):
        if abs(abs(n[i]) - 1.0) < 0.02:
            val = d / (n[i] if abs(n[i]) > 1e-9 else 1.0)
            return f"{ax}={fmt(val)}"
    return f"n=({fmt(n[0])},{fmt(n[1])},{fmt(n[2])})"
