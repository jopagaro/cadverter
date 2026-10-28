"""Feature detection from topology graph.

Every detector returns a confidence score in [0.0, 1.0].
  1.0 = topologically unambiguous
  0.7–0.9 = high confidence, one condition is softer
  0.4–0.6 = plausible but ambiguous geometry
  < 0.4   = suppressed (not emitted)

CONFIDENCE_THRESHOLD controls the emit cutoff.

Detection order matters because claimed_faces prevents double-counting:
  1. Fillets / chamfers          (claim faces before cylinder detector runs)
  2. Coaxial cylinder groups     (counterbores, countersinks)
  3. Individual cylinders        (through holes, blind holes, bosses)
  4. Slots
  5. Pockets
  6. Patterns
"""

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .topology import TopologyGraph, FaceInfo, EdgeInfo
from .utils import fmt, normalize, angle_between_normals


CONFIDENCE_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectedFeature:
    feature_type: str
    face_ids: list[int]
    edge_ids: list[int]
    parameters: dict[str, Any]
    confidence: float = 1.0
    standard_match: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Standard size tables
# ---------------------------------------------------------------------------

_METRIC_CLEARANCE = [
    ("M2 clearance",  2.4), ("M2.5 clearance", 2.9), ("M3 clearance",  3.4),
    ("M4 clearance",  4.5), ("M5 clearance",   5.5), ("M6 clearance",  6.6),
    ("M8 clearance",  9.0), ("M10 clearance", 11.0), ("M12 clearance", 13.5),
    ("M16 clearance", 17.5), ("M20 clearance", 22.0),
]
_METRIC_TAP = [
    ("M2 tap", 1.6), ("M2.5 tap", 2.05), ("M3 tap", 2.5), ("M4 tap", 3.3),
    ("M5 tap", 4.2), ("M6 tap", 5.0), ("M8 tap", 6.8), ("M10 tap", 8.5),
    ("M12 tap", 10.2), ("M16 tap", 14.0), ("M20 tap", 17.5),
]
_FRACTIONAL_INCH = [
    ("1/16\"", 1.588), ("5/64\"", 1.984), ("3/32\"", 2.381), ("7/64\"", 2.778),
    ("1/8\"",  3.175), ("9/64\"", 3.572), ("5/32\"", 3.969), ("11/64\"", 4.366),
    ("3/16\"", 4.762), ("13/64\"", 5.159), ("7/32\"", 5.556),
    ("1/4\"",  6.350), ("9/32\"", 7.144), ("5/16\"", 7.938), ("11/32\"", 8.731),
    ("3/8\"",  9.525), ("7/16\"", 11.113), ("1/2\"", 12.700),
    ("9/16\"", 14.288), ("5/8\"", 15.875), ("11/16\"", 17.463), ("3/4\"", 19.050),
]
_ALL_STANDARDS = _METRIC_CLEARANCE + _METRIC_TAP + _FRACTIONAL_INCH


def _match_standard(diameter_mm: float, tol_mm: float = 0.15) -> str | None:
    best, best_dist = None, tol_mm
    for label, d in _ALL_STANDARDS:
        dist = abs(d - diameter_mm)
        if dist < best_dist:
            best_dist = dist
            best = label
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_features(graph: TopologyGraph) -> list[DetectedFeature]:
    face_by_id  = {f.id: f for f in graph.faces}
    edge_by_id  = {e.id: e for e in graph.edges}

    cyl_fids   = [f.id for f in graph.faces if f.geometry.get("type") == "CYLINDER"]
    cone_fids  = [f.id for f in graph.faces if f.geometry.get("type") == "CONE"]
    torus_fids = [f.id for f in graph.faces if f.geometry.get("type") == "TORUS"]
    plane_fids = [f.id for f in graph.faces if f.geometry.get("type") == "PLANE"]

    claimed: set[int] = set()   # face ids already assigned
    features: list[DetectedFeature] = []

    def _emit(feat: DetectedFeature) -> None:
        if feat.confidence >= CONFIDENCE_THRESHOLD:
            features.append(feat)
            claimed.update(feat.face_ids)

    # ── 1. Fillets ────────────────────────────────────────────────────────────
    for fid in torus_fids:
        _emit(_detect_fillet(fid, face_by_id, edge_by_id))
    # Rounded straight edges — the commonest fillet — are partial cylinders, not tori.
    for fid in cyl_fids:
        feat = _detect_edge_fillet(fid, face_by_id, edge_by_id)
        if feat:
            _emit(feat)

    # ── 2. Chamfers ───────────────────────────────────────────────────────────
    for fid in plane_fids:
        if fid in claimed:
            continue
        feat = _detect_chamfer(fid, face_by_id, edge_by_id)
        if feat:
            _emit(feat)

    # ── 3. Cone faces that are NOT chamfers → countersink seats ───────────────
    for fid in cone_fids:
        if fid in claimed:
            continue
        feat = _detect_countersink_cone(fid, face_by_id, edge_by_id)
        if feat:
            _emit(feat)

    # ── 4. Coaxial cylinder groups → counterbores ─────────────────────────────
    coaxial_groups = _find_coaxial_groups(cyl_fids, face_by_id, edge_by_id)
    for group in coaxial_groups:
        if any(fid in claimed for fid in group):
            continue
        feat = _detect_counterbore(group, face_by_id, edge_by_id, graph)
        if feat:
            _emit(feat)
            continue
        # Not a counterbore — fall through to individual cylinder detection

    # ── 5. Individual cylinders ───────────────────────────────────────────────
    for fid in cyl_fids:
        if fid in claimed:
            continue
        feat = _classify_cylinder(fid, face_by_id, edge_by_id, graph, claimed)
        if feat:
            _emit(feat)

    # ── 6. Slots ──────────────────────────────────────────────────────────────
    slots = _detect_slots(cyl_fids, plane_fids, face_by_id, edge_by_id, claimed)
    for feat in slots:
        _emit(feat)

    # ── 7. Pockets ────────────────────────────────────────────────────────────
    for feat in _detect_pockets(plane_fids, face_by_id, edge_by_id, claimed, graph):
        _emit(feat)

    # ── 8. Patterns ───────────────────────────────────────────────────────────
    for feat in _detect_patterns(features):
        features.append(feat)   # patterns don't claim new faces

    return features


# ---------------------------------------------------------------------------
# Cylinder classification
# ---------------------------------------------------------------------------

def _classify_cylinder(
    fid: int,
    face_by_id: dict,
    edge_by_id: dict,
    graph: TopologyGraph,
    claimed: set[int],
) -> DetectedFeature | None:
    face = face_by_id[fid]
    geom = face.geometry
    axis   = normalize(geom["axis_dir"])
    ax_org = geom["axis_origin"]
    radius = geom["radius"]
    diam   = 2 * radius

    # All full-circle edges (excluding seams)
    circle_edges = [
        edge_by_id[eid] for eid in face.edge_ids
        if _is_real_circle(edge_by_id[eid])
    ]

    if not circle_edges:
        return None

    # Gather end-cap info for each circle edge
    caps = []   # (edge, adj_face, dot_with_axis, is_on_boundary)
    for ce in circle_edges:
        adj_fids = [f for f in ce.face_ids if f != fid]
        if not adj_fids:
            continue
        adj_face = face_by_id[adj_fids[0]]
        ag = adj_face.geometry
        dot = 0.0
        if ag.get("type") == "PLANE":
            dot = abs(sum(normalize(ag["normal"])[i] * axis[i] for i in range(3)))
        elif ag.get("type") == "CYLINDER":
            # hole through curved surface — treat as a valid cap
            dot = 0.5
        else:
            dot = 0.3  # torus/cone/nurbs — partial credit
        caps.append((ce, adj_face, dot, _is_outer_face(adj_face, graph)))

    if len(caps) == 0:
        return None

    # Material inside the cylinder makes it a boss or shaft — never a hole, however many
    # capped ends it has. (Older graphs without the flag fall back to the cap count.)
    concave = geom.get("concave")
    if concave is False or (concave is None and len(caps) == 1):
        return _boss_or_partial(fid, face, geom, axis, radius, diam,
                                caps, face_by_id, edge_by_id)

    # Two or more caps: determine through/blind
    return _through_or_blind(fid, face, geom, axis, ax_org, radius, diam,
                              caps, face_by_id, edge_by_id, graph)


def _through_or_blind(
    fid, face, geom, axis, ax_org, radius, diam, caps,
    face_by_id, edge_by_id, graph,
) -> DetectedFeature:
    # Compute depth from the two outermost circles along the axis
    circle_positions = []
    for ce, adj_face, dot, is_outer in caps:
        c = ce.geometry["center"]
        pos = sum(c[i] * axis[i] for i in range(3))
        circle_positions.append((pos, ce, adj_face, dot, is_outer))
    circle_positions.sort(key=lambda x: x[0])
    top = circle_positions[-1]   # higher along axis
    bot = circle_positions[0]    # lower along axis
    depth = top[0] - bot[0]

    # ── Through or blind: look at what closes each end ─────────────────────────
    # An end is open if the hole runs out through the face there, closed if that face
    # is the hole's floor (or a drill point). Judged from the faces themselves, so a
    # boss or rib standing above the surface doesn't turn through holes "blind" — which
    # the old bounding-box test did. Falls back to that test only when a face's
    # orientation is unknown.
    if len(circle_positions) >= 2 and top[0] - bot[0] > 1e-9:
        down = tuple(-c for c in axis)
        top_closed = _end_is_closed(top[1], top[2], toward_interior=down)
        bot_closed = _end_is_closed(bot[1], bot[2], toward_interior=axis)
    else:
        top_closed = bot_closed = None
    top_is_outer = top[4] if top_closed is None else not top_closed
    bot_is_outer = bot[4] if bot_closed is None else not bot_closed
    entry_cap = exit_cap = None

    if top_is_outer and bot_is_outer:
        # Both on boundary → through hole
        ftype = "THROUGH_HOLE"
        entry_cap, exit_cap = top[2], bot[2]
        confidence = _through_hole_confidence(caps, axis)
    elif top_is_outer and not bot_is_outer:
        ftype = "BLIND_HOLE"
        entry_cap, blind_cap = top[2], bot[2]
        confidence = _blind_hole_confidence(entry_cap, blind_cap, axis, depth, radius)
    elif bot_is_outer and not top_is_outer:
        ftype = "BLIND_HOLE"
        entry_cap, blind_cap = bot[2], top[2]
        confidence = _blind_hole_confidence(entry_cap, blind_cap, axis, depth, radius)
    else:
        # Neither cap is on the part boundary → internal hole or pocket floor
        ftype = "BLIND_HOLE"
        entry_cap, blind_cap = top[2], bot[2]
        confidence = 0.55  # ambiguous

    std = _match_standard(diam)
    params: dict[str, Any] = {
        "diameter": diam,
        "radius": radius,
        "depth": depth,
        "axis": axis,
        "axis_origin": ax_org,
        "center_2d": _center_2d(ax_org, axis),
        "n_circle_edges": len(caps),
    }

    if ftype == "THROUGH_HOLE" and entry_cap and exit_cap:
        params["entry_face_id"] = entry_cap.id
        params["exit_face_id"]  = exit_cap.id
    elif ftype == "BLIND_HOLE" and entry_cap:
        params["entry_face_id"]  = entry_cap.id if entry_cap else None
        params["bottom_face_id"] = blind_cap.id if not top_is_outer else bot[2].id

    return DetectedFeature(
        feature_type=ftype,
        face_ids=[fid],
        edge_ids=[ce.id for ce, _, _, _ in [(c[1], c[2], c[3], c[4]) for c in circle_positions]],
        parameters=params,
        confidence=confidence,
        standard_match=std,
        notes=f"∅{fmt(diam)} mm, depth={fmt(depth)} mm"
              + (f", {len(caps)} cap edges" if len(caps) > 2 else "")
              + (f" — {std}" if std else ""),
    )


def _end_is_closed(cap_edge, adj_face, toward_interior) -> bool | None:
    """Does *adj_face* close this end of a hole? True for a floor or drill point, False if
    the hole opens out through it, None when the face's orientation isn't known.

    *toward_interior* points from this end along the axis into the hole.
    """
    g = adj_face.geometry
    kind = g.get("type")
    if kind == "PLANE":
        n = g.get("outward_normal")
        if n is None:
            return None
        # A floor's outward normal looks back up into the hole; an exit face looks away.
        return sum(n[i] * toward_interior[i] for i in range(3)) > 0.5
    if kind == "CONE":
        # Drill point: the cone's tip lies beyond this end. Countersink: it lies inside.
        apex, c = g["apex"], cap_edge.geometry["center"]
        return sum((apex[i] - c[i]) * toward_interior[i] for i in range(3)) < 0
    # Breaking out into a curved or blended surface still means the hole is open here.
    return False


def _through_hole_confidence(caps, axis) -> float:
    """Score 1.0 for two planar caps with normals aligned to axis, lower for curved caps."""
    score = 1.0
    for ce, adj_face, dot, is_outer in caps:
        if adj_face.geometry.get("type") != "PLANE":
            score -= 0.15   # hole through curved surface — still valid, reduced confidence
        elif dot < 0.95:
            score -= 0.1    # cap plane not perfectly perpendicular to axis
    if len(caps) > 2:
        score -= 0.05 * (len(caps) - 2)  # extra caps (fillets at entry/exit)
    return max(0.4, min(1.0, score))


def _blind_hole_confidence(entry_face, blind_face, axis, depth, radius) -> float:
    score = 0.9
    if entry_face and entry_face.geometry.get("type") != "PLANE":
        score -= 0.2
    if blind_face and blind_face.geometry.get("type") != "PLANE":
        score -= 0.1
    # Aspect ratio sanity: depth < 0.1 * radius is almost certainly not a hole
    if depth > 0 and depth < 0.1 * radius:
        score -= 0.3
    return max(0.4, min(1.0, score))


def _boss_or_partial(
    fid, face, geom, axis, radius, diam, caps, face_by_id, edge_by_id,
) -> DetectedFeature | None:
    if not caps:
        return None
    ce, adj_face, dot, is_outer = caps[0]
    depth = abs(face.area / (2 * math.pi * radius))
    std = _match_standard(diam)
    confidence = 0.75 if is_outer else 0.5
    return DetectedFeature(
        feature_type="BOSS",
        face_ids=[fid],
        edge_ids=[ce.id],
        parameters={
            "diameter": diam, "radius": radius, "height": depth,
            "axis": axis, "axis_origin": geom["axis_origin"],
            "top_face_id": adj_face.id,
        },
        confidence=confidence,
        standard_match=std,
        notes=f"∅{fmt(diam)} mm boss, height≈{fmt(depth)} mm",
    )


# ---------------------------------------------------------------------------
# Coaxial group detection → counterbores
# ---------------------------------------------------------------------------

def _find_coaxial_groups(
    cyl_fids: list[int],
    face_by_id: dict,
    edge_by_id: dict,
) -> list[list[int]]:
    """Return lists of cylinder face IDs that share the same axis (within tolerance)."""
    # Two cylinders belong together if their axes coincide and they touch — either along
    # a shared circular edge, or through a flat step face between them, which is how a
    # counterbore is built (different radii can never share an edge).
    adj: dict[int, set[int]] = defaultdict(set)
    for i, fid_a in enumerate(cyl_fids):
        for fid_b in cyl_fids[i + 1:]:
            if (_are_coaxial(face_by_id[fid_a], face_by_id[fid_b], edge_by_id)
                    or _share_step_face(face_by_id[fid_a], face_by_id[fid_b], face_by_id, edge_by_id)):
                adj[fid_a].add(fid_b)
                adj[fid_b].add(fid_a)

    visited: set[int] = set()
    groups = []
    for fid in cyl_fids:
        if fid in visited:
            continue
        group = []
        queue = [fid]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            queue.extend(adj[cur] - visited)
        if len(group) > 1:
            groups.append(group)
    return groups


def _are_coaxial(
    face_a: FaceInfo,
    face_b: FaceInfo,
    edge_by_id: dict,
) -> bool:
    """True if two cylinder faces share a circular edge (are directly connected) and
    their axes are parallel and collinear within tolerance."""
    ga, gb = face_a.geometry, face_b.geometry
    ax_a = normalize(ga["axis_dir"])
    ax_b = normalize(gb["axis_dir"])

    # Axes must be parallel
    dot = abs(sum(ax_a[i] * ax_b[i] for i in range(3)))
    if dot < 0.999:
        return False

    # Must share a circular edge
    shared = set(face_a.edge_ids) & set(face_b.edge_ids)
    if not shared:
        return False
    for eid in shared:
        if edge_by_id[eid].geometry.get("type") == "CIRCLE":
            # Verify the centers are collinear with the axis
            oc = edge_by_id[eid].geometry["center"]
            ao = ga["axis_origin"]
            v = tuple(oc[i] - ao[i] for i in range(3))
            cross = (
                v[1]*ax_a[2] - v[2]*ax_a[1],
                v[2]*ax_a[0] - v[0]*ax_a[2],
                v[0]*ax_a[1] - v[1]*ax_a[0],
            )
            dist = math.sqrt(sum(x*x for x in cross))
            if dist < 1.0:   # center within 1 mm of axis
                return True
    return False


def _circle_neighbours(face: FaceInfo, edge_by_id: dict) -> set[int]:
    """Faces adjacent to *face* across its full circular edges."""
    return {fid for eid in face.edge_ids if _is_real_circle(edge_by_id[eid])
            for fid in edge_by_id[eid].face_ids if fid != face.id}


def _axes_coincide(ga: dict, gb: dict, tol: float = 0.5) -> bool:
    ax = normalize(ga["axis_dir"])
    if abs(sum(ax[i] * normalize(gb["axis_dir"])[i] for i in range(3))) < 0.999:
        return False
    v = tuple(gb["axis_origin"][i] - ga["axis_origin"][i] for i in range(3))
    along = sum(v[i] * ax[i] for i in range(3))
    off = math.sqrt(max(0.0, sum(x * x for x in v) - along * along))
    return off < tol


def _share_step_face(face_a: FaceInfo, face_b: FaceInfo, face_by_id: dict, edge_by_id: dict) -> bool:
    """Coaxial hole cylinders joined by a planar step face (a counterbore's floor)."""
    ga, gb = face_a.geometry, face_b.geometry
    if ga.get("concave") is False or gb.get("concave") is False:
        return False
    if not _axes_coincide(ga, gb):
        return False
    shared = _circle_neighbours(face_a, edge_by_id) & _circle_neighbours(face_b, edge_by_id)
    return any(face_by_id[f].geometry.get("type") == "PLANE" for f in shared)


def _detect_counterbore(
    group: list[int],
    face_by_id: dict,
    edge_by_id: dict,
    graph: TopologyGraph,
) -> DetectedFeature | None:
    """Two coaxial cylinders, larger on entry side = counterbore."""
    if len(group) != 2:
        return None
    fid_a, fid_b = group
    ga, gb = face_by_id[fid_a].geometry, face_by_id[fid_b].geometry
    r_a, r_b = ga["radius"], gb["radius"]
    if abs(r_a - r_b) < 0.01:
        return None   # same radius → stepped bore, not counterbore

    bore_fid   = fid_a if r_a < r_b else fid_b
    cbore_fid  = fid_b if r_a < r_b else fid_a
    bore_geom  = face_by_id[bore_fid].geometry
    cbore_geom = face_by_id[cbore_fid].geometry
    axis = normalize(bore_geom["axis_dir"])
    if bore_geom.get("concave") is False or cbore_geom.get("concave") is False:
        return None   # a stepped shaft or boss, not a hole

    # Get all circle edges for both cylinders
    all_circle_edges = []
    for fid in [bore_fid, cbore_fid]:
        for eid in face_by_id[fid].edge_ids:
            e = edge_by_id[eid]
            if _is_real_circle(e):
                all_circle_edges.append(e)

    if len(all_circle_edges) < 2:
        return None

    # Depths: project all circle centers onto axis
    positions = sorted(
        [(sum(e.geometry["center"][i] * axis[i] for i in range(3)), e)
         for e in all_circle_edges],
        key=lambda x: x[0]
    )
    total_depth = positions[-1][0] - positions[0][0]
    cbore_depth = _cbore_depth(cbore_fid, bore_fid, face_by_id, edge_by_id, axis)

    # The wide part must be at the entry: entirely on one side of the bore.
    def span(fid):
        ps = [sum(edge_by_id[e].geometry["center"][i] * axis[i] for i in range(3))
              for e in face_by_id[fid].edge_ids if _is_real_circle(edge_by_id[e])]
        return (min(ps), max(ps)) if ps else (0.0, 0.0)
    (c_lo, c_hi), (b_lo, b_hi) = span(cbore_fid), span(bore_fid)
    if not (c_lo >= b_hi - 1e-6 or c_hi <= b_lo + 1e-6):
        return None

    # Does the bore run out the far side, or stop at a floor?
    far_down = c_lo >= b_hi - 1e-6            # counterbore above, bore continues downward
    far_pos = b_lo if far_down else b_hi
    toward_interior = axis if far_down else tuple(-c for c in axis)
    through = True
    for eid in face_by_id[bore_fid].edge_ids:
        e = edge_by_id[eid]
        if _is_real_circle(e) and abs(sum(e.geometry["center"][i] * axis[i] for i in range(3)) - far_pos) < 1e-6:
            others = [f for f in e.face_ids if f != bore_fid]
            if others:
                through = not bool(_end_is_closed(e, face_by_id[others[0]], toward_interior))

    std = _match_standard(2 * bore_geom["radius"])
    return DetectedFeature(
        feature_type="COUNTERBORE",
        face_ids=[bore_fid, cbore_fid],
        edge_ids=[e.id for _, e in positions],
        parameters={
            "bore_diameter":   2 * bore_geom["radius"],
            "cbore_diameter":  2 * cbore_geom["radius"],
            "total_depth":     total_depth,
            "cbore_depth":     cbore_depth,
            "axis":            axis,
            "axis_origin":     bore_geom["axis_origin"],
            "bore_face_id":    bore_fid,
            "cbore_face_id":   cbore_fid,
            "through":         through,
        },
        confidence=0.90,
        standard_match=std,
        notes=(
            f"Counterbore: bore ∅{fmt(2*bore_geom['radius'])} mm depth={fmt(total_depth)} mm, "
            f"cbore ∅{fmt(2*cbore_geom['radius'])} mm depth≈{fmt(cbore_depth)} mm"
            + ("" if through else " (blind)")
        ),
    )


def _cbore_depth(
    cbore_fid: int,
    bore_fid: int,
    face_by_id: dict,
    edge_by_id: dict,
    axis: tuple,
) -> float:
    """Counterbore depth: the height of the wide cylinder itself."""
    ps = [sum(edge_by_id[e].geometry["center"][i] * axis[i] for i in range(3))
          for e in face_by_id[cbore_fid].edge_ids if _is_real_circle(edge_by_id[e])]
    if len(ps) >= 2:
        return max(ps) - min(ps)
    # Fallback for the rare case of cylinders meeting along a shared edge.
    shared_eids = set(face_by_id[cbore_fid].edge_ids) & set(face_by_id[bore_fid].edge_ids)
    for eid in shared_eids:
        e = edge_by_id[eid]
        if e.geometry.get("type") == "CIRCLE":
            transition_pos = sum(e.geometry["center"][i] * axis[i] for i in range(3))
            # Find top of counterbore (outermost circle edge)
            top_pos = transition_pos
            for eid2 in face_by_id[cbore_fid].edge_ids:
                e2 = edge_by_id[eid2]
                if _is_real_circle(e2) and eid2 != eid:
                    pos = sum(e2.geometry["center"][i] * axis[i] for i in range(3))
                    top_pos = max(top_pos, pos)
            return abs(top_pos - transition_pos)
    return 0.0


# ---------------------------------------------------------------------------
# Countersink (cone seat between cylinder and entry plane)
# ---------------------------------------------------------------------------

def _detect_countersink_cone(
    fid: int,
    face_by_id: dict,
    edge_by_id: dict,
) -> DetectedFeature | None:
    face = face_by_id[fid]
    geom = face.geometry
    axis = normalize(geom["axis_dir"])

    # Must adjoin at least one cylinder face (the bore) and one plane (the entry)
    adj_fids = set()
    for eid in face.edge_ids:
        e = edge_by_id[eid]
        for afid in e.face_ids:
            if afid != fid:
                adj_fids.add(afid)

    adj_types = {face_by_id[afid].geometry.get("type") for afid in adj_fids}
    if "CYLINDER" not in adj_types:
        return None

    bore_fid = next(
        (afid for afid in adj_fids
         if face_by_id[afid].geometry.get("type") == "CYLINDER"),
        None
    )
    if bore_fid is None:
        return None
    bore_geom = face_by_id[bore_fid].geometry

    # A countersink widens beyond the bore toward the surface. A drill point — the cone a
    # twist drill leaves at the bottom of a blind hole — never gets wider than the bore;
    # it narrows to a tip. Leave that to the hole classifier, which reads it as a closed end.
    cone_radii = [edge_by_id[e].geometry["radius"] for e in face.edge_ids if _is_real_circle(edge_by_id[e])]
    if not cone_radii or max(cone_radii) <= bore_geom["radius"] + 0.01:
        return None

    # Through, or a blind countersunk hole? Look at the bore's far end.
    cone_edges = set(face.edge_ids)
    through = True
    for eid in face_by_id[bore_fid].edge_ids:
        e = edge_by_id[eid]
        if eid in cone_edges or not _is_real_circle(e):
            continue
        c = e.geometry["center"]
        toward_cone = tuple(geom["apex"][i] - c[i] for i in range(3))
        norm = math.sqrt(sum(v * v for v in toward_cone)) or 1.0
        others = [f for f in e.face_ids if f != bore_fid]
        if others:
            through = not bool(_end_is_closed(e, face_by_id[others[0]], tuple(v / norm for v in toward_cone)))

    return DetectedFeature(
        feature_type="COUNTERSINK",
        face_ids=[fid, bore_fid],
        edge_ids=list(face.edge_ids),
        parameters={
            "cone_half_angle":  geom["half_angle"],
            "bore_diameter":    2 * bore_geom["radius"],
            "axis":             axis,
            "apex":             geom["apex"],
            "bore_face_id":     bore_fid,
            "cone_face_id":     fid,
            "through":          through,
        },
        confidence=0.85,
        notes=(
            f"Countersink: {fmt(geom['half_angle']*2)}° included angle, "
            f"bore ∅{fmt(2*bore_geom['radius'])} mm"
        ),
    )


# ---------------------------------------------------------------------------
# Fillet
# ---------------------------------------------------------------------------

def _detect_fillet(
    fid: int,
    face_by_id: dict,
    edge_by_id: dict,
) -> DetectedFeature:
    face = face_by_id[fid]
    geom = face.geometry
    adj_fids = set()
    for eid in face.edge_ids:
        for afid in edge_by_id[eid].face_ids:
            if afid != fid:
                adj_fids.add(afid)
    return DetectedFeature(
        feature_type="FILLET",
        face_ids=[fid],
        edge_ids=list(face.edge_ids),
        parameters={
            "radius":            geom["minor_radius"],
            "adjacent_face_ids": sorted(adj_fids),
            "center":            geom["center"],
            "axis_dir":          geom["axis_dir"],
        },
        confidence=1.0,
        notes=f"Fillet R={fmt(geom['minor_radius'])} mm",
    )


def _detect_edge_fillet(
    fid: int,
    face_by_id: dict,
    edge_by_id: dict,
) -> DetectedFeature | None:
    """A rounded straight edge: a partial cylinder that blends smoothly (tangent) into
    the faces either side, sweeping less than a half-turn. A slot's end sweeps exactly a
    half-turn and a hole a full turn, so neither is mistaken for one."""
    face = face_by_id[fid]
    geom = face.geometry
    edges = [edge_by_id[eid] for eid in face.edge_ids]
    arcs = [e for e in edges if e.geometry.get("type") == "CIRCLE"]
    if any(e.geometry.get("is_full_circle") for e in arcs):
        return None
    radius = geom["radius"]
    if not arcs or any(e.geometry["arc_length"] / radius >= math.radians(179) for e in arcs):
        return None
    def same_cylinder(other_fid):
        g = face_by_id[other_fid].geometry
        return (g.get("type") == "CYLINDER" and abs(g["radius"] - radius) < 1e-6
                and _axes_coincide(geom, g, tol=1e-3))
    # The smooth joins must be to two other surfaces — a cylinder split along its seam
    # (as a slot's end often is) is joined smoothly to another piece of itself.
    tangent = [e for e in edges if e.convexity == "tangent"
               and not any(f != fid and same_cylinder(f) for f in e.face_ids)]
    if len({f for e in tangent for f in e.face_ids if f != fid}) < 2:
        return None
    adj_fids = sorted({f for e in tangent for f in e.face_ids if f != fid})
    convex = geom.get("concave") is False
    return DetectedFeature(
        feature_type="FILLET",
        face_ids=[fid],
        edge_ids=list(face.edge_ids),
        parameters={
            "radius":            radius,
            "adjacent_face_ids": adj_fids,
            "axis_dir":          geom["axis_dir"],
            "convex":            convex,
        },
        confidence=0.9,
        notes=f"Fillet R={fmt(radius)} mm ({'rounded outside edge' if convex else 'inside corner'})",
    )


# ---------------------------------------------------------------------------
# Chamfer
# ---------------------------------------------------------------------------

def _detect_chamfer(
    fid: int,
    face_by_id: dict,
    edge_by_id: dict,
) -> DetectedFeature | None:
    """A chamfer is a narrow flat bevel cut along an edge: its two long sides are parallel
    lines, each an outside corner onto one of the two faces it bevels between, at a
    shallow angle (a 45° chamfer meets each at 45°, where an ordinary corner is 90°).

    Its short ends may touch any number of faces — bevelling a box edge leaves it touching
    four — so faces are not counted.
    """
    face = face_by_id[fid]
    normal = face.geometry["normal"]
    lines = sorted((edge_by_id[eid] for eid in face.edge_ids
                    if edge_by_id[eid].geometry.get("type") == "LINE"),
                   key=lambda e: -e.geometry["length"])
    if len(lines) < 2:
        return None
    a, b = lines[0], lines[1]
    da, db = a.geometry["direction"], b.geometry["direction"]
    if abs(sum(da[i] * db[i] for i in range(3))) < 0.999:
        return None                                   # long sides must be parallel
    for e in (a, b):
        if e.convexity != "convex" or e.dihedral_angle is None:
            return None
        if not 10.0 <= e.dihedral_angle <= 80.0:      # a bevel, not a square corner
            return None
    beveled = {f for e in (a, b) for f in e.face_ids if f != fid}
    if len(beveled) != 2:
        return None

    # Width: distance between the two long sides.
    pa, pb = a.geometry["start"], b.geometry["start"]
    v = tuple(pb[i] - pa[i] for i in range(3))
    along = sum(v[i] * da[i] for i in range(3))
    width = math.sqrt(max(0.0, sum(x * x for x in v) - along * along))
    if width >= a.geometry["length"]:
        return None                                   # wider than long: a face, not a bevel
    confidence = 0.9 if width < 10.0 else 0.7
    angles = sorted(round(e.dihedral_angle, 1) for e in (a, b))

    return DetectedFeature(
        feature_type="CHAMFER",
        face_ids=[fid],
        edge_ids=list(face.edge_ids),
        parameters={
            "normal":            normal,
            "width":             width,
            "angles":            angles,
            "adjacent_face_ids": sorted(beveled),
        },
        confidence=confidence,
        notes=f"Chamfer, width {fmt(width)} mm at {angles[0]:g}°/{angles[1]:g}°",
    )


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------

def _detect_slots(
    cyl_fids: list[int],
    plane_fids: list[int],
    face_by_id: dict,
    edge_by_id: dict,
    claimed: set[int],
) -> list[DetectedFeature]:
    """A slot has two parallel half-cylinder end-caps and two parallel planar sides.
    Topological signature: two CYLINDER faces with parallel axes and ~180° arc
    (not full circles), connected by two planar faces."""
    plane_set = set(plane_fids)
    slots = []

    for fid in cyl_fids:
        if fid in claimed:
            continue
        face = face_by_id[fid]
        geom = face.geometry

        # Check for partial circles (arc, not full circle) in the edges
        partial_circles = [
            edge_by_id[eid] for eid in face.edge_ids
            if edge_by_id[eid].geometry.get("type") == "CIRCLE"
            and not edge_by_id[eid].geometry.get("is_full_circle")
        ]
        if not partial_circles:
            continue

        # Find adjacent cylinder with same radius and parallel axis
        adj_cyl_fids = []
        for eid in face.edge_ids:
            for afid in edge_by_id[eid].face_ids:
                if afid != fid and face_by_id[afid].geometry.get("type") == "CYLINDER":
                    adj_cyl_fids.append(afid)

        partner_fid = None
        for afid in adj_cyl_fids:
            if afid in claimed:
                continue
            ag = face_by_id[afid].geometry
            if abs(ag["radius"] - geom["radius"]) < 0.01:
                ax_a = normalize(geom["axis_dir"])
                ax_b = normalize(ag["axis_dir"])
                if abs(sum(ax_a[i]*ax_b[i] for i in range(3))) > 0.999:
                    partner_fid = afid
                    break

        if partner_fid is None:
            continue

        # Find the planar side walls between the two cylinders
        shared_planes = []
        for plane_fid in plane_fids:
            if plane_fid in claimed:
                continue
            plane_edges = set(face_by_id[plane_fid].edge_ids)
            touches_a = bool(plane_edges & set(face.edge_ids))
            touches_b = bool(plane_edges & set(face_by_id[partner_fid].edge_ids))
            if touches_a and touches_b:
                shared_planes.append(plane_fid)

        if len(shared_planes) < 1:
            continue

        axis = normalize(geom["axis_dir"])
        radius = geom["radius"]

        # Slot length = distance between the two cylinder axis origins projected onto
        # the direction perpendicular to both axis and the connecting plane normal
        ao_a = geom["axis_origin"]
        ao_b = face_by_id[partner_fid].geometry["axis_origin"]
        slot_length = math.sqrt(sum((ao_b[i]-ao_a[i])**2 for i in range(3)))

        face_ids = [fid, partner_fid] + shared_planes
        edge_ids = list(set(
            list(face.edge_ids)
            + list(face_by_id[partner_fid].edge_ids)
            + [eid for pf in shared_planes for eid in face_by_id[pf].edge_ids]
        ))

        slots.append(DetectedFeature(
            feature_type="SLOT",
            face_ids=face_ids,
            edge_ids=edge_ids,
            parameters={
                "width":          2 * radius,
                "radius":         radius,
                "length":         slot_length,
                "axis":           axis,
                "end_a_face_id":  fid,
                "end_b_face_id":  partner_fid,
                "wall_face_ids":  shared_planes,
            },
            confidence=0.80,
            notes=(
                f"Slot: width={fmt(2*radius)} mm, "
                f"length={fmt(slot_length)} mm, "
                f"axis={tuple(round(x,3) for x in axis)}"
            ),
        ))

    return slots


# ---------------------------------------------------------------------------
# Pocket detection
# ---------------------------------------------------------------------------

def _detect_pockets(
    plane_fids: list[int],
    face_by_id: dict,
    edge_by_id: dict,
    claimed: set[int],
    graph: TopologyGraph | None = None,
) -> list[DetectedFeature]:
    """A pocket is a floor with walls rising from it: a planar face whose every edge to
    the rest of the part is an inside (concave) corner, plus the walls and blends joined
    to it by inside corners.

    Edges into features already found (a hole drilled through the floor, a boss standing
    on it) are ignored, so they don't hide the pocket. Outside corners are never crossed,
    which keeps the part's own outer shape — and a raised block's base, whose outer edges
    are outside corners — from being mistaken for a pocket.
    """
    vertex_point = {v.id: v.point for v in graph.vertices} if graph is not None else {}
    used: set[int] = set()
    pockets = []

    for floor_fid in plane_fids:
        if floor_fid in claimed or floor_fid in used:
            continue
        floor = face_by_id[floor_fid]
        rim = [edge_by_id[eid] for eid in floor.edge_ids
               if any(f != floor_fid and f not in claimed for f in edge_by_id[eid].face_ids)]
        concave_rim = [e for e in rim if e.convexity == "concave"]
        if len(concave_rim) < 2 or any(e.convexity == "convex" for e in rim):
            continue

        # Walk out from the floor across inside corners — and across smooth (tangent)
        # joins into concave blends, so rounded pocket corners come along.
        faces, queue = {floor_fid}, [floor_fid]
        while queue:
            cur = queue.pop()
            for eid in face_by_id[cur].edge_ids:
                e = edge_by_id[eid]
                for nxt in e.face_ids:
                    if nxt == cur or nxt in faces or nxt in claimed:
                        continue
                    g = face_by_id[nxt].geometry
                    blend = g.get("type") in ("CYLINDER", "TORUS") and g.get("concave", True)
                    if e.convexity == "concave" or (e.convexity == "tangent" and blend):
                        faces.add(nxt)
                        queue.append(nxt)
        walls = sorted(faces - {floor_fid})
        if not walls:
            continue
        used |= faces

        params: dict[str, Any] = {
            "floor_face_id": floor_fid,
            "wall_face_ids": walls,
            "n_faces":       len(faces),
        }
        n = floor.geometry.get("outward_normal")
        if n and vertex_point:
            floor_level = sum(n[i] * floor.geometry["origin"][i] for i in range(3))
            heights = [sum(n[i] * vertex_point[vid][i] for i in range(3)) - floor_level
                       for f in walls for eid in face_by_id[f].edge_ids
                       for vid in edge_by_id[eid].vertex_ids if vid in vertex_point]
            if heights:
                params["depth"] = max(heights)

        depth_note = f", depth {fmt(params['depth'])} mm" if "depth" in params else ""
        pockets.append(DetectedFeature(
            feature_type="POCKET",
            face_ids=sorted(faces),
            edge_ids=sorted({eid for f in faces for eid in face_by_id[f].edge_ids}),
            parameters=params,
            confidence=0.85,
            notes=f"Pocket: floor F{floor_fid} with {len(walls)} wall faces{depth_note}",
        ))

    return pockets


def _planar_cluster(start_fid, plane_set, face_by_id, edge_by_id, visited):
    cluster_fids, internal_eids, boundary_eids = set(), set(), set()
    queue = [start_fid]
    while queue:
        fid = queue.pop()
        if fid in visited or fid not in plane_set:
            continue
        visited.add(fid); cluster_fids.add(fid)
        for eid in face_by_id[fid].edge_ids:
            for afid in edge_by_id[eid].face_ids:
                if afid == fid:
                    continue
                if afid in plane_set and afid not in visited:
                    internal_eids.add(eid); queue.append(afid)
                elif afid not in plane_set:
                    boundary_eids.add(eid)
    return (cluster_fids, internal_eids, boundary_eids) if cluster_fids else None


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def _detect_patterns(features: list[DetectedFeature]) -> list[DetectedFeature]:
    groups: dict[tuple, list[DetectedFeature]] = defaultdict(list)
    for feat in features:
        if feat.feature_type in ("THROUGH_HOLE", "BLIND_HOLE", "BOSS", "COUNTERBORE"):
            d = feat.parameters.get("diameter",
                feat.parameters.get("bore_diameter", 0))
            key = (feat.feature_type, round(d, 3))
            groups[key].append(feat)

    patterns = []
    for (ftype, diam), members in groups.items():
        if len(members) < 2:
            continue
        centers = [_feature_center(m) for m in members]
        if None in centers:
            continue
        notes_extra = _pattern_geometry(centers)
        patterns.append(DetectedFeature(
            feature_type="PATTERN",
            face_ids=[fid for m in members for fid in m.face_ids],
            edge_ids=[eid for m in members for eid in m.edge_ids],
            parameters={
                "child_type":     ftype,
                "count":          len(members),
                "diameter":       diam,
                "centers":        centers,
                "child_face_ids": [m.face_ids[0] for m in members if m.face_ids],
            },
            confidence=min(m.confidence for m in members),
            notes=f"{len(members)}× {ftype} ∅{fmt(diam)} mm. {notes_extra}",
        ))
    return patterns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_real_circle(edge: EdgeInfo) -> bool:
    return (
        edge.geometry.get("type") == "CIRCLE"
        and edge.geometry.get("is_full_circle")
        and len(edge.face_ids) == 2
        and edge.face_ids[0] != edge.face_ids[1]
    )


def _is_outer_face(face: FaceInfo, graph: TopologyGraph) -> bool:
    """Heuristic: a face is 'on the boundary' if its plane position matches a
    bounding box face, OR if it's a non-planar face directly on the BB surface."""
    bb = graph.bounding_box
    tol = 1.0
    g = face.geometry
    if g.get("type") != "PLANE":
        return False
    n = g["normal"]
    d = g["d"]
    for i, key in enumerate(("X", "Y", "Z")):
        if abs(abs(n[i]) - 1.0) < 0.05:
            plane_pos = d / (n[i] if abs(n[i]) > 1e-9 else 1.0)
            lo, hi = bb[key]
            if abs(plane_pos - lo) < tol or abs(plane_pos - hi) < tol:
                return True
    return False


def _center_2d(ao: tuple, axis: tuple) -> tuple:
    ax, ay, az = (abs(v) for v in axis)
    if az >= ax and az >= ay:
        return (ao[0], ao[1])
    elif ay >= ax:
        return (ao[0], ao[2])
    else:
        return (ao[1], ao[2])


def _feature_center(feat: DetectedFeature) -> tuple | None:
    ao = feat.parameters.get("axis_origin")
    return tuple(ao) if ao else None


def _pattern_geometry(centers: list[tuple]) -> str:
    n = len(centers)
    if n < 2:
        return ""
    dists = [
        math.sqrt(sum((centers[i][k]-centers[j][k])**2 for k in range(3)))
        for i in range(n) for j in range(i+1, n)
    ]
    min_d = min(dists)
    if n >= 2 and _are_collinear(centers):
        return f"Linear pattern, spacing={fmt(min_d)} mm"
    if n >= 3:
        bcd = min_d / math.sin(math.pi / n)
        return f"Circular pattern, BCD≈{fmt(bcd)} mm, spacing≈{fmt(min_d)} mm"
    return f"Spacing={fmt(min_d)} mm"


def _are_collinear(pts: list[tuple], tol: float = 1e-3) -> bool:
    if len(pts) < 3:
        return True
    p0, v = pts[0], tuple(pts[1][i]-pts[0][i] for i in range(3))
    mag = math.sqrt(sum(x*x for x in v))
    if mag < 1e-9:
        return False
    v = tuple(x/mag for x in v)
    for p in pts[2:]:
        diff = tuple(p[i]-p0[i] for i in range(3))
        cross = (v[1]*diff[2]-v[2]*diff[1], v[2]*diff[0]-v[0]*diff[2], v[0]*diff[1]-v[1]*diff[0])
        if math.sqrt(sum(x*x for x in cross)) > tol:
            return False
    return True
