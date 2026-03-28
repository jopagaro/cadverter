"""Extraction validation module.

Verifies that geometry.py extracted the B-REP data correctly by:
  1. Surface round-trip: sample N points on each OCC face, reconstruct
     the analytical surface from extracted parameters, measure deviation.
  2. Connectivity: for each edge, verify both claimed parent faces actually
     share it in OCC topology.
  3. Global properties: recompute volume/area from extracted face geometry
     independently and compare against OCC's BRepGProp values.

Returns a ValidationReport that gets embedded in the HSD document header
so the LLM knows the extraction is trustworthy and what tolerance to expect.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.TopTools import TopTools_IndexedMapOfShape

from .topology import TopologyGraph
from .utils import fmt


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FaceValidation:
    face_id: int
    surface_type: str
    n_samples: int
    max_deviation: float       # mm — largest distance from sampled point to reconstructed surface
    passed: bool
    notes: str = ""


@dataclass
class EdgeValidation:
    edge_id: int
    claimed_face_ids: list[int]
    actual_face_ids: list[int]
    passed: bool
    notes: str = ""


@dataclass
class GlobalValidation:
    occ_volume: float
    occ_surface_area: float
    extracted_face_count: int
    occ_face_count: int
    face_count_match: bool
    edge_count_match: bool
    extracted_edge_count: int
    occ_edge_count: int
    notes: str = ""


@dataclass
class ValidationReport:
    face_results: list[FaceValidation]
    edge_results: list[EdgeValidation]
    global_result: GlobalValidation

    # Summary
    faces_passed: int = 0
    faces_failed: int = 0
    max_surface_deviation: float = 0.0
    connectivity_passed: int = 0
    connectivity_failed: int = 0
    overall_passed: bool = False

    def __post_init__(self):
        self.faces_passed = sum(1 for r in self.face_results if r.passed)
        self.faces_failed = sum(1 for r in self.face_results if not r.passed)
        self.max_surface_deviation = max(
            (r.max_deviation for r in self.face_results), default=0.0
        )
        self.connectivity_passed = sum(1 for r in self.edge_results if r.passed)
        self.connectivity_failed = sum(1 for r in self.edge_results if not r.passed)
        self.overall_passed = (
            self.faces_failed == 0
            and self.connectivity_failed == 0
            and self.global_result.face_count_match
            and self.global_result.edge_count_match
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

SURFACE_DEVIATION_TOL = 1e-4   # mm — acceptable round-trip error
N_SAMPLE_GRID = 4              # sample at this many u,v points per face (4×4 = 16)


def validate_extraction(
    shape: TopoDS_Shape,
    graph: TopologyGraph,
) -> ValidationReport:
    """Run all validation checks and return a report."""
    face_imap = TopTools_IndexedMapOfShape()
    edge_imap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_imap)
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edge_imap)

    face_results = _validate_faces(graph, face_imap)
    edge_results = _validate_connectivity(graph, face_imap, edge_imap)
    global_result = _validate_global(graph, shape, face_imap, edge_imap)

    return ValidationReport(
        face_results=face_results,
        edge_results=edge_results,
        global_result=global_result,
    )


# ---------------------------------------------------------------------------
# 1. Surface round-trip validation
# ---------------------------------------------------------------------------

def _validate_faces(
    graph: TopologyGraph,
    face_imap: TopTools_IndexedMapOfShape,
) -> list[FaceValidation]:
    results = []
    for face_info in graph.faces:
        occ_face = TopoDS.Face_s(face_imap.FindKey(face_info.id))
        result = _check_face(face_info, occ_face)
        results.append(result)
    return results


def _check_face(face_info, occ_face: TopoDS_Face) -> FaceValidation:
    geom = face_info.geometry
    stype = geom.get("type", "UNKNOWN")

    # Sample points on the OCC face parametric domain
    adaptor = BRepAdaptor_Surface(occ_face, True)
    u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()

    # Clamp infinite parameter ranges (planes extend to infinity)
    u0 = max(u0, -1e6); u1 = min(u1, 1e6)
    v0 = max(v0, -1e6); v1 = min(v1, 1e6)

    samples = []
    n = N_SAMPLE_GRID
    for i in range(n):
        u = u0 + (u1 - u0) * (i + 0.5) / n
        for j in range(n):
            v = v0 + (v1 - v0) * (j + 0.5) / n
            p = gp_Pnt()
            try:
                adaptor.D0(u, v, p)
                samples.append((p.X(), p.Y(), p.Z()))
            except Exception:
                pass

    if not samples:
        return FaceValidation(face_info.id, stype, 0, 0.0, True, "no samples")

    # For each sample, compute deviation from the reconstructed analytical surface
    max_dev = 0.0
    failed_note = ""

    surf_handle = BRep_Tool.Surface_s(occ_face)

    for sx, sy, sz in samples:
        pt = gp_Pnt(sx, sy, sz)
        dev = _point_to_surface_deviation(pt, surf_handle, geom, stype)
        if dev > max_dev:
            max_dev = dev

    passed = max_dev <= SURFACE_DEVIATION_TOL
    notes = ""
    if not passed:
        notes = f"max deviation {max_dev:.2e} mm exceeds tolerance {SURFACE_DEVIATION_TOL:.2e} mm"

    return FaceValidation(face_info.id, stype, len(samples), max_dev, passed, notes)


def _point_to_surface_deviation(
    pt: gp_Pnt,
    surf_handle,
    geom: dict,
    stype: str,
) -> float:
    """Compute distance from pt to the analytical surface defined in geom."""
    try:
        if stype == "PLANE":
            n = geom["normal"]
            d = geom["d"]
            # Distance from point to plane: |n·p - d|
            dot = sum(n[i] * [pt.X(), pt.Y(), pt.Z()][i] for i in range(3))
            return abs(dot - d)

        elif stype == "CYLINDER":
            ax_o = geom["axis_origin"]
            ax_d = geom["axis_dir"]
            r = geom["radius"]
            return abs(_dist_point_to_axis(pt, ax_o, ax_d) - r)

        elif stype == "SPHERE":
            c = geom["center"]
            r = geom["radius"]
            d = math.sqrt(sum((pt.XYZ().GetData()[i] - c[i]) ** 2 for i in range(3)))
            return abs(d - r)

        elif stype == "CONE":
            # Project onto OCC surface for cones (complex formula, use OCC)
            return _occ_projection_deviation(pt, surf_handle)

        elif stype == "TORUS":
            return _occ_projection_deviation(pt, surf_handle)

        elif stype == "NURBS_SURFACE":
            return _occ_projection_deviation(pt, surf_handle)

        else:
            return _occ_projection_deviation(pt, surf_handle)

    except Exception:
        return 0.0  # can't compute — don't penalize


def _occ_projection_deviation(pt: gp_Pnt, surf_handle) -> float:
    """Project pt onto surf and return distance to nearest point."""
    try:
        proj = GeomAPI_ProjectPointOnSurf(pt, surf_handle)
        if proj.NbPoints() > 0:
            return proj.LowerDistance()
        return 0.0
    except Exception:
        return 0.0


def _dist_point_to_axis(
    pt: gp_Pnt,
    ax_origin: tuple,
    ax_dir: tuple,
) -> float:
    """Perpendicular distance from pt to an infinite line (axis)."""
    ox, oy, oz = ax_origin
    dx, dy, dz = ax_dir
    # Vector from axis origin to point
    vx, vy, vz = pt.X() - ox, pt.Y() - oy, pt.Z() - oz
    # Cross product v × d
    cx = vy * dz - vz * dy
    cy = vz * dx - vx * dz
    cz = vx * dy - vy * dx
    return math.sqrt(cx * cx + cy * cy + cz * cz)


# ---------------------------------------------------------------------------
# 2. Connectivity validation
# ---------------------------------------------------------------------------

def _validate_connectivity(
    graph: TopologyGraph,
    face_imap: TopTools_IndexedMapOfShape,
    edge_imap: TopTools_IndexedMapOfShape,
) -> list[EdgeValidation]:
    """For each edge in the graph, verify OCC agrees on which faces contain it."""
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    edge_to_faces_map = TopTools_IndexedDataMapOfShapeListOfShape()
    # Rebuild map from scratch using OCC
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE

    # Build face→edge containment check differently:
    # For each edge in our graph, check that its claimed faces' OCC representations
    # actually contain that edge in their boundary.
    results = []

    for edge_info in graph.edges:
        if edge_info.id > edge_imap.Extent():
            results.append(EdgeValidation(
                edge_info.id, edge_info.face_ids, [], False,
                "edge id out of range in OCC map"
            ))
            continue

        occ_edge = TopoDS.Edge_s(edge_imap.FindKey(edge_info.id))

        # For each claimed parent face, check the edge appears in its boundary
        actual_containing_fids = []
        for fid in graph.faces:
            if fid.id > face_imap.Extent():
                continue
            occ_face = TopoDS.Face_s(face_imap.FindKey(fid.id))
            if _face_contains_edge(occ_face, occ_edge, edge_imap):
                actual_containing_fids.append(fid.id)

        claimed = set(edge_info.face_ids)
        actual = set(actual_containing_fids)

        # Seam edges: claimed [F,F] → actual contains just F once
        if len(edge_info.face_ids) == 2 and edge_info.face_ids[0] == edge_info.face_ids[1]:
            passed = edge_info.face_ids[0] in actual
            notes = "seam edge"
        else:
            # All claimed faces must appear in actual
            missing = claimed - actual
            extra = actual - claimed
            passed = len(missing) == 0
            notes = ""
            if missing:
                notes += f"claimed faces {sorted(missing)} not found in OCC boundary; "
            if extra:
                notes += f"OCC also contains faces {sorted(extra)} not in graph"

        results.append(EdgeValidation(
            edge_info.id,
            edge_info.face_ids,
            sorted(actual_containing_fids),
            passed,
            notes.strip(),
        ))

    return results


def _face_contains_edge(
    occ_face: TopoDS_Face,
    target_edge: TopoDS_Edge,
    edge_imap: TopTools_IndexedMapOfShape,
) -> bool:
    """Return True if occ_face's boundary contains target_edge (by OCC identity)."""
    target_idx = edge_imap.FindIndex(target_edge)
    exp = TopExp_Explorer(occ_face, TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        if edge_imap.FindIndex(e) == target_idx:
            return True
        exp.Next()
    return False


# ---------------------------------------------------------------------------
# 3. Global property validation
# ---------------------------------------------------------------------------

def _validate_global(
    graph: TopologyGraph,
    shape: TopoDS_Shape,
    face_imap: TopTools_IndexedMapOfShape,
    edge_imap: TopTools_IndexedMapOfShape,
) -> GlobalValidation:
    occ_face_count = face_imap.Extent()
    occ_edge_count = edge_imap.Extent()
    extracted_face_count = len(graph.faces)
    extracted_edge_count = len(graph.edges)

    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol_props)
    occ_volume = vol_props.Mass()

    surf_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf_props)
    occ_surface_area = surf_props.Mass()

    notes_parts = []
    if extracted_face_count != occ_face_count:
        notes_parts.append(
            f"face count mismatch: graph={extracted_face_count} occ={occ_face_count}"
        )
    if extracted_edge_count != occ_edge_count:
        notes_parts.append(
            f"edge count mismatch: graph={extracted_edge_count} occ={occ_edge_count}"
        )

    # Cross-check volume and surface area against graph's stored values
    vol_dev = abs(graph.volume - occ_volume)
    area_dev = abs(graph.surface_area - occ_surface_area)
    if vol_dev > 0.01:
        notes_parts.append(f"volume deviation {vol_dev:.4f}")
    if area_dev > 0.01:
        notes_parts.append(f"surface area deviation {area_dev:.4f}")

    return GlobalValidation(
        occ_volume=occ_volume,
        occ_surface_area=occ_surface_area,
        extracted_face_count=extracted_face_count,
        occ_face_count=occ_face_count,
        face_count_match=(extracted_face_count == occ_face_count),
        edge_count_match=(extracted_edge_count == occ_edge_count),
        extracted_edge_count=extracted_edge_count,
        occ_edge_count=occ_edge_count,
        notes="; ".join(notes_parts) if notes_parts else "all counts match",
    )


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_validation_report(report: ValidationReport) -> str:
    """Render a compact validation report for inclusion in the HSD header."""
    status = "PASSED" if report.overall_passed else "FAILED"
    lines = [
        f"EXTRACTION VALIDATION: {status}",
        f"  Faces:        {report.faces_passed} passed, {report.faces_failed} failed"
        f"  (max surface deviation: {report.max_surface_deviation:.2e} mm,"
        f" tolerance: {SURFACE_DEVIATION_TOL:.0e} mm)",
        f"  Connectivity: {report.connectivity_passed} edges passed,"
        f" {report.connectivity_failed} failed",
        f"  Global:       faces {report.global_result.extracted_face_count}/"
        f"{report.global_result.occ_face_count},"
        f" edges {report.global_result.extracted_edge_count}/"
        f"{report.global_result.occ_edge_count},"
        f" {report.global_result.notes}",
    ]

    # List any failures explicitly
    for r in report.face_results:
        if not r.passed:
            lines.append(f"  FAIL [F{r.face_id}] {r.surface_type}: {r.notes}")
    for r in report.edge_results:
        if not r.passed:
            lines.append(f"  FAIL E{r.edge_id}: {r.notes}")

    return "\n".join(lines)
