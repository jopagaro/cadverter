"""High-level one-call API for CADVERT.

Most users only need this::

    import cadvert
    result = cadvert.analyze("bracket.step")
    print(result.to_text())          # LLM-ready structured text (the flagship)
    data  = result.to_dict()         # JSON-safe dict for programmatic consumers
    graph = result.to_graph()        # networkx face-adjacency graph for GNNs

The five-stage pipeline (ingest → topology → features → spatial → document)
is still available module-by-module for advanced use; this wrapper just wires
it together, handles the mesh/B-REP split, and exposes every representation
from a single object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ingest import load_step, IngestError, PartMetadata, SUPPORTED_EXTENSIONS

__all__ = ["analyze", "CadvertResult", "IngestError", "SUPPORTED_EXTENSIONS"]


@dataclass
class CadvertResult:
    """Everything CADVERT knows about one CAD file, in every representation.

    Attributes are plain data — no OCC objects leak out except ``shape``,
    which is kept only so downstream helpers (rendering, cadquery edits) can
    reuse the already-loaded geometry.
    """

    source_path: Path
    metadata: PartMetadata
    body_count: int

    # B-REP analysis (None for mesh formats)
    graph: Any = None                       # TopologyGraph
    features: list = field(default_factory=list)   # list[DetectedFeature]
    spatial: list = field(default_factory=list)    # list[SpatialRelationship]

    # Kept for optional downstream reuse (rendering, cadquery). Not serialized.
    shape: Any = field(default=None, repr=False)
    _mesh_info: Optional[dict] = field(default=None, repr=False)

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def is_mesh(self) -> bool:
        return self.metadata.is_mesh

    @property
    def units(self) -> str:
        return self.metadata.units

    @property
    def format(self) -> str:
        return self.metadata.source_format

    # ── Text representation (the flagship LLM context) ────────────────────────
    def to_text(self) -> str:
        """Compact, LLM-ready structured document (a few KB).

        This is the representation to drop into a prompt or system message.
        """
        from .document import render_tier0

        return render_tier0(
            self.graph,
            self.source_path,
            features=self.features,
            spatial=self.spatial,
            units=self.metadata.units,
            gdt_annotations=self.metadata.gdt_annotations or None,
            mesh_info=self._mesh_info,
        )

    def to_full_text(self, rendered_views=None, validation_report=None) -> str:
        """Complete multi-section HSD document (full face/edge dump)."""
        from .document import render_document

        return render_document(
            self.graph,
            self.source_path,
            features=self.features,
            spatial=self.spatial,
            rendered_views=rendered_views,
            validation_report=validation_report,
            units=self.metadata.units,
            gdt_annotations=self.metadata.gdt_annotations or None,
            mesh_info=self._mesh_info,
        )

    # ── Structured dict / JSON (programmatic consumers) ───────────────────────
    def to_dict(self) -> dict:
        """JSON-safe dict of the analysis — for quoting engines, CI checks, DBs."""
        d: dict[str, Any] = {
            "part": self.source_path.stem,
            "file": self.source_path.name,
            "format": self.metadata.source_format,
            "units": self.metadata.units,
            "is_mesh": self.metadata.is_mesh,
            "schema": self.metadata.schema or None,
            "originating_system": self.metadata.originating_system or None,
        }

        if self.metadata.is_mesh:
            d["global"] = _jsonify(self._mesh_info or {})
            return d

        g = self.graph
        d["global"] = {
            "body_count": g.body_count,
            "face_count": len(g.faces),
            "edge_count": len(g.edges),
            "vertex_count": len(g.vertices),
            "bounding_box": _jsonify(g.bounding_box),
            "volume": g.volume,
            "surface_area": g.surface_area,
            "center_of_mass": _jsonify(g.center_of_mass),
        }
        d["faces"] = [
            {
                "id": f.id,
                "type": f.geometry.get("type", "UNKNOWN"),
                "area": f.area,
                "edge_ids": list(f.edge_ids),
                "geometry": _jsonify(f.geometry),
            }
            for f in g.faces
        ]
        d["edges"] = [
            {
                "id": e.id,
                "type": e.geometry.get("type", "UNKNOWN"),
                "face_ids": list(e.face_ids),
                "dihedral_angle": e.dihedral_angle,
                "convexity": e.convexity,
            }
            for e in g.edges
        ]
        d["features"] = [
            {
                "type": ft.feature_type,
                "face_ids": list(ft.face_ids),
                "edge_ids": list(ft.edge_ids),
                "parameters": _jsonify(ft.parameters),
                "confidence": ft.confidence,
                "standard_match": ft.standard_match,
                "notes": ft.notes,
            }
            for ft in self.features
        ]
        d["spatial"] = [
            {
                "description": r.description,
                "value": r.value,
                "from_ref": r.from_ref,
                "to_ref": r.to_ref,
                "notes": r.notes,
            }
            for r in self.spatial
        ]
        if self.metadata.gdt_annotations:
            d["gdt"] = [
                {
                    "annotation_type": a.annotation_type,
                    "symbol": a.symbol,
                    "gdt_type": a.gdt_type,
                    "tolerance_value": a.tolerance_value,
                    "tolerance_lower": a.tolerance_lower,
                    "tolerance_upper": a.tolerance_upper,
                    "datum_refs": list(a.datum_refs),
                }
                for a in self.metadata.gdt_annotations
            ]
        return d

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize :meth:`to_dict` to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    # ── Graph representation (occwl-style, for GNNs) ──────────────────────────
    def to_graph(self):
        """Face-adjacency graph as a ``networkx.Graph``.

        Nodes are faces (attrs: type, area, geometry); an edge connects two
        faces that share a B-REP edge (attrs: edge id, curve type, dihedral
        angle, convexity). This is the representation UV-Net / GNN pipelines
        consume. Requires the ``graph`` extra: ``pip install cadvert[graph]``.
        """
        if self.metadata.is_mesh or self.graph is None:
            raise ValueError(
                "to_graph() requires B-REP topology; mesh formats (STL/OBJ) "
                "have no face-adjacency graph."
            )
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "to_graph() needs networkx. Install with: pip install cadvert[graph]"
            ) from exc

        g = nx.Graph()
        for f in self.graph.faces:
            g.add_node(
                f.id,
                type=f.geometry.get("type", "UNKNOWN"),
                area=f.area,
                geometry=f.geometry,
            )
        for e in self.graph.edges:
            if len(e.face_ids) == 2:
                a, b = e.face_ids
                g.add_edge(
                    a, b,
                    edge_id=e.id,
                    curve_type=e.geometry.get("type", "UNKNOWN"),
                    dihedral_angle=e.dihedral_angle,
                    convexity=e.convexity,
                )
        return g

    # ── Point-cloud representation (for PointNet / 3D CNNs) ────────────────────
    def to_points(self, count: int = 2048, *, deflection: float = 0.1, seed: int = 0):
        """Sample ``count`` surface points as an (N, 3) numpy array.

        Works for both B-REP and mesh inputs (B-REP is tessellated on demand).
        Area-weighted via trimesh when installed; otherwise falls back to
        tessellation vertices. Install ``cadvert[mesh]`` for proper sampling.
        """
        from .mesh import sample_points

        return sample_points(self.shape, count, deflection=deflection, seed=seed)

    # ── Rendering passthrough ─────────────────────────────────────────────────
    def render(self, output_dir: str | Path, image_size=(1200, 900)) -> list[Path]:
        """Render orthographic + isometric PNG views. Requires cadvert[server]."""
        from .renderer import render_shape

        return render_shape(
            self.shape, Path(output_dir),
            image_size=image_size,
            stem=self.source_path.stem,
        )


def analyze(
    path: str | Path,
    *,
    features: bool = True,
    spatial: bool = True,
    pull_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> CadvertResult:
    """Load a CAD file and run the full analysis pipeline in one call.

    Supports ``.step .stp .iges .igs .brep`` (full B-REP analysis),
    ``.stl .obj`` (global properties only), and ``.dxf`` (2D drawing —
    returns a :class:`~cadvert.dxf.DxfResult`). For 3D inputs, returns a
    :class:`CadvertResult` exposing every representation — ``.to_text()``,
    ``.to_dict()``, ``.to_json()``, ``.to_graph()``, ``.to_points()``.

    Args:
        path: CAD file to analyze.
        features: run manufacturing-feature detection (B-REP only).
        spatial: compute spatial relationships (requires ``features=True``).
        pull_direction: draft/undercut analysis direction for spatial stage.

    Raises:
        IngestError: unsupported format or parse failure.
    """
    src = Path(path)

    # DXF is 2D — it has its own light pipeline and result type.
    if src.suffix.lower() == ".dxf":
        from .dxf import load_dxf
        return load_dxf(src)

    shape, body_count, metadata = load_step(src)

    result = CadvertResult(
        source_path=src,
        metadata=metadata,
        body_count=body_count,
        shape=shape,
    )

    if metadata.is_mesh:
        result._mesh_info = _compute_mesh_info(metadata, shape)
        return result

    from .topology import build_topology
    result.graph = build_topology(shape, body_count)

    if features:
        from .features import detect_features
        result.features = detect_features(result.graph)

        if spatial:
            from .spatial import compute_spatial_relationships
            result.spatial = compute_spatial_relationships(
                result.graph, result.features,
                shape=shape,
                pull_direction=pull_direction,
            )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
    """Recursively convert tuples/sets to lists so the result is JSON-safe."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    return obj


def _compute_mesh_info(metadata: PartMetadata, shape) -> dict:
    """Basic global properties for a mesh shape (no topology available)."""
    info: dict[str, Any] = {
        "format": metadata.source_format,
        "triangle_count": metadata.triangle_count,
        "units": metadata.units,
    }
    try:
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        vol = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vol)
        info["volume"] = vol.Mass()

        surf = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, surf)
        info["surface_area"] = surf.Mass()

        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if not box.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            info["bbox"] = {"X": (xmin, xmax), "Y": (ymin, ymax), "Z": (zmin, zmax)}
    except Exception:
        pass
    return info
