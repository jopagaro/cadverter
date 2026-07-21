"""DXF (2D drawing) ingest — the flat-part counterpart to the B-REP pipeline.

DXF is the format of 2D shop drawings, laser/waterjet/plasma cutting, and
sheet-metal flat patterns. It carries no 3D solid, so it gets its own light
pipeline: read the vector entities, layers, dimensions and title-block text,
and emit the same kind of LLM-ready structured document CADVERT produces for
3D parts — plus a JSON-safe dict.

Requires the ``dxf`` extra: ``pip install cadvert[dxf]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ingest import IngestError

# AutoCAD $INSUNITS code → unit name
_INSUNITS = {
    0: "unitless", 1: "inch", 2: "feet", 4: "mm", 5: "cm",
    6: "m", 8: "microinch", 9: "mil", 10: "yd", 11: "angstrom",
    12: "nm", 13: "micron", 14: "dm", 21: "us_survey_foot",
}


@dataclass
class DxfEntity:
    kind: str                       # LINE, CIRCLE, ARC, LWPOLYLINE, TEXT, ...
    layer: str = "0"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DxfResult:
    """Analysis of a 2D DXF drawing. Mirrors CadvertResult's output interface."""

    source_path: Path
    units: str = "unitless"
    dxf_version: str = ""
    layers: list[str] = field(default_factory=list)
    entities: list[DxfEntity] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] | None = None
    format: str = "DXF"
    is_mesh: bool = False

    # ── Counts by entity kind ─────────────────────────────────────────────────
    @property
    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for e in self.entities:
            c[e.kind] = c.get(e.kind, 0) + 1
        return c

    # ── LLM-ready text ────────────────────────────────────────────────────────
    def to_text(self) -> str:
        lines: list[str] = []
        p = self.source_path
        lines.append(f"DRAWING: {p.stem}")
        lines.append(f"FILE: {p.name}  |  UNITS: {self.units}  |  DXF: {self.dxf_version}")
        lines.append("FORMAT: DXF (2D vector drawing — no 3D solid geometry)")
        lines.append("")

        lines.append("[EXTENTS]")
        if self.bounds:
            for ax in ("X", "Y"):
                lo, hi = self.bounds[ax]
                lines.append(f"{ax}[{lo:.3f}, {hi:.3f}]  span {hi - lo:.3f}")
        else:
            lines.append("(no measurable extents)")
        lines.append("")

        lines.append(f"[LAYERS — {len(self.layers)}]")
        lines.append("  " + ", ".join(self.layers) if self.layers else "  (none)")
        lines.append("")

        counts = self.counts
        total = len(self.entities)
        lines.append(f"[ENTITIES — {total} total]")
        for kind in sorted(counts):
            lines.append(f"  {counts[kind]}× {kind}")
        lines.append("")

        # Dimensions and text are the numbers a reader cares about most
        dims = [e for e in self.entities if e.kind == "DIMENSION"]
        if dims:
            lines.append(f"[DIMENSIONS — {len(dims)}]")
            for e in dims:
                m = e.params.get("measurement")
                txt = e.params.get("text", "")
                mstr = f"{m:.3f}" if isinstance(m, (int, float)) else "?"
                lines.append(f"  {mstr} {self.units}" + (f'  "{txt}"' if txt else ""))
            lines.append("")

        texts = [e for e in self.entities if e.kind in ("TEXT", "MTEXT")]
        if texts:
            lines.append(f"[TEXT / ANNOTATIONS — {len(texts)}]")
            for e in texts:
                content = e.params.get("text", "").strip()
                if content:
                    lines.append(f'  [{e.layer}] "{content}"')
            lines.append("")

        # Notable geometry: circles/arcs (holes, radii) with exact values
        circles = [e for e in self.entities if e.kind in ("CIRCLE", "ARC")]
        if circles:
            lines.append(f"[CIRCLES / ARCS — {len(circles)}]")
            for e in circles:
                r = e.params.get("radius", 0.0)
                cx, cy = e.params.get("center", (0.0, 0.0))[:2]
                tag = "⌀%.3f" % (2 * r) if e.kind == "CIRCLE" else "R%.3f" % r
                lines.append(f"  {e.kind} {tag} at ({cx:.3f}, {cy:.3f})")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ── JSON-safe dict ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "drawing": self.source_path.stem,
            "file": self.source_path.name,
            "format": "DXF",
            "units": self.units,
            "dxf_version": self.dxf_version,
            "is_mesh": False,
            "bounds": {k: list(v) for k, v in self.bounds.items()} if self.bounds else None,
            "layers": list(self.layers),
            "entity_counts": self.counts,
            "entities": [
                {"kind": e.kind, "layer": e.layer, "params": _jsonify(e.params)}
                for e in self.entities
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


def load_dxf(path: str | Path) -> DxfResult:
    """Read a DXF file into a :class:`DxfResult`.

    Raises IngestError on a missing file or an unreadable/corrupt DXF.
    """
    src = Path(path)
    if not src.exists():
        raise IngestError(f"File not found: {src}")

    try:
        import ezdxf
        from ezdxf import bbox
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise IngestError(
            "DXF support needs ezdxf. Install with: pip install cadvert[dxf]"
        ) from exc

    try:
        doc = ezdxf.readfile(str(src))
    except (IOError, ezdxf.DXFStructureError) as exc:
        raise IngestError(f"Could not read DXF: {exc}") from exc

    units_code = int(doc.header.get("$INSUNITS", 0))
    result = DxfResult(
        source_path=src,
        units=_INSUNITS.get(units_code, "unitless"),
        dxf_version=doc.dxfversion,
        layers=sorted(layer.dxf.name for layer in doc.layers),
    )

    msp = doc.modelspace()
    for e in msp:
        try:
            result.entities.append(_read_entity(e))
        except Exception:
            # Never let one odd entity sink the whole read
            result.entities.append(DxfEntity(kind=e.dxftype(), layer=_layer_of(e)))

    # Overall drawing extents
    try:
        ext = bbox.extents(msp)
        if ext.has_data:
            result.bounds = {
                "X": (ext.extmin.x, ext.extmax.x),
                "Y": (ext.extmin.y, ext.extmax.y),
            }
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Entity readers
# ---------------------------------------------------------------------------

def _layer_of(e) -> str:
    try:
        return e.dxf.layer
    except Exception:
        return "0"


def _read_entity(e) -> DxfEntity:
    kind = e.dxftype()
    layer = _layer_of(e)
    p: dict[str, Any] = {}

    if kind == "LINE":
        s, t = e.dxf.start, e.dxf.end
        p["start"] = (s.x, s.y)
        p["end"] = (t.x, t.y)
        p["length"] = math.dist((s.x, s.y), (t.x, t.y))
    elif kind == "CIRCLE":
        c = e.dxf.center
        p["center"] = (c.x, c.y)
        p["radius"] = e.dxf.radius
        p["diameter"] = 2 * e.dxf.radius
    elif kind == "ARC":
        c = e.dxf.center
        p["center"] = (c.x, c.y)
        p["radius"] = e.dxf.radius
        p["start_angle"] = e.dxf.start_angle
        p["end_angle"] = e.dxf.end_angle
    elif kind == "LWPOLYLINE":
        pts = [(x, y) for x, y, *_ in e.get_points()]
        p["points"] = pts
        p["closed"] = bool(e.closed)
        p["vertex_count"] = len(pts)
    elif kind == "POLYLINE":
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        p["points"] = pts
        p["closed"] = bool(e.is_closed)
        p["vertex_count"] = len(pts)
    elif kind == "TEXT":
        p["text"] = e.dxf.text
    elif kind == "MTEXT":
        p["text"] = e.text
    elif kind == "DIMENSION":
        try:
            p["measurement"] = float(e.get_measurement())
        except Exception:
            pass
        try:
            p["text"] = e.dxf.text
        except Exception:
            pass
    elif kind == "ELLIPSE":
        c = e.dxf.center
        p["center"] = (c.x, c.y)
        p["ratio"] = e.dxf.ratio
    elif kind == "POINT":
        loc = e.dxf.location
        p["location"] = (loc.x, loc.y)
    elif kind == "INSERT":
        p["block"] = e.dxf.name
        ip = e.dxf.insert
        p["at"] = (ip.x, ip.y)

    return DxfEntity(kind=kind, layer=layer, params=p)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    return obj
