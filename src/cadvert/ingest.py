"""CAD file ingestion — multi-format loader.

Supported formats
-----------------
Full B-REP pipeline (exact analytical geometry, features, topology):
  .step / .stp   ISO-10303 STEP (AP203, AP214, AP242)
  .iges / .igs   IGES 5.x
  .brep          OCC native B-REP

Mesh pipeline (global properties + rendering only, no feature detection):
  .stl           STereoLithography
  .obj           Wavefront OBJ  (requires trimesh: pip install trimesh)

The returned PartMetadata.is_mesh flag tells the caller which pipeline to use.
Mesh files skip topology, feature detection, spatial analysis, and validation —
those stages require exact analytical surface definitions that mesh formats
do not contain.

Unit detection for STEP: reads the DATA section via regex (OCC's StepBasic
API is unreliable across versions).  IGES and BREP default to mm.
AP242 GD&T is extracted from STEP files only.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from OCP.TopoDS import TopoDS_Shape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL


class IngestError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GDTAnnotation:
    """One GD&T annotation extracted from an AP242 STEP file."""
    annotation_type: str            # "DIMENSIONAL" | "GEOMETRIC"
    symbol: str                     # human-readable: "±0.050", "⊕0.010 |A|"
    gdt_type: str = ""              # "FLATNESS", "POSITION", "PLUS_MINUS", ...
    tolerance_value: float | None = None
    tolerance_lower: float | None = None
    tolerance_upper: float | None = None
    datum_refs: list[str] = field(default_factory=list)
    face_step_ids: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class PartMetadata:
    """File-level metadata from any supported CAD format."""
    source_format: str = "STEP"     # "STEP", "IGES", "BREP", "STL", "OBJ"
    is_mesh: bool = False           # True → mesh pipeline, False → B-REP pipeline
    units: str = "mm"
    unit_scale_to_mm: float = 1.0
    schema: str = ""                # AP203 / AP214 / AP242 (STEP only)
    originating_system: str = ""
    description: str = ""
    triangle_count: int = 0         # populated for mesh formats
    gdt_annotations: list[GDTAnnotation] = field(default_factory=list)

# Keep old name as alias so existing imports don't break
StepMetadata = PartMetadata


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Map extension → format tag
_EXT_FORMAT: dict[str, str] = {
    ".step": "STEP",
    ".stp":  "STEP",
    ".iges": "IGES",
    ".igs":  "IGES",
    ".brep": "BREP",
    ".stl":  "STL",
    ".obj":  "OBJ",
}

SUPPORTED_EXTENSIONS = sorted(_EXT_FORMAT.keys())


def load_step(path: str | Path) -> tuple[TopoDS_Shape, int, PartMetadata]:
    """Load any supported CAD file and return (shape, body_count, metadata).

    body_count is 0 for mesh files (no concept of solids in a mesh).
    Raises IngestError on failure.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")

    ext = path.suffix.lower()
    fmt = _EXT_FORMAT.get(ext)
    if fmt is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise IngestError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )

    if fmt == "STEP":
        return _load_step(path)
    elif fmt == "IGES":
        return _load_iges(path)
    elif fmt == "BREP":
        return _load_brep(path)
    elif fmt == "STL":
        return _load_stl(path)
    elif fmt == "OBJ":
        return _load_obj(path)
    else:
        raise IngestError(f"Internal error: unhandled format {fmt}")


# ---------------------------------------------------------------------------
# B-REP loaders (full pipeline)
# ---------------------------------------------------------------------------

def _load_step(path: Path) -> tuple[TopoDS_Shape, int, PartMetadata]:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise IngestError(f"STEP reader failed (status {status}): {path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise IngestError(f"No usable shape in STEP file: {path}")

    body_count = _count_bodies(shape)
    meta = _parse_step_metadata(path)
    meta.source_format = "STEP"
    return shape, body_count, meta


def _load_iges(path: Path) -> tuple[TopoDS_Shape, int, PartMetadata]:
    from OCP.IGESControl import IGESControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = IGESControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise IngestError(f"IGES reader failed (status {status}): {path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise IngestError(f"No usable shape in IGES file: {path}")

    body_count = _count_bodies(shape)
    meta = PartMetadata(source_format="IGES", units="mm", unit_scale_to_mm=1.0)
    # IGES files sometimes encode units in their global section
    meta.units, meta.unit_scale_to_mm = _parse_iges_units(path)
    return shape, body_count, meta


def _load_brep(path: Path) -> tuple[TopoDS_Shape, int, PartMetadata]:
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools

    builder = BRep_Builder()
    shape = TopoDS_Shape()
    ok = BRepTools.Read_s(shape, str(path), builder)
    if not ok or shape.IsNull():
        raise IngestError(f"Failed to read BREP file: {path}")

    body_count = _count_bodies(shape)
    meta = PartMetadata(source_format="BREP", units="mm", unit_scale_to_mm=1.0)
    return shape, body_count, meta


# ---------------------------------------------------------------------------
# Mesh loaders (partial pipeline)
# ---------------------------------------------------------------------------

def _load_stl(path: Path) -> tuple[TopoDS_Shape, int, PartMetadata]:
    """Load STL as a triangulated shell. Returns is_mesh=True metadata."""
    shape = _stl_to_shape(path)
    tri_count = _count_triangles(shape)
    meta = PartMetadata(
        source_format="STL",
        is_mesh=True,
        units="mm",           # STL has no unit encoding — mm is the convention
        unit_scale_to_mm=1.0,
        triangle_count=tri_count,
    )
    return shape, 0, meta


def _stl_to_shape(path: Path) -> TopoDS_Shape:
    """Try multiple OCC STL APIs (they vary across OCC versions)."""
    # Method 1: RWStl (preferred, available in OCC 7.4+)
    try:
        from OCP.RWStl import RWStl
        from OCP.Message import Message_ProgressRange
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Shape as _Shape

        tri_shape = RWStl.ReadFile_s(str(path), Message_ProgressRange())
        if tri_shape is not None and not tri_shape.IsNull():
            # Wrap triangulated data into a TopoDS_Shape via sewing
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
            sewing = BRepBuilderAPI_Sewing(1e-3)
            sewing.Add(tri_shape)
            sewing.Perform()
            result = sewing.SewedShape()
            if result is not None and not result.IsNull():
                return result
            return tri_shape
    except Exception:
        pass

    # Method 2: StlAPI_Reader (older OCC)
    try:
        from OCP.StlAPI import StlAPI_Reader
        reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        reader.Read(shape, str(path))
        if not shape.IsNull():
            return shape
    except Exception:
        pass

    raise IngestError(
        f"Could not read STL file with available OCC APIs: {path}\n"
        "Ensure pythonocc-core 7.4+ is installed."
    )


def _load_obj(path: Path) -> tuple[TopoDS_Shape, int, PartMetadata]:
    """Load OBJ by converting to STL via trimesh, then loading as mesh."""
    try:
        import trimesh  # type: ignore
    except ImportError:
        raise IngestError(
            "OBJ support requires trimesh: pip install trimesh\n"
            "Alternatively, convert your OBJ to STEP using FreeCAD or Blender."
        )

    try:
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as e:
        raise IngestError(f"trimesh failed to load OBJ: {e}")

    if mesh is None or (hasattr(mesh, "is_empty") and mesh.is_empty):
        raise IngestError(f"OBJ file contains no geometry: {path}")

    # Write a temporary STL and reload via OCC so the rest of the pipeline
    # works with a proper TopoDS_Shape
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        mesh.export(str(tmp_path))
        shape = _stl_to_shape(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    tri_count = len(mesh.faces)
    meta = PartMetadata(
        source_format="OBJ",
        is_mesh=True,
        units="mm",
        unit_scale_to_mm=1.0,
        triangle_count=tri_count,
    )
    return shape, 0, meta


# ---------------------------------------------------------------------------
# Body / triangle counting
# ---------------------------------------------------------------------------

def _count_bodies(shape: TopoDS_Shape) -> int:
    count = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        count += 1
        exp.Next()
    if count > 0:
        return count

    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        count += 1
        exp.Next()
    return max(count, 1)


def _count_triangles(shape: TopoDS_Shape) -> int:
    """Count triangles in the triangulation of a mesh shape."""
    try:
        from OCP.BRep import BRep_Tool
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopoDS import TopoDS
        total = 0
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            loc = face.Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is not None:
                total += tri.NbTriangles()
            exp.Next()
        return total
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# IGES unit extraction
# ---------------------------------------------------------------------------

def _parse_iges_units(path: Path) -> tuple[str, float]:
    """Read the IGES Global Section to extract the length unit."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # IGES Global Section: parameter 15 is unit flag, 16 is unit name
        # Format: fixed 72-char columns, section ID 'G' in col 73
        global_lines = [l[:72] for l in text.splitlines() if len(l) > 72 and l[72] == "G"]
        global_section = "".join(global_lines).replace("\n", "")
        params = global_section.split(",")
        # Parameter 15 (0-indexed: 14) is the unit flag
        if len(params) > 14:
            unit_flag = params[14].strip().strip(";")
            _iges_unit_map = {
                "1": ("inch", 25.4), "2": ("mm", 1.0), "3": ("ft", 304.8),
                "4": ("mi", 1_609_344.0), "5": ("m", 1000.0), "6": ("km", 1_000_000.0),
                "7": ("mil", 0.0254), "8": ("µm", 0.001), "9": ("cm", 10.0),
                "10": ("µin", 0.0000254),
            }
            if unit_flag in _iges_unit_map:
                return _iges_unit_map[unit_flag]
    except Exception:
        pass
    return "mm", 1.0


# ---------------------------------------------------------------------------
# STEP metadata parsing
# ---------------------------------------------------------------------------

def _parse_step_metadata(path: Path) -> PartMetadata:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return PartMetadata()

    meta = PartMetadata()
    meta.schema = _parse_schema(text)
    meta.originating_system = _parse_originating_system(text)
    meta.description = _parse_description(text)
    meta.units, meta.unit_scale_to_mm = _parse_units(text)

    if _is_ap242(meta.schema):
        meta.gdt_annotations = _parse_gdt_ap242(text)

    return meta


def _is_ap242(schema: str) -> bool:
    s = schema.upper()
    return "AP242" in s or "242" in s


def _parse_schema(text: str) -> str:
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text, re.IGNORECASE)
    return m.group(1).split(";")[0].strip() if m else ""


def _parse_originating_system(text: str) -> str:
    m = re.search(r"ORIGINATING_SYSTEM\s*\(\s*'([^']*)'", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(
        r"FILE_NAME\s*\([^,]*,[^,]*,\([^)]*\),\([^)]*\),\s*'[^']*'\s*,\s*'([^']*)'",
        text, re.IGNORECASE,
    )
    return m.group(1) if m else ""


def _parse_description(text: str) -> str:
    m = re.search(r"FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'", text, re.IGNORECASE)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Unit extraction (STEP)
# ---------------------------------------------------------------------------

_SI_PREFIX_SCALE = {
    ".ATTO.": 1e-18, ".FEMTO.": 1e-15, ".PICO.": 1e-12, ".NANO.": 1e-9,
    ".MICRO.": 1e-6, ".MILLI.": 1e-3, ".CENTI.": 1e-2, ".DECI.": 1e-1,
    "$": 1.0, "": 1.0,
    ".DECA.": 1e1, ".HECTO.": 1e2, ".KILO.": 1e3, ".MEGA.": 1e6, ".GIGA.": 1e9,
}

_SI_BASE_LENGTH = {".METRE.": (1000.0, "m")}

_NAMED_UNIT_MM = {
    "INCH": (25.4, "inch"), "IN": (25.4, "inch"), '"': (25.4, "inch"),
    "FOOT": (304.8, "foot"), "FT": (304.8, "foot"),
    "YARD": (914.4, "yard"), "MILE": (1_609_344.0, "mile"),
    "MM": (1.0, "mm"), "MILLIMETER": (1.0, "mm"), "MILLIMETRE": (1.0, "mm"),
    "CM": (10.0, "cm"), "CENTIMETER": (10.0, "cm"), "CENTIMETRE": (10.0, "cm"),
    "M": (1000.0, "m"), "METER": (1000.0, "m"), "METRE": (1000.0, "m"),
}


def _parse_units(text: str) -> tuple[str, float]:
    # Compound entity form: LENGTH_UNIT() ... SI_UNIT(.PREFIX.,.UNIT.)
    m = re.search(
        r"LENGTH_UNIT\s*\(\s*\).*?SI_UNIT\s*\(\s*([^,)]*)\s*,\s*([^)]*)\s*\)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return _si_decode(m.group(1).strip(), m.group(2).strip())

    # Bare SI_UNIT with a length unit name
    for m in re.finditer(r"SI_UNIT\s*\(\s*([^,)]*)\s*,\s*([^)]*)\s*\)", text, re.IGNORECASE):
        prefix, unit = m.group(1).strip(), m.group(2).strip()
        if unit in _SI_BASE_LENGTH:
            return _si_decode(prefix, unit)

    # CONVERSION_BASED_UNIT('name', ...)
    m = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'", text, re.IGNORECASE)
    if m:
        name = m.group(1).upper().strip()
        for key, (mm_val, short) in _NAMED_UNIT_MM.items():
            if key in name:
                return short, mm_val

    return "mm", 1.0


def _si_decode(prefix: str, unit: str) -> tuple[str, float]:
    base_mm, base_short = _SI_BASE_LENGTH.get(unit, (1000.0, "m"))
    scale_to_mm = base_mm * _SI_PREFIX_SCALE.get(prefix, 1.0)

    if base_short == "m":
        short = {
            ".MILLI.": "mm", ".CENTI.": "cm", ".DECI.": "dm",
            ".KILO.": "km", "$": "m", "": "m",
        }.get(prefix, f"{prefix}m")
        return short, scale_to_mm
    return base_short, scale_to_mm


# ---------------------------------------------------------------------------
# AP242 GD&T parser
# ---------------------------------------------------------------------------

_GDT_SYMBOLS: dict[str, str] = {
    "FLATNESS_TOLERANCE": "⏥",
    "STRAIGHTNESS_TOLERANCE": "—",
    "CIRCULARITY_TOLERANCE": "○",
    "CYLINDRICITY_TOLERANCE": "⌭",
    "PROFILE_OF_A_LINE_TOLERANCE": "⌒",
    "PROFILE_OF_A_SURFACE_TOLERANCE": "⌓",
    "PERPENDICULARITY_TOLERANCE": "⊥",
    "ANGULARITY_TOLERANCE": "∠",
    "PARALLELISM_TOLERANCE": "∥",
    "POSITION_TOLERANCE": "⊕",
    "CONCENTRICITY_TOLERANCE": "◎",
    "SYMMETRY_TOLERANCE": "≡",
    "CIRCULAR_RUNOUT_TOLERANCE": "↗",
    "TOTAL_RUNOUT_TOLERANCE": "⟳",
}


def _parse_gdt_ap242(text: str) -> list[GDTAnnotation]:
    entity_map = _build_entity_map(text)
    annotations: list[GDTAnnotation] = []
    annotations.extend(_extract_plus_minus(entity_map))
    annotations.extend(_extract_geometric_tolerances(entity_map))
    return annotations


def _build_entity_map(text: str) -> dict[int, tuple[str, str]]:
    entity_map: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"#(\d+)\s*=\s*([A-Z_]+)\s*\((.*)\)\s*;?\s*$", line, re.IGNORECASE)
        if m:
            entity_map[int(m.group(1))] = (m.group(2).upper(), m.group(3))
    return entity_map


def _extract_plus_minus(entity_map: dict) -> list[GDTAnnotation]:
    annotations = []
    for eid, (ename, args) in entity_map.items():
        if ename != "PLUS_MINUS_TOLERANCE":
            continue
        refs = [int(r) for r in re.findall(r"#(\d+)", args)]
        lower = upper = None
        for ref_id in refs:
            ref_name, ref_args = entity_map.get(ref_id, ("", ""))
            if ref_name == "TOLERANCE_VALUE":
                nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", ref_args)
                if len(nums) >= 2:
                    lower, upper = float(nums[0]), float(nums[1])
                break
        if lower is None and upper is None:
            nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", args)
            if nums:
                v = abs(float(nums[0]))
                lower, upper = -v, v
        if upper is not None:
            sym = f"+{upper}/{lower}" if lower != -upper else f"±{upper}"
            tol = (abs(lower or 0) + abs(upper)) / 2
            annotations.append(GDTAnnotation(
                annotation_type="DIMENSIONAL", symbol=sym, gdt_type="PLUS_MINUS",
                tolerance_value=tol, tolerance_lower=lower, tolerance_upper=upper,
                face_step_ids=_resolve_to_advanced_faces(refs, entity_map),
            ))
    return annotations


def _extract_geometric_tolerances(entity_map: dict) -> list[GDTAnnotation]:
    annotations = []
    for eid, (ename, args) in entity_map.items():
        symbol = _GDT_SYMBOLS.get(ename)
        if symbol is None:
            continue
        gdt_type = ename.replace("_TOLERANCE", "")
        refs = [int(r) for r in re.findall(r"#(\d+)", args)]
        nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", args)
        tol_value = float(nums[0]) if nums else None
        for ref_id in refs:
            ref_name, ref_args = entity_map.get(ref_id, ("", ""))
            if ref_name in ("MEASURE_WITH_UNIT", "LENGTH_MEASURE_WITH_UNIT"):
                sub_nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", ref_args)
                if sub_nums:
                    tol_value = float(sub_nums[0])
                    break
        datum_refs = _resolve_datum_refs(refs, entity_map)
        face_ids = _resolve_to_advanced_faces(refs, entity_map)
        sym_str = f"{symbol}{tol_value}" if tol_value is not None else symbol
        if datum_refs:
            sym_str += " |" + "|".join(datum_refs) + "|"
        annotations.append(GDTAnnotation(
            annotation_type="GEOMETRIC", symbol=sym_str, gdt_type=gdt_type,
            tolerance_value=tol_value, datum_refs=datum_refs, face_step_ids=face_ids,
        ))
    return annotations


def _resolve_to_advanced_faces(
    ref_ids: list[int], entity_map: dict, max_depth: int = 5,
) -> list[int]:
    found: list[int] = []
    visited: set[int] = set()

    def follow(eid: int, depth: int) -> None:
        if depth > max_depth or eid in visited:
            return
        visited.add(eid)
        entry = entity_map.get(eid)
        if not entry:
            return
        ename, eargs = entry
        if ename == "ADVANCED_FACE":
            found.append(eid)
            return
        for sub in re.findall(r"#(\d+)", eargs):
            follow(int(sub), depth + 1)

    for r in ref_ids:
        follow(r, 0)
    return found


def _resolve_datum_refs(ref_ids: list[int], entity_map: dict) -> list[str]:
    labels: list[str] = []
    for rid in ref_ids:
        entry = entity_map.get(rid)
        if not entry:
            continue
        ename, eargs = entry
        if ename in (
            "DATUM", "DATUM_REFERENCE", "DATUM_REFERENCE_COMPARTMENT",
            "DATUM_REFERENCE_ELEMENT", "DATUM_FEATURE",
        ):
            m = re.search(r"'([A-Za-z0-9_]{1,4})'", eargs)
            if m:
                labels.append(m.group(1))
    return labels
