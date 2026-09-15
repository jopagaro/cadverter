"""Assembly structure: which named component each face belongs to.

A STEP assembly names its parts. Catalog components arrive as real order codes
(``Belt S5M-300``, ``DIN 625 T1 - 6205``, ``ISO 4762 - M8 x 20``) and those names say
more about what a machine *does* than the geometry alone ever will. Without this module
cadvert sees 4,322 anonymous faces and, separately, a list of names; with it, every face
belongs to a named part, quantities are known, and questions like "what do the screws
weigh" become arithmetic over measured geometry instead of guesswork.

The structure is read with OpenCASCADE's XDE reader (``STEPCAFControl_Reader``), which
returns the same shapes as the plain reader *plus* the product tree. Everything here is
already inside the ``cadquery-ocp`` wheel, so this costs no extra install size.

One rule matters: face indices are only comparable when they come from the *same* loaded
shape. The plain and XDE readers build disjoint shape trees, so a mapping built against
one is meaningless against the other. :func:`extract_assembly` therefore takes the very
shape the rest of the pipeline uses and indexes faces exactly the way ``topology.py``
does — ``TopExp.MapShapes_s`` over ``TopAbs_FACE`` — so the IDs line up by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "ComponentInstance", "AssemblyInfo", "StepStructure",
    "extract_assembly", "read_step_with_structure",
]


# CAD translators stamp their own name on a shape that had none. Surfacing
# "Open CASCADE STEP translator 7.9 1" as a component would be worse than saying nothing.
_PLACEHOLDER_PATTERNS = (
    "step translator", "iges translator", "open cascade", "opencascade",
    "shape", "compound", "solid", "unnamed", "product", "part", "assembly",
)


def _is_placeholder_name(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return True
    if any(p in n for p in _PLACEHOLDER_PATTERNS):
        # "Belt S5M-300" must survive; only reject when the name is *mostly* boilerplate.
        stripped = n
        for p in _PLACEHOLDER_PATTERNS:
            stripped = stripped.replace(p, "")
        return len(stripped.strip(" -_.0123456789")) == 0
    return False


@dataclass
class ComponentInstance:
    """One *occurrence* of a part. Four identical shafts are four instances."""

    index: int                      # 0-based, unique per occurrence
    name: str                       # as written in the CAD file
    path: tuple[str, ...] = ()      # enclosing sub-assemblies, outermost first
    face_ids: list[int] = field(default_factory=list)   # 1-based, matches TopologyGraph
    volume: float = 0.0             # mm³ of the modelled solid
    bbox: Optional[dict] = None     # {"X": (min,max), ...}

    @property
    def location(self) -> str:
        return " / ".join(self.path) if self.path else "(top level)"


@dataclass
class AssemblyInfo:
    """The product tree of one file, plus the face → component mapping."""

    instances: list[ComponentInstance] = field(default_factory=list)
    face_owner: dict[int, int] = field(default_factory=dict)   # face id → instance index
    root_name: str = ""

    # ── convenience ───────────────────────────────────────────────────────────
    def __bool__(self) -> bool:
        return bool(self.instances)

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def part_count(self) -> int:
        return len({i.name for i in self.instances})

    def owner_of_face(self, face_id: int) -> Optional[ComponentInstance]:
        idx = self.face_owner.get(face_id)
        return self.instances[idx] if idx is not None else None

    def owner_of_faces(self, face_ids) -> Optional[ComponentInstance]:
        """The component a feature belongs to — the one owning most of its faces."""
        from collections import Counter
        votes = Counter(
            self.face_owner[f] for f in face_ids or () if f in self.face_owner
        )
        if not votes:
            return None
        return self.instances[votes.most_common(1)[0][0]]

    def bill_of_materials(self) -> list[dict]:
        """One row per distinct part name, quantity first.

        ``volume_each`` is the modelled solid volume of one instance, so a mass follows
        from any density the caller supplies. Threads are usually not modelled on
        fasteners, which makes such a mass a slight over-estimate — state that when
        quoting.
        """
        from collections import defaultdict

        rows: dict[str, dict] = defaultdict(
            lambda: {"name": "", "quantity": 0, "faces_each": 0, "volume_each": 0.0, "locations": set()}
        )
        for inst in self.instances:
            r = rows[inst.name]
            r["name"] = inst.name
            r["quantity"] += 1
            r["faces_each"] = max(r["faces_each"], len(inst.face_ids))
            r["volume_each"] = max(r["volume_each"], inst.volume)
            r["locations"].add(inst.location)
        out = []
        for r in rows.values():
            r["locations"] = sorted(r["locations"])
            r["volume_total"] = r["volume_each"] * r["quantity"]
            out.append(r)
        out.sort(key=lambda r: (-r["quantity"], r["name"]))
        return out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@dataclass
class StepStructure:
    """Shape plus product tree from one XDE read.

    The document must outlive the shape tool — OpenCASCADE's tool borrows it rather than
    owning it — so all three are held together here instead of being returned loose.
    """

    shape: Any = None
    shape_tool: Any = None
    document: Any = field(default=None, repr=False)

    def __bool__(self) -> bool:
        return self.shape is not None and self.shape_tool is not None


def read_step_with_structure(path) -> StepStructure:
    """Read a STEP file with the XDE reader.

    Returns an empty :class:`StepStructure` when the file has no usable product
    structure; the plain reader then remains the source of truth for the geometry.
    """
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_LabelSequence
    from OCP.IFSelect import IFSelect_RetDone

    doc = TDocStd_Document(TCollection_ExtendedString("cadvert"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        return StepStructure()
    if not reader.Transfer(doc):
        return StepStructure()

    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    if roots.Length() == 0:
        return StepStructure()

    if roots.Length() == 1:
        shape = tool.GetShape_s(roots.Value(1))
    else:                                    # several roots — compound them
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        builder = BRep_Builder()
        shape = TopoDS_Compound()
        builder.MakeCompound(shape)
        for i in range(1, roots.Length() + 1):
            builder.Add(shape, tool.GetShape_s(roots.Value(i)))
    return StepStructure(shape=shape, shape_tool=tool, document=doc)


def extract_assembly(shape, shape_tool, *, measure: bool = True,
                     max_instances: int = 20000) -> AssemblyInfo:
    """Map every face of ``shape`` to the named component that owns it.

    ``shape`` must be the shape returned alongside ``shape_tool`` by
    :func:`read_step_with_structure`; face IDs are only meaningful within one load.
    Set ``measure=False`` to skip per-part volume when only names are wanted.
    """
    from OCP.XCAFDoc import XCAFDoc_ShapeTool
    from OCP.TDF import TDF_LabelSequence, TDF_Label
    from OCP.TDataStd import TDataStd_Name
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer

    info = AssemblyInfo()
    if shape is None or shape_tool is None:
        return info

    # Index faces exactly as topology.py does, so IDs agree.
    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)

    def label_name(label) -> str:
        attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
            name = str(attr.Get().ToExtString()).strip()
            return "" if _is_placeholder_name(name) else name
        return ""

    measured: dict[str, tuple[float, dict]] = {}     # master label tag → (volume, bbox)

    def measure_master(label) -> tuple[float, Optional[dict]]:
        if not measure:
            return 0.0, None
        tag = label.Tag()
        key = f"{tag}"
        if key in measured:
            return measured[key]
        try:
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib

            master = XCAFDoc_ShapeTool.GetShape_s(label)
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(master, props)
            vol = abs(props.Mass())
            box = Bnd_Box()
            BRepBndLib.Add_s(master, box)
            bbox = None
            if not box.IsVoid():
                x0, y0, z0, x1, y1, z1 = box.Get()
                bbox = {"X": (x0, x1), "Y": (y0, y1), "Z": (z0, z1)}
            measured[key] = (vol, bbox)
            return vol, bbox
        except Exception:
            measured[key] = (0.0, None)
            return 0.0, None

    def attribute(instance: ComponentInstance, located_shape) -> None:
        exp = TopExp_Explorer(located_shape, TopAbs_FACE)
        while exp.More():
            idx = fmap.FindIndex(exp.Current())
            if idx > 0 and idx not in info.face_owner:
                info.face_owner[idx] = instance.index
                instance.face_ids.append(idx)
            exp.Next()

    def walk(label, loc, path: tuple[str, ...]) -> None:
        if len(info.instances) >= max_instances:
            return
        children = TDF_LabelSequence()
        XCAFDoc_ShapeTool.GetComponents_s(label, children)
        for i in range(1, children.Length() + 1):
            comp = children.Value(i)
            # Compose this occurrence's placement with everything above it. Skipping
            # this is the classic bug: nested instances then fail to match any face.
            comp_loc = loc * XCAFDoc_ShapeTool.GetLocation_s(comp)
            referred = TDF_Label()
            target = referred if XCAFDoc_ShapeTool.GetReferredShape_s(comp, referred) else comp
            name = label_name(target) or label_name(comp) or "unnamed"

            if XCAFDoc_ShapeTool.IsAssembly_s(target):
                walk(target, comp_loc, path + (name,))
                continue

            vol, bbox = measure_master(target)
            inst = ComponentInstance(index=len(info.instances), name=name,
                                     path=path, volume=vol, bbox=bbox)
            info.instances.append(inst)
            try:
                placed = XCAFDoc_ShapeTool.GetShape_s(target).Moved(comp_loc)
            except Exception:
                continue
            attribute(inst, placed)

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        root = roots.Value(i)
        info.root_name = info.root_name or label_name(root)
        if XCAFDoc_ShapeTool.IsAssembly_s(root):
            walk(root, TopLoc_Location(), ())
        else:
            # A single part: one instance covering everything.
            vol, bbox = measure_master(root)
            inst = ComponentInstance(index=len(info.instances),
                                     name=label_name(root) or "", volume=vol, bbox=bbox)
            info.instances.append(inst)
            attribute(inst, XCAFDoc_ShapeTool.GetShape_s(root))

    # An instance that matched no face is noise from an empty or failed placement.
    info.instances = [i for i in info.instances if i.face_ids]
    # A lone unnamed instance tells the reader nothing; treat it as "no structure".
    if len(info.instances) <= 1 and not any(i.name for i in info.instances):
        return AssemblyInfo()
    for new_index, inst in enumerate(info.instances):
        if inst.index != new_index:
            for f in inst.face_ids:
                info.face_owner[f] = new_index
            inst.index = new_index
    return info
