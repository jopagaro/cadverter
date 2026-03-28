# CADVERT — Design Reference

Convert CAD files (STEP) into **Hierarchical Spatial Documents** (HSD) that
preserve exact dimensional data and topological connectivity in a form an LLM
can reason over with geometric fidelity.

---

## Output Format: Hierarchical Spatial Document (HSD)

### Header
```
PART: <name>
SOURCE: <filename> (SHA256: <hash>)
UNITS: mm
PRECISION: all values exact from B-REP analytical definitions
```

---

### Section 1 — GLOBAL PROPERTIES
Exact values from B-REP:
- Bounding box: X[min, max] Y[min, max] Z[min, max]
- Volume (mm³)
- Surface area (mm²)
- Center of mass (x, y, z)
- Total faces / edges / vertices
- Total unique bodies

---

### Section 2 — TOPOLOGY MAP
Every face listed with its exact analytical definition, and every edge showing
which two faces it connects plus edge geometry and dihedral angle.

**Face types and their required fields:**

| Type | Fields |
|------|--------|
| PLANE | normal vector (unit), offset d (ax+by+cz=d), orientation |
| CYLINDER | axis direction, axis point, radius, z-extent |
| CONE | apex, axis, half-angle |
| SPHERE | center, radius |
| TORUS | center, axis, major radius, minor radius |
| NURBS_SURFACE | degree (u,v), control point grid, knot vectors U+V, sampled boundary, surface characteristics |

**Edge types:**

| Type | Fields |
|------|--------|
| LINE | start point, end point, length |
| CIRCLE | center, normal, radius |
| ELLIPSE | center, normal, major radius, minor radius |
| NURBS_CURVE | degree, control points, knot vector |

**Edge entry format:**
```
├─ E<n> (<type>, <key params>) → connects to [F<n>] — <angle>° <convex|concave>
```

Dihedral angle convention:
- **Convex** = outside corner (material on inside), angle > 180° measured through material
- **Concave** = inside corner (material on outside), angle < 180° measured through material
- Reported as the supplement: the geometric angle between face normals

---

### Section 3 — IDENTIFIED FEATURES
Manufacturing-relevant features detected from topology:

- **Through hole** — full cylindrical face bounded by two circular edges on parallel planes
- **Blind hole** — cylindrical face with one flat bottom
- **Pocket** — recessed planar region surrounded by walls
- **Boss** — raised cylindrical protrusion
- **Fillet** — concave blend between two faces (toroidal or cylindrical)
- **Chamfer** — angled planar face between two faces
- **Keyway** — rectangular slot inside a bore
- **Thread** — helical surface (detected from NURBS or helix edges)
- **Pattern** — 2+ identical features with regular spacing/rotation

Each feature references exact face/edge IDs from the topology map and includes
standard size matching (e.g. "6.35 mm → 1/4 inch clearance hole").

---

### Section 4 — SPATIAL RELATIONSHIPS
Computed distances an engineer would put on a drawing:
- Wall thicknesses
- Hole-to-edge distances
- Feature-to-feature distances
- Clearances

---

### Section 5 — RENDERED VIEWS
6 orthographic views (front/back/top/bottom/left/right) + isometric + section
views through detected features. PNG output.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| STEP parsing & B-REP traversal | `pythonocc-core` (OCC bindings) |
| Geometry math | `numpy` |
| Rendering | OCC's `Graphic3d` / `V3d` offscreen, or `cadquery` SVG export |

## Module Structure

```
src/cadvert/
  __init__.py
  cli.py              # entry point: cadvert <file.step>
  ingest.py           # STEP → OCC shape
  topology.py         # B-REP walker: faces, edges, vertices + geometry extraction
  geometry.py         # analytical surface/curve classifiers and parameter extractors
  features.py         # feature detection from topology graph
  spatial.py          # distance/clearance computations
  renderer.py         # image rendering
  document.py         # HSD document assembly and formatting
  utils.py            # shared helpers (units, rounding, hashing)
```

## Key OCC APIs

```python
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, ...
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_Reader
```

## Precision Notes

- All geometry extracted directly from OCC analytical definitions — no tessellation
- Round to 3 decimal places for display; store full float internally
- NURBS: dump full control point grid and knot vectors verbatim
- Dihedral angles computed via face normal vectors at shared edge midpoint

## Implementation Order

1. **Ingest** — load STEP, validate, extract body count
2. **Topology walker** — enumerate faces/edges/vertices with IDs
3. **Geometry extraction** — classify each face/edge, extract analytical params
4. **Connectivity graph** — map face↔edge adjacency, compute dihedral angles
5. **Global properties** — bounding box, volume, surface area, CoM via GProp
6. **Document formatter** — render sections 1 & 2
7. **Feature detection** — section 3
8. **Spatial relationships** — section 4
9. **Renderer** — section 5
