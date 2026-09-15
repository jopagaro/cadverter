# CADVERT

**Turn CAD files into something an AI can actually read.**

A STEP file is a boundary representation — dense surface equations and topology
built for CAD kernels, not for language models. Paste one into an LLM and it
chokes; show it a screenshot and the model can see the shape but can't tell you
the wall is exactly 11.000 mm thick. In engineering, the number is the point.

CADVERT reads the CAD file directly and extracts the **exact** geometry the
engineer authored — dimensions, holes, fillets, wall thicknesses, tolerances —
as clean, structured text (and other representations) any model can reason over.
Geometry is analytical, straight from the B-REP — never tessellated or
approximated.

```bash
pip install cadvert
```

## Quick start

```python
import cadvert

result = cadvert.analyze("bracket.step")

print(result.to_text())     # LLM-ready structured document (the flagship)
result.to_dict()            # JSON-safe analysis for quoting / CI / DBs
result.to_json()
result.to_graph()           # networkx face-adjacency graph  (GNN / UV-Net)
result.to_points(2048)      # (N,3) surface point cloud       (PointNet / 3D-CNN)
```

For an assembly, CADVERT reads the product tree: every face is attributed to a named
component, so you get a bill of materials with quantities and per-part volume alongside
the geometry.

```
DESIGN: Cylindrical face chamfering machine
AUTHORED IN: Autodesk Inventor 2021
ASSEMBLY — 183 parts of 45 types
   qty  part                                 volume each
    36  ISO 4762 - M5 x 10                     435.1 mm³
     8  LBHSW20                             26,318.7 mm³
     4  DIN 625 T1 - 6205 - 25 x 52 x 15    15,744.5 mm³
```

Which turns questions of quantity into arithmetic over measured geometry:

```python
result = cadvert.analyze("machine.step")
result.assembly.bill_of_materials()          # qty, volume, where each part is fitted
result.component_of_face(1500).name          # 'LBHSW20'
result.mass_properties(7.85, "ISO 4762")     # 66 screws, 337.1 g, 11.89 oz
```

Files without a product tree (a single part, IGES, a mesh) simply report no assembly
rather than inventing one.

Example `to_text()` output:

```
PART: bracket
[GLOBAL PROPERTIES]
Bodies: 1  |  Faces: 8  |  Edges: 18
Bbox: X[0,60]  Y[0,40]  Z[0,20]
[FEATURES]
  hole_1: THROUGH HOLE  d=8.000mm  depth=20.000mm  [5/16"]
  pattern_1: PATTERN  2× through hole  d=8.000mm
[SPATIAL RELATIONSHIPS]
  Wall thickness: hole_1 bore → outer surface = 11.000 mm
```

## Supported formats

| Extension | Pipeline | Notes |
|-----------|----------|-------|
| `.step` `.stp` | Full B-REP | Features, spatial, GD&T (AP242) |
| `.iges` `.igs` | Full B-REP | Features, spatial |
| `.brep` | Full B-REP | Features, spatial |
| `.stl` `.obj` | Mesh | Global properties + point cloud |
| `.dxf` | 2D drawing | Entities, layers, dimensions, text |

## Representations

| Method | Output | For |
|--------|--------|-----|
| `.to_text()` | Structured text | Prompting an LLM (Claude, GPT) |
| `.to_dict()` / `.to_json()` | JSON | Quoting engines, CI checks, databases |
| `.to_graph()` | `networkx` graph (lossless) | Graph neural networks |
| `.to_points(n)` | `(N,3)` array | PointNet, 3D CNNs |

## Optional extras

```bash
pip install cadvert[graph]   # networkx  → .to_graph()
pip install cadvert[mesh]    # trimesh   → area-weighted .to_points(), OBJ
pip install cadvert[dxf]     # ezdxf     → .dxf drawings
pip install cadvert[server]  # FastAPI REST server + web UI
pip install cadvert[llm]     # OpenAI + Claude chat providers (server / desktop app)
pip install cadvert[full]    # everything
```

## Mac, iPad and iPhone app

A native SwiftUI app with the same design and flows as the web UI lives in
[`APPLE OS APP/`](APPLE%20OS%20APP/README.md). On the Mac it bundles the Python +
OpenCASCADE engine and works fully offline; on iPad/iPhone it connects to a hosted
`cadvert-server`. Questions can be answered by Apple's on-device model (macOS 26 /
iOS 26, no key), OpenAI, or Claude — the server's `/chat` takes an `X-Provider` header
and the web UI's model menu lists both vendors. See that folder's README for building and DISTRIBUTION.md for
selling it (Mac App Store, direct DMG, iOS).

## CLI

```bash
cadvert part.step -o out.hsd.txt      # convert to an HSD document
cadvert-server                        # REST API on http://localhost:8000
```

## Building an AI CAD tool

Register `analyze` as a tool in your LLM framework and the model can answer
questions about real parts using real dimensions:

```python
def analyze_cad(file_path: str) -> str:
    return cadvert.analyze(file_path).to_text()
```

Requires Python 3.10+. Core install ships `cadquery-ocp` (OCCT bindings) and
`numpy` — no other heavy dependencies.
