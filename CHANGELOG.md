# Changelog

## 0.3.1

### Fixed

- **Four of the six geometry tools were broken.** `get_feature`, `get_face`, `get_edge` and
  `get_neighbors` read `edge.length` and `face.convexity`, neither of which exists, so every
  call returned an error string. The dispatcher caught the exception and handed the model that
  string, so the failure was silent: answers fell back to the summary and looked plausible.
  Edge length now comes from the curve geometry (`length` for lines, `arc_length` for circles,
  so hole rims report a real value), and faces report the convexity of their boundary edges,
  which is where convexity actually lives. `get_neighbors` reports the convexity of the edge
  joining two faces rather than asking a face for its own.
- `originating_system` was empty whenever a CAD tool wrapped the `FILE_NAME` entity across
  lines, which is most real files.

- **`pip install cadvert` was broken on new installs.** `cadquery-ocp` was unpinned, so a
  fresh install resolved to 8.x, where OpenCASCADE 8 relocated the classes `topology.py`
  imports; `analyze()` died with `ImportError: cannot import name
  'TopTools_IndexedMapOfShape'`. The dependency is now pinned to the 7.x line.

### Added

- **Assembly component names.** STEP `PRODUCT` names are retained and reported through
  `metadata.components`, `to_dict()` and the LLM summary. For an assembly these are the
  strongest clue to what the thing *is*: catalog parts arrive as real order codes
  (`Belt S5M-300`, `DIN 625 T1 - 6205`, `ISO 4762 - M8 x 20`) that describe function in a
  way raw geometry cannot. Catalog parts are listed ahead of internal drawing numbers, and
  the summary caps the list to protect the prompt budget on large assemblies.
- `metadata.project` — the design name from the file path, decoded from ISO 10303-21
  character escapes, so non-ASCII names (Vietnamese, Japanese, German) read correctly.
- **Claude as a chat provider** alongside OpenAI, selected with an `X-Provider` header or
  inferred from a `claude-*` model name. Both providers drive the same six geometry tools
  and emit the same stream events, so existing clients need no change.
- `POST /tool/{session}` runs one geometry tool directly, for clients that drive their own
  model, and `GET /tools` returns the definitions.
- `/convert` now returns `tier0`, the compact summary, for use as system context.
- The `llm` extra installs `anthropic` alongside `openai`.

## 0.3.0

- `to_graph()` became a lossless topology translation.
- One-call `analyze()` API with multi-representation output.
- Point-cloud sampling and DXF ingest.
