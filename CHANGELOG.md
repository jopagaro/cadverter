# Changelog

## Unreleased

### Fixed

- **Cached parts were never expiring.** Cleanup walked only the in-memory session table,
  so anything left behind when the app quit was invisible to the next run and stayed
  forever — five-day-old files under a one-day limit. The sweep now walks the directory,
  runs once at startup as well as hourly, and collects orphans from previous runs.

### Added

- `CADVERT_DATA_DIR` chooses where cached parts live. The desktop app points it at its
  Caches directory, which is where Apple says regenerable data belongs and which resolves
  inside the sandbox container automatically; the CLI still falls back to the temp folder.
- `GET /cache` and `DELETE /cache` report and clear what is held on disk, so an app can
  show "26 MB across 15 parts" and offer a Clear button. Everything cached rebuilds from
  the original CAD file, so clearing is always safe.

### Removed

- **Accounts, plans and payment.** cadvert is a local engine: it runs on the machine that
  owns the file. Stripe checkout and webhooks, Google sign-in, the user database, tier
  logic, per-user file and message quotas and the upgrade walls are all gone, along with
  the `/auth/verify`, `/create-checkout` and `/stripe-webhook` endpoints. `server.py` lost
  about 400 lines.
- Chat now has one rule: the key is the caller's own, from `X-OpenAI-Key` /
  `X-Anthropic-Key` or from the environment the engine was started with. Without one the
  server returns a plain `api_key_required` instead of a sales wall.
- `/config` reports `local_only`, the available providers and the file cap. It no longer
  reports auth or payment state.

### Added

- **Assembly structure.** STEP files are now read with OpenCASCADE's XDE reader, which
  returns the same geometry plus the product tree. Every face is attributed to the named
  component that owns it, giving a bill of materials with quantities, per-part volume and
  bounding box. On a 4,322-face machine assembly that is 183 instances across 45 part
  types, all faces attributed, in about 4 ms. No new dependency and no change in install
  size — the XDE reader is already inside `cadquery-ocp`.
- `result.assembly`, `result.component_of_face()`, `result.component_of_feature()` and
  `result.mass_properties(density, name_filter)`, which computes per-part and total mass
  in g, kg, oz and lb. The arithmetic deliberately lives in the library rather than in a
  language model's head.
- Two tools for LLM callers: `get_component` (one part's quantity, volume, faces and
  features) and `compute_mass`. `get_face` and `get_feature` now also name the component
  they belong to, so "which part is this hole in" is answerable.
- Volumes are reported for the modelled solid, and the summary says so: fastener threads
  are usually not modelled, which makes a fastener mass slightly high.

### Fixed

- Names stamped by a translator (`Open CASCADE STEP translator 7.9 1` and similar) are no
  longer surfaced as component names; a file whose only "component" is such a placeholder
  reports no assembly at all rather than inventing one.
- Ingest falls back to the plain STEP reader whenever the structure-aware read fails, so
  no file that loaded before can stop loading.

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
