# CADVERT for Mac, iPad and iPhone

A native SwiftUI app with the same design system, layout and flows as the CADVERT
web app (`src/cadvert/static/index.html`): drop a CAD file, see the rendered views and
part info, read the Hierarchical Spatial Document, and chat with an AI about the exact
geometry.

| Platform | Status | Engine |
|----------|--------|--------|
| **macOS 14+** (primary) | Standalone, offline. The Python + OpenCASCADE engine is bundled inside the app. | Local (bundled) — or a remote server |
| **iPadOS 17+** | Same two-column layout as the Mac. | Remote `cadvert-server` |
| **iOS 17+** | Single-column layout; part info in a sheet. | Remote `cadvert-server` |

```
APPLE OS APP/
├── project.yml            XcodeGen spec → CADVERT.xcodeproj (committed, regenerate with `make generate`)
├── CADVERT/
│   ├── App/               CADVERTApp (scenes, macOS menu commands, AppDelegate), AppModel (all state)
│   ├── Design/            Palette (the CSS variables, light + dark), Typography, FlowLayout, Components
│   ├── Models/            API response types, errors, chat/session models, constants copied from the web UI
│   ├── Services/          CadvertClient (REST + SSE), SSEParser, MarkdownBlocks, KeychainStore,
│   │                      AppSettings, LocalEngine + EngineLocator (macOS only)
│   ├── Views/             RootView, HeaderBar, SidebarView, ViewsStrip, ChatView, MessageBubble,
│   │                      InputBar, Overlays (processing / toast / HSD / engine log), WallSheets,
│   │                      SettingsView, AboutView
│   └── Resources/         Assets (icon, accent colour), entitlements
├── CADVERTTests/          Unit tests + a real local-engine integration test
├── Engine/                Bundled Python engine lands here (gitignored)
├── scripts/               bundle-engine.sh, build-release.sh, copy-engine.sh, sign-engine.sh, make-dmg.sh
├── Makefile
└── DISTRIBUTION.md        How to sell it: Mac App Store, direct DMG, iPad/iPhone
```

## How it maps to the web app

The web app is a thin client over `cadvert-server` (FastAPI). This app is the same
client, native:

```
┌────────────────────────────┐   HTTP (multipart /convert, SSE /chat)   ┌──────────────────────────┐
│  CADVERT.app (SwiftUI)     │ ───────────────────────────────────────▶ │  cadvert-server          │
│  AppModel ⇄ CadvertClient  │ ◀─────────────────────────────────────── │  (FastAPI + OCCT + VTK)  │
└────────────────────────────┘        JSON + PNG views + event stream   └──────────────────────────┘
        macOS: the server is a child process launched from Contents/Resources/engine on 127.0.0.1:<random>
        iPad / iPhone: the server is wherever you host it (Settings → Remote server)
```

| Web (`index.html`) | Native |
|---|---|
| `:root` CSS variables, light/dark | `Palette.light` / `Palette.dark`, `ThemePreference` (System / Light / Dark) |
| `<header>` logo, About, For Developers, theme pills | `HeaderBar` (+ engine status pill, Settings) |
| `.sidebar` dropzone, progress, info card | `SidebarView` (`DropzoneView`, `ProgressRow`, `PartInfoCard`) |
| `.views-strip` thumbnails + lightbox | `ViewsStrip`, `LightboxView` |
| `.messages`, chips, bubbles, typing dots, tool indicator | `ChatView`, `MessageBubble`, `TypingDots`, `ToolIndicator` |
| `.processing-overlay` with staged messages | `ProcessingOverlay` (`ProcessingStages` copied verbatim) |
| HSD modal | `HSDSheet` (+ Copy, Save… to `.hsd.txt`) |
| BYOK / key / file-limit walls | `WallSheets` |
| `uploadFile()` / `_sendText()` | `AppModel.importData()` / `AppModel.send()` |

Sign-in: the hosted web app verifies a Google ID token. The native app does not do
Google sign-in (that needs an iOS OAuth client and a one-line server change to accept
its audience). Today it works against servers started with `DISABLE_AUTH=1` — which is
what the bundled Mac engine does — and accepts a pasted bearer token under
Settings → Advanced for anything else. See DISTRIBUTION.md for the recommended path
(Sign in with Apple + StoreKit) before shipping the iOS build to strangers.

## Choosing who answers: Apple on-device, OpenAI, or Claude

Settings → AI has a provider picker. All three drive the same six exact-geometry tools
(`get_feature`, `get_face`, `get_edge`, `measure_distance`, `get_neighbors`, `search_faces`).

| Provider | Where the model runs | Key | Good for | Watch out |
|---|---|---|---|---|
| **Apple Intelligence** (default on macOS 26 / iOS 26) | On the device, via the Foundation Models framework | none | Lookups, quick questions, privacy, offline | ~4k-token context: big assemblies get a truncated summary; weaker engineering judgement than the hosted models |
| **OpenAI** | OpenAI's API through the engine | user's key (Keychain) | Strong all-rounder, model picker (GPT-4o mini … GPT-5.4) | costs per message |
| **Claude** | Anthropic's API through the engine | user's key (Keychain) | Multi-step tool use and reasoning; Claude Opus 5 is the default | costs per message |

How the Apple path works: the app builds a `LanguageModelSession` with the part's Tier-0
summary as instructions and six Swift `Tool`s (`Services/AppleIntelligenceChat.swift`).
When the model calls a tool, the app posts to the engine's `POST /tool/{session}` and
feeds the JSON back; the answer streams into the same chat UI. On context overflow it
retries once with a tighter summary, then suggests switching provider. Availability is
checked with `SystemLanguageModel.default.availability` and explained in Settings
(device not eligible / Apple Intelligence off / model downloading).

The hosted providers are one server code path: `POST /chat` takes `X-Provider`
(`openai` | `anthropic`, inferred from a `claude-*` model when absent) and the key in
`X-OpenAI-Key` / `X-Anthropic-Key`; the server translates the tool definitions and
streams the same `content` / `tool_call` events either way. The web app got the same
Claude models in its model menu.

## Build

Requirements: Xcode 26, [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).

```bash
make generate     # regenerate CADVERT.xcodeproj from project.yml
make build-mac    # Debug build → build/Build/Products/Debug/CADVERT.app
make run-mac
make build-ios    # iPad/iPhone simulator build
make test         # 27 tests: SSE parser, models, markdown, providers, local engine, on-device AI
```

Or open `CADVERT.xcodeproj` in Xcode and run the `CADVERT` scheme on "My Mac" or any
iPad/iPhone simulator.

### The bundled engine (what makes the Mac app standalone)

```bash
make engine            # this Mac's architecture, from the repo source
make engine-intel      # Intel (x86_64) — cross-builds on Apple silicon via Rosetta
make engine-all        # both, for a universal build
```

`bundle-engine.sh` downloads a relocatable CPython 3.12 (python-build-standalone),
installs `cadvert[server,llm,mesh,graph,dxf]` into it with `cadquery-ocp` pinned to the
release cadvert is tested against, prunes caches, and writes `Engine/engine-<arch>/`
(about 800 MB each — OpenCASCADE and VTK are large).

**One engine per architecture.** Engines live at `Engine/engine-arm64` and
`Engine/engine-x86_64`, and are copied to `CADVERT.app/Contents/Resources/engine-<arch>`.
The build copies whichever architectures it targets (Xcode's `ARCHS`, overridable with
`CADVERT_ENGINE_ARCHS`), and the app picks the folder matching the slice it is running —
including the Intel engine when a universal build runs under Rosetta. If the bundle has
an engine but not for this Mac, the app says so plainly instead of failing generically.

Cross-building the Intel engine on Apple silicon needs Rosetta 2:

```bash
softwareupdate --install-rosetta --agree-to-license
```

The macOS build's two post-build phases rsync the engine(s) in and code-sign every Mach-O
inside with the app's identity (hardened runtime + `scripts/engine.entitlements`).

### Shipping

**1.0 ships Apple silicon only** (`make release-arm64`). The Intel engine builds and passes
every functional test under Rosetta, but Rosetta runs it against an Apple silicon GPU, so the
OpenGL path VTK uses for rendered views is untested on real Intel/AMD graphics. Apple
Intelligence is also unavailable on every Intel Mac, so those users would need an OpenAI or
Anthropic key (the app detects this and does not offer the on-device option).

```bash
make release-both      # two ~800 MB DMGs, one per architecture
make archive-universal # one ~1.6 GB universal archive carrying both engines
```

Per-architecture DMGs keep each download to roughly 800 MB and are the better option for
direct sale. The App Store cannot thin a `Resources` folder, so an App Store build must be
universal and carries both engines. `scripts/build-release.sh` takes
`--identity "Developer ID Application: …"` to produce a distributable, signed app;
without it the build is ad-hoc signed and runs only on this Mac.

At launch the Mac app looks for an engine in this order, validating each with
`python -c "import cadvert.server, uvicorn"`:

1. Custom path from Settings (Debug builds only — the App Sandbox in Release blocks
   executables outside the bundle)
2. `CADVERT.app/Contents/Resources/engine/bin/python3` (bundled)
3. `~/.cadvert/engine/bin/python3`, `~/.cadvert/venv/bin/python3`

It then runs `uvicorn cadvert.server:app --host 127.0.0.1 --port <free port>` with
`DISABLE_AUTH=1` (the purchase is the paywall; no Google sign-in locally) and the
user's OpenAI key in the environment, polls `/config` until ready, and stops the
process on quit. The engine is never exposed beyond loopback.

Without a bundled engine the app still builds and runs — point Settings at a Python
environment that has `pip install cadvert[server]` (for example this repo's `.venv`),
or switch to a remote server.

### Keys on the Mac

With `DISABLE_AUTH=1` the server treats every caller as "pro" and uses its own
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. The app passes the user's keys (Keychain) into the
engine's environment, so choosing OpenAI or Claude asks for that vendor's key on first
use. Changing a key restarts the engine; the last file is re-processed automatically.
Apple Intelligence needs no key. File analysis, views and the HSD never need a key.

## Launch arguments (QA / automation)

```bash
open build/Build/Products/Debug/CADVERT.app --args --open part.step --ask "What is the thinnest wall?"
xcrun simctl launch <udid> com.cadvert.CADVERT --open /path/part.step --show hsd
```

`--open <file>` imports a file after the engine is ready; `--ask "<q>"` sends a message
once the part is loaded; `--show hsd|about|developers|settings` opens that sheet.
`CADVERT_OPEN_FILE=<path>` works like `--open`. These flags exist only in **Debug**
builds (`#if DEBUG`); a Release/App Store build ignores them and always starts empty.

## Verified so far (this machine, Xcode 26.6)

- macOS Debug + iOS Simulator builds succeed; 22/22 tests pass, including a real
  integration test that boots the engine from the repo `.venv`, converts
  `samples/test_block_with_holes.step` (8 faces, 18 edges, 7 rendered views) and cleans up.
- Mac app launched with the bundled 1.0 GB engine: the child interpreter runs from
  inside the `.app` and serves `/config` and `/convert` on loopback.
- iPad Pro and iPhone 17 Pro simulators against a mock `cadvert-server`: upload
  overlay, views strip, part info, suggestion chips, streaming answer with tool
  indicators, markdown (bold / inline code / bullets / code block), HSD sheet, dark mode.
- Screenshots from that run are in the session scratchpad; regenerate with the launch
  arguments above.

## Not done yet / decisions for you

- **App Store sign-in for iOS.** Needs Sign in with Apple on the server (see
  DISTRIBUTION.md). Until then the iOS build is for servers you control.
- **Fonts.** The web app loads Satoshi from a CDN; the app uses SF Pro / SF Mono (the
  CSS fallback stack). Bundle Satoshi (Fontshare licence) if you want pixel parity.
- **Google sign-in / Stripe checkout** from the web app are intentionally absent:
  Stripe links for digital features are not allowed inside iOS apps, and the Mac app is
  itself the paid product.
