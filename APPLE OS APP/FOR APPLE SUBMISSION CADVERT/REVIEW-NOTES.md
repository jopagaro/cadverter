# App Review notes

Paste the section marked **FOR THE REVIEW NOTES FIELD** into App Store Connect →
App Review Information → Notes. The rest is background for you.

---

## FOR THE REVIEW NOTES FIELD

CADVERT analyses CAD files (STEP, IGES, BREP, STL, OBJ) and reports exact engineering
geometry — dimensions, holes, fillets, wall thicknesses — read directly from the file's
analytical definitions rather than measured from a rendering.

**No account, no sign-in, no subscription.** The app is a one-time purchase and has no
server of ours behind it. Nothing is uploaded to us and we operate no backend.

**About the bundled interpreter — please read before testing.**
The app contains a Python interpreter and the OpenCASCADE geometry kernel at
`CADVERT.app/Contents/Resources/engine-<architecture>/`. This is the geometry engine, not
downloaded or executed code from the internet. On launch the app starts it as a child
process bound to `127.0.0.1` on a randomly chosen free port, and communicates with it over
local HTTP. It is not reachable from outside the machine. The interpreter is shipped inside
the bundle, is code-signed with the app's identity, and no code is downloaded or updated at
runtime. This accounts for the app's size.

**How to test it in under two minutes.**
1. Launch the app. The status pill in the top right reads "Local engine · ready" after a
   few seconds.
2. Drag the enclosed `test_block_with_holes.step` onto the drop zone (or use ⌘O).
3. Analysis completes in about two seconds. Seven rendered views appear along the top and
   the sidebar fills with exact values: 8 faces, 18 edges, 2 through holes, 1 pattern.
4. Click "View HSD →" to see the full geometry document.

Steps 1–4 need no AI, no API key and no network connection. **You can disconnect the
machine from the internet and the entire analysis still works.**

**About the optional AI.**
Asking questions about a part is optional and never required to use the app. There are
three choices in Settings:
- **Apple Intelligence** (default where supported) — runs on device, needs no key, sends
  nothing anywhere.
- **OpenAI** or **Anthropic** — only if the user enters their own API key, which is stored
  in the macOS Keychain and sent only to that provider. We have no API keys in the app and
  receive no data.

If you are testing on a Mac with Apple Intelligence enabled, you can ask a question with no
setup at all. If not, the app explains why and offers the key field; the analysis features
above remain fully usable.

**Network use.** Only two things ever leave the machine, both optional and user-initiated:
a request to OpenAI or Anthropic when the user has entered their own key and asks a
question, and a request to that provider listing which models the key can use. There is no
telemetry, no analytics and no server of ours.

**Contact.** [your email] — happy to walk through the engine architecture if useful.

---

## Background for you (not for Apple)

**Why a reviewer might flag this app**

1. *Unexpected executable content.* Guideline 2.5.2 covers downloading and executing code.
   We do neither — the interpreter ships inside the bundle and nothing is fetched. The notes
   above say so explicitly, which is the point of including them.
2. *Size.* A universal build is around 1.6 GB because OpenCASCADE and VTK are large and the
   App Store cannot thin a Resources folder. Nothing to fix; just expect it.
3. *"Where's the sign-in?"* There isn't one, which is unusual enough to confuse a reviewer
   looking for an account wall. The notes state it plainly.

**Entitlements and why each is needed** (`CADVERT/Resources/CADVERT-macOS.entitlements`)

| Entitlement | Why |
|---|---|
| `app-sandbox` | Required for the App Store. |
| `network.client` | The app talks to its own engine on loopback, and to OpenAI/Anthropic when the user supplies a key. |
| `network.server` | The engine binds a loopback port. |
| `files.user-selected.read-write` | Opening the CAD file the user chose. |
| `cs.disable-library-validation` | The interpreter loads several hundred third-party compiled extensions (OpenCASCADE, VTK, numpy). They are re-signed with the app's identity at build time, but library validation still rejects them without this. |
| `cs.allow-unsigned-executable-memory` | Required by the Python runtime. |

If a reviewer questions the last two, the honest answer is that they are what a bundled
CPython requires; they are also the two most likely to draw a question, so expect it.

**What to have ready**
- A sample STEP file attached to the submission (`samples/test_block_with_holes.step` in
  the repo — small, neutral, not a customer's design).
- The exact steps above, which a reviewer can follow without engineering knowledge.
