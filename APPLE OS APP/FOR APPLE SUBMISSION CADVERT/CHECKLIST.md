# Submission checklist

Ordered so nothing blocks on something later. Items marked **[you]** need your developer
account and can't be done from the code side.

## 1. Account and certificates — **[you]**

- [ ] Decide which account owns the app: personal or the LLC. **This is close to permanent.**
      Moving an app between accounts afterwards is a formal transfer with restrictions, not a
      settings change. If the LLC matters for liability or for how customers see you, wait.
- [ ] Accept the current Apple Developer Program Licence Agreement (submissions silently
      block until you do).
- [ ] Create an **Apple Distribution** certificate.
- [ ] Create a **Mac App Store provisioning profile** for `com.conviviumllc.cadvert`.
- [ ] Apply for the **Small Business Program** — it cuts Apple's commission from 30% to 15%
      under $1M a year. It is **not retroactive**, so do it before your first sale.

Verify locally when done:
```
security find-identity -v -p codesigning | grep "Apple Distribution"
```

## 2. Hosting — **[you]**

- [ ] Run `website/fill.sh` (`site.config` is already filled in: admin@conviviumllc.com,
      Convivium LLC, $9.99).
- [ ] conviviumllc.com runs WordPress, so publish as pages, not files: a **CADVERT** page
      (slug `cadvert`) with child pages **Privacy Policy** (`privacy`) and **Support**
      (`support`), each a Custom HTML block pasted from `website/dist/wordpress/`. Steps in
      `website/README.md`.
- [x] Published and checked: `https://conviviumllc.com/cadvert/privacy` and
      `https://conviviumllc.com/cadvert/support` (no trailing slash — that's how WordPress
      serves them). App Store Connect takes these plus `https://conviviumllc.com/cadvert`
      as the Marketing URL; see the URL table in `APP-STORE-LISTING.md`.

Both URLs are required fields; you cannot submit without them, and Apple opens them during
review.

## 3. Build

- [ ] `make engine-all` — both architecture engines present (slow, downloads ~800 MB each).
- [ ] **`./scripts/refresh-engine.sh`** — reinstall the current cadvert into them.
- [ ] `make archive-universal` — produces the universal archive (~1.6 GB). It also rebuilds
      the in-app Acknowledgements from the engines (`make licenses`); if that changes
      `CADVERT/Resources/Acknowledgements.json`, commit it.
- [ ] In Xcode Organizer: **Validate App** first. It runs Apple's upload checks (signing,
      sandbox entitlements on every executable, Info.plist) without submitting anything.
      This matters more than usual here, because the engine's binaries live in
      `Contents/Resources`, an unusual place for code.
- [ ] Then Distribute App → App Store Connect → Upload.

**Do not skip the refresh step.** The engines are built once and reused, so they keep
whatever version of cadvert was current when they were built. Shipping a stale one is easy
to do and hard to notice: the app launches, analysis works, and only the newest endpoints
are quietly missing. It had already happened here — the Release build was carrying a server
that still had Stripe configuration in it.

Verify before archiving:
```
curl -s http://127.0.0.1:<port>/config | grep local_only
```
(the port is in Settings → Engine log)

Expect the upload to be slow. It's mostly the two bundled engines.

## 4. App Store Connect — **[you]**

- [ ] New macOS app, bundle ID `com.conviviumllc.cadvert`, SKU of your choosing.
- [ ] Paste name, subtitle, description, keywords from `APP-STORE-LISTING.md`.
- [ ] Price: USD 9.99 in Pricing and Availability (Apple converts it for other storefronts).
- [ ] Upload screenshots from `screenshots/app-store/` — they must be exactly 2880×1800 (or
      another size Apple lists). See `SCREENSHOTS.md`.
- [ ] Paste the review notes from `REVIEW-NOTES.md` into App Review Information → Notes.
      **Do not skip this.** It explains the bundled Python interpreter before a reviewer
      finds it and wonders whether you are downloading executable code.
- [ ] Attach `samples/test_block_with_holes.step` so the reviewer has something to open.
- [ ] Age rating: answer everything "None". The app has no objectionable content.
- [ ] Export compliance: see below.

## 5. Export compliance

Already answered in the app: Info.plist sets `ITSAppUsesNonExemptEncryption` to `NO`, so App
Store Connect won't ask on each upload.

The reasoning, if anyone asks: CADVERT implements no encryption of its own. It makes HTTPS
calls to OpenAI and Anthropic only when the user supplies a key, which falls under the
standard exemption for apps that use only HTTPS.

## 6. Verified already — you do not need to retest these

- The **App Sandbox works end to end**: launching the engine as a child process, loopback
  networking, writing to the container, and VTK rendering all function under the real
  sandbox with the release entitlements. This was the biggest unknown, because Debug builds
  run unsandboxed and never exercise it.
- The **universal build** produces both architectures with both engines correctly signed,
  and the app selects the right one at runtime.
- **Cached parts land inside the app container**, where Apple expects regenerable data, and
  the Storage control in Settings reports and clears them.
- **The bundled interpreter is itself sandboxed** (`app-sandbox` + `inherit`); the build
  stops if it isn't. Without this, validation rejects the upload even though the app runs.
- **No keys, no CAD files, no build-machine paths ship.** Searched every file in the bundle
  for your actual key values, for any key-shaped string, for CAD formats, and for your
  username.
- **Release builds offer only the bundled engine.** The remote-server option and custom
  Python field exist in Debug builds only, so nothing in the shipped app can send a file
  off the Mac.

## 7. After approval

- [ ] Verify the download on a Mac that has never run the app — this is the only way to
      catch a signing or sandbox problem that never appears on a development machine.
- [ ] Specifically test on a Mac **without** Apple Intelligence, since that path shows the
      guidance sheet rather than on-device answers.

## Known before you start

**Size.** Roughly 1.6 GB universal. Unavoidable: OpenCASCADE and VTK are large, and the App
Store cannot thin a Resources folder.

**Three-year rule.** Apple removes apps not updated in three years, and requires builds made
with the current Xcode and SDK. Plan on periodic maintenance releases.

**No paid upgrades.** Everyone who buys gets every future version free. There is no Apple
mechanism to charge existing customers for a new version. If you later want paid new
capability, an in-app purchase for genuinely new features is the sanctioned route.
