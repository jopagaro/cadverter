# Selling CADVERT for Apple platforms

Three routes, in the order I'd do them. The Mac app is the product the web app already
advertises ("Get the Desktop App — $20: unlimited files, full analysis, offline").

## 0. One-time setup

1. Join the Apple Developer Program (US$99/yr) with the Apple ID that owns Team
   `AZVC5A74YD` (the team already in `project.yml`; change it if you sell under a company).
2. In App Store Connect create the app record: bundle ID `com.cadvert.CADVERT`,
   name "CADVERT", SKU `cadvert-1`. Turn on **Universal Purchase** if you want Mac + iPad
   in one purchase.
3. Legal pages you'll be asked for: privacy policy (mention: files are processed on-device
   on the Mac; on iOS they are uploaded to the server you configure; with Apple
   Intelligence nothing leaves the device; with OpenAI or Claude the part summary and
   your question go to that vendor using your own key), support URL, marketing URL.
4. Bump `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION` in `project.yml` for each release,
   then `make generate`.

## 1. Mac App Store (paid app — recommended first)

Why: Apple handles payment, licensing, updates and refunds; a one-time price like
$19.99 is exactly the "desktop app" tier. Apple keeps 15% (Small Business Program) or 30%.

Steps

**Decision: version 1.0 ships Apple silicon only.** The Intel build works and is kept in the
repo, but it has never run on real Intel hardware — only under Rosetta, which cannot exercise
Intel/AMD graphics drivers (see "Intel" below). Apple silicon is the large majority of active
Macs and Apple stopped selling Intel ones in 2023, so holding the Intel build costs very little.
Release it when a customer asks, and verify rendering on their Mac or a rented Intel host first.

1. Build the engine for the architecture you sell to: `make engine` (this Mac). For Intel,
   `make engine-intel` cross-builds through Rosetta, or `make engine-all` builds both.
   Engines land in `Engine/engine-arm64` / `Engine/engine-x86_64`, are copied into the app
   as `Contents/Resources/engine-<arch>`, and the app picks the one matching the slice it
   is running — including the Intel engine under Rosetta. Choose a shape:

   - `make release-arm64` — the shipping build for 1.0: one Apple silicon DMG, about 800 MB.
   - `make release-both` — adds the Intel DMG alongside it, once Intel is verified.
   - `make archive-universal` — one universal app carrying both engines, about 1.6 GB.
     Required for the App Store, which cannot thin a `Resources` folder.

   If a buyer somehow runs the wrong build, the app says which download they need rather
   than failing with a generic engine error.
2. Release config uses `CADVERT-macOS.entitlements`: App Sandbox **on**, network client +
   server (the engine binds a loopback port), user-selected file access, and the two
   Hardened Runtime relaxations the Python interpreter needs. Debug builds use
   `CADVERT-macOS-debug.entitlements` (no sandbox) so a developer venv works.
3. `make archive-mac` (or Xcode → Product → Archive), then Organizer → Distribute →
   App Store Connect. The post-build phases already sign every `.so`/`.dylib` and the
   interpreter inside `Contents/Resources/engine`, so validation passes the
   "nested code must be signed" check.
4. Review notes to include: "The app launches a bundled Python interpreter
   (Contents/Resources/engine) as a child process that listens only on 127.0.0.1; it is
   the geometry engine. Chat uses Apple Intelligence on-device by default; OpenAI/Claude
   are optional and use the user's own key." Provide a sample STEP file
   (`samples/test_block_with_holes.step`); the reviewer can test chat with no key on a
   Mac that has Apple Intelligence enabled.
5. Price: Pricing and Availability → Paid → pick the tier nearest $19.99. Consider a
   free 7-day trial via a Free app + non-consumable IAP if you want try-before-buy; the
   simplest launch is a straight paid app.
6. Upload screenshots (1280×800 or 2560×1600 for Mac; the layout is the same as the
   web app so reuse your marketing shots).

Gotchas

- App size will be ~1 GB compressed less; Apple allows it, but say so on the store page.
- If review objects to "unsigned executable memory", you can drop that entitlement:
  CPython does not JIT; it is only there for ctypes/cffi corner cases.
- First launch under sandbox writes the engine's temp files to the app container —
  already handled (the engine uses `TMPDIR`).

## 2. Direct download (DMG) — sell outside the App Store

Why: no 15–30% cut, no review delays, you can offer a trial and license keys.

1. Create a **Developer ID Application** certificate in your developer account and let
   Xcode manage signing (`CODE_SIGN_STYLE = Automatic` is set).
2. Archive Release, then Organizer → Distribute → **Developer ID** → Upload (notarize).
   Or from the terminal:
   ```bash
   make archive-mac
   xcodebuild -exportArchive -archivePath build/CADVERT.xcarchive -exportPath build/export \
     -exportOptionsPlist scripts/ExportOptions-developer-id.plist   # method: developer-id
   xcrun notarytool submit build/export/CADVERT.app --keychain-profile AC_PROFILE --wait
   xcrun stapler staple build/export/CADVERT.app
   make dmg            # scripts/make-dmg.sh → build/CADVERT.dmg (staple the DMG too)
   ```
3. Sell through a merchant of record that handles VAT/sales tax: Paddle, Lemon Squeezy,
   or Gumroad. All three can issue license keys; add a "License" field in Settings and
   validate it against their API (about 40 lines in `AppModel`) if you want to gate
   the app, or simply sell the download link.
4. Host the DMG on the website next to the web app; the existing "Get the Desktop App"
   button already exists in `index.html` — point it at the store page.

You can keep the sandbox on for direct distribution too (recommended), or turn it off
in `CADVERT-macOS.entitlements` if you want users to point the app at their own Python.

## 3. iPad and iPhone (App Store)

The iOS app is a client: it needs a hosted `cadvert-server` (the FastAPI app in this
repo). Order of work:

1. **Host the server** — `pip install cadvert[full]`, run `cadvert-server` behind HTTPS
   (Fly.io, Render, a small VPS with Caddy). Set `ALLOWED_ORIGINS`, `MAX_FILE_MB`,
   `OPENAI_API_KEY` (if you pay for chat) and keep `DISABLE_AUTH=0`.
2. **Auth.** The server verifies Google ID tokens; iOS review prefers Sign in with Apple
   whenever you offer third-party login. Add `POST /auth/apple` on the server (verify the
   identity token against Apple's JWKS, upsert the user with `apple_id` as the key) and
   use `SignInWithAppleButton` in the app; send the resulting token as
   `Authorization: Bearer`. `CadvertClient` already sends that header.
3. **Payments.** Stripe links for digital features are not allowed in iOS apps. Either
   (a) make the iOS app free and let it use the account tier bought on the web (allowed
   if the app never links to the purchase page — the "reader app" model), or (b) sell a
   StoreKit subscription mirroring the web's Pro/BYOK tiers and report entitlements to the
   server. (a) is far less work.
4. Archive with the iOS destination, upload, TestFlight, then submit. Screenshot sizes:
   iPad 13" (2064×2752) and iPhone 6.9" (1320×2868).

Until 2–3 are done, ship the iOS build only through TestFlight / Ad Hoc to people who run
their own server with `DISABLE_AUTH=1`.

## Release checklist

- [ ] `make test` green; `make build-ios` green
- [ ] Engine bundled for every architecture you ship (`Engine/engine-<arch>/.complete` exists)
- [ ] Listing states "Apple silicon (M1 or later)" as a requirement
- [ ] If shipping Intel: launched and a part rendered on real Intel hardware, not just Rosetta
- [ ] Version bumped in `project.yml`; `make generate`
- [ ] Release archive signed with the right identity (App Store vs Developer ID)
- [ ] Launch the exported app on a clean Mac (no Python installed): drop
      `samples/test_block_with_holes.step`, confirm views + HSD, add a key, ask a question
- [ ] Privacy policy + support page live
