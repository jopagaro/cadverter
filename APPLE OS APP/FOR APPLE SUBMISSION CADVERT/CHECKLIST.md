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
- [ ] Create a **Mac App Store provisioning profile** for `com.cadvert.CADVERT`.
- [ ] Apply for the **Small Business Program** — it cuts Apple's commission from 30% to 15%
      under $1M a year. It is **not retroactive**, so do it before your first sale.

Verify locally when done:
```
security find-identity -v -p codesigning | grep "Apple Distribution"
```

## 2. Hosting — **[you]**

- [ ] Publish the privacy policy somewhere public (`PRIVACY-POLICY.md`) and note the URL.
- [ ] Publish the support page (`SUPPORT-PAGE.md`) and note the URL.

Both URLs are required fields; you cannot submit without them.

## 3. Build

- [ ] `make engine-all` — both architecture engines present.
- [ ] `make archive-universal` — produces the universal archive (~1.6 GB).
- [ ] In Xcode Organizer: Distribute App → App Store Connect → Upload.

Expect the upload to be slow. It's mostly the two bundled engines.

## 4. App Store Connect — **[you]**

- [ ] New macOS app, bundle ID `com.cadvert.CADVERT`, SKU of your choosing.
- [ ] Paste name, subtitle, description, keywords from `APP-STORE-LISTING.md`.
- [ ] Price: $49 (Tier varies by storefront; Apple converts).
- [ ] Upload screenshots — see `SCREENSHOTS.md` for what to capture and at which sizes.
- [ ] Paste the review notes from `REVIEW-NOTES.md` into App Review Information → Notes.
      **Do not skip this.** It explains the bundled Python interpreter before a reviewer
      finds it and wonders whether you are downloading executable code.
- [ ] Attach `samples/test_block_with_holes.step` so the reviewer has something to open.
- [ ] Age rating: answer everything "None". The app has no objectionable content.
- [ ] Export compliance: see below.

## 5. Export compliance

You will be asked whether the app uses encryption. It does not implement any of its own,
but it does make HTTPS calls to OpenAI and Anthropic when the user supplies a key.

That falls under the standard exemption for apps using only HTTPS. Answer that you use
encryption, then that it qualifies for the exemption. This avoids the annual self-
classification report. If in doubt, Apple's own questionnaire walks you through it.

## 6. After approval

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
