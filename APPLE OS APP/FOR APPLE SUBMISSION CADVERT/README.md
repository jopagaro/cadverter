# For Apple submission — CADVERT

Everything needed to submit the Mac app, drafted and ready to edit.

| File | What it is |
|---|---|
| `CHECKLIST.md` | **Start here.** Ordered steps, with the ones only you can do marked |
| `REVIEW-NOTES.md` | Paste into App Review Information. Explains the bundled interpreter |
| `APP-STORE-LISTING.md` | Name, subtitle, description, keywords, category, pricing note |
| `PRIVACY-POLICY.md` | Draft to host at a public URL (required) |
| `SUPPORT-PAGE.md` | Draft to host at a public URL (required) |
| `SCREENSHOTS.md` | What to capture, how to capture windows cleanly, accepted sizes |
| `screenshots/` | What has been captured so far |
| `make-screenshots.swift` | Optional: compose captures onto a 2880×1800 canvas with a caption |

## The three things that actually gate submission

1. **An Apple Distribution certificate.** You have none today. Nothing ships without it.
2. **Two public URLs** — privacy policy and support. Drafts are here; they need hosting.
3. **Screenshots.** Two are captured; you need a few more, and one of the two shows a
   client's design and should not be published.

Everything else — the build, the icons, the entitlements, the engine — is done and verified.

## The single most important file

`REVIEW-NOTES.md`. The app ships a Python interpreter and the OpenCASCADE kernel inside the
bundle and runs them as a child process. That is entirely legitimate and is what makes the
app work offline, but it is exactly the shape of thing App Review asks about. The notes say
what it is, why it is there, and that nothing is downloaded or executed from the network —
before a reviewer has to guess.
