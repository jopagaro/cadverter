# Screenshots — what to capture

Apple accepts **only 1280×800, 1440×900, 2560×1600 or 2880×1800** for Mac apps, up to 10
per listing. A window capture is whatever size the window happens to be (2360×1520, say),
which is almost never one of those — App Store Connect rejects it. Every capture goes
through the composer at the bottom of this page before upload.

## How to capture a clean window (no desktop, no personal files)

```
⌘⇧4 then press Space, then click the CADVERT window
```

The cursor becomes a camera and the capture contains **only the window**, with its rounded
corners and a drop shadow, saved to your Desktop. Hold **Option** while clicking to drop the
shadow if you prefer a tighter crop.

Do not use ⌘⇧3 (full screen) — it captures your desktop, your other windows and your file
names. The first shot I took did exactly that and showed your folders.

## The five worth having, in order

| # | What to show | How to get there |
|---|---|---|
| 1 | **A real assembly analysed** — sidebar full of detected features, the 7 rendered views along the top | Open a large STEP assembly and wait for it to finish |
| 2 | **A question answered** — your question and the reply visible in the chat pane | Ask "What are the hole diameters and how thick is the thinnest wall?" |
| 3 | **The geometry document** — exact analytical values, the thing competitors can't produce | Click "View HSD →" in the sidebar |
| 4 | **Settings** — Apple Intelligence selected, "Local engine · ready", the Storage row | Open Settings (⌘,) |
| 5 | **Dark mode** — same main view, dark theme | Toggle Dark in the header, reopen a part |

Shot 1 is the one that sells it. The feature list down the side (236 fillets, 186
countersinks, 110 bosses) is the proof that it read the geometry rather than looked at a
picture.

## Screenshots are not committed

`screenshots/` is gitignored. Screenshots contain whatever part happened to be open, and
that part is often someone else's model with their part numbers in it. Keeping them out of
the repo means a published repository can never leak a design that is not yours.

## One caution

Only show parts you own. The large assembly used while developing CADVERT was downloaded
from GrabCAD — it is someone else's model, with their part numbers in it, and a shared
download is not permission to use it in your marketing. `samples/test_block_with_holes.step`
(generated in this repo) or a part you designed yourself is safe.

## Make them the right size (required)

`make-screenshots.swift` places a window capture on a 2880×1800 canvas with a soft shadow,
and optionally a headline above it:

```
swift make-screenshots.swift screenshots/app-store/03-settings.png screenshots/settings.png
swift make-screenshots.swift screenshots/app-store/04-answer.png screenshots/answer.png "Ask it anything about the part"
```

Upload the files from `screenshots/app-store/`. The two you captured are already done:
`01-hero-2880x1800.png` and `02-geometry-document-2880x1800.png`.

The hero shot shows the empty start screen. It's valid, but shot 1 above — a part already
analysed, features down the side — does far more to sell the app if you take one.
