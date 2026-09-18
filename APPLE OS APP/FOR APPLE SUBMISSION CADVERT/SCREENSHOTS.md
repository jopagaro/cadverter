# Screenshots — what to capture

Apple accepts **1280×800, 1440×900, 2560×1600 or 2880×1800** for Mac apps, up to 10 per
listing. Your Mac is Retina, so a window capture is already 2× and lands in range.

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

## One caution

The most impressive capture I made shows your Vietnamese client's assembly, including its
part number and a full breakdown of its components. **Don't publish that unless you have
the customer's permission** — a parts list is commercially sensitive and it would be on a
public product page indefinitely. Use your own test part, or a part you own outright.

## Optional polish

If you want captions above each image, there is a composer at `make-screenshots.swift`:

```
swift make-screenshots.swift out.png window-capture.png "Exact geometry, not a guess"
```

It places the capture on a 2880×1800 canvas with a soft shadow and an optional headline.
