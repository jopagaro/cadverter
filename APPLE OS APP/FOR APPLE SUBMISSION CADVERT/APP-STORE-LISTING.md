# App Store listing copy

Every field App Store Connect asks for, with a draft. Edit freely — these are written to be
usable as-is, not to be admired.

---

## Name (30 characters max)

```
CADVERT
```

## Subtitle (30 characters max)

```
Exact CAD geometry, offline AI
```

That is exactly 30 characters, Apple's limit. Alternatives if you want a different emphasis:
- `Read CAD files with AI` (28)
- `CAD analysis that's exact` (25)

## Promotional text (170 max — editable without a new review)

```
Analyse STEP, IGES and BREP files entirely on your Mac. Exact dimensions straight from the CAD geometry — never measured from a render, never uploaded anywhere.
```

162 characters. It names only the formats that get exact dimensions; STL and OBJ are meshes
and report overall size only, which the description already says.

## Description (4000 max)

```
CADVERT reads CAD files the way an engineer does — from the geometry the designer
authored, not from a picture of it.

Open a STEP, IGES or BREP file and CADVERT extracts the exact analytical
geometry: hole diameters, wall thicknesses, fillet radii, bounding boxes, volumes and
centres of mass. Every value comes straight from the file's own definitions, so a wall
that is 11.000 mm thick is reported as 11.000 mm — not as an estimate from a mesh.

EXACT, NOT APPROXIMATE
Most tools convert CAD into triangles before doing anything with it, which throws away the
precision that made it CAD. CADVERT walks the boundary representation directly. It knows a
face rather than guessing from a few thousand triangles.

WHAT IT FINDS
• Through holes, blind holes, counterbores and countersinks, with standard size matching
• Fillets, chamfers, bosses, pockets and repeating patterns
• Wall thicknesses, hole-to-hole distances and edge clearances
• Full topology: every face and edge, with dihedral angles and connectivity
• Assembly structure — which named part each face belongs to, with quantities
• GD&T annotations from AP242 files

ASK QUESTIONS ABOUT YOUR PART
Ask "what size tap do these holes need" or "is this machinable from one side" and get an
answer grounded in real measurements. On Macs with Apple Intelligence this runs entirely
on device, free, with no key and no account. You can also use your own OpenAI or Anthropic
key if you prefer those models.

EVERYTHING RUNS ON YOUR MAC
The geometry engine is built into the app. There is no server, no account, no
subscription, and your files never leave your machine. It works with the network switched
off. For anyone under NDA, that is the point.

RENDERED VIEWS
Six orthographic views and an isometric, generated automatically, so you can see what you
are analysing.

EXPORT
Copy or save the full geometry document as text, ready to paste into a report, a quote, or
an AI tool of your choice.

SUPPORTED FORMATS
STEP (.step, .stp), IGES (.iges, .igs), BREP — full analysis with features and
measurements. STL and OBJ — global properties only, since a mesh has no exact geometry to
read.

REQUIREMENTS
macOS 14 or later on Apple silicon or Intel. Apple Intelligence features need macOS 26 and
an Apple silicon Mac; without it, analysis works exactly the same and questions need an API key.
```

## Keywords (100 characters max, comma-separated, no spaces after commas)

```
CAD,STEP,IGES,engineering,mechanical,geometry,CNC,machining,DFM,manufacturing,BREP,quoting
```

That is 89 characters. Do not repeat the app name — Apple already indexes it.

## Category

- **Primary:** Developer Tools *(matches the current `LSApplicationCategoryType`)*
- **Alternative:** Graphics & Design, or Productivity

Developer Tools is the least competitive of the three and matches how the app behaves.
Graphics & Design gets more browsing traffic but puts you next to illustration apps.

## URLs

All three point at CADVERT's own pages under conviviumllc.com — never the Convivium
homepage, which lists every app. Paste them exactly; WordPress serves them without a
trailing slash.

| App Store Connect field | URL | Shown on the App Store as |
|---|---|---|
| Support URL (required) | `https://conviviumllc.com/cadvert/support` | "App Support" |
| Privacy Policy URL (required) | `https://conviviumllc.com/cadvert/privacy` | "Privacy Policy" |
| Marketing URL (optional) | `https://conviviumllc.com/cadvert` | "Developer Website" |

The Privacy Policy URL is entered under App Privacy, not on the version page.

## Copyright

```
© 2026 Convivium LLC
```

## Version and What's New

```
1.0.0
```

For a first release, "What's New" can simply be:

```
First release.
```

---

## Price

**USD 9.99**, one-time. Set it in App Store Connect → CADVERT → Pricing and Availability;
Apple converts it for other storefronts. The website (`website/site.config`) says the same,
so change both together if it ever moves.

Apple takes 15% under the Small Business Program (applied for), so you keep about **$8.49**
per US sale, before any sales tax Apple collects on top in some regions.
