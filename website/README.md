# Convivium website

Static HTML. No build step, no dependencies, no JavaScript. One shared stylesheet so every
app's pages look like the same site.

## Layout

```
/                      Convivium home — lists the apps
/cadvert/              CADVERT product page
/cadvert/privacy/      privacy policy   ← Apple requires this URL
/cadvert/support/      support page     ← Apple requires this URL
/assets/style.css      shared design system for the whole site
/_template/            copy this to add the next app (not published)
```

Pages are `index.html` inside a directory, so the URLs are clean (`/cadvert/privacy`) on any
static host without rewrite rules. Links are absolute from the root, so the site must be
served at a domain root — `convivium.com/cadvert/privacy`, not `example.com/site/cadvert/privacy`.

## Publishing

1. Fill in `site.config`.
2. Run `./fill.sh` — it writes `dist/` and **fails if any `{{TOKEN}}` is still unfilled**.
3. Publish it one of two ways:

**On conviviumllc.com (WordPress).** Don't upload files — make pages, so nothing touches the
existing homepage. In WordPress → Pages → Add New:

| Page title | Parent | Slug | Content |
|---|---|---|---|
| CADVERT | (none) | `cadvert` | `dist/wordpress/cadvert.html` |
| Privacy Policy | CADVERT | `privacy` | `dist/wordpress/cadvert-privacy.html` |
| Support | CADVERT | `support` | `dist/wordpress/cadvert-support.html` |

For the content, add a **Custom HTML** block and paste the whole file. The parent page is
what puts the child pages at `/cadvert/privacy/` and `/cadvert/support/`; set it under
Page → Attributes → Parent, and the slug under Page → URL. The pages take your WordPress
theme's look rather than `assets/style.css` — that's fine; Apple only needs them public
and readable.

**On a static host** (Netlify, GitHub Pages, S3…): upload everything in `dist/` except
`wordpress/` to the web root.

Tokens used across the site:

| Token | What it is |
|---|---|
| `{{LEGAL_ENTITY}}` | The name in the copyright line and on the privacy policy. Use the LLC once it's approved. |
| `{{SUPPORT_EMAIL}}` | Public contact address. Worth a real forwarding address rather than a personal inbox. |
| `{{PRICE}}` | Shown on the CADVERT page. |
| `{{APP_STORE_URL}}` | The download button. Leave `#` until the app is live. |

`dist/` is generated — don't edit it, and don't commit it.

## Adding another app

```sh
cp -R _template mynewapp
grep -rl APPSLUG mynewapp | xargs sed -i '' 's/APPSLUG/mynewapp/g'
grep -rl APPNAME mynewapp | xargs sed -i '' 's/APPNAME/My New App/g'
```

Then add a card for it in `index.html`, and set that app's accent colour in the `<style>`
block at the top of each of its pages. Nothing else needs changing — the shared stylesheet
handles type, spacing, dark mode and mobile.

## For the App Store submission

App Store Connect asks for two URLs. Once this is live, they are:

- Privacy Policy URL — `https://conviviumllc.com/cadvert/privacy`
- Support URL — `https://conviviumllc.com/cadvert/support`
- Marketing URL — `https://conviviumllc.com/cadvert` (the CADVERT page, not the homepage)

Both must be publicly reachable **before** you submit; Apple checks them during review.
The page content matches `APPLE OS APP/FOR APPLE SUBMISSION CADVERT/`, so if you change the
app's data handling, change it in both places.
