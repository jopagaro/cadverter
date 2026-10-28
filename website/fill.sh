#!/usr/bin/env bash
# Build the publishable site into dist/ by filling the {{TOKENS}} from site.config.
#
#   ./fill.sh
#
# Source files keep their tokens, so you can re-run this after any edit. The build
# fails if a token is left unfilled — that is deliberate: Apple rejects a privacy
# URL that still says {{SUPPORT_EMAIL}}.
set -euo pipefail
cd "$(dirname "$0")"

[ -f site.config ] || { echo "site.config not found"; exit 1; }

rm -rf dist
mkdir -p dist
# Copy the site, minus the things that are not part of it.
for item in *; do
  case "$item" in
    dist|fill.sh|site.config|README.md|_template|wordpress-embed.css) continue ;;
  esac
  cp -R "$item" dist/
done

# Substitute every KEY=VALUE from site.config.
while IFS='=' read -r key value; do
  case "$key" in ''|\#*) continue ;; esac
  # shellcheck disable=SC2001
  esc=$(printf '%s' "$value" | sed 's/[&|]/\\&/g')
  find dist -type f -name '*.html' -exec sed -i '' "s|{{$key}}|$esc|g" {} +
done < site.config

# No store link yet (APP_STORE_URL=#): show "coming soon" rather than a button to nowhere.
if grep -q '^APP_STORE_URL=#$' site.config; then
  find dist -type f -name '*.html' -exec sed -i '' \
    's|<a class="btn" href="#">Download on the Mac App Store</a>|<span class="btn btn-soon">Coming soon to the Mac App Store</span>|' {} +
fi

# Nothing should be left.
if leftover=$(grep -rno '{{[A-Z_]*}}' dist --include='*.html'); then
  echo "Unfilled tokens — add them to site.config:"
  echo "$leftover"
  exit 1
fi

# WordPress: the same pages as paste-ready content for a Custom HTML block. Just what's
# inside <main>, minus the eyebrow and <h1> (WordPress shows the page title itself). Make a
# page "CADVERT" (slug cadvert) with child pages "Privacy" and "Support" and the URLs come
# out the same: /cadvert/, /cadvert/privacy/, /cadvert/support/.
mkdir -p dist/wordpress
python3 - <<'PY_WP'
import pathlib, re
for src, name, wide in (("dist/index.html", "home", True),
                        ("dist/cadvert/index.html", "cadvert", False),
                        ("dist/cadvert/privacy/index.html", "cadvert-privacy", False),
                        ("dist/cadvert/support/index.html", "cadvert-support", False)):
    html = pathlib.Path(src).read_text()
    body = re.search(r"<main>\s*<div class=\"wrap[^\"]*\">(.*)</div>\s*</main>", html, re.S).group(1)
    body = re.sub(r"\s*<!--.*?-->", "", body, flags=re.S)
    # WordPress serves pages without a trailing slash (/cadvert/ answers with a redirect).
    body = re.sub(r'href="(/[^"]*?)/"', r'href="\1"', body)
    body = re.sub(r'\s*<p class="eyebrow">.*?</p>', "", body, count=1, flags=re.S)
    body = re.sub(r"\s*<h1>.*?</h1>", "", body, count=1, flags=re.S)
    css = pathlib.Path("wordpress-embed.css").read_text()
    page = f'<style>\n{css}</style>\n<div class="cadvert-page{" wide" if wide else ""}">\n{body.strip()}\n</div>\n'
    # Plain ASCII, with every other character as an HTML entity (— → &#8212;), so no
    # clipboard or editor along the way can garble the dashes, × and ³.
    pathlib.Path(f"dist/wordpress/{name}.html").write_text(page.encode("ascii", "xmlcharrefreplace").decode())
PY_WP

echo "Built dist/:"
echo "  Static host: upload everything except wordpress/ to the web root."
echo "  WordPress:   paste dist/wordpress/*.html into Custom HTML blocks (see README)."
find dist -name '*.html' | sort | sed 's|^dist|  |'
