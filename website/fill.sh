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
    dist|fill.sh|site.config|README.md|_template) continue ;;
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

# Nothing should be left.
if leftover=$(grep -rno '{{[A-Z_]*}}' dist --include='*.html'); then
  echo "Unfilled tokens — add them to site.config:"
  echo "$leftover"
  exit 1
fi

echo "Built dist/ — upload its contents to the web root."
find dist -name '*.html' | sort | sed 's|^dist|  |'
