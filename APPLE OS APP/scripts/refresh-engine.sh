#!/bin/bash
# Reinstall the current cadvert source into engines that already exist.
#
# The engines take a long time to build because they download CPython and ~800 MB of
# wheels. Once built, the only thing that changes between releases is cadvert itself, so
# this reinstalls just that — seconds instead of many minutes.
#
# Run it after ANY change to src/cadvert before archiving. Shipping an engine built
# before your latest server changes is easy to do and hard to notice: the app launches,
# analysis works, and only the newest endpoints are quietly missing.
#
#   ./scripts/refresh-engine.sh              # every engine present
#   ./scripts/refresh-engine.sh arm64        # just one
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ARCHS=("$@")
if [ ${#ARCHS[@]} -eq 0 ]; then
  ARCHS=()
  for d in "$HERE"/Engine/engine-*; do
    [ -d "$d" ] && ARCHS+=("$(basename "$d" | sed 's/^engine-//')")
  done
fi
[ ${#ARCHS[@]} -gt 0 ] || { echo "no engines found — run ./scripts/bundle-engine.sh first"; exit 1; }

HOST="$(uname -m)"
for arch in "${ARCHS[@]}"; do
  ENGINE="$HERE/Engine/engine-$arch"
  PY="$ENGINE/bin/python3"
  [ -x "$PY" ] || { echo "skip $arch: no interpreter at $PY"; continue; }

  RUNNER=""
  if [ "$arch" != "$HOST" ] && [ "$arch" = "x86_64" ] && [ "$HOST" = "arm64" ]; then
    /usr/bin/pgrep -q oahd || { echo "skip x86_64: Rosetta not installed"; continue; }
    RUNNER="arch -x86_64"
  fi

  echo "▸ $arch: reinstalling cadvert from $REPO"
  $RUNNER "$PY" -m pip install -q --no-deps --force-reinstall "$REPO"
  VER="$($RUNNER "$PY" -c 'import cadvert; print(cadvert.__version__)')"
  $RUNNER "$PY" -c 'import cadvert.server as s; assert hasattr(s, "list_models"); assert not hasattr(s, "create_checkout")' \
    && echo "  cadvert $VER installed and current"
done

find "$HERE"/Engine/engine-* -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
for d in "$HERE"/Engine/engine-*; do [ -d "$d" ] && date > "$d/.complete"; done
echo "▸ done — rebuild the app so the refreshed engine is copied in"
