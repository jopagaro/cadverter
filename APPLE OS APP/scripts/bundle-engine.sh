#!/bin/bash
# Build a self-contained, relocatable Python engine for the Mac app.
#
#   ./scripts/bundle-engine.sh                 # Apple silicon, cadvert from PyPI
#   ./scripts/bundle-engine.sh --arch x86_64   # Intel Macs (works on an Apple silicon host)
#   ./scripts/bundle-engine.sh --source ..     # install this repo's own source tree
#   ./scripts/bundle-engine.sh --source "cadvert==0.3.0"
#
# Uses python-build-standalone (astral-sh) — a CPython build with no absolute paths,
# so it can live inside CADVERT.app/Contents/Resources/engine and run on any Mac.
# Output: Engine/engine/ (gitignored). Expect ~700-900 MB: OpenCASCADE + VTK are big.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="arm64"
SOURCE="cadvert"
PY_MINOR="3.12"
PBS_TAG=""          # empty = latest release
# OpenCASCADE bindings: pin to the release cadvert is developed and tested against.
# (cadquery-ocp 8.0.x — OpenCASCADE 8 — changes the module layout: e.g.
#  OCP.TopTools.TopTools_IndexedMapOfShape is gone, which breaks cadvert.server.)
OCP_VERSION="7.9.3.1"

while [ $# -gt 0 ]; do
  case "$1" in
    --arch)   ARCH="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --python) PY_MINOR="$2"; shift 2 ;;
    --tag)    PBS_TAG="$2"; shift 2 ;;
    --ocp)    OCP_VERSION="$2"; shift 2 ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

case "$ARCH" in
  arm64|aarch64) ARCH="arm64";  TRIPLE="aarch64-apple-darwin" ;;
  x86_64|intel)  ARCH="x86_64"; TRIPLE="x86_64-apple-darwin" ;;
  *) echo "unsupported --arch $ARCH (arm64|x86_64)"; exit 1 ;;
esac

# One engine per architecture: Engine/engine-arm64, Engine/engine-x86_64.
# The macOS build copies the slice(s) it needs and the app picks the matching one at runtime.
OUT="$HERE/Engine/engine-$ARCH"
DL="$HERE/Engine/downloads"
mkdir -p "$DL"

if [ -z "$PBS_TAG" ]; then
  echo "▸ resolving latest python-build-standalone release…"
  PBS_TAG=$(curl -fsSL https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest \
    | grep -oE '"tag_name": *"[^"]+"' | head -1 | sed -E 's/.*"([^"]+)"$/\1/')
fi
echo "▸ python-build-standalone tag: $PBS_TAG"

ASSET_URL=$(curl -fsSL "https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/$PBS_TAG" \
  | grep -oE '"browser_download_url": *"[^"]+"' \
  | grep -E "cpython-${PY_MINOR}\.[0-9]+(\+|%2B)${PBS_TAG}-${TRIPLE}-install_only_stripped\.tar\.gz\"" \
  | head -1 | sed -E 's/.*"(https[^"]+)"$/\1/' || true)
if [ -z "$ASSET_URL" ]; then
  echo "could not find a cpython ${PY_MINOR} ${TRIPLE} install_only_stripped asset in $PBS_TAG"; exit 1
fi
TARBALL="$DL/$(basename "$ASSET_URL" | sed 's/%2B/+/g')"
if [ ! -f "$TARBALL" ]; then
  echo "▸ downloading $(basename "$ASSET_URL")"
  curl -fL --progress-bar -o "$TARBALL" "$ASSET_URL"
fi

echo "▸ unpacking into $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
tar -xzf "$TARBALL" -C "$OUT" --strip-components=1   # tarball root is "python/"

PY="$OUT/bin/python3"

# Cross-building an Intel engine on Apple silicon: run the x86_64 interpreter under
# Rosetta so pip resolves x86_64 wheels. (Native builds use no prefix.)
RUN=()
HOST_ARCH="$(uname -m)"
if [ "$ARCH" != "$HOST_ARCH" ]; then
  if [ "$ARCH" = "x86_64" ] && [ "$HOST_ARCH" = "arm64" ]; then
    if ! /usr/bin/pgrep -q oahd; then
      echo "Rosetta 2 is required to build an Intel engine on Apple silicon."
      echo "Install it with:  softwareupdate --install-rosetta --agree-to-license"
      exit 1
    fi
    RUN=(arch -x86_64)
    echo "▸ cross-building for x86_64 via Rosetta"
  else
    echo "cannot build a $ARCH engine on a $HOST_ARCH host"; exit 1
  fi
fi

"${RUN[@]}" "$PY" --version

echo "▸ installing cadvert engine ($SOURCE) with server, llm, mesh, graph, dxf extras"
"${RUN[@]}" "$PY" -m pip install --upgrade pip >/dev/null
"${RUN[@]}" "$PY" -m pip install --no-cache-dir "cadquery-ocp==${OCP_VERSION}"
if [ -d "$SOURCE" ]; then
  "${RUN[@]}" "$PY" -m pip install --no-cache-dir "${SOURCE}[server,llm,mesh,graph,dxf]"
else
  # Allow both "cadvert" and "cadvert==x.y.z"
  if [[ "$SOURCE" == *"["* ]]; then
    "${RUN[@]}" "$PY" -m pip install --no-cache-dir "$SOURCE"
  else
    "${RUN[@]}" "$PY" -m pip install --no-cache-dir "${SOURCE}[server,llm,mesh,graph,dxf]"
  fi
fi

echo "▸ pruning caches, tests and tooling we don't ship"
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT/lib" -type d \( -name "tests" -o -name "test" \) -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$OUT/lib/python${PY_MINOR}/idlelib" "$OUT/lib/python${PY_MINOR}/tkinter" \
       "$OUT/lib/python${PY_MINOR}/turtledemo" "$OUT/share" 2>/dev/null || true

echo "▸ smoke test: import cadvert.server + uvicorn"
"$PY" - <<'PYEOF'
import cadvert, cadvert.server, uvicorn, fastapi
print("cadvert", getattr(cadvert, "__version__", "?"), "· fastapi", fastapi.__version__, "· uvicorn", uvicorn.__version__)
PYEOF

# Reset the pip shebangs so nothing points at an absolute build path.
for f in "$OUT"/bin/pip* "$OUT"/bin/cadvert* "$OUT"/bin/uvicorn* "$OUT"/bin/f2py*; do
  [ -f "$f" ] || continue
  if head -1 "$f" | grep -q "^#!"; then
    tmp="$f.tmp"; { echo "#!/bin/sh"; echo '"exec" "$(dirname "$0")/python3" "$0" "$@"'; tail -n +2 "$f"; } > "$tmp"
    mv "$tmp" "$f"; chmod +x "$f"
  fi
done

# Marker so the Xcode copy phase never picks up a half-built engine.
date > "$OUT/.complete"
echo "▸ done: $(du -sh "$OUT" | cut -f1) at $OUT  (arch: $ARCH)"
echo "  Build the macOS app (Release) and the folder is copied to CADVERT.app/Contents/Resources/engine"
