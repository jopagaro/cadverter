#!/bin/bash
# Xcode post-build phase (macOS only): copy the bundled Python engine(s) into the .app.
#
# One engine per architecture lives at Engine/engine-<arch> (built by scripts/bundle-engine.sh)
# and is copied to CADVERT.app/Contents/Resources/engine-<arch>. The app picks the folder
# matching the slice it is running. Whatever architectures this build targets (Xcode's ARCHS)
# are copied, so a universal build carries both and a native build carries one.
#
# Set CADVERT_ENGINE_ARCHS to override (e.g. "arm64 x86_64" to force a universal payload).
set -euo pipefail
if [ "${PLATFORM_NAME:-}" != "macosx" ]; then exit 0; fi

RES="${CODESIGNING_FOLDER_PATH}/Contents/Resources"
WANTED="${CADVERT_ENGINE_ARCHS:-${ARCHS:-$(uname -m)}}"
mkdir -p "$RES"

# Drop engines this build no longer wants, so switching arch never leaves a stale copy.
for existing in "$RES"/engine-*; do
  [ -d "$existing" ] || continue
  arch="$(basename "$existing" | sed 's/^engine-//')"
  case " $WANTED " in *" $arch "*) ;; *) echo "copy-engine: removing stale $(basename "$existing")"; rm -rf "$existing" ;; esac
done

copied=0
for arch in $WANTED; do
  SRC="${SRCROOT}/Engine/engine-${arch}"
  # Fall back to the legacy single-arch folder when it matches the host.
  if [ ! -x "$SRC/bin/python3" ] && [ "$arch" = "$(uname -m)" ] && [ -x "${SRCROOT}/Engine/engine/bin/python3" ]; then
    SRC="${SRCROOT}/Engine/engine"
  fi
  if [ ! -x "$SRC/bin/python3" ] || [ ! -f "$SRC/.complete" ]; then
    echo "copy-engine: no engine for $arch at $SRC — build ./scripts/bundle-engine.sh --arch $arch"
    continue
  fi
  DST="$RES/engine-${arch}"
  mkdir -p "$DST"
  rsync -a --delete "$SRC/" "$DST/"
  echo "copy-engine: $arch engine copied ($(du -sh "$DST" | cut -f1))"
  copied=$((copied+1))
done

# Legacy path: nothing left in the bundle under the old name.
rm -rf "$RES/engine"

if [ "$copied" -eq 0 ]; then
  echo "copy-engine: WARNING — no engine bundled; the app will look in ~/.cadvert/engine or a custom Python."
fi
