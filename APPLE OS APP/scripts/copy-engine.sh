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
  # The shipped copy leaves out what the app never runs and what would leak this machine:
  #  - pip: a package installer has no place in a sandboxed app (App Review guideline 2.5.2).
  #    The source engine keeps it, so scripts/refresh-engine.sh still works. ensurepip (a
  #    second copy of pip) and pydoc_data (offline help text) go for the same reason.
  #  - bin/ console scripts other than the interpreter: their shebangs hold this machine's
  #    path (the app runs `python -c`, never these).
  #  - direct_url.json: pip's record of the local folder a package was installed from.
  #  - Tcl/Tk and everything that loads it: CADVERT never opens a Tk window, and Tk's macOS
  #    backend uses private AppKit notifications (NSWindowDid/WillOrderOnScreenNotification),
  #    which App Review rejects.
  # --delete-excluded clears anything an older build copied in before these rules existed.
  rsync -a --delete --delete-excluded \
    --exclude '/lib/python3*/site-packages/pip/' \
    --exclude '/lib/python3*/site-packages/pip-*.dist-info/' \
    --exclude '/bin/python*-config' \
    --include '/bin/python*' \
    --exclude '/bin/*' \
    --exclude 'direct_url.json' \
    --exclude '/lib/libtcl*' --exclude '/lib/tcl9*' --exclude '/lib/tk9*' \
    --exclude '/lib/itcl*' --exclude '/lib/thread[0-9]*' \
    --exclude '/lib/python3*/lib-dynload/_tkinter*' \
    --exclude '/lib/python3*/tkinter/' --exclude '/lib/python3*/idlelib/' --exclude '/lib/python3*/turtledemo/' \
    --exclude '/lib/python3*/ensurepip/' --exclude '/lib/python3*/pydoc_data/' \
    --exclude '/lib/python3*/site-packages/PIL/_imagingtk*' \
    --exclude '/lib/python3*/site-packages/matplotlib/backends/_tkagg*' \
    "$SRC/" "$DST/"

  # App Review scans every file for text, not just code that runs:
  #  - urllib.parse lists 'itms-services' among URL schemes; the scanner reads that as
  #    installing apps (guideline 2.5.2). CADVERT never parses such URLs.
  #  - a networkx docstring cites a blog on an Iranian domain, which U.S. sanctions bar
  #    from the App Store (guideline 5). It's a reference in documentation, nothing more.
  PY="$(ls -d "$DST"/lib/python3.* | head -1)"
  sed -i '' "s/'ws', 'wss', 'itms-services'\]/'ws', 'wss']/" "$PY/urllib/parse.py"
  NX="$PY/site-packages/networkx/algorithms/operators/product.py"
  if [ -f "$NX" ]; then sed -i '' -e '/\[2\] A\. Faraji, "Corona Product/d' -e '/alifaraji\.ir/d' "$NX"; fi
  find "$DST" -path '*/__pycache__/*' \( -name 'parse.cpython-*' -o -name 'product.cpython-*' \) -delete
  # Beyond those two: drop citations and links from every library's documentation text
  # (docstrings, comments, package descriptions). Licence and copyright notices stay.
  # Any engine can run it — it only edits text — so use the one native to this Mac.
  HOSTPY="${SRCROOT}/Engine/engine-$(uname -m)/bin/python3"
  [ -x "$HOSTPY" ] || HOSTPY="$SRC/bin/python3"
  "$HOSTPY" -I "${SRCROOT}/scripts/strip-citations.py" "$PY"

  # Release builds are what App Review sees: refuse to ship if either kind of text is back,
  # e.g. after an engine rebuild picks up a new library version.
  if [ "${CONFIGURATION:-}" = "Release" ]; then
    hits="$(LC_ALL=C grep -rlaE "itms-services|https?://[A-Za-z0-9.-]+\.(ir|cu|sy|kp)([/:\"' <>)]|$)" "$DST" || true)"
    if [ -n "$hits" ]; then
      echo "error: copy-engine: $arch engine contains text App Review rejects:"; echo "$hits"
      exit 1
    fi
  fi
  echo "copy-engine: $arch engine copied ($(du -sh "$DST" | cut -f1))"
  copied=$((copied+1))
done

# Legacy path: nothing left in the bundle under the old name.
rm -rf "$RES/engine"

if [ "$copied" -eq 0 ]; then
  # A Release app can only run the engine inside it, so without one it can't open anything.
  # Fail here rather than upload an app that can't work (e.g. from a CI machine with no engine).
  if [ "${CONFIGURATION:-}" = "Release" ]; then
    echo "error: copy-engine: no engine bundled — build one with ./scripts/bundle-engine.sh"
    exit 1
  fi
  echo "copy-engine: WARNING — no engine bundled; the app will look in ~/.cadvert/engine or a custom Python."
fi
