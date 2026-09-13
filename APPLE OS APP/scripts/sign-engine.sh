#!/bin/sh
# Xcode post-build phase: deep-sign every Mach-O inside the bundled Python engine(s) so the
# app passes Hardened Runtime / notarization / App Store validation. POSIX sh (Xcode runs /bin/sh).
# Runs only for macOS builds. Handles one engine per architecture (engine-arm64, engine-x86_64).
set -eu
[ "${PLATFORM_NAME:-}" = "macosx" ] || exit 0

RES="${CODESIGNING_FOLDER_PATH:-}/Contents/Resources"
IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:-}"
[ -n "$IDENTITY" ] || IDENTITY="-"
ENTITLEMENTS="${SRCROOT}/scripts/engine.entitlements"
export IDENTITY ENTITLEMENTS

found=0
for ENGINE_DIR in "$RES"/engine "$RES"/engine-*; do
  [ -d "$ENGINE_DIR" ] || continue
  found=$((found + 1))
  echo "sign-engine: signing Mach-O files in $(basename "$ENGINE_DIR") with identity '$IDENTITY'"

  # 1. Shared libraries / Python extensions.
  find "$ENGINE_DIR" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 \
    | xargs -0 -P 8 -n 40 sh -c '
      for f in "$@"; do
        codesign --force --sign "$IDENTITY" --timestamp=none "$f" >/dev/null 2>&1 \
          || echo "sign-engine: warning: could not sign $f"
      done' sh

  # 2. Executables (the interpreter itself) get the hardened runtime + engine entitlements.
  find "$ENGINE_DIR/bin" -type f -perm -u+x -print0 \
    | xargs -0 -n 1 sh -c '
      f="$1"
      magic=$(head -c 4 "$f" | od -An -tx1 | tr -d " \n")
      case "$magic" in
        cffaedfe|feedface|feedfacf|cafebabe|bebafeca)
          codesign --force --sign "$IDENTITY" --options runtime --timestamp=none \
            --entitlements "$ENTITLEMENTS" "$f" >/dev/null 2>&1 \
            || codesign --force --sign "$IDENTITY" "$f" ;;
      esac' sh

  echo "sign-engine: $(basename "$ENGINE_DIR") done ($(find "$ENGINE_DIR" -type f \( -name "*.so" -o -name "*.dylib" \) | wc -l | tr -d " ") libraries)"
done

[ "$found" -gt 0 ] || echo "sign-engine: no bundled engine under $RES — skipping"
