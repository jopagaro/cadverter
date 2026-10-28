#!/bin/sh
# Xcode post-build phase: deep-sign every Mach-O inside the bundled Python engine(s) so the
# app passes Hardened Runtime / notarization / App Store validation. POSIX sh (Xcode runs /bin/sh).
# Runs only for macOS builds. Handles one engine per architecture (engine-arm64, engine-x86_64).
set -eu
[ "${PLATFORM_NAME:-}" = "macosx" ] || exit 0

RES="${CODESIGNING_FOLDER_PATH:-}/Contents/Resources"
IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:-}"
[ -n "$IDENTITY" ] || IDENTITY="-"

# The interpreter's entitlements must match the app's sandbox. In a sandboxed app it needs
# app-sandbox + inherit (App Store validation rejects any executable without app-sandbox),
# but an inherit helper traps at launch when its parent isn't sandboxed — as in Debug.
APP_ENTITLEMENTS="${SRCROOT}/${CODE_SIGN_ENTITLEMENTS:-}"
SANDBOXED=0
if [ -f "$APP_ENTITLEMENTS" ] && \
   [ "$(/usr/libexec/PlistBuddy -c 'Print :com.apple.security.app-sandbox' "$APP_ENTITLEMENTS" 2>/dev/null)" = "true" ]; then
  SANDBOXED=1
  ENTITLEMENTS="${SRCROOT}/scripts/engine.entitlements"
else
  ENTITLEMENTS="${SRCROOT}/scripts/engine-debug.entitlements"
fi
echo "sign-engine: app sandboxed=$SANDBOXED — interpreter entitlements $(basename "$ENTITLEMENTS")"
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
  #    A failure stops the build: an interpreter signed without them would run here (it
  #    inherits the app's sandbox) but be rejected by App Store validation.
  for f in "$ENGINE_DIR"/bin/*; do
    [ -f "$f" ] && [ ! -L "$f" ] && [ -x "$f" ] || continue
    magic=$(head -c 4 "$f" | od -An -tx1 | tr -d " \n")
    case "$magic" in
      cffaedfe|feedface|feedfacf|cafebabe|bebafeca) ;;
      *) continue ;;
    esac
    if ! codesign --force --sign "$IDENTITY" --options runtime --timestamp=none \
         --entitlements "$ENTITLEMENTS" "$f"; then
      echo "error: sign-engine: could not sign $f with $(basename "$ENTITLEMENTS")" >&2
      exit 1
    fi
    if [ "$SANDBOXED" = 1 ] && ! codesign -d --entitlements - --xml "$f" 2>/dev/null | grep -q "com.apple.security.app-sandbox"; then
      echo "error: sign-engine: $f is missing the app-sandbox entitlement" >&2
      exit 1
    fi
  done

  echo "sign-engine: $(basename "$ENGINE_DIR") done ($(find "$ENGINE_DIR" -type f \( -name "*.so" -o -name "*.dylib" \) | wc -l | tr -d " ") libraries)"
done

[ "$found" -gt 0 ] || echo "sign-engine: no bundled engine under $RES — skipping"
