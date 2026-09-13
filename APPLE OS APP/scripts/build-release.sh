#!/bin/bash
# Build a signed, per-architecture Release .app (and optionally a .dmg) for direct sale.
#
#   ./scripts/build-release.sh --arch arm64      # Apple silicon
#   ./scripts/build-release.sh --arch x86_64     # Intel
#   ./scripts/build-release.sh --arch arm64 --identity "Developer ID Application: Name (TEAMID)"
#
# One architecture per build keeps each download ~800 MB instead of ~1.6 GB.
# Without --identity the app is ad-hoc signed: fine for local testing, NOT distributable
# (Gatekeeper blocks it on other Macs — you need a Developer ID Application certificate).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

ARCH="$(uname -m)"
IDENTITY=""
MAKE_DMG=1

while [ $# -gt 0 ]; do
  case "$1" in
    --arch)     ARCH="$2"; shift 2 ;;
    --identity) IDENTITY="$2"; shift 2 ;;
    --no-dmg)   MAKE_DMG=0; shift ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

case "$ARCH" in
  arm64|aarch64) ARCH="arm64" ;;
  x86_64|intel)  ARCH="x86_64" ;;
  *) echo "unsupported --arch $ARCH (arm64|x86_64)"; exit 1 ;;
esac

ENGINE="$HERE/Engine/engine-$ARCH"
if [ ! -x "$ENGINE/bin/python3" ] || [ ! -f "$ENGINE/.complete" ]; then
  echo "No $ARCH engine at $ENGINE"
  echo "Build it first:  ./scripts/bundle-engine.sh --arch $ARCH --source .."
  exit 1
fi

OUT="$HERE/build-release/$ARCH"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "▸ building Release for $ARCH"
CADVERT_ENGINE_ARCHS="$ARCH" xcodebuild \
  -project CADVERT.xcodeproj -scheme CADVERT \
  -configuration Release -destination 'generic/platform=macOS' \
  ARCHS="$ARCH" ONLY_ACTIVE_ARCH=NO \
  CONFIGURATION_BUILD_DIR="$OUT" \
  CODE_SIGNING_ALLOWED=NO \
  build 2>&1 | grep -E "error:|copy-engine:|sign-engine:|BUILD" | tail -8

APP="$OUT/CADVERT.app"
[ -d "$APP" ] || { echo "build produced no .app"; exit 1; }

# Verify the right engine, and only that engine, is inside.
echo "▸ engine payload:"
for d in "$APP/Contents/Resources"/engine*; do
  [ -d "$d" ] || continue
  echo "   $(basename "$d")  $(du -sh "$d" | cut -f1)  ($(lipo -archs "$d/bin/python3" 2>/dev/null || echo '?'))"
done

ENTITLEMENTS="$HERE/CADVERT/Resources/CADVERT-macOS.entitlements"
if [ -n "$IDENTITY" ]; then
  echo "▸ signing with: $IDENTITY"
  # Inner Mach-O first (deep signing an engine this large needs the explicit pass), then the app.
  PLATFORM_NAME=macosx CODESIGNING_FOLDER_PATH="$APP" SRCROOT="$HERE" \
    EXPANDED_CODE_SIGN_IDENTITY="$IDENTITY" sh scripts/sign-engine.sh
  codesign --force --sign "$IDENTITY" --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" "$APP"
  codesign --verify --deep --strict --verbose=1 "$APP" 2>&1 | tail -2
else
  echo "▸ ad-hoc signing (local testing only — not distributable)"
  PLATFORM_NAME=macosx CODESIGNING_FOLDER_PATH="$APP" SRCROOT="$HERE" \
    EXPANDED_CODE_SIGN_IDENTITY="-" sh scripts/sign-engine.sh
  codesign --force --sign - --options runtime --entitlements "$ENTITLEMENTS" "$APP"
fi

echo "▸ app: $APP  ($(du -sh "$APP" | cut -f1))"

if [ "$MAKE_DMG" -eq 1 ]; then
  DMG="$OUT/CADVERT-$ARCH.dmg"
  ./scripts/make-dmg.sh "$APP" "$DMG" >/dev/null
  echo "▸ dmg: $DMG  ($(du -sh "$DMG" | cut -f1))"
fi

if [ -z "$IDENTITY" ]; then
  cat <<'NOTE'

To ship this to other Macs you still need to:
  1. Get a "Developer ID Application" certificate (Apple Developer Program).
  2. Rebuild with --identity "Developer ID Application: Your Name (TEAMID)".
  3. Notarize:  xcrun notarytool submit <dmg> --keychain-profile <profile> --wait
  4. Staple:    xcrun stapler staple <dmg>
NOTE
fi
