#!/bin/bash
# Package an exported CADVERT.app into a compressed DMG for direct (non-App Store) sale.
#   ./scripts/make-dmg.sh path/to/CADVERT.app [out.dmg]
set -euo pipefail
APP="${1:?path to CADVERT.app}"
OUT="${2:-$(dirname "$APP")/CADVERT.dmg}"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$OUT"
hdiutil create -volname "CADVERT" -srcfolder "$STAGE" -ov -format UDZO "$OUT"
rm -rf "$STAGE"
echo "▸ wrote $OUT"
echo "  Notarize before shipping:  xcrun notarytool submit \"$OUT\" --keychain-profile <profile> --wait && xcrun stapler staple \"$OUT\""
