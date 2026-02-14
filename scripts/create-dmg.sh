#!/usr/bin/env bash

# Creates a macOS DMG installer from the built .app bundle.
# Expects the app to be at the standard Release derived data path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

APP_PATH="${APP_PATH:-$PROJECT_DIR/build/DerivedData/Build/Products/Release/OfflineTranscriptionMac.app}"
if [ -n "${VERSION:-}" ]; then
  VERSION="$VERSION"
elif [ -n "${GITHUB_SHA:-}" ]; then
  VERSION="${GITHUB_SHA:0:12}"
else
  VERSION="${GITHUB_REF_NAME:-dev}"
fi
DMG_PATH="$PROJECT_DIR/build/VoicePingOfflineTranscribe-${VERSION}.dmg"
STAGING="$PROJECT_DIR/build/dmg-staging"

if [ ! -d "$APP_PATH" ]; then
  echo "ERROR: App bundle not found at $APP_PATH" >&2
  echo "Build with: xcodebuild build -scheme OfflineTranscriptionMac -configuration Release" >&2
  exit 1
fi

rm -rf "$STAGING"
mkdir -p "$STAGING"

# Use ditto for app bundles to preserve metadata and avoid signature breakage.
APP_STAGING_PATH="$STAGING/$(basename "$APP_PATH")"
ditto "$APP_PATH" "$APP_STAGING_PATH"

# Re-sign ad-hoc so Gatekeeper does not treat stale signatures as corrupted.
codesign --force --deep --sign - "$APP_STAGING_PATH"
codesign --verify --deep --strict --verbose=2 "$APP_STAGING_PATH"

ln -s /Applications "$STAGING/Applications"

hdiutil create \
  -volname "VoicePing Offline Transcribe" \
  -srcfolder "$STAGING" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

# Verify the DMG checksum/integrity before publishing.
hdiutil verify "$DMG_PATH"

rm -rf "$STAGING"
echo "DMG created at: $DMG_PATH"
