#!/usr/bin/env bash
# Wrapper to run Android E2E benchmarks from this repo (monorepo workspace).
# Delegates to ../android-offline-transcribe/scripts/android-e2e-test.sh when available.
#
# Usage:
#   EVAL_WAV_PATH=... INSTRUMENT_TIMEOUT_SEC=... scripts/android-e2e-test.sh [model_id ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SIBLING_SCRIPT="$PROJECT_DIR/../android-offline-transcribe/scripts/android-e2e-test.sh"

if [ ! -x "$SIBLING_SCRIPT" ]; then
  echo "ERROR: Android benchmark script not found or not executable:" >&2
  echo "  $SIBLING_SCRIPT" >&2
  echo "" >&2
  echo "Run Android benchmarks from the Android repo instead:" >&2
  echo "  cd ../android-offline-transcribe && scripts/android-e2e-test.sh" >&2
  exit 1
fi

# Default to storing Android evidence under this repo so the cross-platform report
# can pick it up without additional path wiring.
export EVIDENCE_DIR="${EVIDENCE_DIR:-$PROJECT_DIR/artifacts/e2e/android}"

exec "$SIBLING_SCRIPT" "$@"

