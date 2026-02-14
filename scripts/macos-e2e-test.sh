#!/usr/bin/env bash
# macOS E2E Test Script - Runs the OfflineTranscriptionMac app in --auto-test mode per model.
# Usage: ./scripts/macos-e2e-test.sh [--app /path/to/OfflineTranscriptionMac.app] [--skip-build] [model_id ...]
# If no model_ids provided, runs the default macOS model list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_FILE="$PROJECT_DIR/VoicePingIOSOfflineTranscribe.xcodeproj"
SCHEME="OfflineTranscriptionMac"
CONFIGURATION="Debug" # --auto-test task is behind #if DEBUG
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-$PROJECT_DIR/build/DerivedData}"

EVIDENCE_DIR="${EVIDENCE_DIR:-$PROJECT_DIR/artifacts/e2e/macos}"
WAV_SOURCE="${EVAL_WAV_PATH:-$PROJECT_DIR/artifacts/benchmarks/long_en_eval.wav}"

APP_BUNDLE_OVERRIDE=""
SKIP_BUILD=false

ALL_MODELS=(
  "sensevoice-small"
  "whisper-tiny"
  "whisper-base"
  "whisper-small"
  "whisper-large-v3-turbo"
  "whisper-large-v3-turbo-compressed"
  "moonshine-tiny"
  "moonshine-base"
  "zipformer-20m"
  "omnilingual-300m"
  "parakeet-tdt-v3"
  "qwen3-asr-0.6b"
  "qwen3-asr-0.6b-onnx"
  "apple-speech"
)

# Parse args
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP_BUNDLE_OVERRIDE="${2:-}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  MODELS=("${POSITIONAL[@]}")
else
  MODELS=("${ALL_MODELS[@]}")
fi

model_timeout_sec() {
  local model_id="$1"
  case "$model_id" in
    whisper-large-v3-turbo*) echo 2400 ;;
    whisper-small|parakeet-tdt-v3) echo 1200 ;;
    qwen3-asr-0.6b|qwen3-asr-0.6b-onnx) echo 1800 ;;
    omnilingual-300m) echo 1800 ;;
    whisper-base) echo 900 ;;
    *) echo 600 ;;
  esac
}

echo "=== macOS E2E Test Suite ==="
echo "Project:        $PROJECT_DIR"
echo "Scheme:         $SCHEME ($CONFIGURATION)"
echo "Models to test: ${MODELS[*]}"
echo "Audio fixture:  $WAV_SOURCE"
echo "Evidence dir:   $EVIDENCE_DIR"
if [[ -n "$APP_BUNDLE_OVERRIDE" ]]; then
  echo "App bundle:     $APP_BUNDLE_OVERRIDE"
fi
echo ""

mkdir -p "$EVIDENCE_DIR"

if [[ ! -f "$WAV_SOURCE" ]]; then
  echo "ERROR: WAV source not found: $WAV_SOURCE" >&2
  exit 1
fi

# Place the evaluation WAV on the host path used by the app's auto-test fallback.
cp "$WAV_SOURCE" /private/tmp/test_speech.wav
echo "Test WAV placed at /private/tmp/test_speech.wav"
echo ""

if [[ -n "$APP_BUNDLE_OVERRIDE" ]]; then
  APP_PATH="$APP_BUNDLE_OVERRIDE"
elif [[ "$SKIP_BUILD" = true ]]; then
  APP_PATH="$DERIVED_DATA_PATH/Build/Products/$CONFIGURATION/OfflineTranscriptionMac.app"
else
  echo "Building OfflineTranscriptionMac..."
  xcodebuild \
    -project "$PROJECT_FILE" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -derivedDataPath "$DERIVED_DATA_PATH" \
    CODE_SIGNING_ALLOWED=NO \
    build >/tmp/macos_e2e_build.log

  APP_PATH="$DERIVED_DATA_PATH/Build/Products/$CONFIGURATION/OfflineTranscriptionMac.app"
fi

APP_EXEC="$APP_PATH/Contents/MacOS/OfflineTranscriptionMac"
if [[ ! -x "$APP_EXEC" ]]; then
  echo "ERROR: built app executable not found at $APP_EXEC" >&2
  exit 1
fi

# Prefer container temp output when sandboxed, but also accept /tmp when running unsigned.
BUNDLE_ID="$(python3 -c 'import plistlib,sys; pl=plistlib.load(open(sys.argv[1],"rb")); print(pl.get("CFBundleIdentifier",""))' "$APP_PATH/Contents/Info.plist")"
SANDBOX_TMP="$HOME/Library/Containers/$BUNDLE_ID/Data/tmp"
GROUP_ROOT="$HOME/Library/Group Containers"
GROUP_TMP=""
if [[ -d "$GROUP_ROOT" ]]; then
  for d in "$GROUP_ROOT"/*; do
    [[ -d "$d" ]] || continue
    base="$(basename "$d")"
    if [[ "$base" == "group.com.voiceping.transcribe" || "$base" == *.group.com.voiceping.transcribe ]]; then
      GROUP_TMP="$d"
      break
    fi
  done
fi
if [[ -z "$GROUP_TMP" ]]; then
  GROUP_TMP="$GROUP_ROOT/group.com.voiceping.transcribe"
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

CURRENT_PID=""
cleanup() {
  if [[ -n "${CURRENT_PID:-}" ]]; then
    kill "$CURRENT_PID" >/dev/null 2>&1 || true
    wait "$CURRENT_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for MODEL_ID in "${MODELS[@]}"; do
  MODEL_DIR="$EVIDENCE_DIR/$MODEL_ID"
  rm -rf "$MODEL_DIR"
  mkdir -p "$MODEL_DIR"

  RESULT_HOST_TMP="/tmp/e2e_result_${MODEL_ID}.json"
  RESULT_GROUP_TMP="$GROUP_TMP/e2e_result_${MODEL_ID}.json"
  RESULT_SANDBOX_TMP="$SANDBOX_TMP/e2e_result_${MODEL_ID}.json"
  rm -f "$RESULT_HOST_TMP" "$RESULT_GROUP_TMP" "$RESULT_SANDBOX_TMP"

  TIMEOUT_SEC="$(model_timeout_sec "$MODEL_ID")"
  echo "--- Testing: $MODEL_ID (timeout: ${TIMEOUT_SEC}s) ---"

  # Launch the app.
  # Most models can run by executing the Mach-O directly. For Apple Speech, launch via `open`
  # so TCC associates the process with the app bundle (direct exec can still TCC-abort).
  CURRENT_PID=""
  if [[ "$MODEL_ID" == "apple-speech" ]]; then
    open -n "$APP_PATH" --args --auto-test --model-id "$MODEL_ID" >/dev/null 2>&1 || true
  else
    "$APP_EXEC" --auto-test --model-id "$MODEL_ID" >/dev/null 2>&1 &
    CURRENT_PID="$!"
  fi

  START_TS="$(date +%s)"
  RESULT_PATH=""
  while [[ -z "$RESULT_PATH" ]]; do
    if [[ -f "$RESULT_SANDBOX_TMP" ]]; then RESULT_PATH="$RESULT_SANDBOX_TMP"; break; fi
    if [[ -f "$RESULT_GROUP_TMP" ]]; then RESULT_PATH="$RESULT_GROUP_TMP"; break; fi
    if [[ -f "$RESULT_HOST_TMP" ]]; then RESULT_PATH="$RESULT_HOST_TMP"; break; fi
    NOW_TS="$(date +%s)"
    # If the app process has already exited and no result file was written, fail fast.
    if (( NOW_TS - START_TS >= 2 )); then
      if [[ -n "${CURRENT_PID:-}" ]]; then
        if ! kill -0 "$CURRENT_PID" >/dev/null 2>&1; then
          break
        fi
      else
        if ! pgrep -x OfflineTranscriptionMac >/dev/null 2>&1; then
          break
        fi
      fi
    fi
    if (( NOW_TS - START_TS >= TIMEOUT_SEC )); then
      break
    fi
    sleep 0.2
  done

  if [[ -n "$RESULT_PATH" ]]; then
    cp "$RESULT_PATH" "$MODEL_DIR/result.json"
  else
    cat >"$MODEL_DIR/result.json" <<JSON
{
  "model_id": "$MODEL_ID",
  "engine": "",
  "pass": false,
  "skipped": false,
  "error": "app exited or timed out waiting for E2E result file",
  "transcript": "",
  "duration_ms": 0.0,
  "tokens_per_second": 0.0
}
JSON
  fi

  # Stop the app instance for this model (best-effort; it should self-terminate).
  if [[ -n "${CURRENT_PID:-}" ]]; then
    kill "$CURRENT_PID" >/dev/null 2>&1 || true
    wait "$CURRENT_PID" >/dev/null 2>&1 || true
    CURRENT_PID=""
  fi

  STATUS="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('SKIP' if r.get('skipped', False) else ('PASS' if r.get('pass', False) else 'FAIL'))" "$MODEL_DIR/result.json" 2>/dev/null || echo UNKNOWN)"
  DURATION="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"{r.get('duration_ms',0):.0f}ms\")" "$MODEL_DIR/result.json" 2>/dev/null || echo '')"
  TRANSCRIPT="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print((r.get('transcript','') or '')[:80])" "$MODEL_DIR/result.json" 2>/dev/null || echo '')"
  if [[ "$STATUS" == "PASS" ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "  PASS ($DURATION) - $TRANSCRIPT"
  elif [[ "$STATUS" == "SKIP" ]]; then
    SKIP_COUNT=$((SKIP_COUNT + 1))
    REASON="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print((r.get('error','') or '')[:120])" "$MODEL_DIR/result.json" 2>/dev/null || echo '')"
    echo "  SKIP - $REASON"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    REASON="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print((r.get('error','') or '')[:120])" "$MODEL_DIR/result.json" 2>/dev/null || echo '')"
    echo "  FAIL ($DURATION) - $REASON"
  fi
  echo ""
done

echo "=== macOS E2E Test Summary ==="
echo "Total: ${#MODELS[@]} | Pass: $PASS_COUNT | Fail: $FAIL_COUNT | Skip: $SKIP_COUNT"
echo "Evidence directory: $EVIDENCE_DIR"
