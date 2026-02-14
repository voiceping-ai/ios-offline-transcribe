#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WAV_PATH="${QWEN_BENCH_WAV:-$PROJECT_DIR/artifacts/benchmarks/long_en_eval.wav}"
MODEL_REPO="${QWEN_MODEL_REPO:-jima/qwen3-asr-0.6b-onnx-int8}"
MODEL_DIR_DEFAULT="$HOME/Library/Application Support/SherpaModels/${MODEL_REPO//\//_}"
MODEL_DIR="${QWEN_MODEL_DIR:-$MODEL_DIR_DEFAULT}"

REQUIRED_FILES=(
  "encoder.int8.onnx"
  "decoder_prefill.int8.onnx"
  "decoder_decode.int8.onnx"
  "embed_tokens.fp16.npy"
  "vocab.json"
  "config.json"
  "tokens.json"
)

has_local_bundle() {
  [[ -f "$MODEL_DIR/vocab.json" ]] && \
  [[ -f "$MODEL_DIR/config.json" ]] && \
  [[ -f "$MODEL_DIR/tokens.json" ]] && \
  ([[ -f "$MODEL_DIR/embed_tokens.fp16.npy" ]] || [[ -f "$MODEL_DIR/embed_tokens.npy" ]]) && \
  ([[ -f "$MODEL_DIR/encoder.int8.onnx" ]] || [[ -f "$MODEL_DIR/encoder.onnx" ]]) && \
  ([[ -f "$MODEL_DIR/decoder_prefill.int8.onnx" ]] || [[ -f "$MODEL_DIR/decoder_prefill.onnx" ]]) && \
  ([[ -f "$MODEL_DIR/decoder_decode.int8.onnx" ]] || [[ -f "$MODEL_DIR/decoder_decode.onnx" ]])
}

download_if_missing() {
  if has_local_bundle; then
    echo "Using local model bundle: $MODEL_DIR"
    return
  fi
  mkdir -p "$MODEL_DIR"
  local auth_args=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer $HF_TOKEN")
  fi
  for f in "${REQUIRED_FILES[@]}"; do
    local dst="$MODEL_DIR/$f"
    if [[ -f "$dst" ]]; then
      continue
    fi
    local url="https://huggingface.co/$MODEL_REPO/resolve/main/$f"
    echo "Downloading $f ..."
    if ! curl -L --fail --retry 3 --retry-delay 2 "${auth_args[@]}" -o "$dst" "$url"; then
      echo "Failed to download $f from $MODEL_REPO" >&2
      if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "If this repo is private/gated, set HF_TOKEN and rerun." >&2
      fi
      exit 1
    fi
  done
}

if [[ ! -f "$WAV_PATH" ]]; then
  echo "WAV file not found: $WAV_PATH" >&2
  exit 1
fi

download_if_missing

swift run --package-path "$PROJECT_DIR/tools/qwen-bench" -c release qwen-bench \
  "$MODEL_DIR" "$WAV_PATH"
