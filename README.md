<p align="center">
  <img src="OfflineTranscription/Resources/Assets.xcassets/AppIcon.appiconset/AppIcon.png" width="120" alt="App Icon"/>
</p>

# VoicePing iOS Offline Transcribe

[![iOS Build](https://github.com/voiceping-ai/ios-offline-transcribe/actions/workflows/ios-build.yml/badge.svg)](https://github.com/voiceping-ai/ios-offline-transcribe/actions/workflows/ios-build.yml)
[![iOS](https://img.shields.io/badge/iOS-17%2B-black)](#requirements)
[![Swift](https://img.shields.io/badge/Swift-5.9-orange)](#tech-stack)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#license)

A multi-engine iOS speech-to-text app that runs fully on-device.
No cloud transcription, no API keys, and no network dependency after model download.

Combines WhisperKit, sherpa-onnx, and FluidAudio into one production-oriented app with model switching, live streaming, telemetry, and E2E coverage.

> **Related repos:**
> [Android Transcription](https://github.com/voiceping-ai/android-offline-transcribe) ·
> [iOS + Android Translation](https://github.com/voiceping-ai/ios-android-offline-speech-translation)

## Why This Project

- Fully offline transcription pipeline from microphone capture to export
- Multiple ASR engines in one app with side-by-side practical benchmarks
- Real-time streaming path (Zipformer) plus high-accuracy offline paths (Whisper, Parakeet)
- Built-in operational safeguards: VAD gating, storage checks, interruption handling, resource telemetry
- Designed for real-device validation, not simulator-only demos

## Benchmark Snapshot (Real iPad, 30s Fixture)

Baseline: `artifacts/e2e/ios/final-verify-20260209-151128`

**11/11 tested, 9 PASS, 2 FAIL, 0 TIMEOUT**

| Model | tok/s | Time | Verdict | Notes |
|---|---:|---:|---|---|
| Moonshine Tiny | 76.5 | 0.8s | PASS | Fastest overall |
| Zipformer Streaming | 75.2 | 0.7s | PASS | Best real-time responsiveness |
| Moonshine Base | 40.6 | 1.4s | PASS | Strong speed/quality balance |
| SenseVoice Small | 16.2 | 3.6s | PASS | Best multilingual punctuation balance |
| Whisper Tiny | 10.6 | 5.5s | PASS | Lightweight Whisper option |
| Parakeet TDT 0.6B | 9.0 | 6.4s | PASS | High-quality punctuation and casing |
| Whisper Base | 7.9 | 26.3s | FAIL | Quality regression (repetitive/hallucinated text) |
| Whisper Small | 0.9 | 65.3s | PASS | Best stable Whisper quality |
| Whisper Large V3 Turbo | 0.24 | 241.8s | PASS | High quality, high latency |
| Whisper Large V3 Turbo (Compressed) | 0.13 | 428.8s | PASS | Slowest passing model |
| Omnilingual 300M | 0.0 | 97.6s | FAIL | Empty or non-English output on English fixture |

Token speed is computed from transcript token count over `duration_ms` in E2E output and is intended for relative comparison.

### Recommended Models by Goal

- **Lowest latency:** `moonshine-tiny`, `zipformer-20m`
- **Fast with better readability:** `moonshine-base`, `sensevoice-small`
- **Best Whisper quality:** `whisper-small`
- **Highest quality punctuation/casing:** `parakeet-tdt-v3`

## Supported Models

| Model | Engine | Size | Params | Languages |
|---|---|---:|---:|---|
| Whisper Tiny | WhisperKit (CoreML) | ~80 MB | 39M | 99 languages |
| Whisper Base | WhisperKit (CoreML) | ~150 MB | 74M | 99 languages |
| Whisper Small | WhisperKit (CoreML) | ~500 MB | 244M | 99 languages |
| Whisper Large V3 Turbo | WhisperKit (CoreML) | ~600 MB | 809M | 99 languages |
| Whisper Large V3 Turbo (Compressed) | WhisperKit (CoreML) | ~1 GB | 809M | 99 languages |
| Moonshine Tiny | sherpa-onnx offline | ~125 MB | 27M | English |
| Moonshine Base | sherpa-onnx offline | ~280 MB | 61M | English |
| SenseVoice Small | sherpa-onnx offline | ~240 MB | 234M | zh/en/ja/ko/yue |
| Omnilingual 300M | sherpa-onnx offline | ~365 MB | 300M | 1,600+ languages |
| Zipformer Streaming | sherpa-onnx streaming | ~46 MB | 20M | English |
| Parakeet TDT 0.6B | FluidAudio (CoreML) | ~600 MB | 600M | 25 European languages |

## Features

### Transcription
- Live microphone transcription with rolling hypothesis and confirmed text
- Streaming and offline decode paths under a unified engine protocol
- Model download, load, and runtime switching in-app

### Reliability
- VAD and RMS gating to reduce silence hallucinations
- Audio session interruption and route-change handling
- Chunked decoding and adaptive scheduling for real-device stability

### UX and Export
- Timestamp support
- Session audio save (`audio.wav`) plus transcript export (`ZIP`)
- Waveform playback with seek and progress visualization
- CPU, memory, and throughput telemetry while recording

## Architecture

```
WhisperService (orchestrator)
 ├── ASREngine protocol
 │    ├── WhisperKitEngine
 │    ├── SherpaOnnxOfflineEngine
 │    ├── SherpaOnnxStreamingEngine
 │    └── FluidAudioEngine
 ├── AudioRecorder (AVAudioEngine)
 ├── SessionFileManager + WAVWriter + ZIPExporter
 └── SystemMetrics (CPU/memory telemetry)

UI: SwiftUI views + view models
    TranscriptionView, ModelSetupView, WaveformPlaybackView
Persistence: SwiftData (TranscriptionRecord)
```

## Quick Start

### Requirements

- macOS
- Xcode 15+
- iOS 17+ device or simulator
- `xcodegen` (`brew install xcodegen`)

### Setup

```bash
git clone <repo-url>
cd ios-offline-transcribe
./scripts/generate-ios-project.sh
open VoicePingIOSOfflineTranscribe.xcodeproj
```

For physical iPhone/iPad builds, add local signing overrides (kept out of git):

```bash
cp project.local.yml.example project.local.yml
# Edit with your DEVELOPMENT_TEAM and PRODUCT_BUNDLE_IDENTIFIER
./scripts/generate-ios-project.sh
```

### Build

```bash
# Simulator
xcodebuild -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'generic/platform=iOS Simulator' build

# Physical device (requires project.local.yml)
xcodebuild -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'platform=iOS,id=<device-udid>' \
  -allowProvisioningUpdates build
```

### Tests

```bash
# Unit tests
xcodebuild test -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
  -only-testing:OfflineTranscriptionTests
```

## E2E Validation

### Run E2E with evidence output

```bash
IOS_DEVICE_ID=<your-device-udid> \
EVIDENCE_DIR=artifacts/e2e/ios/verify-$(date +%Y%m%d-%H%M%S) \
scripts/ios-e2e-test.sh whisper-tiny whisper-small moonshine-base
```

### Validate artifacts

Each model folder contains `01_model_loading.png`, `02_model_loaded.png`, `03_inference_result.png`, and `result.json`.

```bash
# Count screenshots
find artifacts/e2e/ios/verify-* -maxdepth 2 -name '*.png' | wc -l

# Inspect results
python3 - <<'PY'
import json,glob
for p in sorted(glob.glob("artifacts/e2e/ios/verify-*/**/result.json", recursive=True)):
    r=json.load(open(p))
    print(p, "PASS" if r.get("pass") else "FAIL", str(r.get("duration_ms",0))+"ms", (r.get("transcript","")[:80]))
PY
```

### UI flow validation

```bash
EVIDENCE_DIR=artifacts/ui-flow-tests/ios/verify-$(date +%Y%m%d-%H%M%S) \
scripts/ios-ui-flow-tests.sh
```

## Known Issues

- `whisper-base`: quality regression on iPad E2E (repetitive/hallucinated output)
- `omnilingual-300m`: fails English fixture validation (empty or non-English output)

These are inference-quality issues, not app crashes.

## Tech Stack

- Swift 5.9, SwiftUI, SwiftData
- WhisperKit (CoreML)
- sherpa-onnx (ONNX Runtime, local SPM package)
- FluidAudio (CoreML, Parakeet)
- swift-transformers (HuggingFace model fetch)

## Privacy

- Audio and transcripts are processed on-device
- Network is used only for model download
- No cloud transcription or analytics dependency in the ASR path

## License

Apache License 2.0. See `LICENSE`.

Model weights are downloaded at runtime and keep their own licenses. See `NOTICE`.

## Creator

Created by **Akinori Nakajima** ([atyenoria](https://github.com/atyenoria)).
