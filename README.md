# VoicePing iOS Offline Transcribe

Offline-first iOS transcription app focused on local speech recognition.
All inference runs on-device after model download.

## Current Scope (Code-Accurate)

- Live transcription with confirmed text plus rolling hypothesis.
- Audio source switching:
- `Voice` (microphone)
- `System` (ReplayKit Broadcast Upload Extension + shared ring buffer)
- In-app model download/load/switch.
- Runtime stats while recording (`CPU`, `RAM`, `tok/s`, elapsed audio).
- Recording controls and test-audio transcription (`test_speech.wav`).
- Settings toggles for `Voice Activity Detection` and `timestamps`.
- Apple Speech built-in engine support.
- No cloud ASR dependency in runtime path.

Note: translation APIs exist in the service layer, but this repo build is transcription-focused and uses a no-op translation implementation.

## Supported Models

Defined in `OfflineTranscription/Models/ModelInfo.swift`.

| Model ID | Display Name | Engine | Languages |
|---|---|---|---|
| `sensevoice-small` | SenseVoice Small | sherpa-onnx offline | `zh/en/ja/ko/yue` |
| `whisper-tiny` | Whisper Tiny | WhisperKit (CoreML) | `99 languages` |
| `whisper-base` | Whisper Base | WhisperKit (CoreML) | `English` |
| `whisper-small` | Whisper Small | WhisperKit (CoreML) | `99 languages` |
| `whisper-large-v3-turbo` | Whisper Large V3 Turbo | WhisperKit (CoreML) | `99 languages` |
| `whisper-large-v3-turbo-compressed` | Whisper Large V3 Turbo (Compressed) | WhisperKit (CoreML) | `99 languages` |
| `moonshine-tiny` | Moonshine Tiny | sherpa-onnx offline | `English` |
| `moonshine-base` | Moonshine Base | sherpa-onnx offline | `English` |
| `zipformer-20m` | Zipformer Streaming | sherpa-onnx streaming | `English` |
| `omnilingual-300m` | Omnilingual 300M | sherpa-onnx offline | `1,600+ languages` |
| `parakeet-tdt-v3` | Parakeet TDT 0.6B | FluidAudio (CoreML) | `25 European languages` |
| `apple-speech` | Apple Speech | SFSpeechRecognizer | `50+ languages` |

`parakeet-tdt-v3` is filtered at runtime when device capability checks fail.

## Architecture

- Orchestrator: `OfflineTranscription/Services/WhisperService.swift`
- Engines:
- `WhisperKitEngine`
- `SherpaOnnxOfflineEngine`
- `SherpaOnnxStreamingEngine`
- `FluidAudioEngine`
- `AppleSpeechEngine`
- Audio capture: `OfflineTranscription/Services/AudioRecorder.swift`
- System capture bridge:
- `BroadcastUploadExtension/SampleHandler.swift`
- `OfflineTranscription/Services/SystemAudioSource.swift`
- UI: `OfflineTranscription/Views/TranscriptionView.swift`, `OfflineTranscription/Views/ModelSetupView.swift`

## Requirements

- macOS
- Xcode 15+
- iOS 17+
- `xcodegen`

## Setup

```bash
git clone <repo-url>
cd ios-offline-transcribe
scripts/setup-ios-deps.sh
scripts/generate-ios-project.sh
open VoicePingIOSOfflineTranscribe.xcodeproj
```

For local signing overrides:

```bash
cp project.local.yml.example project.local.yml
scripts/generate-ios-project.sh
```

## Build

```bash
xcodebuild -project VoicePingIOSOfflineTranscribe.xcodeproj \
  -scheme OfflineTranscription \
  -destination 'generic/platform=iOS Simulator' build
```

## Tests and Automation

```bash
scripts/ci-ios-unit-test.sh
scripts/ios-e2e-test.sh
scripts/ios-ui-flow-tests.sh
```

## Privacy

- Audio is processed locally on device.
- Network is used for model downloads only.
- No cloud transcription service is required for runtime inference.

## License

Apache License 2.0. See `LICENSE`.
