import SwiftUI

@main
struct OfflineTranscriptionApp: App {
    let whisperService: WhisperService

    init() {
        // Clear persisted state BEFORE WhisperService.init() reads UserDefaults
        if ProcessInfo.processInfo.arguments.contains("--reset-state") {
            UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
        }
        whisperService = WhisperService()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(whisperService)
        }
    }
}

struct RootView: View {
    @Environment(WhisperService.self) private var whisperService

    private static var autoTestModelId: String? {
        let args = ProcessInfo.processInfo.arguments
        guard let idx = args.firstIndex(of: "--model-id"), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }

    var body: some View {
        Group {
            switch whisperService.modelState {
            case .loaded:
                TranscriptionRootView()
            case .loading, .downloading, .downloaded:
                VStack(spacing: 8) {
                    ProgressView(whisperService.modelState == .downloading
                        ? "Downloading model..." : "Loading model...")
                    if whisperService.modelState == .downloading {
                        Text("\(Int(whisperService.downloadProgress * 100))%")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if !whisperService.loadingStatusMessage.isEmpty {
                        Text(whisperService.loadingStatusMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    AppVersionLabel()
                }
            default:
                // Only keep TranscriptionRootView if a model was previously loaded
                // (e.g. during model switch). Prevents navigating to transcription
                // when download/load failed and no model is ready.
                if whisperService.activeEngine?.modelState == .loaded {
                    TranscriptionRootView()
                } else {
                    ModelSetupView()
                }
            }
        }
        .task {
            let resetState = ProcessInfo.processInfo.arguments.contains("--reset-state")

            // --reset-state: clear UserDefaults for clean UI test runs
            if resetState {
                UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
            }

            // Auto-load only for E2E / UI testing (--model-id / --auto-test).
            // Normal launch always starts on the model selection screen.
            if let modelId = Self.autoTestModelId,
               let model = ModelInfo.availableModels.first(where: { $0.id == modelId }) {
                await whisperService.switchModel(to: model)
            } else if ProcessInfo.processInfo.arguments.contains("--auto-test") {
                await whisperService.loadModelIfAvailable()
            }
        }
    }
}

struct TranscriptionRootView: View {
    var body: some View {
        NavigationStack {
            TranscriptionView()
        }
        .accessibilityIdentifier("main_tab_view")
    }
}
