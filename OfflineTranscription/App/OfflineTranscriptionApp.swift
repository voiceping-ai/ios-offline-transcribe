import SwiftUI
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

enum AutoTestAudioPath {
    static func resolve() -> String {
        let env = ProcessInfo.processInfo.environment["EVAL_WAV_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let env, !env.isEmpty {
            return env
        }

        // Prefer bundle resource — always readable even when sandboxed (macOS).
        // /tmp paths may pass fileExists but fail AVAudioFile read under sandbox.
        if let bundled = Bundle.main.path(forResource: "test_speech", ofType: "wav") {
            return bundled
        }

        let fm = FileManager.default
        for p in ["/private/tmp/test_speech.wav", "/tmp/test_speech.wav"] {
            if fm.fileExists(atPath: p) {
                return p
            }
        }

        return "/tmp/test_speech.wav"
    }
}

#if os(macOS)
/// Avoid macOS state-restoration crash prompts during automation runs.
/// AppKit may show a blocking modal ("Ignore persistent state?") after a crash.
final class OfflineTranscriptionAppDelegate: NSObject, NSApplicationDelegate {
    private var isAutoTest: Bool {
        ProcessInfo.processInfo.arguments.contains("--auto-test")
    }

    func application(_ application: NSApplication, shouldSaveApplicationState coder: NSCoder) -> Bool {
        !isAutoTest
    }

    func application(_ application: NSApplication, shouldRestoreApplicationState coder: NSCoder) -> Bool {
        !isAutoTest
    }
}
#endif

@main
struct OfflineTranscriptionApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(OfflineTranscriptionAppDelegate.self) private var appDelegate
    #endif

    let whisperService: WhisperService

    init() {
        #if DEBUG
        // Debugging aid for automation: verify which arguments the app actually sees.
        // When E2E automation appears to "do nothing", this file is the first thing to check.
        let initArgs = ProcessInfo.processInfo.arguments
        let initDump = initArgs.joined(separator: "\n") + "\n"
        if let data = initDump.data(using: .utf8) {
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("args_init.txt")
            try? data.write(to: tmp, options: .atomic)
            print("[ArgsInit] wrote: \(tmp.path)")
        }
        print("[ArgsInit] argv: \(initArgs)")
        #endif

        // Clear persisted state BEFORE WhisperService.init() reads UserDefaults
        let args = ProcessInfo.processInfo.arguments
        if args.contains("--reset-state") || args.contains("--auto-test") {
            UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
            UserDefaults.standard.removeObject(forKey: "selectedModelCardId")
            UserDefaults.standard.removeObject(forKey: "selectedInferenceBackend")
        }
        // --backend <value>: pre-set the inference backend before WhisperService reads it
        if let idx = args.firstIndex(of: "--backend"), idx + 1 < args.count {
            let backendArg = args[idx + 1]
            if InferenceBackend(rawValue: backendArg) != nil {
                UserDefaults.standard.set(backendArg, forKey: "selectedInferenceBackend")
            }
        }
        whisperService = WhisperService()

        #if os(macOS)
        // macOS automation can launch with 0 windows (previous state had no windows),
        // which prevents SwiftUI view lifecycle hooks like `.task` from running.
        // Run the E2E driver from App.init so benchmarking is not coupled to UI state.
        let service = whisperService
        Task { @MainActor in
            await Self.runAutoTestIfNeeded(service: service)
        }
        #endif
    }

    #if os(macOS)
    @MainActor
    private static var autoTestStarted: Bool = false

    @MainActor
    private static func runAutoTestIfNeeded(service: WhisperService) async {
        let args = ProcessInfo.processInfo.arguments
        guard args.contains("--auto-test") else { return }
        guard !autoTestStarted else { return }
        autoTestStarted = true

        let modelId: String? = {
            guard let idx = args.firstIndex(of: "--model-id"), idx + 1 < args.count else { return nil }
            return args[idx + 1]
        }()
        let backendOverride: InferenceBackend? = {
            guard let idx = args.firstIndex(of: "--backend"), idx + 1 < args.count else { return nil }
            return InferenceBackend(rawValue: args[idx + 1])
        }()

        func loadTimeoutSeconds(for modelId: String) -> TimeInterval {
            if modelId.contains("large") || modelId.contains("omnilingual") { return 900 }
            if modelId.contains("qwen3-asr") { return 900 }
            if modelId.contains("parakeet") { return 600 }
            return 300
        }

        func transcribeTimeoutSeconds(for modelId: String) -> TimeInterval {
            if modelId.contains("large") || modelId.contains("omnilingual") { return 1800 }
            if modelId.contains("qwen3-asr") { return 900 }
            if modelId.contains("parakeet") { return 900 }
            return 600
        }

        let selectedModelId = modelId ?? service.selectedModel.id
        if let modelId, let model = ModelInfo.availableModels.first(where: { $0.id == modelId }) {
            // Bound the entire model switch (download + session creation) so the process can't hang forever.
            let timeout = loadTimeoutSeconds(for: modelId)
            let didLoad: Bool = await withTaskGroup(of: Bool.self) { group in
                group.addTask { @MainActor in
                    await service.switchModel(to: model, backendOverride: backendOverride)
                    return service.modelState == .loaded
                }
                group.addTask {
                    try? await Task.sleep(for: .seconds(timeout))
                    return false
                }
                let result = await group.next() ?? false
                group.cancelAll()
                return result
            }
            if !didLoad {
                service.writeE2EFailure(reason: "model switch timed out after \(Int(timeout))s for \(modelId)")
                NSApplication.shared.terminate(nil)
                return
            }
        } else {
            // No explicit model id; best-effort load of saved selection.
            let timeout = loadTimeoutSeconds(for: selectedModelId)
            let didLoad: Bool = await withTaskGroup(of: Bool.self) { group in
                group.addTask { @MainActor in
                    if let backend = backendOverride {
                        service.setSelectedInferenceBackend(backend)
                    }
                    await service.loadModelIfAvailable()
                    return service.modelState == .loaded
                }
                group.addTask {
                    try? await Task.sleep(for: .seconds(timeout))
                    return false
                }
                let result = await group.next() ?? false
                group.cancelAll()
                return result
            }
            if !didLoad {
                service.writeE2EFailure(reason: "model load timed out after \(Int(timeout))s for \(selectedModelId)")
                NSApplication.shared.terminate(nil)
                return
            }
        }

        // Trigger file transcription on the service directly (no dependency on a visible window).
        service.transcribeTestFile(AutoTestAudioPath.resolve())

        let deadline = Date().addingTimeInterval(transcribeTimeoutSeconds(for: selectedModelId))
        while service.e2eOverlayPayload.isEmpty && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(200))
        }
        if service.e2eOverlayPayload.isEmpty {
            service.writeE2EFailure(reason: "transcription timed out for \(selectedModelId)")
        }

        // Ensure the process exits so automation doesn't leave stray instances.
        NSApplication.shared.terminate(nil)
    }
    #endif

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(whisperService)
                #if os(macOS)
                .frame(minWidth: 500, minHeight: 600)
                #endif
        }
        #if os(macOS)
        Settings {
            MacSettingsPanel()
                .environment(whisperService)
                .frame(minWidth: 520, minHeight: 700)
        }
        #endif
    }
}

struct RootView: View {
    @Environment(WhisperService.self) private var whisperService

    init() {
        #if DEBUG
        print("[RootView] init")
        #endif
    }

    private static var autoTestModelId: String? {
        let args = ProcessInfo.processInfo.arguments
        guard let idx = args.firstIndex(of: "--model-id"), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }

    private static var autoTestBackend: InferenceBackend? {
        let args = ProcessInfo.processInfo.arguments
        guard let idx = args.firstIndex(of: "--backend"), idx + 1 < args.count else { return nil }
        return InferenceBackend(rawValue: args[idx + 1])
    }

    private static var isAutoTestRun: Bool {
        ProcessInfo.processInfo.arguments.contains("--auto-test")
    }

    private static func autoTestLoadTimeoutSeconds(for modelId: String) -> TimeInterval {
        if modelId.contains("large") || modelId.contains("omnilingual") { return 480 }
        if modelId == "whisper-small" || modelId == "parakeet-tdt-v3" { return 300 }
        if modelId == "whisper-base" { return 240 }
        return 120
    }

    private static func autoTestTranscribeTimeoutSeconds(for modelId: String) -> TimeInterval {
        if modelId.contains("large") || modelId.contains("omnilingual") { return 900 }
        if modelId.contains("qwen3-asr") { return 600 }
        return 300
    }

    var body: some View {
        ZStack {
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

            // Surface E2E result overlay even when model fails to load (e.g. Apple Speech
            // with Dictation disabled). Without this, the XCUITest can't see the result
            // because TranscriptionView is never shown.
            #if DEBUG
            if Self.isAutoTestRun, !whisperService.e2eOverlayPayload.isEmpty,
               whisperService.modelState != .loaded {
                VStack {
                    HStack {
                        Spacer()
                        VStack(alignment: .trailing, spacing: 4) {
                            Text("E2E Result Ready")
                                .font(.caption2)
                                .fontWeight(.semibold)
                            Text(whisperService.e2eOverlayPayload)
                                .font(.caption2.monospaced())
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .accessibilityIdentifier("e2e_overlay_payload")
                        }
                        .padding(8)
                        .background(.black.opacity(0.72))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .padding(8)
                        .accessibilityIdentifier("e2e_overlay")
                        .accessibilityLabel("e2e_overlay")
                        .accessibilityValue(whisperService.e2eOverlayPayload)
                    }
                    Spacer()
                }
            }
            #endif
        }
        .task {
            #if os(macOS)
            // macOS auto-test is orchestrated from App.init to avoid coupling to window lifecycle.
            if Self.isAutoTestRun { return }
            #endif

            #if DEBUG
            do {
                let marker = FileManager.default.temporaryDirectory.appendingPathComponent("root_task_started.txt")
                try "started\n".data(using: .utf8)?.write(to: marker, options: .atomic)
                print("[RootTask] started: \(marker.path)")
            } catch {
                print("[RootTask] failed to write marker: \(error)")
            }
            #endif

            let resetState = ProcessInfo.processInfo.arguments.contains("--reset-state")
            configureIdleTimerForUITests()

            // --reset-state: clear UserDefaults for clean UI test runs
            if resetState {
                UserDefaults.standard.removeObject(forKey: "selectedModelVariant")
                UserDefaults.standard.removeObject(forKey: "selectedModelCardId")
                UserDefaults.standard.removeObject(forKey: "selectedInferenceBackend")
            }

            // Auto-load only for E2E / UI testing (--model-id / --auto-test).
            // Normal launch always starts on the model selection screen.
            if let modelId = Self.autoTestModelId,
               let model = ModelInfo.availableModels.first(where: { $0.id == modelId }) {
                await whisperService.switchModel(to: model, backendOverride: Self.autoTestBackend)
            } else if Self.isAutoTestRun {
                if let backend = Self.autoTestBackend {
                    whisperService.setSelectedInferenceBackend(backend)
                }
                await whisperService.loadModelIfAvailable()
            }

            // Auto-test runner: avoid coupling E2E to UI navigation/state restoration.
            if Self.isAutoTestRun {
                let args = ProcessInfo.processInfo.arguments
                if let argsDump = (args.joined(separator: "\n")).data(using: .utf8) {
                    // Debugging aid: locate args in sandbox temp when /tmp is unavailable.
                    let url = FileManager.default.temporaryDirectory.appendingPathComponent("args_runtime.txt")
                    try? argsDump.write(to: url, options: .atomic)
                }

                let modelId = Self.autoTestModelId ?? whisperService.selectedModel.id
                let loadDeadline = Date().addingTimeInterval(Self.autoTestLoadTimeoutSeconds(for: modelId))
                while whisperService.modelState != .loaded && Date() < loadDeadline {
                    if whisperService.modelState == .error || whisperService.modelState == .unloaded {
                        break
                    }
                    try? await Task.sleep(for: .milliseconds(200))
                }
                guard whisperService.modelState == .loaded else {
                    whisperService.writeE2EFailure(
                        reason: "model load failed/timed out for \(modelId) (state=\(whisperService.modelState.rawValue))"
                    )
                    #if os(macOS)
                    NSApplication.shared.terminate(nil)
                    #endif
                    return
                }

                // Trigger file transcription on the service directly (no dependency on TranscriptionView.task).
                whisperService.transcribeTestFile(AutoTestAudioPath.resolve())

                let transcribeDeadline = Date().addingTimeInterval(Self.autoTestTranscribeTimeoutSeconds(for: modelId))
                while whisperService.e2eOverlayPayload.isEmpty && Date() < transcribeDeadline {
                    try? await Task.sleep(for: .milliseconds(200))
                }
                if whisperService.e2eOverlayPayload.isEmpty {
                    whisperService.writeE2EFailure(reason: "transcription timed out for \(modelId)")
                }

                #if os(macOS)
                // Ensure the process exits so automation doesn't leave stray instances.
                NSApplication.shared.terminate(nil)
                #endif
            }
        }
    }

    private func configureIdleTimerForUITests() {
        #if os(iOS)
        guard Self.isAutoTestRun else { return }
        UIApplication.shared.isIdleTimerDisabled = true
        #endif
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
