import Foundation
import Observation
import AVFoundation

@MainActor
@Observable
final class TranscriptionViewModel {
    let whisperService: WhisperService

    var showError: Bool = false
    var errorMessage: String = ""
    var showPermissionDenied: Bool = false

    var isRecording: Bool { whisperService.isRecording }
    var confirmedText: String { whisperService.confirmedText }
    var hypothesisText: String { whisperService.hypothesisText }
    var bufferEnergy: [Float] { whisperService.bufferEnergy }
    var bufferSeconds: Double { whisperService.bufferSeconds }
    var tokensPerSecond: Double { whisperService.tokensPerSecond }
    var cpuPercent: Double { whisperService.cpuPercent }
    var memoryMB: Double { whisperService.memoryMB }
    var selectedModel: ModelInfo { whisperService.selectedModel }
    var fullText: String { whisperService.fullTranscriptionText }
    var sessionState: SessionState { whisperService.sessionState }

    /// True when the engine has a mid-session error to surface.
    var hasEngineError: Bool { whisperService.lastError != nil }

    /// True when the session was interrupted (phone call, etc.)
    var isInterrupted: Bool { whisperService.sessionState == .interrupted }

    init(whisperService: WhisperService) {
        self.whisperService = whisperService
    }

    func toggleRecording() async {
        if isRecording || isInterrupted {
            stopRecording()
        } else {
            await startRecording()
        }
    }

    func startRecording() async {
        do {
            showPermissionDenied = false
            try await whisperService.startRecording()
        } catch let error as AppError {
            if case .microphonePermissionDenied = error {
                showPermissionDenied = true
            }
            showError = true
            errorMessage = error.localizedDescription
        } catch {
            showError = true
            errorMessage = error.localizedDescription
        }
    }

    func stopRecording() {
        whisperService.stopRecording()
    }

    /// Surface any engine error via the shared error alert.
    func surfaceEngineError() {
        if let error = whisperService.lastError {
            showError = true
            errorMessage = error.localizedDescription
            whisperService.clearLastError()
        }
    }

    func openSettings() {
        PermissionManager.openAppSettings()
    }

    var isTranscribingFile: Bool { whisperService.isTranscribingFile }

    func transcribeFile(_ url: URL) {
        whisperService.transcribeFile(url)
    }

    func clearTranscription() {
        showPermissionDenied = false
        whisperService.clearTranscription()
    }

    func transcribeTestFile(_ path: String) {
        whisperService.transcribeTestFile(path)
    }
}
