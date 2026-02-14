#if canImport(Qwen3ASR)
import Foundation
import Qwen3ASR

@MainActor
final class MLXEngine: ASREngine {
    var isStreaming: Bool { false }
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0
    private(set) var loadingStatusMessage: String = ""

    var audioSamples: [Float] { recorder.audioSamples }
    var relativeEnergy: [Float] { recorder.relativeEnergy }

    private let recorder = AudioRecorder()
    private var model: Qwen3ASRModel?
    private var segmentIdCounter: Int = 0

    /// HuggingFace model ID for the 4-bit quantized 0.6B model (~400 MB).
    private static let defaultModelId = "mlx-community/Qwen3-ASR-0.6B-4bit"

    func setupModel(_ modelInfo: ModelInfo) async throws {
        guard BackendCapabilities.isBackendSupported(.mlx) else {
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "MLXEngine", code: -20,
                userInfo: [NSLocalizedDescriptionKey: "MLX backend is not supported on this platform"]
            ))
        }

        modelState = .downloading
        loadingStatusMessage = "Downloading MLX model..."

        do {
            let loadedModel = try await Qwen3ASRModel.fromPretrained(
                modelId: Self.defaultModelId,
                progressHandler: { [weak self] progress, status in
                    Task { @MainActor in
                        self?.downloadProgress = progress
                        self?.loadingStatusMessage = status
                    }
                }
            )

            self.model = loadedModel
            modelState = .loaded
            downloadProgress = 1
            loadingStatusMessage = ""
        } catch {
            modelState = .error
            loadingStatusMessage = ""
            throw AppError.modelLoadFailed(underlying: error)
        }
    }

    func loadModel(_ modelInfo: ModelInfo) async throws {
        guard isModelDownloaded(modelInfo) else {
            modelState = .unloaded
            return
        }

        modelState = .loading
        loadingStatusMessage = "Loading MLX model..."

        do {
            let loadedModel = try await Qwen3ASRModel.fromPretrained(
                modelId: Self.defaultModelId,
                progressHandler: { [weak self] progress, status in
                    Task { @MainActor in
                        self?.downloadProgress = progress
                        self?.loadingStatusMessage = status
                    }
                }
            )

            self.model = loadedModel
            modelState = .loaded
            downloadProgress = 1
            loadingStatusMessage = ""
        } catch {
            modelState = .error
            loadingStatusMessage = ""
            throw AppError.modelLoadFailed(underlying: error)
        }
    }

    func isModelDownloaded(_ modelInfo: ModelInfo) -> Bool {
        let sanitizedId = Self.defaultModelId.replacingOccurrences(of: "/", with: "_")
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("qwen3-speech")
            .appendingPathComponent(sanitizedId)

        guard FileManager.default.fileExists(atPath: cacheDir.path) else { return false }
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: cacheDir.path)) ?? []
        return contents.contains(where: { $0.hasSuffix(".safetensors") })
    }

    func unloadModel() async {
        stopRecording()
        model = nil
        modelState = .unloaded
        downloadProgress = 0
        loadingStatusMessage = ""
    }

    func startRecording(captureMode: AudioCaptureMode) async throws {
        try await recorder.startRecording(captureMode: captureMode)
    }

    func stopRecording() {
        recorder.stopRecording()
    }

    func transcribe(audioArray: [Float], options: ASRTranscriptionOptions) async throws -> ASRResult {
        guard let model else { throw AppError.modelNotReady }

        let text = model.transcribe(
            audio: audioArray,
            sampleRate: 16000,
            language: options.language
        )

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ASRResult(text: "", segments: [], language: options.language)
        }

        let duration = Float(audioArray.count) / 16000
        let segment = ASRSegment(
            id: segmentIdCounter,
            text: " " + trimmed,
            start: 0,
            end: duration
        )
        segmentIdCounter += 1

        return ASRResult(text: trimmed, segments: [segment], language: options.language)
    }
}

#else
// iOS stub — MLX engine is gated by BackendCapabilities and never instantiated on iOS.
import Foundation

@MainActor
final class MLXEngine: ASREngine {
    var isStreaming: Bool { false }
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0
    private(set) var loadingStatusMessage: String = ""

    var audioSamples: [Float] { [] }
    var relativeEnergy: [Float] { [] }

    func setupModel(_ model: ModelInfo) async throws {
        throw AppError.modelLoadFailed(underlying: NSError(
            domain: "MLXEngine", code: -20,
            userInfo: [NSLocalizedDescriptionKey: "MLX backend is not available on this platform"]
        ))
    }

    func loadModel(_ model: ModelInfo) async throws {
        throw AppError.modelLoadFailed(underlying: NSError(
            domain: "MLXEngine", code: -20,
            userInfo: [NSLocalizedDescriptionKey: "MLX backend is not available on this platform"]
        ))
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool { false }

    func unloadModel() async {
        modelState = .unloaded
    }

    func startRecording(captureMode: AudioCaptureMode) async throws {
        throw AppError.modelLoadFailed(underlying: NSError(
            domain: "MLXEngine", code: -20,
            userInfo: [NSLocalizedDescriptionKey: "MLX backend is not available on this platform"]
        ))
    }

    func stopRecording() {}

    func transcribe(audioArray: [Float], options: ASRTranscriptionOptions) async throws -> ASRResult {
        throw AppError.modelNotReady
    }
}
#endif
