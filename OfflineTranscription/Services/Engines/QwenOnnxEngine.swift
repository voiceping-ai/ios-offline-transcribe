import Foundation
import QwenASRKit

@MainActor
final class QwenOnnxEngine: ASREngine {
    var isStreaming: Bool { false }
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0
    private(set) var loadingStatusMessage: String = ""

    var audioSamples: [Float] { recorder.audioSamples }
    var relativeEnergy: [Float] { recorder.relativeEnergy }

    private let recorder = AudioRecorder()
    private let downloader = ModelDownloader()
    private var onnx: QwenOnnxASR?
    private var segmentIdCounter: Int = 0

    func setupModel(_ model: ModelInfo) async throws {
        guard model.qwenModelConfig != nil else {
            throw AppError.noModelSelected
        }

        let logger = InferenceLogger.shared

        modelState = .downloading
        loadingStatusMessage = "Downloading Qwen ONNX models..."
        downloader.onProgress = { [weak self] value in
            self?.downloadProgress = value
        }

        let modelDir: URL
        do {
            logger.log("[QwenOnnxEngine] downloading model files...")
            modelDir = try await downloader.downloadModel(model)
            logger.log("[QwenOnnxEngine] download done: \(modelDir.path)")
        } catch {
            logger.log("[QwenOnnxEngine] download FAILED: \(error)")
            modelState = .error
            throw AppError.modelDownloadFailed(underlying: error)
        }

        // List files in model directory for verification
        if let files = try? FileManager.default.contentsOfDirectory(atPath: modelDir.path) {
            for f in files.sorted() {
                let attrs = try? FileManager.default.attributesOfItem(atPath: modelDir.appendingPathComponent(f).path)
                let size = (attrs?[.size] as? Int64) ?? 0
                logger.log("[QwenOnnxEngine]   \(f): \(size / 1_000_000) MB")
            }
        }

        modelState = .loading
        loadingStatusMessage = "Loading ONNX Runtime sessions..."
        logger.log("[QwenOnnxEngine] calling QwenOnnxASR(modelDir:)...")

        guard let runtime = QwenOnnxASR(modelDir: modelDir.path) else {
            let onnxError = QwenOnnxASR.lastError
            let logger = InferenceLogger.shared
            logger.log("[QwenOnnxEngine] ONNX load failed: \(onnxError)")
            modelState = .error
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "QwenOnnxEngine",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to initialize ONNX Runtime sessions: \(onnxError)"]
            ))
        }

        onnx = runtime
        modelState = .loaded
        downloadProgress = 1
        loadingStatusMessage = ""
    }

    func loadModel(_ model: ModelInfo) async throws {
        guard model.qwenModelConfig != nil else {
            throw AppError.noModelSelected
        }
        guard downloader.isModelDownloaded(model),
              let modelDir = downloader.modelDirectory(for: model) else {
            modelState = .unloaded
            return
        }

        let logger = InferenceLogger.shared
        modelState = .loading
        loadingStatusMessage = "Loading ONNX Runtime sessions..."
        logger.log("[QwenOnnxEngine] loadModel: \(modelDir.path)")

        guard let runtime = QwenOnnxASR(modelDir: modelDir.path) else {
            let onnxError = QwenOnnxASR.lastError
            let logger = InferenceLogger.shared
            logger.log("[QwenOnnxEngine] ONNX load failed: \(onnxError)")
            modelState = .error
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "QwenOnnxEngine",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to initialize ONNX Runtime sessions: \(onnxError)"]
            ))
        }

        onnx = runtime
        modelState = .loaded
        loadingStatusMessage = ""
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        downloader.isModelDownloaded(model)
    }

    func unloadModel() async {
        stopRecording()
        onnx?.release()
        onnx = nil
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
        guard let onnx else { throw AppError.modelNotReady }

        guard let text = onnx.transcribe(samples: audioArray)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty else {
            return ASRResult(text: "", segments: [], language: options.language)
        }

        let duration = Float(audioArray.count) / 16000
        let segment = ASRSegment(
            id: segmentIdCounter,
            text: " " + text,
            start: 0,
            end: duration
        )
        segmentIdCounter += 1

        return ASRResult(text: text, segments: [segment], language: options.language)
    }
}
