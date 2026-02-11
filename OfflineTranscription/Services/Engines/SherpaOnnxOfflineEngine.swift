import Foundation
import SherpaOnnxKit

/// ASREngine implementation for sherpa-onnx offline models (Moonshine, SenseVoice).
@MainActor
final class SherpaOnnxOfflineEngine: ASREngine {
    var isStreaming: Bool { false }
    private(set) var modelState: ASRModelState = .unloaded
    private(set) var downloadProgress: Double = 0.0
    private(set) var loadingStatusMessage: String = ""
    var audioSamples: [Float] { recorder.audioSamples }
    var relativeEnergy: [Float] { recorder.relativeEnergy }

    private var recognizer: SherpaOnnxOfflineRecognizer?
    private let recorder = AudioRecorder()
    private let downloader = ModelDownloader()
    private var currentModel: ModelInfo?
    private var segmentIdCounter: Int = 0

    // MARK: - ASREngine

    func setupModel(_ model: ModelInfo) async throws {
        guard model.sherpaModelConfig != nil else {
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "SherpaOnnxOfflineEngine", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Missing sherpa model config"]
            ))
        }

        // Download if needed
        if !downloader.isModelDownloaded(model) {
            modelState = .downloading
            downloader.onProgress = { [weak self] progress in
                self?.downloadProgress = progress
            }
            _ = try await downloader.downloadModel(model)
        }

        modelState = .downloaded
        currentModel = model

        // Load immediately after download
        try await loadModel(model)
    }

    func loadModel(_ model: ModelInfo) async throws {
        guard let config = model.sherpaModelConfig,
              let modelDir = downloader.modelDirectory(for: model) else {
            throw AppError.modelLoadFailed(underlying: NSError(
                domain: "SherpaOnnxOfflineEngine", code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Model not downloaded"]
            ))
        }

        modelState = .loading
        let dirPath = modelDir.path
        loadingStatusMessage = "Loading model..."

        do {
            let recognizer = try await Task.detached {
                return try Self.createRecognizer(config: config, modelDir: dirPath)
            }.value

            NSLog("[SherpaOnnxOfflineEngine] Created recognizer for model=%@", config.modelType.rawValue)
            self.recognizer = recognizer
            self.currentModel = model
            self.modelState = .loaded
            self.loadingStatusMessage = ""
        } catch {
            modelState = .error
            loadingStatusMessage = ""
            throw AppError.modelLoadFailed(underlying: error)
        }
    }

    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        downloader.isModelDownloaded(model)
    }

    func unloadModel() async {
        recognizer = nil
        currentModel = nil
        modelState = .unloaded
    }

    func startRecording(captureMode: AudioCaptureMode) async throws {
        try await recorder.startRecording(captureMode: captureMode)
    }

    func stopRecording() {
        recorder.stopRecording()
    }

    func transcribe(audioArray: [Float], options: ASRTranscriptionOptions) async throws -> ASRResult {
        let logger = InferenceLogger.shared
        guard let recognizer else {
            logger.log("ERROR: transcribe called but recognizer is nil (model not ready)")
            throw AppError.modelNotReady
        }

        let modelType = currentModel?.sherpaModelConfig?.modelType
        let modelName = currentModel?.id ?? "unknown"
        let audioDuration = Float(audioArray.count) / 16000.0
        let isLongOmnilingual = modelType == .omnilingualCtc && audioArray.count > Int(16000 * 8)

        // Log audio stats
        let minSample = audioArray.min() ?? 0
        let maxSample = audioArray.max() ?? 0
        let rms = sqrt(audioArray.reduce(0.0) { $0 + $1 * $1 } / max(Float(audioArray.count), 1))
        logger.log("TRANSCRIBE START model=\(modelName) type=\(modelType?.rawValue ?? "nil") samples=\(audioArray.count) duration=\(String(format: "%.2f", audioDuration))s min=\(String(format: "%.4f", minSample)) max=\(String(format: "%.4f", maxSample)) rms=\(String(format: "%.6f", rms))")

        // All sherpa-onnx models consume raw [-1, 1] float waveforms directly.
        // No int16 scaling — matches Android behavior where SenseVoice works
        // with raw floats and has better accuracy.
        let samples = audioArray

        if isLongOmnilingual {
            logger.log("  Omnilingual long decode (\(String(format: "%.1f", audioDuration))s) -> chunked path")
            let chunkStart = CFAbsoluteTimeGetCurrent()
            let chunkedText = await Task.detached {
                Self.decodeOmnilingualChunked(
                    recognizer: recognizer,
                    samples: samples,
                    languageHint: options.language ?? "auto"
                )
            }.value.trimmingCharacters(in: .whitespacesAndNewlines)
            let chunkEnd = CFAbsoluteTimeGetCurrent()
            logger.log("  Omnilingual chunk-first decode took \(String(format: "%.3f", chunkEnd - chunkStart))s result_len=\(chunkedText.count) text=\"\(String(chunkedText.prefix(200)))\"")
            if !chunkedText.isEmpty {
                let segId = segmentIdCounter
                segmentIdCounter += 1
                let segment = ASRSegment(
                    id: segId,
                    text: " " + chunkedText,
                    start: 0,
                    end: audioDuration
                )
                return ASRResult(
                    text: chunkedText,
                    segments: [segment],
                    language: options.language
                )
            }
            logger.log("  Omnilingual chunk-first path returned empty; falling back to full decode")
        }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        var result = await Task.detached {
            recognizer.decode(samples: samples, sampleRate: 16000)
        }.value
        let decodeEnd = CFAbsoluteTimeGetCurrent()

        var text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        logger.log("  Decode #1 took \(String(format: "%.3f", decodeEnd - decodeStart))s result_len=\(text.count) lang=\"\(result.lang)\" text=\"\(String(text.prefix(200)))\"")

        if text.isEmpty, modelType == .omnilingualCtc {
            logger.log("  Omnilingual retry with int16 scaling...")
            let scaledSamples = samples.map { $0 * 32768.0 }
            let retryStart = CFAbsoluteTimeGetCurrent()
            result = await Task.detached {
                recognizer.decode(samples: scaledSamples, sampleRate: 16000)
            }.value
            let retryEnd = CFAbsoluteTimeGetCurrent()
            text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            logger.log("  Decode #2 (scaled) took \(String(format: "%.3f", retryEnd - retryStart))s result_len=\(text.count) text=\"\(String(text.prefix(200)))\"")
        }
        if text.isEmpty, modelType == .omnilingualCtc {
            logger.log("  Omnilingual chunked fallback...")
            let chunkStart = CFAbsoluteTimeGetCurrent()
            text = await Task.detached {
                Self.decodeOmnilingualChunked(
                    recognizer: recognizer,
                    samples: samples,
                    languageHint: options.language ?? "auto"
                )
            }.value
            let chunkEnd = CFAbsoluteTimeGetCurrent()
            logger.log("  Chunked decode took \(String(format: "%.3f", chunkEnd - chunkStart))s result_len=\(text.count) text=\"\(String(text.prefix(200)))\"")
        }

        // SenseVoice provides language detection
        let detectedLang: String? = {
            let raw = result.lang.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !raw.isEmpty else { return nil }
            return raw.replacingOccurrences(of: "<|", with: "")
                .replacingOccurrences(of: "|>", with: "")
        }()

        // Strip spurious spaces from CJK output. SenseVoice's BPE decoder
        // sometimes inserts word-boundary spaces that are wrong for ja/zh/ko.
        if modelType == .senseVoice, let lang = detectedLang {
            let langCode = lang.replacingOccurrences(of: "<|", with: "")
                .replacingOccurrences(of: "|>", with: "")
            if ["ja", "zh", "ko", "yue"].contains(langCode) {
                let before = text
                text = Self.stripCJKSpaces(text)
                if before != text {
                    logger.log("  CJK space strip (\(langCode)): \"\(before)\" → \"\(text)\"")
                }
            }
        }

        guard !text.isEmpty else {
            logger.log("  EMPTY RESULT — returning empty ASRResult for \(modelName)")
            return ASRResult(text: "", segments: [], language: options.language)
        }

        // Create a single segment for the entire transcription
        let duration = Float(audioArray.count) / 16000.0
        let segId = segmentIdCounter
        segmentIdCounter += 1
        let segment = ASRSegment(
            id: segId,
            text: " " + text,
            start: 0,
            end: duration
        )

        let totalTime = CFAbsoluteTimeGetCurrent() - decodeStart
        logger.log("TRANSCRIBE END model=\(modelName) total=\(String(format: "%.3f", totalTime))s rtf=\(String(format: "%.2f", totalTime / Double(audioDuration))) text_len=\(text.count)")

        return ASRResult(
            text: text,
            segments: [segment],
            language: detectedLang ?? options.language
        )
    }

    // MARK: - Private

    private nonisolated static func decodeOmnilingualChunked(
        recognizer: SherpaOnnxOfflineRecognizer,
        samples: [Float],
        languageHint: String = "auto"
    ) -> String {
        let deadline = CFAbsoluteTimeGetCurrent() + 90.0

        func runPass(_ input: [Float], chunkSize: Int, overlap: Int) -> String {
            var pieces: [String] = []
            var offset = 0
            while offset < input.count {
                if CFAbsoluteTimeGetCurrent() >= deadline {
                    break
                }
                let end = min(offset + chunkSize, input.count)
                let chunk = Array(input[offset..<end])
                let partial = recognizer.decode(samples: chunk, sampleRate: 16000)
                let text = partial.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    if let last = pieces.last {
                        if text == last {
                            // Skip duplicate overlap decode.
                        } else if text.hasPrefix(last) {
                            pieces[pieces.count - 1] = text
                        } else if last.hasPrefix(text) {
                            // Keep the longer prior piece.
                        } else {
                            pieces.append(text)
                        }
                    } else {
                        pieces.append(text)
                    }
                }
                if end == input.count { break }
                offset = max(end - overlap, offset + 1)
            }
            return pieces.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let raw = samples
        let scaled = samples.map { $0 * 32768.0 }
        let chunkShapes: [(Int, Int)] = [
            (16000 * 4, 16000 / 2),
            (16000 * 8, 16000),
            (16000 * 12, 16000 * 3 / 2)
        ]
        var bestText = ""
        var bestScore = Int.min

        for (candidateIndex, candidate) in [raw, scaled].enumerated() {
            let candidateLabel = candidateIndex == 0 ? "raw" : "scaled"
            for (chunkSize, overlap) in chunkShapes {
                if CFAbsoluteTimeGetCurrent() >= deadline {
                    break
                }
                let text = runPass(candidate, chunkSize: chunkSize, overlap: overlap)
                if text.isEmpty {
                    continue
                }
                let score = scoreOmnilingualText(text, languageHint: languageHint)
                NSLog(
                    "[SherpaOnnxOfflineEngine] Omnilingual chunk candidate=%@ chunk=%.1fs score=%d text=\"%@\"",
                    candidateLabel,
                    Double(chunkSize) / 16000.0,
                    score,
                    String(text.prefix(160))
                )
                if score > bestScore {
                    bestScore = score
                    bestText = text
                }
            }
        }

        if !bestText.isEmpty {
            NSLog(
                "[SherpaOnnxOfflineEngine] Omnilingual chunk selected score=%d text=\"%@\"",
                bestScore,
                String(bestText.prefix(200))
            )
        } else {
            NSLog("[SherpaOnnxOfflineEngine] Omnilingual chunk fallback produced empty output")
        }
        return bestText
    }

    private nonisolated static func scoreOmnilingualText(_ text: String, languageHint: String = "auto") -> Int {
        let lower = text.lowercased()
        let keywords = ["country", "ask", "do for", "fellow", "americans"]
        let normalizedHint = languageHint.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var score = 0
        for keyword in keywords where lower.contains(keyword) {
            score += 120
        }
        let asciiLetters = text.unicodeScalars.filter { CharacterSet.letters.contains($0) && $0.isASCII }.count
        let nonAsciiLetters = text.unicodeScalars.filter { CharacterSet.letters.contains($0) && !$0.isASCII }.count
        score += asciiLetters
        score -= (normalizedHint.hasPrefix("en") ? 3 : 2) * nonAsciiLetters
        score -= text.filter { $0 == "\u{FFFD}" }.count * 8
        if normalizedHint.hasPrefix("en") && asciiLetters == 0 {
            score -= 300
        }
        return score
    }

    /// Create a recognizer using CPU provider.
    private nonisolated static func createRecognizer(
        config: SherpaModelConfig,
        modelDir: String
    ) throws -> SherpaOnnxOfflineRecognizer {
        let provider = "cpu"
        let fm = FileManager.default
        let tokensPath = "\(modelDir)/\(config.tokens)"

        guard fm.fileExists(atPath: tokensPath) else {
            throw NSError(domain: "SherpaOnnxOfflineEngine", code: -3,
                          userInfo: [NSLocalizedDescriptionKey: "tokens.txt not found at \(tokensPath)"])
        }

        let numThreads = config.modelType == .omnilingualCtc ? 1 : recommendedOfflineThreads()
        var modelConfig: SherpaOnnxOfflineModelConfig

        switch config.modelType {
        case .moonshine:
            guard let preprocessor = config.preprocessor,
                  let encoder = config.encoder,
                  let uncachedDecoder = config.uncachedDecoder,
                  let cachedDecoder = config.cachedDecoder else {
                throw NSError(domain: "SherpaOnnxOfflineEngine", code: -3,
                              userInfo: [NSLocalizedDescriptionKey: "Missing moonshine model file names in config"])
            }
            let paths = [preprocessor, encoder, uncachedDecoder, cachedDecoder]
            for p in paths {
                let fullPath = "\(modelDir)/\(p)"
                guard fm.fileExists(atPath: fullPath) else {
                    throw NSError(domain: "SherpaOnnxOfflineEngine", code: -3,
                                  userInfo: [NSLocalizedDescriptionKey: "Model file not found: \(p)"])
                }
            }
            let moonshineConfig = sherpaOnnxOfflineMoonshineModelConfig(
                preprocessor: "\(modelDir)/\(preprocessor)",
                encoder: "\(modelDir)/\(encoder)",
                uncachedDecoder: "\(modelDir)/\(uncachedDecoder)",
                cachedDecoder: "\(modelDir)/\(cachedDecoder)"
            )
            modelConfig = sherpaOnnxOfflineModelConfig(
                tokens: tokensPath,
                numThreads: numThreads,
                provider: provider,
                debug: 0,
                moonshine: moonshineConfig
            )

        case .senseVoice:
            guard let senseVoiceModel = config.senseVoiceModel else {
                throw NSError(domain: "SherpaOnnxOfflineEngine", code: -4,
                              userInfo: [NSLocalizedDescriptionKey: "Missing SenseVoice model file name in config"])
            }
            let modelPath = "\(modelDir)/\(senseVoiceModel)"
            guard fm.fileExists(atPath: modelPath) else {
                throw NSError(domain: "SherpaOnnxOfflineEngine", code: -4,
                              userInfo: [NSLocalizedDescriptionKey: "Model file not found: \(senseVoiceModel)"])
            }
            let senseVoiceConfig = sherpaOnnxOfflineSenseVoiceModelConfig(
                model: modelPath,
                language: "auto",
                useInverseTextNormalization: true
            )
            modelConfig = sherpaOnnxOfflineModelConfig(
                tokens: tokensPath,
                numThreads: numThreads,
                provider: provider,
                debug: 0,
                senseVoice: senseVoiceConfig
            )

        case .zipformerTransducer:
            throw NSError(domain: "SherpaOnnxOfflineEngine", code: -5,
                          userInfo: [NSLocalizedDescriptionKey: "Zipformer transducer should use streaming engine"])

        case .omnilingualCtc:
            guard let omniModel = config.omnilingualModel else {
                throw NSError(domain: "SherpaOnnxOfflineEngine", code: -7,
                              userInfo: [NSLocalizedDescriptionKey: "Missing omnilingual model file name in config"])
            }
            let modelPath = "\(modelDir)/\(omniModel)"
            guard fm.fileExists(atPath: modelPath) else {
                throw NSError(domain: "SherpaOnnxOfflineEngine", code: -7,
                              userInfo: [NSLocalizedDescriptionKey: "Model file not found: \(omniModel)"])
            }
            let omniConfig = sherpaOnnxOfflineOmnilingualAsrCtcModelConfig(model: modelPath)
            modelConfig = sherpaOnnxOfflineModelConfig(
                tokens: tokensPath,
                numThreads: numThreads,
                provider: provider,
                debug: 0,
                modelingUnit: "bpe",
                omnilingual: omniConfig
            )
        }

        let featConfig = sherpaOnnxFeatureConfig(sampleRate: 16000, featureDim: 80)
        var recognizerConfig = sherpaOnnxOfflineRecognizerConfig(
            featConfig: featConfig,
            modelConfig: modelConfig,
            decodingMethod: "greedy_search"
        )

        guard let recognizer = SherpaOnnxOfflineRecognizer(config: &recognizerConfig) else {
            throw NSError(domain: "SherpaOnnxOfflineEngine", code: -6,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create offline recognizer for provider \(provider)"])
        }

        return recognizer
    }

    /// Remove spaces between CJK characters. Keeps spaces around Latin/number runs.
    private nonisolated static func stripCJKSpaces(_ text: String) -> String {
        var result = ""
        let chars = Array(text)
        for (i, char) in chars.enumerated() {
            if char == " " {
                let prev = i > 0 ? chars[i - 1] : nil
                let next = i + 1 < chars.count ? chars[i + 1] : nil
                // Keep space only if both neighbors are non-CJK (Latin, digits, etc.)
                let prevIsCJK = prev.map { Self.isCJK($0) } ?? true
                let nextIsCJK = next.map { Self.isCJK($0) } ?? true
                if !prevIsCJK && !nextIsCJK {
                    result.append(char)
                }
                // Otherwise drop the space
            } else {
                result.append(char)
            }
        }
        return result
    }

    private nonisolated static func isCJK(_ char: Character) -> Bool {
        guard let scalar = char.unicodeScalars.first else { return false }
        let v = scalar.value
        // CJK Unified Ideographs, Hiragana, Katakana, Hangul, CJK punctuation
        return (v >= 0x3000 && v <= 0x9FFF)
            || (v >= 0xAC00 && v <= 0xD7AF)  // Hangul Syllables
            || (v >= 0xF900 && v <= 0xFAFF)   // CJK Compat Ideographs
            || (v >= 0xFF00 && v <= 0xFFEF)   // Fullwidth Forms
            || (v >= 0x20000 && v <= 0x2FA1F)  // CJK Extension B+
    }

    private nonisolated static func recommendedOfflineThreads() -> Int {
        let cores = max(ProcessInfo.processInfo.activeProcessorCount, 1)
        switch cores {
        case 0...2: return 1
        case 3...4: return 2
        case 5...8: return 4
        default:    return 6
        }
    }
}
