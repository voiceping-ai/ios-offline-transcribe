import AVFoundation
import Foundation
import QwenASRKit

struct BenchmarkResult: Codable {
    let modelDir: String
    let wavPath: String
    let audioSec: Double
    let wallMs: Double
    let rtf: Double
    let textLength: Int
    let melMs: Double
    let encoderMs: Double
    let prefillMs: Double
    let decodeMs: Double
    let totalMs: Double
    let tokens: Int
    let textPreview: String
}

enum BenchError: Error, LocalizedError {
    case usage
    case loadModelFailed(String)
    case transcribeFailed
    case audioConversionFailed(String)

    var errorDescription: String? {
        switch self {
        case .usage:
            return "Usage: qwen-bench <model_dir> <wav_path>"
        case .loadModelFailed(let path):
            return "Failed to load Qwen ONNX model from: \(path)"
        case .transcribeFailed:
            return "Qwen ONNX transcribe failed"
        case .audioConversionFailed(let msg):
            return msg
        }
    }
}

func loadAudioFile(url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let fileFormat = file.processingFormat

    guard let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16000,
        channels: 1,
        interleaved: false
    ) else {
        throw BenchError.audioConversionFailed("Cannot create target 16kHz mono format")
    }

    let fileFrameCount = AVAudioFrameCount(file.length)
    guard fileFrameCount > 0 else {
        throw BenchError.audioConversionFailed("Audio file is empty: \(url.path)")
    }

    guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: fileFrameCount) else {
        throw BenchError.audioConversionFailed("Failed to allocate source buffer")
    }
    try file.read(into: sourceBuffer)

    if fileFormat.sampleRate == 16000 &&
        fileFormat.channelCount == 1 &&
        fileFormat.commonFormat == .pcmFormatFloat32 {
        guard let floatData = sourceBuffer.floatChannelData else {
            throw BenchError.audioConversionFailed("No float channel data in source buffer")
        }
        return Array(UnsafeBufferPointer(start: floatData[0], count: Int(sourceBuffer.frameLength)))
    }

    guard let converter = AVAudioConverter(from: fileFormat, to: targetFormat) else {
        throw BenchError.audioConversionFailed("Cannot create AVAudioConverter")
    }

    let ratio = 16000.0 / fileFormat.sampleRate
    let outputFrameCount = AVAudioFrameCount(ceil(Double(sourceBuffer.frameLength) * ratio))
    guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
        throw BenchError.audioConversionFailed("Failed to allocate output buffer")
    }

    var conversionError: NSError?
    var inputConsumed = false
    converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
        if inputConsumed {
            outStatus.pointee = .endOfStream
            return nil
        }
        outStatus.pointee = .haveData
        inputConsumed = true
        return sourceBuffer
    }

    if let conversionError {
        throw conversionError
    }

    guard let floatData = outputBuffer.floatChannelData, outputBuffer.frameLength > 0 else {
        throw BenchError.audioConversionFailed("Audio conversion produced no output")
    }
    return Array(UnsafeBufferPointer(start: floatData[0], count: Int(outputBuffer.frameLength)))
}

func run() throws {
    guard CommandLine.arguments.count == 3 else {
        throw BenchError.usage
    }

    let modelDir = CommandLine.arguments[1]
    let wavPath = CommandLine.arguments[2]
    let wavURL = URL(fileURLWithPath: wavPath)

    let samples = try loadAudioFile(url: wavURL)
    let audioSec = Double(samples.count) / 16000.0

    guard let runtime = QwenOnnxASR(modelDir: modelDir) else {
        throw BenchError.loadModelFailed(modelDir)
    }

    let start = CFAbsoluteTimeGetCurrent()
    guard let text = runtime.transcribe(samples: samples) else {
        throw BenchError.transcribeFailed
    }
    let wallMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

    let timing = runtime.getLastTiming()
    let rtf = audioSec > 0 ? (wallMs / 1000.0) / audioSec : 0

    let result = BenchmarkResult(
        modelDir: modelDir,
        wavPath: wavPath,
        audioSec: audioSec,
        wallMs: wallMs,
        rtf: rtf,
        textLength: text.count,
        melMs: timing.melMs,
        encoderMs: timing.encoderMs,
        prefillMs: timing.prefillMs,
        decodeMs: timing.decodeMs,
        totalMs: timing.totalMs,
        tokens: timing.nTokens,
        textPreview: String(text.prefix(160))
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(result)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

do {
    try run()
} catch {
    fputs("[qwen-bench] ERROR: \(error.localizedDescription)\n", stderr)
    exit(1)
}
