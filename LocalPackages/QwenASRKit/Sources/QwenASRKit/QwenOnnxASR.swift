import Foundation
import QwenASRCLib

@_silgen_name("qwen_onnx_load")
private func c_qwen_onnx_load(_ modelDir: UnsafePointer<CChar>?) -> OpaquePointer?

@_silgen_name("qwen_onnx_free")
private func c_qwen_onnx_free(_ ctx: OpaquePointer?)

@_silgen_name("qwen_onnx_transcribe")
private func c_qwen_onnx_transcribe(
    _ ctx: OpaquePointer?,
    _ samples: UnsafePointer<Float>?,
    _ nSamples: Int32
) -> UnsafeMutablePointer<CChar>?

@_silgen_name("qwen_onnx_get_last_timing")
private func c_qwen_onnx_get_last_timing(
    _ mel: UnsafeMutablePointer<Double>?,
    _ enc: UnsafeMutablePointer<Double>?,
    _ prefill: UnsafeMutablePointer<Double>?,
    _ decode: UnsafeMutablePointer<Double>?,
    _ total: UnsafeMutablePointer<Double>?,
    _ nTokens: UnsafeMutablePointer<Int32>?
)

@_silgen_name("qwen_onnx_get_last_error")
private func c_qwen_onnx_get_last_error() -> UnsafePointer<CChar>?

@_silgen_name("qwen_onnx_set_log_file")
private func c_qwen_onnx_set_log_file(_ path: UnsafePointer<CChar>?)

/// Thread-safe Swift wrapper around the Qwen3-ASR ONNX Runtime inference.
public final class QwenOnnxASR: @unchecked Sendable {
    private var ctx: OpaquePointer?
    private let lock = NSLock()

    /// Last error message from the C layer (empty if no error).
    public static var lastError: String {
        guard let ptr = c_qwen_onnx_get_last_error() else { return "" }
        return String(cString: ptr)
    }

    /// Load ONNX models from a directory containing encoder, decoder_prefill,
    /// decoder_decode ONNX models, embed_tokens.npy, and vocab.json.
    /// Returns nil if loading fails.
    public init?(modelDir: String) {
        NSLog("[QwenOnnxASR] Loading from: %@", modelDir)

        // Set up C-level log file for device diagnostics
        // Use both Documents and tmp — jetsam may prevent Documents flush
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        let logPath = docs?.appendingPathComponent("qwen_onnx_log.txt").path
            ?? NSTemporaryDirectory() + "qwen_onnx_log.txt"
        logPath.withCString { c_qwen_onnx_set_log_file($0) }

        guard let c = modelDir.withCString({ c_qwen_onnx_load($0) }) else {
            NSLog("[QwenOnnxASR] FAILED to load: %@", Self.lastError)
            return nil
        }
        NSLog("[QwenOnnxASR] Loaded successfully")
        self.ctx = c
    }

    deinit {
        lock.lock()
        if let c = ctx { c_qwen_onnx_free(c) }
        ctx = nil
        lock.unlock()
    }

    /// Timing breakdown from the last transcribe call (all in milliseconds).
    public struct Timing {
        public let melMs: Double
        public let encoderMs: Double
        public let prefillMs: Double
        public let decodeMs: Double
        public let totalMs: Double
        public let nTokens: Int
    }

    /// Transcribe Float32 audio samples (16kHz mono, range [-1, 1]).
    /// Returns transcribed text, or nil on failure.
    public func transcribe(samples: [Float]) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let c = ctx else { return nil }
        let result = samples.withUnsafeBufferPointer { buf in
            c_qwen_onnx_transcribe(c, buf.baseAddress, Int32(buf.count))
        }
        guard let result else { return nil }
        let text = String(cString: result)
        free(result)

        // Log timing breakdown
        let timing = getLastTiming()
        let audioSec = Double(samples.count) / 16000.0
        NSLog("[QwenOnnx] mel=%.0fms enc=%.0fms prefill=%.0fms decode=%.0fms total=%.0fms (%d tokens, %.1f ms/tok, RTF=%.3f, audio=%.1fs)",
              timing.melMs, timing.encoderMs, timing.prefillMs, timing.decodeMs, timing.totalMs,
              timing.nTokens, timing.nTokens > 0 ? timing.decodeMs / Double(timing.nTokens) : 0,
              audioSec > 0 ? (timing.totalMs / 1000.0) / audioSec : 0, audioSec)

        return text
    }

    /// Get timing breakdown from the last transcribe() call.
    public func getLastTiming() -> Timing {
        var mel: Double = 0, enc: Double = 0, prefill: Double = 0
        var decode: Double = 0, total: Double = 0
        var nTokens: Int32 = 0
        c_qwen_onnx_get_last_timing(&mel, &enc, &prefill, &decode, &total, &nTokens)
        return Timing(melMs: mel, encoderMs: enc, prefillMs: prefill,
                      decodeMs: decode, totalMs: total, nTokens: Int(nTokens))
    }

    /// Release all resources. Safe to call multiple times.
    public func release() {
        lock.lock()
        if let c = ctx { c_qwen_onnx_free(c) }
        ctx = nil
        lock.unlock()
    }
}
