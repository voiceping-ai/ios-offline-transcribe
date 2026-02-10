import Foundation

/// Appends timestamped inference log entries to Documents/inference_log.txt.
/// Retrieve from device: `xcrun devicectl device copy from --device <ID> --source <app-container>/Documents/inference_log.txt .`
final class InferenceLogger {
    static let shared = InferenceLogger()

    private let fileURL: URL
    private let queue = DispatchQueue(label: "InferenceLogger", qos: .utility)

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        fileURL = docs.appendingPathComponent("inference_log.txt")
        // Truncate on each app launch so logs don't grow unbounded
        try? "".write(to: fileURL, atomically: true, encoding: .utf8)
        log("=== Inference Logger initialized ===")
    }

    func log(_ message: String) {
        let ts = Self.timestamp()
        let line = "[\(ts)] \(message)"
        // print() goes to stdout, captured by devicectl --console
        print("[InferenceLog] \(line)")
        // Also persist to file
        let fileLine = line + "\n"
        queue.async { [fileURL] in
            if let data = fileLine.data(using: .utf8) {
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    if let handle = try? FileHandle(forWritingTo: fileURL) {
                        handle.seekToEndOfFile()
                        handle.write(data)
                        handle.closeFile()
                    }
                } else {
                    try? data.write(to: fileURL)
                }
            }
        }
    }

    var logFilePath: String { fileURL.path }

    private static func timestamp() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm:ss.SSS"
        return fmt.string(from: Date())
    }
}
