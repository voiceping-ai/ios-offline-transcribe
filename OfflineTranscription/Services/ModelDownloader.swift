import Foundation

/// Downloads individual sherpa-onnx model files from HuggingFace.
@MainActor
final class ModelDownloader: NSObject, @unchecked Sendable {
    private(set) var progress: Double = 0.0
    var onProgress: ((Double) -> Void)?

    private var downloadTask: URLSessionDownloadTask?
    nonisolated(unsafe) private var session: URLSession?
    nonisolated(unsafe) private var continuation: CheckedContinuation<URL, Error>?
    private let continuationLock = NSLock()

    /// Tracks multi-file download progress.
    nonisolated(unsafe) private var currentFileIndex: Int = 0
    nonisolated(unsafe) private var totalFilesToDownload: Int = 1

    private static let defaultHuggingFaceOrg = "csukuangfj"
    nonisolated(unsafe) private static let fileManager = FileManager.default
    private static let downloadSessionConfiguration: URLSessionConfiguration = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 15 * 60
        config.waitsForConnectivity = true
        return config
    }()

    /// Directory where model files are stored.
    static var modelsDirectory: URL {
        #if os(macOS)
        // Use an App Group container so models persist across (signed) app reinstalls and
        // remain shared between sandboxed and non-sandboxed debug runs.
        let appGroupId = "group.com.voiceping.transcribe"
        let groupRoot = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library/Group Containers", isDirectory: true)
        let suffix = "." + appGroupId

        if let groupURL = fileManager.containerURL(forSecurityApplicationGroupIdentifier: appGroupId) {
            // When running unsigned (or outside the app sandbox), macOS can return an unprefixed
            // directory (~/Library/Group Containers/group.com.voiceping.transcribe). If a TeamID-
            // prefixed container exists, prefer it so we reuse the real sandbox container.
            if groupURL.lastPathComponent == appGroupId,
               let entries = try? fileManager.contentsOfDirectory(
                   at: groupRoot,
                   includingPropertiesForKeys: nil,
                   options: [.skipsHiddenFiles]
               ),
               let match = entries
                   .filter({ $0.lastPathComponent.hasSuffix(suffix) })
                   .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                   .first {
                return match.appendingPathComponent("SherpaModels", isDirectory: true)
            }
            return groupURL.appendingPathComponent("SherpaModels", isDirectory: true)
        }
        // Fallback for unsigned local runs (no entitlements): still prefer the same path.
        // When the app is signed, macOS typically prefixes the directory with the Team ID
        // (e.g. "<TEAMID>.group.com.voiceping.transcribe"). Try to reuse it if present.
        if let entries = try? fileManager.contentsOfDirectory(
            at: groupRoot,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            // Prefer the TeamID-prefixed container (e.g. "<TEAMID>.group.com.voiceping.transcribe")
            // over the bare group id directory, which can be created accidentally by unsigned runs.
            if let match = entries
                .filter({ $0.lastPathComponent.hasSuffix(suffix) })
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                .first {
                return match.appendingPathComponent("SherpaModels", isDirectory: true)
            }
            if let match = entries
                .filter({ $0.lastPathComponent == appGroupId })
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                .first {
                return match.appendingPathComponent("SherpaModels", isDirectory: true)
            }
        }
        return groupRoot
            .appendingPathComponent(appGroupId, isDirectory: true)
            .appendingPathComponent("SherpaModels", isDirectory: true)
        #else
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("SherpaModels", isDirectory: true)
        #endif
    }

    /// Check if all required model files are already downloaded.
    func isModelDownloaded(_ model: ModelInfo) -> Bool {
        if let config = model.sherpaModelConfig {
            let modelDir = Self.modelsDirectory.appendingPathComponent(config.repoName)
            return config.allFiles.allSatisfy { file in
                Self.fileManager.fileExists(atPath: modelDir.appendingPathComponent(file).path)
            }
        }
        if let config = model.qwenModelConfig {
            let modelDir = Self.modelsDirectory.appendingPathComponent(config.localDirName)
            return config.files.allSatisfy { file in
                Self.fileManager.fileExists(atPath: modelDir.appendingPathComponent(file).path)
            }
        }
        return false
    }

    /// Get the local directory path for a downloaded model.
    func modelDirectory(for model: ModelInfo) -> URL? {
        let dirName: String
        if let config = model.sherpaModelConfig {
            dirName = config.repoName
        } else if let config = model.qwenModelConfig {
            dirName = config.localDirName
        } else {
            return nil
        }
        let dir = Self.modelsDirectory.appendingPathComponent(dirName)
        guard Self.fileManager.fileExists(atPath: dir.path) else { return nil }
        return dir
    }

    /// Download all model files individually from HuggingFace. Returns the local model directory.
    func downloadModel(_ model: ModelInfo) async throws -> URL {
        // Determine repo, directory name, and file list
        let repoPath: String
        let localDirName: String
        let allFiles: [String]

        if let config = model.sherpaModelConfig {
            repoPath = config.repoName.contains("/") ? config.repoName : "\(Self.defaultHuggingFaceOrg)/\(config.repoName)"
            localDirName = config.repoName
            allFiles = config.allFiles
        } else if let config = model.qwenModelConfig {
            repoPath = config.repoId
            localDirName = config.localDirName
            allFiles = config.files
        } else {
            throw AppError.modelDownloadFailed(underlying: NSError(
                domain: "ModelDownloader", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "No model config for \(model.id)"]
            ))
        }

        let modelDir = Self.modelsDirectory.appendingPathComponent(localDirName)
        progress = 0

        if isModelDownloaded(model) {
            progress = 1
            return modelDir
        }

        // Create model directory
        try Self.fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Determine which files still need downloading
        let filesToDownload = allFiles.filter { filename in
            !Self.fileManager.fileExists(atPath: modelDir.appendingPathComponent(filename).path)
        }

        guard !filesToDownload.isEmpty else { return modelDir }

        totalFilesToDownload = filesToDownload.count

        for (index, filename) in filesToDownload.enumerated() {
            currentFileIndex = index

            let url = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(filename)")!
            let tempFile = try await downloadFile(from: url)

            let destPath = modelDir.appendingPathComponent(filename)
            // Remove partial file if it exists
            try? Self.fileManager.removeItem(at: destPath)
            try Self.fileManager.moveItem(at: tempFile, to: destPath)
        }

        guard isModelDownloaded(model) else {
            throw AppError.modelDownloadFailed(underlying: NSError(
                domain: "ModelDownloader", code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Downloaded files but model validation failed"]
            ))
        }

        progress = 1
        return modelDir
    }

    /// Delete a downloaded model.
    func deleteModel(_ model: ModelInfo) throws {
        let dirName: String
        if let config = model.sherpaModelConfig {
            dirName = config.repoName
        } else if let config = model.qwenModelConfig {
            dirName = config.localDirName
        } else {
            return
        }
        let modelDir = Self.modelsDirectory.appendingPathComponent(dirName)
        if Self.fileManager.fileExists(atPath: modelDir.path) {
            try Self.fileManager.removeItem(at: modelDir)
        }
    }

    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        session?.invalidateAndCancel()
        session = nil

        // Resume any waiting continuation so the caller doesn't hang
        resumeContinuation(with: .failure(CancellationError()))
    }

    deinit {
        session?.invalidateAndCancel()
    }

    // MARK: - Private

    private static func fileURL(repo: String, filename: String) -> URL {
        let repoPath: String
        if repo.contains("/") {
            repoPath = repo
        } else {
            repoPath = "\(defaultHuggingFaceOrg)/\(repo)"
        }
        return URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(filename)")!
    }

    private func downloadFile(from url: URL) async throws -> URL {
        return try await withCheckedThrowingContinuation { continuation in
            let session = URLSession(
                configuration: Self.downloadSessionConfiguration,
                delegate: self,
                delegateQueue: nil
            )
            self.session = session

            continuationLock.lock()
            self.continuation = continuation
            continuationLock.unlock()

            self.downloadTask = session.downloadTask(with: url)
            self.downloadTask?.resume()
        }
    }

    nonisolated private func resumeContinuation(with result: Result<URL, Error>) {
        continuationLock.lock()
        let cont = continuation
        continuation = nil
        continuationLock.unlock()
        switch result {
        case .success(let url):
            cont?.resume(returning: url)
        case .failure(let error):
            cont?.resume(throwing: error)
        }
    }
}

// MARK: - URLSessionDownloadDelegate

extension ModelDownloader: URLSessionDownloadDelegate {
    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Validate HTTP status to avoid caching an auth/error body as a "model file".
        if let http = downloadTask.response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            let urlString = downloadTask.originalRequest?.url?.absoluteString ?? "(unknown url)"
            let snippet: String = (try? String(contentsOf: location, encoding: .utf8))
                .map { String($0.prefix(200)) } ?? ""
            let message = "HTTP \(http.statusCode) downloading \(urlString): \(snippet)"
            session.finishTasksAndInvalidate()
            resumeContinuation(with: .failure(NSError(
                domain: "ModelDownloader",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: message]
            )))
            return
        }

        let tempDir = Self.fileManager.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent(UUID().uuidString)

        do {
            try Self.fileManager.copyItem(at: location, to: tempFile)
            session.finishTasksAndInvalidate()
            resumeContinuation(with: .success(tempFile))
        } catch {
            session.finishTasksAndInvalidate()
            resumeContinuation(with: .failure(error))
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fileFraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        let total = Double(max(1, totalFilesToDownload))
        let overallFraction = (Double(currentFileIndex) + fileFraction) / total
        Task { @MainActor [weak self] in
            self?.progress = overallFraction
            self?.onProgress?(overallFraction)
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error else { return }
        resumeContinuation(with: .failure(error))
    }
}
