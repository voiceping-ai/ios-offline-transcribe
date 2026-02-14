#if os(macOS)
import AppKit
import SwiftUI

/// Non-modal model switcher + quick settings for macOS.
///
/// This is used both as:
/// - The main window inspector panel (fast model switching, doesn't block transcription UI)
/// - The app's Settings window content (Command+,)
struct MacSettingsPanel: View {
    @Environment(WhisperService.self) private var whisperService
    @State private var viewModel: ModelManagementViewModel?
    @State private var searchText: String = ""

    private var isBusy: Bool {
        whisperService.modelState == .downloading
            || whisperService.modelState == .downloaded
            || whisperService.modelState == .loading
    }

    private var fullText: String {
        whisperService.fullTranscriptionText
    }

    private var filteredGroups: [(family: ModelFamily, cards: [ModelCard])] {
        let q = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !q.isEmpty else { return whisperService.modelCardsByFamily }

        return whisperService.modelCardsByFamily.compactMap { group in
            let cards = group.cards.filter { card in
                card.displayName.lowercased().contains(q)
                    || card.description.lowercased().contains(q)
                    || card.id.lowercased().contains(q)
                    || card.languages.lowercased().contains(q)
            }
            guard !cards.isEmpty else { return nil }
            return (family: group.family, cards: cards)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            TextField("Search models", text: $searchText)
                .textFieldStyle(.roundedBorder)

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    ForEach(filteredGroups, id: \.family) { group in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(group.family.displayName)
                                .font(.headline)
                                .foregroundStyle(.secondary)

                            ForEach(group.cards) { card in
                                modelRow(card)
                            }
                        }
                    }
                }
                .padding(.vertical, 4)
            }

            Divider()

            transcriptionToggles

            footer
        }
        .padding(12)
        .frame(minWidth: 380, idealWidth: 460, maxWidth: 560, minHeight: 560)
        .task {
            if viewModel == nil {
                viewModel = ModelManagementViewModel(whisperService: whisperService)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Current Model")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(whisperService.selectedModel.displayName)
                        .font(.headline)
                    Text(whisperService.effectiveRuntimeLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    copyTextToClipboard()
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                .disabled(fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                Button(role: .destructive) {
                    whisperService.clearTranscription()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .disabled(fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }

            if let warning = whisperService.backendFallbackWarning {
                Text(warning)
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }

            if let err = whisperService.lastError {
                Text(err.localizedDescription)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
    }

    @ViewBuilder
    private func modelRow(_ card: ModelCard) -> some View {
        let isSelectedCard = whisperService.selectedModelCardId == card.id
        let availableBackends = whisperService.availableBackends(for: card)
        let selectedBackend = isSelectedCard
            ? whisperService.selectedInferenceBackend
            : card.preferredBackend()
        let resolvedModel = whisperService.resolvedModelInfo(
            for: card,
            requestedBackend: selectedBackend
        )
        let downloaded = resolvedModel.map { viewModel?.isModelDownloaded($0) ?? false } ?? false

        ModelPickerRow(
            card: card,
            isSelected: isSelectedCard,
            selectedBackend: selectedBackend,
            availableBackends: availableBackends,
            effectiveBackend: isSelectedCard
                ? whisperService.effectiveInferenceBackend
                : selectedBackend,
            effectiveRuntimeLabel: isSelectedCard
                ? whisperService.effectiveRuntimeLabel
                : whisperService.runtimeLabel(for: card, requestedBackend: selectedBackend),
            fallbackWarning: isSelectedCard ? whisperService.backendFallbackWarning : nil,
            isDownloaded: downloaded,
            isDownloading: whisperService.modelState == .downloading && isSelectedCard,
            downloadProgress: whisperService.downloadProgress,
            isLoading: whisperService.modelState == .loading && isSelectedCard,
            onBackendChange: { backend in
                whisperService.setSelectedModelCard(card.id)
                whisperService.setSelectedInferenceBackend(backend)
            },
            onTap: {
                guard !isBusy else { return }
                whisperService.setSelectedModelCard(card.id)
                Task {
                    await viewModel?.downloadAndSetup()
                }
            }
        )
        .disabled(isBusy)
    }

    private var transcriptionToggles: some View {
        @Bindable var service = whisperService
        return VStack(alignment: .leading, spacing: 8) {
            Text("Transcription")
                .font(.headline)
                .foregroundStyle(.secondary)

            Toggle("Voice Activity Detection", isOn: $service.useVAD)
            Toggle("Enable Timestamps", isOn: $service.enableTimestamps)
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button("Open Dictation Settings") {
                    PermissionManager.openSpeechRecognitionSettings()
                }
                Button("Reveal Model Folder") {
                    let dir = ModelDownloader.modelsDirectory
                    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                    NSWorkspace.shared.activateFileViewerSelecting([dir])
                }
                Spacer()
                AppVersionLabel()
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            Text("Models are cached locally at: \(ModelDownloader.modelsDirectory.path)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(2)
        }
    }

    private func copyTextToClipboard() {
        let text = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
#endif
