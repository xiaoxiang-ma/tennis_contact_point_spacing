// SessionView.swift
// TennisContact — iOS
//
// New session flow:
//   1. PhotosPicker → video file URL + metadata (Task 4)
//   2. "Analyse Video" → ProcessingPipeline → live stage progress (Task 5)
//   3. Results summary shown inline; full UI in Tasks 7/8/11
//
// See docs/implementation_v3.md Section 2.2

import SwiftUI
import PhotosUI
import AVFoundation
import UniformTypeIdentifiers

// MARK: - VideoFile (Transferable)

/// Copies the picked video to a temp directory and exposes its on-disk URL.
struct VideoFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoFile(url: dest)
        }
    }
}

// MARK: - SessionView

struct SessionView: View {
    @Environment(\.dismiss) private var dismiss

    // Video selection state
    @State private var selectedItem: PhotosPickerItem?
    @State private var videoURL: URL?
    @State private var videoName: String = ""
    @State private var videoDuration: String = ""
    @State private var isLoadingVideo = false
    @State private var loadError: String?

    // Processing state
    @State private var isProcessing = false
    @State private var currentStage: ProcessingStage?
    @State private var completedStages: Set<ProcessingStage> = []
    @State private var processedShots: [ProcessedShot] = []
    @State private var processingError: String?
    @State private var processingTask: Task<Void, Never>?

    // The ordered stages shown in the progress list
    private let stages: [ProcessingStage] = [
        .extractingAudio, .detectingContacts, .estimatingPose, .computingStatistics
    ]

    var body: some View {
        NavigationStack {
            Form {

                // MARK: Video picker
                Section {
                    PhotosPicker(
                        selection: $selectedItem,
                        matching: .videos,
                        photoLibrary: .shared()
                    ) {
                        Label(
                            videoURL == nil ? "Select Video" : "Change Video",
                            systemImage: "video.badge.plus"
                        )
                    }
                    .disabled(isProcessing)
                    .onChange(of: selectedItem) { _, newItem in
                        resetProcessing()
                        loadVideo(from: newItem)
                    }
                } header: {
                    Text("Video")
                } footer: {
                    Text("Record from a tripod behind the baseline, 4–6 ft height, 60 fps.")
                        .font(.caption)
                }

                // MARK: Video info
                if isLoadingVideo {
                    Section {
                        HStack {
                            ProgressView().padding(.trailing, 8)
                            Text("Loading video…").foregroundStyle(.secondary)
                        }
                    }
                } else if let error = loadError {
                    Section {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                    }
                } else if videoURL != nil {
                    Section("Video Info") {
                        LabeledContent("File", value: videoName)
                        LabeledContent("Duration", value: videoDuration)
                    }

                    // MARK: Analyse button / progress / results
                    if isProcessing {
                        progressSection
                    } else if !processedShots.isEmpty {
                        resultsSection
                    } else if let err = processingError {
                        Section {
                            Label(err, systemImage: "exclamationmark.triangle")
                                .foregroundStyle(.red)
                        }
                        analyseButton
                    } else {
                        analyseButton
                    }
                }
            }
            .navigationTitle("New Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        processingTask?.cancel()
                        dismiss()
                    }
                }
            }
        }
    }

    // MARK: - Sub-views

    private var analyseButton: some View {
        Section {
            Button("Analyse Video") { startProcessing() }
                .frame(maxWidth: .infinity)
        }
    }

    /// Live four-stage progress list shown during processing.
    private var progressSection: some View {
        Section("Processing") {
            ForEach(stages, id: \.self) { stage in
                HStack(spacing: 12) {
                    stageIcon(for: stage)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(stage.rawValue)
                            .font(.subheadline)
                        if stage == currentStage && !completedStages.contains(stage) {
                            Text("In progress…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }

    @ViewBuilder
    private func stageIcon(for stage: ProcessingStage) -> some View {
        if completedStages.contains(stage) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        } else if stage == currentStage {
            ProgressView()
                .controlSize(.small)
        } else {
            Image(systemName: "circle")
                .foregroundStyle(.tertiary)
        }
    }

    /// Summary shown after pipeline completes.
    private var resultsSection: some View {
        Section("Results") {
            LabeledContent("Contacts detected", value: "\(processedShots.count)")

            let withJoints = processedShots.filter { !$0.joints.isEmpty }.count
            LabeledContent("Pose estimated", value: "\(withJoints) / \(processedShots.count)")

            // TODO (Task 8): navigate to ShotDetailView
            // TODO (Task 11): navigate to StatisticsView
            Text("Detailed views coming in the next update.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Actions

    private func startProcessing() {
        guard let url = videoURL else { return }

        resetProcessing()
        isProcessing = true
        processingError = nil

        processingTask = Task {
            do {
                let pipeline = ProcessingPipeline()
                for try await event in await pipeline.process(videoURL: url) {
                    await MainActor.run {
                        switch event {
                        case .stage(let stage):
                            if stage != .complete {
                                if let prev = currentStage { completedStages.insert(prev) }
                                currentStage = stage
                            } else {
                                if let prev = currentStage { completedStages.insert(prev) }
                                currentStage = .complete
                            }
                        case .completed(let shots):
                            processedShots = shots
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    processingError = error.localizedDescription
                }
            }
            await MainActor.run { isProcessing = false }
        }
    }

    private func resetProcessing() {
        processingTask?.cancel()
        isProcessing        = false
        currentStage        = nil
        completedStages     = []
        processedShots      = []
        processingError     = nil
    }

    // MARK: - Video loading

    private func loadVideo(from item: PhotosPickerItem?) {
        guard let item else { return }

        isLoadingVideo = true
        loadError = nil
        videoURL = nil
        videoName = ""
        videoDuration = ""

        Task {
            do {
                guard let file = try await item.loadTransferable(type: VideoFile.self) else {
                    throw VideoLoadError.transferFailed
                }
                let asset    = AVURLAsset(url: file.url)
                let duration = try await asset.load(.duration)
                let seconds  = CMTimeGetSeconds(duration)

                await MainActor.run {
                    videoURL      = file.url
                    videoName     = file.url.deletingPathExtension().lastPathComponent
                    videoDuration = formatDuration(seconds)
                    isLoadingVideo = false
                }
            } catch {
                await MainActor.run {
                    loadError      = error.localizedDescription
                    isLoadingVideo = false
                }
            }
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "Unknown" }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return mins > 0 ? "\(mins)m \(secs)s" : "\(secs)s"
    }
}

// MARK: - Errors

private enum VideoLoadError: LocalizedError {
    case transferFailed
    var errorDescription: String? { "Could not load the selected video. Please try again." }
}

// MARK: - Preview

#Preview { SessionView() }
