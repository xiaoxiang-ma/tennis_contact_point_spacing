// SessionView.swift
// TennisContact — iOS
//
// New session flow: video picker (PhotosUI) → shows filename and duration.
// No processing happens here yet — that is wired in Task 5 (ProcessingPipeline).
// See docs/implementation_v3.md Section 2.2 and Task 4

import SwiftUI
import PhotosUI
import AVFoundation
import UniformTypeIdentifiers

// MARK: - VideoFile (Transferable)

/// Copies the picked video to the app's temp directory and exposes its URL.
/// Using FileRepresentation lets us get a real on-disk URL without loading
/// the entire video into memory, which is important for long recordings.
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

    @State private var selectedItem: PhotosPickerItem?
    @State private var videoURL: URL?
    @State private var videoName: String = ""
    @State private var videoDuration: String = ""
    @State private var isLoading = false
    @State private var loadError: String?

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
                    .onChange(of: selectedItem) { _, newItem in
                        loadVideo(from: newItem)
                    }
                } header: {
                    Text("Video")
                } footer: {
                    Text("Record from a tripod behind the baseline at 4–6 ft height, 60 fps.")
                        .font(.caption)
                }

                // MARK: Video info (shown after selection)
                if isLoading {
                    Section {
                        HStack {
                            ProgressView()
                                .padding(.trailing, 8)
                            Text("Loading video info…")
                                .foregroundStyle(.secondary)
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

                    Section {
                        // TODO (Task 5): wire to ProcessingPipeline and navigate to
                        // the four-stage progress screen when tapped.
                        Button("Analyse Video") {
                            // placeholder — Task 5
                        }
                        .frame(maxWidth: .infinity)
                        .disabled(true)
                    } footer: {
                        Text("Processing will be enabled in the next update.")
                            .font(.caption)
                    }
                }
            }
            .navigationTitle("New Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    // MARK: - Video loading

    private func loadVideo(from item: PhotosPickerItem?) {
        guard let item else { return }

        isLoading = true
        loadError = nil
        videoURL = nil
        videoName = ""
        videoDuration = ""

        Task {
            do {
                guard let file = try await item.loadTransferable(type: VideoFile.self) else {
                    throw VideoLoadError.transferFailed
                }

                let asset = AVURLAsset(url: file.url)
                let duration = try await asset.load(.duration)
                let seconds = CMTimeGetSeconds(duration)

                await MainActor.run {
                    videoURL = file.url
                    videoName = file.url.deletingPathExtension().lastPathComponent
                    videoDuration = formatDuration(seconds)
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    loadError = error.localizedDescription
                    isLoading = false
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

    var errorDescription: String? {
        "Could not load the selected video. Please try again."
    }
}

// MARK: - Preview

#Preview {
    SessionView()
}
