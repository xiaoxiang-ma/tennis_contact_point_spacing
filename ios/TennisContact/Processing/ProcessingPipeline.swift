// ProcessingPipeline.swift
// TennisContact — iOS
//
// Orchestrates all four processing components in sequence:
//   AudioDetector → PoseEstimator → CoordinateTransform → StatisticsEngine
//
// Returns an AsyncThrowingStream of PipelineEvent so the UI can update
// in real time as each stage completes, without blocking the main thread.
//
// Stage 1/4: Extracting audio track
// Stage 2/4: Detecting contact frames
// Stage 3/4: Running body pose estimation
// Stage 4/4: Computing statistics and swing arcs
//
// See docs/implementation_v3.md Section 3.1 and Task 5

import Foundation
import AVFoundation

// MARK: - ProcessedShot

/// Value type produced by the pipeline for each detected contact.
/// Converted to a CoreData Shot entity in Task 12.
struct ProcessedShot {
    /// Timestamp (seconds from video start) of the canonical contact frame.
    let timestamp: TimeInterval
    /// Frame index of the canonical contact frame.
    let frameIndex: Int
    /// Confidence score from AudioDetector (0.4–1.0).
    let audioConfidence: Float
    /// Raw joint positions in Vision camera space, keyed by joint name.
    /// Transform to pelvis-relative coordinates in Task 6 (CoordinateTransform).
    let joints: [String: SIMD3<Float>]
}

// MARK: - PipelineEvent

enum PipelineEvent {
    case stage(ProcessingStage)
    case completed([ProcessedShot])
}

// MARK: - ProcessingStage

enum ProcessingStage: String {
    case extractingAudio    = "Extracting audio track"
    case detectingContacts  = "Detecting contact frames"
    case estimatingPose     = "Running body pose estimation"
    case computingStatistics = "Computing statistics and swing arcs"
    case complete           = "Complete"

    var stepNumber: Int {
        switch self {
        case .extractingAudio:    return 1
        case .detectingContacts:  return 2
        case .estimatingPose:     return 3
        case .computingStatistics: return 4
        case .complete:           return 4
        }
    }

    var displayText: String { "Stage \(stepNumber)/4: \(rawValue)" }
}

// MARK: - PipelineError

enum PipelineError: LocalizedError {
    case noVideoTrack
    case noAudioTrack
    case noContactsDetected

    var errorDescription: String? {
        switch self {
        case .noVideoTrack:
            return "The video file has no video track."
        case .noAudioTrack:
            return "The video file has no audio track. Ensure your recording includes sound."
        case .noContactsDetected:
            return "No ball-racket contacts were detected. Try adjusting the recording position or reducing background noise."
        }
    }
}

// MARK: - ProcessingPipeline

actor ProcessingPipeline {

    // MARK: Public API

    /// Run the full four-stage pipeline and stream progress + final results back to the caller.
    ///
    /// Usage in SwiftUI:
    /// ```swift
    /// for try await event in ProcessingPipeline().process(videoURL: url) {
    ///     switch event {
    ///     case .stage(let s):   currentStage = s
    ///     case .completed(let shots): self.shots = shots
    ///     }
    /// }
    /// ```
    func process(videoURL: URL) -> AsyncThrowingStream<PipelineEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // ── Stage 1: read video metadata ────────────────────────
                    continuation.yield(.stage(.extractingAudio))
                    let (frameRate, _) = try await videoInfo(url: videoURL)

                    // ── Stage 2: audio contact detection ────────────────────
                    continuation.yield(.stage(.detectingContacts))
                    let rawContacts = try await AudioDetector().detect(videoURL: videoURL)

                    // Assign frame indices now that we have the frame rate
                    let contacts = rawContacts.map { c in
                        ContactCandidate(
                            timestamp: c.timestamp,
                            frameIndex: Int((c.timestamp * frameRate).rounded()),
                            confidence: c.confidence
                        )
                    }

                    // ── Stage 3: body pose estimation ────────────────────────
                    continuation.yield(.stage(.estimatingPose))
                    let shots = try await PoseEstimator().estimate(
                        videoURL: videoURL,
                        contacts: contacts,
                        frameRate: frameRate
                    )

                    // ── Stage 4: statistics (placeholder until Task 10) ──────
                    continuation.yield(.stage(.computingStatistics))
                    // StatisticsEngine.compute(shots:) wired in Task 10.
                    // CoordinateTransform applied in Task 6.

                    // ── Done ─────────────────────────────────────────────────
                    continuation.yield(.completed(shots))
                    continuation.yield(.stage(.complete))
                    continuation.finish()

                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: Helpers

    private func videoInfo(url: URL) async throws -> (frameRate: Double, duration: Double) {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw PipelineError.noVideoTrack }
        let fps      = try await Double(track.load(.nominalFrameRate))
        let duration = CMTimeGetSeconds(try await asset.load(.duration))
        return (fps > 0 ? fps : 30.0, duration)
    }
}
