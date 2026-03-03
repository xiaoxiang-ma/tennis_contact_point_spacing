// PoseEstimator.swift
// TennisContact — iOS
//
// Python reference: python/src/pose_estimation.py
// Replaces MediaPipe with Apple Vision VNDetectHumanBodyPose3DRequest (iOS 17+).
//
// For each audio-detected contact, extracts ±5 frames (11 frames total) from
// the video and runs pose estimation on each. The frame with the highest total
// joint confidence is selected as the canonical contact frame.
//
// Joint mapping (Apple Vision → our string keys, matching Python LANDMARK_MAP):
//   .leftShoulder / .rightShoulder → "left_shoulder" / "right_shoulder"
//   .leftElbow    / .rightElbow    → "left_elbow"    / "right_elbow"
//   .leftWrist    / .rightWrist    → "left_wrist"    / "right_wrist"
//   .leftHip      / .rightHip     → "left_hip"      / "right_hip"
//   .root                          → "root"
//   .head                          → "head"
//   (synthetic) midpoint of hips   → "pelvis"  (mirrors Python _add_synthetic())
//
// NOTE: VNDetectHumanBodyPose3DRequest returns metric-space coordinates on
// LiDAR devices (iPhone 12 Pro+) and relative coordinates on all others.
// Consistency tracking works on both; label metric vs. relative in the UI (Task 11).
//
// See docs/architecture.md Component 2 and docs/implementation_v3.md §3.3

import Foundation
import Vision
import AVFoundation

struct PoseEstimator {

    // MARK: - Joint mapping

    /// Apple Vision joint names → our canonical string keys (matches Python).
    static let jointKeyMap: [VNHumanBodyPose3DObservation.JointName: String] = [
        .leftShoulder:   "left_shoulder",
        .rightShoulder:  "right_shoulder",
        .leftElbow:      "left_elbow",
        .rightElbow:     "right_elbow",
        .leftWrist:      "left_wrist",
        .rightWrist:     "right_wrist",
        .leftHip:        "left_hip",
        .rightHip:       "right_hip",
        .root:           "root",
        .head:           "head",
        .centerShoulder: "neck",
    ]

    // MARK: - Public API

    /// Run pose estimation for every contact and return one ProcessedShot per contact.
    func estimate(
        videoURL: URL,
        contacts: [ContactCandidate],
        frameRate: Double
    ) async throws -> [ProcessedShot] {
        guard !contacts.isEmpty else { return [] }

        let asset = AVURLAsset(url: videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        // Allow ±half-frame tolerance so we get the nearest available frame
        let halfFrame = CMTime(seconds: 0.5 / frameRate, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = halfFrame
        generator.requestedTimeToleranceAfter  = halfFrame

        var shots: [ProcessedShot] = []
        for contact in contacts {
            let shot = await processContact(
                generator: generator,
                contact: contact,
                frameRate: frameRate
            )
            shots.append(shot)
        }
        return shots
    }

    // MARK: - Per-contact window

    private func processContact(
        generator: AVAssetImageGenerator,
        contact: ContactCandidate,
        frameRate: Double
    ) async -> ProcessedShot {
        // ±5 frames around the audio-detected contact (11 frames total)
        let windowIndices = (-5...5).map { contact.frameIndex + $0 }
        let times = windowIndices.map { idx in
            CMTime(seconds: max(0, Double(idx) / frameRate), preferredTimescale: 600)
        }

        // Extract all frames in one batch call
        let frames = await extractFrames(generator: generator, times: times)

        // Run pose on each frame; keep the one with highest total joint confidence
        var bestJoints: [String: SIMD3<Float>] = [:]
        var bestConfidence: Float = -1
        var bestFrameIndex = contact.frameIndex

        for (i, (_, cgImage)) in frames.enumerated() {
            guard let image = cgImage else { continue }
            let (joints, confidence) = runPose(on: image)
            if confidence > bestConfidence {
                bestConfidence  = confidence
                bestJoints      = joints
                bestFrameIndex  = windowIndices[i]
            }
        }

        return ProcessedShot(
            timestamp:       Double(bestFrameIndex) / frameRate,
            frameIndex:      bestFrameIndex,
            audioConfidence: contact.confidence,
            joints:          bestJoints
        )
    }

    // MARK: - Frame extraction

    /// Batch-extract CGImages at the requested times using the async callback API.
    /// Thread-safe: all writes to `results` happen on a private serial queue.
    private func extractFrames(
        generator: AVAssetImageGenerator,
        times: [CMTime]
    ) async -> [(CMTime, CGImage?)] {
        await withCheckedContinuation { continuation in
            var results = [(CMTime, CGImage?)](repeating: (.zero, nil), count: times.count)
            let queue   = DispatchQueue(label: "pose.frame-extraction")
            var remaining = times.count

            generator.generateCGImagesAsynchronously(
                forTimes: times.map { NSValue(time: $0) }
            ) { requestedTime, image, _, result, _ in
                queue.sync {
                    if let idx = times.firstIndex(of: requestedTime) {
                        results[idx] = (requestedTime, result == .succeeded ? image : nil)
                    }
                    remaining -= 1
                    if remaining == 0 {
                        continuation.resume(returning: results)
                    }
                }
            }
        }
    }

    // MARK: - Pose inference

    /// Run VNDetectHumanBodyPose3DRequest on a single CGImage.
    /// Returns (joints, totalConfidence). joints is empty if no person detected.
    private func runPose(on image: CGImage) -> ([String: SIMD3<Float>], Float) {
        let request = VNDetectHumanBodyPose3DRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])

        do {
            try handler.perform([request])
        } catch {
            return ([:], 0)
        }

        guard let observation = request.results?.first else { return ([:], 0) }

        var joints: [String: SIMD3<Float>] = [:]
        var totalConfidence: Float = 0

        for (visionName, key) in Self.jointKeyMap {
            guard let joint = try? observation.recognizedPoint(visionName),
                  joint.confidence > 0.1 else { continue }
            // The 4×4 position matrix: translation is in column 3
            let col = joint.position.columns.3
            joints[key] = SIMD3<Float>(col.x, col.y, col.z)
            totalConfidence += joint.confidence
        }

        // Synthesise pelvis = midpoint of hips (mirrors Python's _add_synthetic())
        if let l = joints["left_hip"], let r = joints["right_hip"] {
            joints["pelvis"] = (l + r) * 0.5
        }

        return (joints, totalConfidence)
    }
}
