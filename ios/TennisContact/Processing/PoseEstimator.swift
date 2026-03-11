// PoseEstimator.swift
// TennisContact — iOS
//
// Python reference: python/src/pose_estimation.py
// Replaces MediaPipe with Apple Vision VNDetectHumanBodyPose3DRequest (iOS 17+).
//
// For each audio-detected contact, extracts ±10 frames (21 frames total) from
// the video and runs pose estimation on each. The frame with the highest total
// joint confidence is selected as the canonical contact frame. ALL frames are
// stored in windowFrames for skeleton animation in ShotDetailView.
//
// Joint mapping (Apple Vision → our string keys, matching Python LANDMARK_MAP):
//   .leftShoulder / .rightShoulder → "left_shoulder" / "right_shoulder"
//   .leftElbow    / .rightElbow    → "left_elbow"    / "right_elbow"
//   .leftWrist    / .rightWrist    → "left_wrist"    / "right_wrist"
//   .leftHip      / .rightHip     → "left_hip"      / "right_hip"
//   .root                          → "root"
//   (synthetic) midpoint of hips   → "pelvis"  (mirrors Python _add_synthetic())
//
// NOTE: VNHumanBodyPose3DObservation is a subclass of VNHumanBodyPoseObservation,
// so we can also call the 2D recognizedPoint() API for pixel-space coordinates.
// Vision 2D coords have Y=0 at the bottom of the image; we flip to UIKit convention.
//
// See docs/architecture.md Component 2 and docs/implementation_v3.md §3.3

import Foundation
import Vision
import AVFoundation

// MARK: - FrameData

/// One frame in the pose window: frame index + 3D joint positions.
/// rawJoints holds camera-space coords when created by PoseEstimator;
/// ProcessingPipeline Stage 4 replaces them with pelvis-centred transformed coords.
struct FrameData {
    let frameIndex: Int
    let rawJoints:  [String: SIMD3<Float>]
}

// MARK: - PoseEstimator

struct PoseEstimator {

    // MARK: - Joint mapping (3D)

    /// Apple Vision 3D joint names → our canonical string keys.
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
        // ±10 frames around the audio-detected contact (21 frames total)
        let windowIndices = (-10...10).map { contact.frameIndex + $0 }
        let times = windowIndices.map { idx in
            CMTime(seconds: max(0, Double(idx) / frameRate), preferredTimescale: 600)
        }

        // Extract all frames in one batch call
        let frames = await extractFrames(generator: generator, times: times)

        // Run pose on each frame; collect all results and track the best
        var bestJoints:      [String: SIMD3<Float>] = [:]
        var bestImagePoints: [String: CGPoint]       = [:]
        var bestQuality:     Float = -1
        var bestFrameIndex   = contact.frameIndex
        var windowFrames:    [FrameData]             = []

        for (i, (_, cgImage)) in frames.enumerated() {
            guard let image = cgImage else { continue }
            let (joints, imagePoints, quality) = runPose(on: image)
            if joints.isEmpty { continue }

            // Collect every frame that yielded pose data
            windowFrames.append(FrameData(frameIndex: windowIndices[i], rawJoints: joints))

            if quality > bestQuality {
                bestQuality      = quality
                bestJoints       = joints
                bestImagePoints  = imagePoints
                bestFrameIndex   = windowIndices[i]
            }
        }

        // Pick the dominant-wrist image point; Pipeline Stage 4 will select correct side.
        // Store both; Pipeline picks based on dominantSide setting.
        let wristImagePoint = bestImagePoints["right_wrist"] ?? bestImagePoints["left_wrist"]

        return ProcessedShot(
            timestamp:         Double(bestFrameIndex) / frameRate,
            frameIndex:        bestFrameIndex,
            frameRate:         frameRate,
            audioConfidence:   contact.confidence,
            joints:            bestJoints,
            transformedJoints: [:],         // filled in by ProcessingPipeline Stage 4
            windowFrames:      windowFrames,
            wristImagePoint:   wristImagePoint
        )
    }

    // MARK: - Frame extraction

    /// Batch-extract CGImages at the requested times using the async callback API.
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
    ///
    /// Returns (joints3D, imagePoints2D, quality).
    /// - joints3D: world-space 3D positions keyed by our string names.
    /// - imagePoints2D: normalized screen-space (0–1, Y flipped for UIKit) for wrists.
    /// - quality: number of joints detected; used to pick the best frame.
    private func runPose(
        on image: CGImage
    ) -> (joints: [String: SIMD3<Float>], imagePoints: [String: CGPoint], quality: Float) {
        let request = VNDetectHumanBodyPose3DRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])

        do {
            try handler.perform([request])
        } catch {
            return ([:], [:], 0)
        }

        guard let observation = request.results?.first else { return ([:], [:], 0) }

        // 3D joints
        var joints: [String: SIMD3<Float>] = [:]
        for (visionName, key) in Self.jointKeyMap {
            guard let joint = try? observation.recognizedPoint(visionName) else { continue }
            let col = joint.position.columns.3
            joints[key] = SIMD3<Float>(col.x, col.y, col.z)
        }

        // Synthesise pelvis = midpoint of hips (mirrors Python's _add_synthetic())
        if let l = joints["left_hip"], let r = joints["right_hip"] {
            joints["pelvis"] = (l + r) * 0.5
        }

        // 2D image-space wrist positions (for ArcOverlay — no projection needed)
        // VNHumanBodyPose3DObservation inherits from VNHumanBodyPoseObservation,
        // giving access to the 2D recognizedPoint() API.
        // Vision Y=0 is at the bottom of the image; flip for UIKit (Y=0 at top).
        var imagePoints: [String: CGPoint] = [:]
        let obs2D = observation as VNHumanBodyPoseObservation
        if let rw = try? obs2D.recognizedPoint(.rightWrist), rw.confidence > 0.3 {
            imagePoints["right_wrist"] = CGPoint(x: CGFloat(rw.x), y: 1.0 - CGFloat(rw.y))
        }
        if let lw = try? obs2D.recognizedPoint(.leftWrist), lw.confidence > 0.3 {
            imagePoints["left_wrist"] = CGPoint(x: CGFloat(lw.x), y: 1.0 - CGFloat(lw.y))
        }

        return (joints, imagePoints, Float(joints.count))
    }
}
