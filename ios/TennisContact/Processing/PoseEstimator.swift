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
//   .leftKnee     / .rightKnee    → "left_knee"     / "right_knee"
//   .leftAnkle    / .rightAnkle   → "left_ankle"    / "right_ankle"
//   .root                          → "root"
//   .centerShoulder                → "neck"
//   .head                          → "head"
//   (synthetic) midpoint of hips   → "pelvis"  (mirrors Python _add_synthetic())
//
// Frames are extracted sequentially with copyCGImage(at:actualTime:) so each
// CGImage is released immediately after Vision processes it — prevents OOM.
// Frames before video start (idx < 0) are skipped to avoid CMTime collisions.
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
        .centerShoulder: "neck",
        .head:           "head",
        .leftShoulder:   "left_shoulder",
        .rightShoulder:  "right_shoulder",
        .leftElbow:      "left_elbow",
        .rightElbow:     "right_elbow",
        .leftWrist:      "left_wrist",
        .rightWrist:     "right_wrist",
        .root:           "root",
        .leftHip:        "left_hip",
        .rightHip:       "right_hip",
        .leftKnee:       "left_knee",
        .rightKnee:      "right_knee",
        .leftAnkle:      "left_ankle",
        .rightAnkle:     "right_ankle",
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
        var bestJoints:    [String: SIMD3<Float>] = [:]
        var bestQuality:   Float = -1
        var bestFrameIndex = contact.frameIndex
        var windowFrames:  [FrameData] = []

        // Process one frame at a time so each CGImage is released before the next loads.
        // Skip frames before video start (idx < 0) to avoid CMTime collisions at t=0.
        for offset in -10...10 {
            let idx = contact.frameIndex + offset
            guard idx >= 0 else { continue }

            let t = CMTime(seconds: Double(idx) / frameRate, preferredTimescale: 600)
            guard let cgImage = try? generator.copyCGImage(at: t, actualTime: nil) else { continue }

            let (joints, quality) = runPose(on: cgImage)
            // cgImage goes out of scope here — released immediately

            guard !joints.isEmpty else { continue }

            windowFrames.append(FrameData(frameIndex: idx, rawJoints: joints))

            if quality > bestQuality {
                bestQuality    = quality
                bestJoints     = joints
                bestFrameIndex = idx
            }
        }

        return ProcessedShot(
            timestamp:         Double(bestFrameIndex) / frameRate,
            frameIndex:        bestFrameIndex,
            frameRate:         frameRate,
            audioConfidence:   contact.confidence,
            joints:            bestJoints,
            transformedJoints: [:],         // filled in by ProcessingPipeline Stage 4
            windowFrames:      windowFrames,
            wristImagePoint:   nil
        )
    }

    // MARK: - Pose inference

    /// Run pose estimation on a single CGImage.
    ///
    /// Returns (joints3D, quality).
    /// - joints3D: world-space 3D positions keyed by our string names.
    /// - quality: number of 3D joints detected; used to pick the best frame.
    private func runPose(
        on image: CGImage
    ) -> (joints: [String: SIMD3<Float>], quality: Float) {
        let request3D = VNDetectHumanBodyPose3DRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])

        do {
            try handler.perform([request3D])
        } catch {
            return ([:], 0)
        }

        guard let observation = request3D.results?.first else { return ([:], 0) }

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

        return (joints, Float(joints.count))
    }
}
