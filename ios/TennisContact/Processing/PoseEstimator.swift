// PoseEstimator.swift
// TennisContact — iOS
//
// Python reference: python/src/pose_estimation.py
// Replaces MediaPipe with Apple Vision VNDetectHumanBodyPose3DRequest.
//
// Processing window: ±5 frames around each audio contact timestamp (11 frames).
// Selects the frame with highest joint confidence as the canonical contact frame.
//
// Joint mapping (Apple Vision → our names):
//   .leftShoulder  / .rightShoulder   — shoulder line
//   .leftElbow     / .rightElbow
//   .leftWrist     / .rightWrist      — contact point (dominant wrist is primary)
//   .leftHip       / .rightHip        — pelvis derivation (midpoint)
//   .head
//
// See docs/architecture.md Component 2 and docs/implementation_v3.md Section 3.3

import Foundation
import Vision

struct PoseEstimator {
    // TODO (Task 5): implement joints(videoURL:contactTimestamps:) -> [[String: simd_float3]]
}
