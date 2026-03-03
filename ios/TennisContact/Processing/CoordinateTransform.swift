// CoordinateTransform.swift
// TennisContact — iOS
//
// Python reference: python/utils/coordinate_transforms.py
// Ports pelvis_origin_transform(), estimate_ground_plane(), apply_ground_plane()
// using simd. Adds handedness normalisation (new vs. Python).
//
// Output coordinate system (pelvis-centered, camera-angle invariant):
//   X = lateral  (positive = toward dominant hand side)
//   Y = vertical (positive = upward)
//   Z = depth    (positive = toward camera, i.e. behind player)
//
// After pelvisOriginTransform, the dominant wrist always has positive X
// regardless of handedness, so all downstream measurements are consistent.
//
// See docs/architecture.md Component 3 and docs/implementation_v3.md §3.4

import Foundation
import simd

// MARK: - DominantSide

/// The player's dominant (racket) hand.
/// Stored in UserDefaults during onboarding (Task 14).
/// Used here to normalise the lateral axis so dominant wrist = positive X.
enum DominantSide: String, Codable, CaseIterable {
    case right
    case left
}

// MARK: - CoordinateTransform

struct CoordinateTransform {

    // MARK: Pelvis-origin transform

    /// Translate all joints so the pelvis is at the origin.
    ///
    /// For left-handed players the X axis is negated so that the dominant
    /// wrist always has positive X — mirroring Python's convention where
    /// consistent laterality is assumed for all measurements.
    ///
    /// If `"pelvis"` is absent from `joints`, the dict is returned unchanged.
    static func pelvisOriginTransform(
        _ joints: [String: SIMD3<Float>],
        dominantSide: DominantSide = .right
    ) -> [String: SIMD3<Float>] {
        guard let pelvis = joints["pelvis"] else { return joints }

        let xSign: Float = dominantSide == .left ? -1 : 1

        return joints.mapValues { pos in
            let centered = pos - pelvis
            return SIMD3<Float>(centered.x * xSign, centered.y, centered.z)
        }
    }

    // MARK: Ground-plane estimation

    /// Estimate the ground-plane y-value from ankle landmarks.
    ///
    /// In camera space Y increases upward, so ankle joints have the smallest
    /// (most negative) Y values after pelvis centering. The ground reference
    /// is the mean ankle Y.  Returns 0 if no ankle landmarks are present.
    ///
    /// Mirrors Python's `estimate_ground_plane()` which returns the mean
    /// ankle z-value in MediaPipe world coordinates.
    static func estimateGroundPlane(joints: [String: SIMD3<Float>]) -> Float {
        let ankleKeys = ["left_ankle", "right_ankle"]
        let ankleY = ankleKeys.compactMap { joints[$0]?.y }
        guard !ankleY.isEmpty else { return 0 }
        return ankleY.reduce(0, +) / Float(ankleY.count)
    }

    // MARK: Apply ground plane

    /// Shift all joints so the estimated ground plane sits at y = 0.
    ///
    /// Subtracts `groundY` from each joint's y component, matching Python's
    /// `apply_ground_plane()` which subtracts `ground_z` from each z.
    static func applyGroundPlane(
        joints: [String: SIMD3<Float>],
        groundY: Float
    ) -> [String: SIMD3<Float>] {
        joints.mapValues { pos in
            SIMD3<Float>(pos.x, pos.y - groundY, pos.z)
        }
    }
}
