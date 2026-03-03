// CoordinateTransform.swift
// TennisContact — iOS
//
// Python reference: python/utils/coordinate_transforms.py
// Ports pelvis_origin_transform() and estimate_ground_plane() using simd.
//
// Output coordinate system (pelvis-centered, camera-angle invariant):
//   X = forward  (positive = in front of player's chest)
//   Y = lateral  (positive = toward dominant hand side)
//   Z = vertical (positive = upward)
//
// All measurements are reported in this system so "30cm forward" is
// independent of camera distance or angle.
//
// See docs/architecture.md Component 3 and docs/implementation_v3.md Section 3.4

import Foundation
import simd

struct CoordinateTransform {
    // TODO (Task 6): implement pelvisOriginTransform(_:dominantSide:)
    // TODO (Task 6): implement estimateGroundPlane(joints:) -> simd_float4
}
