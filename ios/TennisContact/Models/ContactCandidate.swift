// ContactCandidate.swift
// TennisContact — iOS
//
// Value type representing a candidate contact moment produced by AudioDetector.
// Carries the timestamp, frame index, and composite confidence score.
// See docs/architecture.md Data Models

import Foundation

struct ContactCandidate {
    /// Seconds from the start of the video.
    let timestamp: TimeInterval
    /// Video frame number corresponding to the timestamp.
    let frameIndex: Int
    /// Composite score 0.0–1.0 from shape analysis
    /// (20% amplitude + 40% narrowness + 40% symmetry).
    let confidence: Float
}
