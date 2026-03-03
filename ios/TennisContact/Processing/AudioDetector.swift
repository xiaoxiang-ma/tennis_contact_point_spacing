// AudioDetector.swift
// TennisContact — iOS
//
// Python reference: python/src/audio_detection.py
// Ports detect_contacts_audio() using AVFoundation (audio extraction) and
// Accelerate vDSP (bandpass filter + envelope + peak detection).
//
// Pipeline:
//   AVAssetReader → PCM Float32 (44.1kHz)
//   → vDSP bandpass (1–4 kHz)
//   → amplitude envelope (5ms window)
//   → adaptive threshold (75th percentile × 3.0)
//   → shape scoring: 20% amplitude + 40% narrowness + 40% symmetry
//   → NMS (300ms minimum gap)
//   → [ContactCandidate]
//
// Validation: output timestamps must agree with Python within ±50ms.
// See docs/architecture.md Component 1 and docs/implementation_v3.md Section 3.2

import Foundation
import AVFoundation
import Accelerate

struct AudioDetector {
    // MARK: — Parameters (must match Python defaults)
    static let lowFreq: Float       = 1_000   // Hz
    static let highFreq: Float      = 4_000   // Hz
    static let thresholdFactor: Float = 3.0
    static let noisePercentile: Float = 75.0  // percent
    static let minGapMs: Float      = 300     // ms
    static let maxImpactFwhmMs: Float = 40    // ms
    // Scoring weights
    static let wAmplitude: Float    = 0.20
    static let wNarrowness: Float   = 0.40
    static let wSymmetry: Float     = 0.40

    // TODO (Task 3): implement detect(videoURL:) -> [ContactCandidate]
}
