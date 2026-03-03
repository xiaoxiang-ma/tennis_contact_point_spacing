// StatisticsEngine.swift
// TennisContact — iOS
//
// Python reference: python/src/measurements.py
// Ports compute_measurements() to produce per-shot and session-level analytics.
//
// Per shot:
//   contactForward  = wrist.X - pelvis.X   (positive = in front)
//   contactLateral  = wrist.Y - pelvis.Y   (positive = toward dominant side)
//   contactHeight   = wrist.Z              (absolute height from ground)
//   wristVelocity   = |Δwrist| / Δtime    (across ±3 frames)
//
// Session aggregates:
//   consistencyScore = 100 × exp(−0.5 × (σForward² + σLateral² + σHeight²))
//
// See docs/architecture.md Component 4 and docs/implementation_v3.md Section 3.5

import Foundation

struct StatisticsEngine {
    // TODO (Task 10): implement compute(shots:) -> SessionStats
}
