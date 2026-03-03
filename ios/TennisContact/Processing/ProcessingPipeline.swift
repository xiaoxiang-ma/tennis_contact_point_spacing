// ProcessingPipeline.swift
// TennisContact — iOS
//
// Orchestrates all four processing components in sequence:
//   AudioDetector → PoseEstimator → CoordinateTransform → StatisticsEngine
//
// Runs asynchronously on a background thread and emits four-stage progress
// updates to the UI without blocking the main thread.
//
// Stage 1/4: Extracting audio track
// Stage 2/4: Detecting contact frames
// Stage 3/4: Running body pose estimation
// Stage 4/4: Computing statistics and swing arcs
//
// See docs/implementation_v3.md Section 3.1 and Task 5

import Foundation

actor ProcessingPipeline {
    // TODO (Task 5): implement process(videoURL:) async throws -> [Shot]
    // with AsyncStream<ProcessingStage> progress updates
}

enum ProcessingStage {
    case extractingAudio
    case detectingContacts
    case estimatingPose
    case computingStatistics
    case complete
}
