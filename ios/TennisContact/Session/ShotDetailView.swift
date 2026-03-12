// ShotDetailView.swift
// TennisContact — iOS
//
// Shot review screen: video on top, scrub slider in the middle, animated 3D
// skeleton on the bottom. A single Slider is the source of truth for both panels —
// dragging it seeks the video and updates the skeleton joints simultaneously,
// eliminating the looper/observer jitter of the previous approach.
//
// Layout (portrait iPhone, landscape video):
//   ┌─────────────────────────┐
//   │   VideoPlayer (16:9)    │  ← tap for AVKit controls (play/pause)
//   ├─────────────────────────┤
//   │  ◄────●────►  −3 frames │  ← Slider: drives both panels
//   ├─────────────────────────┤
//   │  SkeletonRenderer       │  ← draggable 3D; contact-frame stats bar below
//   └─────────────────────────┘
//
// See docs/implementation_v3.md Section 2.3 and Task 8

import AVKit
import SwiftUI
import simd

// MARK: - ShotDetailView

struct ShotDetailView: View {
    let shot: ProcessedShot
    let dominantSide: DominantSide

    // Precomputed once in init
    private let sortedFrames: [FrameData]   // window frames sorted by frameIndex
    private let sliderMax: Double           // sortedFrames.count - 1

    // State
    @State private var player: AVPlayer
    @State private var sliderValue: Double
    @State private var currentJoints: [String: SIMD3<Float>]

    // MARK: - Init

    init(shot: ProcessedShot, videoURL: URL, dominantSide: DominantSide) {
        self.shot = shot
        self.dominantSide = dominantSide

        let sorted = shot.windowFrames.sorted { $0.frameIndex < $1.frameIndex }
        self.sortedFrames = sorted
        self.sliderMax = Double(max(sorted.count - 1, 1))

        // Slider starts at the contact frame position within the window
        let contactIdx = sorted.firstIndex(where: { $0.frameIndex == shot.frameIndex })
                         ?? (sorted.count / 2)

        _player       = State(initialValue: AVPlayer(url: videoURL))
        _sliderValue  = State(initialValue: Double(contactIdx))
        _currentJoints = State(initialValue: shot.transformedJoints)
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            videoPanel
            sliderPanel
            skeletonPanel
        }
        .navigationTitle(shotTitle)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            // Seek video to contact frame on open; leave paused
            let t = CMTime(seconds: shot.timestamp, preferredTimescale: 600)
            player.seek(to: t, toleranceBefore: .zero, toleranceAfter: .zero)
        }
        .onDisappear { player.pause() }
    }

    // MARK: - Video panel

    private var videoPanel: some View {
        VideoPlayer(player: player)
            .aspectRatio(16 / 9, contentMode: .fit)
    }

    // MARK: - Slider panel

    private var sliderPanel: some View {
        VStack(spacing: 4) {
            Slider(value: $sliderValue, in: 0...sliderMax)
                .padding(.horizontal, 16)
                .onChange(of: sliderValue) { _, val in scrub(to: val) }

            Text(frameLabel)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
    }

    private var frameLabel: String {
        guard !sortedFrames.isEmpty else { return "No pose data" }
        let idx   = Int(sliderValue.rounded()).clamped(to: 0...(sortedFrames.count - 1))
        let delta = sortedFrames[idx].frameIndex - shot.frameIndex
        if delta == 0 { return "Contact frame" }
        return delta > 0 ? "+\(delta) frames" : "\(delta) frames"
    }

    // MARK: - Skeleton panel

    private var skeletonPanel: some View {
        VStack(spacing: 0) {
            SkeletonRenderer(joints: currentJoints, dominantSide: dominantSide)

            let wristKey = dominantSide == .right ? "right_wrist" : "left_wrist"
            if let wrist = currentJoints[wristKey] {
                contactStatsBar(wrist: wrist)
            }
        }
    }

    private func contactStatsBar(wrist: SIMD3<Float>) -> some View {
        HStack(spacing: 0) {
            statCell(label: "Forward", value: wrist.z * 100)
            Divider().frame(height: 36)
            statCell(label: "Lateral", value: wrist.x * 100)
            Divider().frame(height: 36)
            statCell(label: "Height",  value: wrist.y * 100)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color(.secondarySystemBackground))
    }

    private func statCell(label: String, value: Float) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(String(format: "%.1f cm", value))
                .font(.caption.monospacedDigit().bold())
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Scrub

    /// Seeks the video and updates skeleton joints to match the slider position.
    /// Called synchronously on slider change — no observer latency.
    private func scrub(to val: Double) {
        guard !sortedFrames.isEmpty else { return }
        let idx   = Int(val.rounded()).clamped(to: 0...(sortedFrames.count - 1))
        let frame = sortedFrames[idx]
        let fps   = shot.frameRate > 0 ? shot.frameRate : 30.0

        // Half-frame tolerance: fast enough for smooth scrubbing, precise enough for review
        let halfFrame = CMTime(value: 1, timescale: CMTimeScale(fps * 2))
        let t = CMTime(seconds: Double(frame.frameIndex) / fps, preferredTimescale: 600)
        player.seek(to: t, toleranceBefore: halfFrame, toleranceAfter: halfFrame)

        if !frame.rawJoints.isEmpty {
            currentJoints = frame.rawJoints
        }
    }

    // MARK: - Helpers

    private var shotTitle: String {
        let t = shot.timestamp
        let m = Int(t) / 60
        let s = Int(t) % 60
        return m > 0
            ? "Shot at \(m):\(String(format: "%02d", s))"
            : "Shot at 0:\(String(format: "%02d", s))"
    }
}

// MARK: - Int clamping

private extension Int {
    func clamped(to range: ClosedRange<Int>) -> Int {
        Swift.max(range.lowerBound, Swift.min(range.upperBound, self))
    }
}

// MARK: - Preview

#Preview {
    let contactJoints: [String: SIMD3<Float>] = [
        "pelvis":         SIMD3( 0.00,  0.00,  0.00),
        "left_hip":       SIMD3(-0.10, -0.05,  0.00),
        "right_hip":      SIMD3( 0.10, -0.05,  0.00),
        "left_shoulder":  SIMD3(-0.20,  0.45,  0.00),
        "right_shoulder": SIMD3( 0.20,  0.45,  0.00),
        "neck":           SIMD3( 0.00,  0.50,  0.00),
        "head":           SIMD3( 0.00,  0.62,  0.00),
        "left_elbow":     SIMD3(-0.32,  0.25,  0.00),
        "right_elbow":    SIMD3( 0.36,  0.20,  0.10),
        "left_wrist":     SIMD3(-0.38,  0.05,  0.00),
        "right_wrist":    SIMD3( 0.52,  0.10,  0.20),
    ]

    NavigationStack {
        ShotDetailView(
            shot: ProcessedShot(
                timestamp: 5.0,
                frameIndex: 300,
                frameRate: 60.0,
                audioConfidence: 0.85,
                joints: [:],
                transformedJoints: contactJoints,
                windowFrames: [
                    FrameData(frameIndex: 298, rawJoints: contactJoints),
                    FrameData(frameIndex: 299, rawJoints: contactJoints),
                    FrameData(frameIndex: 300, rawJoints: contactJoints),
                    FrameData(frameIndex: 301, rawJoints: contactJoints),
                    FrameData(frameIndex: 302, rawJoints: contactJoints),
                ],
                wristImagePoint: nil
            ),
            videoURL: URL(string: "about:blank")!,
            dominantSide: .right
        )
    }
}
