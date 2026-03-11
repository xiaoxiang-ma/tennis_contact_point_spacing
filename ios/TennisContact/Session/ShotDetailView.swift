// ShotDetailView.swift
// TennisContact — iOS
//
// Primary UX: split-screen with original video on the left (AVPlayer + wrist
// contact ring overlay) and a draggable 3D skeleton on the right (SceneKit).
//
// Left panel:  VideoPlayer (AVKit) with ArcOverlay CAShapeLayer showing the
//              dominant wrist position as a glowing orange ring derived from the
//              Vision 2D landmark (no projection math — exact pixel position).
// Right panel: SkeletonRenderer animated in sync with the video via a periodic
//              time observer that updates joint positions each frame.
//
// Windowed looping: AVQueuePlayer + AVPlayerLooper clips to ±10 frames around the
// contact timestamp and loops that window continuously.
//
// Tap the video panel to toggle play/pause. Both sides pause/resume together.
//
// See docs/implementation_v3.md Section 2.3 and Task 8

import AVKit
import SwiftUI
import simd

// MARK: - ShotDetailView

struct ShotDetailView: View {
    let shot: ProcessedShot
    let videoURL: URL
    let dominantSide: DominantSide

    // AVQueuePlayer + looper for windowed looping
    @State private var player: AVQueuePlayer
    @State private var looper: AVPlayerLooper?

    // Skeleton animation state — driven by periodic time observer
    @State private var currentJoints: [String: SIMD3<Float>] = [:]
    @State private var timeObserverToken: Any?

    // MARK: - Init

    init(shot: ProcessedShot, videoURL: URL, dominantSide: DominantSide) {
        self.shot = shot
        self.videoURL = videoURL
        self.dominantSide = dominantSide

        // Window: ±10 frames around the contact timestamp
        let fps         = shot.frameRate > 0 ? shot.frameRate : 30.0
        let windowSecs  = 10.0 / fps
        let windowStart = max(0, shot.timestamp - windowSecs)
        let windowEnd   = shot.timestamp + windowSecs

        let item = AVPlayerItem(url: videoURL)
        item.forwardPlaybackEndTime  = CMTime(seconds: windowEnd,   preferredTimescale: 600)
        item.reversePlaybackEndTime  = CMTime(seconds: windowStart, preferredTimescale: 600)

        let queuePlayer = AVQueuePlayer(playerItem: item)
        _player = State(initialValue: queuePlayer)
        _looper = State(initialValue: AVPlayerLooper(player: queuePlayer, templateItem: item))
    }

    // MARK: - Body

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 0) {
                videoPanel
                    .frame(width: geo.size.width / 2, height: geo.size.height)

                Divider()

                skeletonPanel
                    .frame(width: geo.size.width / 2, height: geo.size.height)
            }
        }
        .navigationTitle(shotTitle)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { setupPlayback() }
        .onDisappear { teardownPlayback() }
    }

    // MARK: - Sub-views

    private var videoPanel: some View {
        ZStack {
            VideoPlayer(player: player)
                .onTapGesture { togglePlayback() }

            // Contact ring — position from Vision 2D landmark (no projection)
            if let pt = shot.wristImagePoint {
                ArcOverlayRepresentable(imagePoint: pt)
                    .allowsHitTesting(false)
            }
        }
    }

    private var skeletonPanel: some View {
        VStack(spacing: 0) {
            SkeletonRenderer(
                joints: currentJoints.isEmpty ? shot.transformedJoints : currentJoints,
                dominantSide: dominantSide
            )

            // Per-shot stats bar
            if let wristPos = wristPosition(from: currentJoints.isEmpty
                                             ? shot.transformedJoints : currentJoints) {
                contactStatsBar(wrist: wristPos)
            }
        }
    }

    private func contactStatsBar(wrist: SIMD3<Float>) -> some View {
        HStack(spacing: 0) {
            statCell(label: "Forward",  value: wrist.z * 100)
            Divider().frame(height: 36)
            statCell(label: "Lateral",  value: wrist.x * 100)
            Divider().frame(height: 36)
            statCell(label: "Height",   value: wrist.y * 100)
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

    // MARK: - Lifecycle

    private func setupPlayback() {
        let fps = shot.frameRate > 0 ? shot.frameRate : 30.0
        let windowSecs = 10.0 / fps
        let windowStart = max(0, shot.timestamp - windowSecs)

        // Seek to window start then play
        let seekTime = CMTime(seconds: windowStart, preferredTimescale: 600)
        player.seek(to: seekTime, toleranceBefore: .zero, toleranceAfter: .zero) { _ in
            self.player.play()
        }

        // Start with the contact-frame skeleton
        currentJoints = shot.transformedJoints

        // Periodic observer: update skeleton joints each video frame
        let interval = CMTime(value: 1, timescale: CMTimeScale(fps))
        timeObserverToken = player.addPeriodicTimeObserver(
            forInterval: interval, queue: .main
        ) { [fps] time in
            let fi = Int((time.seconds * fps).rounded())
            if let frame = shot.windowFrames.first(where: { $0.frameIndex == fi }),
               !frame.rawJoints.isEmpty {
                currentJoints = frame.rawJoints
            }
        }
    }

    private func teardownPlayback() {
        if let token = timeObserverToken {
            player.removeTimeObserver(token)
            timeObserverToken = nil
        }
        player.pause()
    }

    private func togglePlayback() {
        if player.timeControlStatus == .playing {
            player.pause()
        } else {
            player.play()
        }
    }

    // MARK: - Helpers

    private func wristPosition(from joints: [String: SIMD3<Float>]) -> SIMD3<Float>? {
        let key = dominantSide == .right ? "right_wrist" : "left_wrist"
        return joints[key]
    }

    private var shotTitle: String {
        let t = shot.timestamp
        let m = Int(t) / 60
        let s = Int(t) % 60
        return m > 0
            ? "Shot at \(m):\(String(format: "%02d", s))"
            : "Shot at 0:\(String(format: "%02d", s))"
    }
}

// MARK: - ArcOverlayRepresentable

/// Wraps ArcOverlayHostView (UIView + CAShapeLayer) for SwiftUI.
/// imagePoint is normalized (0–1), Y=0 at top (UIKit convention).
private struct ArcOverlayRepresentable: UIViewRepresentable {
    let imagePoint: CGPoint

    func makeUIView(context: Context) -> ArcOverlayHostView { ArcOverlayHostView() }

    func updateUIView(_ view: ArcOverlayHostView, context: Context) {
        // Defer until the view has been laid out and has a valid non-zero bounds
        DispatchQueue.main.async {
            guard view.bounds.width > 0, view.bounds.height > 0 else { return }
            view.arcLayer.update(imagePoint: self.imagePoint, viewSize: view.bounds.size)
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        ShotDetailView(
            shot: ProcessedShot(
                timestamp: 5.0,
                frameIndex: 300,
                frameRate: 60.0,
                audioConfidence: 0.85,
                joints: [:],
                transformedJoints: [
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
                ],
                windowFrames: [],
                wristImagePoint: CGPoint(x: 0.65, y: 0.45)
            ),
            videoURL: URL(string: "about:blank")!,
            dominantSide: .right
        )
    }
}
