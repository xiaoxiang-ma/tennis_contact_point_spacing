// ShotDetailView.swift
// TennisContact — iOS
//
// Primary UX: split-screen with original video on the left (AVPlayer + wrist
// contact ring overlay) and a draggable 3D skeleton on the right (SceneKit).
//
// Left panel:  VideoPlayer (AVKit) with ArcOverlay CAShapeLayer showing the
//              dominant wrist contact position as a glowing orange ring.
// Right panel: SkeletonRenderer with joint positions from transformedJoints.
//
// The video seeks to 1 second before the contact timestamp on appear so the
// contact moment is visible shortly after playback begins.
//
// See docs/implementation_v3.md Section 2.3 and Task 8

import AVKit
import SceneKit
import SwiftUI
import simd

// MARK: - ShotDetailView

struct ShotDetailView: View {
    let shot: ProcessedShot
    let videoURL: URL
    let dominantSide: DominantSide

    @State private var player: AVPlayer

    init(shot: ProcessedShot, videoURL: URL, dominantSide: DominantSide) {
        self.shot = shot
        self.videoURL = videoURL
        self.dominantSide = dominantSide
        _player = State(initialValue: AVPlayer(url: videoURL))
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
        .onAppear {
            let seekSeconds = max(0, shot.timestamp - 1.0)
            let seekTime = CMTime(seconds: seekSeconds, preferredTimescale: 600)
            player.seek(to: seekTime, toleranceBefore: .zero, toleranceAfter: .zero)
        }
        .onDisappear { player.pause() }
    }

    // MARK: - Sub-views

    private var videoPanel: some View {
        ZStack {
            VideoPlayer(player: player)
                .onTapGesture { togglePlayback() }

            // Arc overlay — contact ring drawn over dominant wrist position
            if let wristPos = wristPosition {
                ArcOverlayRepresentable(wristPosition: wristPos)
                    .allowsHitTesting(false)
            }
        }
    }

    private var skeletonPanel: some View {
        VStack(spacing: 0) {
            SkeletonRenderer(
                joints: shot.transformedJoints,
                dominantSide: dominantSide
            )

            // Contact stats bar
            if let wristPos = wristPosition {
                contactStatsBar(wrist: wristPos)
            }
        }
    }

    private func contactStatsBar(wrist: SIMD3<Float>) -> some View {
        HStack(spacing: 0) {
            statCell(label: "Forward",  value: wrist.x * 100)
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

    // MARK: - Helpers

    private var wristPosition: SIMD3<Float>? {
        let key = dominantSide == .right ? "right_wrist" : "left_wrist"
        return shot.transformedJoints[key]
    }

    private var shotTitle: String {
        let t = shot.timestamp
        let m = Int(t) / 60
        let s = Int(t) % 60
        return m > 0 ? "Shot at \(m):\(String(format: "%02d", s))" : "Shot at 0:\(String(format: "%02d", s))"
    }

    private func togglePlayback() {
        if player.timeControlStatus == .playing {
            player.pause()
        } else {
            player.play()
        }
    }
}

// MARK: - ArcOverlayRepresentable

/// Wraps ArcOverlayHostView (UIView + CAShapeLayer) for SwiftUI.
private struct ArcOverlayRepresentable: UIViewRepresentable {
    let wristPosition: SIMD3<Float>

    func makeUIView(context: Context) -> ArcOverlayHostView {
        ArcOverlayHostView()
    }

    func updateUIView(_ view: ArcOverlayHostView, context: Context) {
        // Defer so the view has a valid bounds from its first layout pass
        DispatchQueue.main.async {
            view.arcLayer.update(wristPosition: self.wristPosition,
                                 viewSize: view.bounds.size)
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
                ]
            ),
            videoURL: URL(string: "about:blank")!,
            dominantSide: .right
        )
    }
}
