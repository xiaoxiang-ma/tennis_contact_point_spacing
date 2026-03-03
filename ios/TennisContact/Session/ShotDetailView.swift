// ShotDetailView.swift
// TennisContact — iOS
//
// Primary UX: split-screen with original video on the left (AVPlayer + wrist arc
// CALayer overlay) and draggable 3D skeleton on the right (SceneKit), both synced
// to a shared playback timeline.
// See docs/implementation_v3.md Section 2.3 and Task 8

import SwiftUI

struct ShotDetailView: View {
    var body: some View {
        // TODO (Task 8): AVPlayer left panel + SceneKit right panel, shared timeline
        Text("Shot Detail — Split Screen")
    }
}

#Preview {
    ShotDetailView()
}
