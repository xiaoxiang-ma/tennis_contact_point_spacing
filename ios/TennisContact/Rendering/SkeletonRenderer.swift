// SkeletonRenderer.swift
// TennisContact — iOS
//
// Python reference: python/src/visualization_3d.py
// SceneKit SCNScene with 17-joint stick figure.
// Dominant wrist node glows at the contact frame.
// Pelvis shown as a white reference crosshair.
// User can drag to rotate the 3D view freely.
//
// See docs/implementation_v3.md Section 3.3 and Task 9

import SceneKit
import SwiftUI

struct SkeletonRenderer: UIViewRepresentable {
    // TODO (Task 9): accept joint positions [String: simd_float3] and frameIndex
    // TODO (Task 9): animate SCNNode positions along the stored joint arrays

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = SCNScene()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}
