// SkeletonRenderer.swift
// TennisContact — iOS
//
// Python reference: python/src/visualization_3d.py
// SceneKit SCNScene with a stick-figure skeleton built from joint positions.
// Dominant wrist node glows orange at the contact frame.
// Pelvis shown as a white reference crosshair.
// User can drag to rotate the 3D view freely (allowsCameraControl = true).
//
// Animated use: ShotDetailView drives `joints` from a periodic time observer.
// updateUIView calls updatePositions() (moves existing nodes in-place) rather
// than rebuild() to avoid per-frame node churn at 60fps.
//
// Coordinate system matches CoordinateTransform output:
//   X = lateral (positive = dominant hand side)
//   Y = vertical (positive = upward)
//   Z = depth (positive = toward camera)
// SceneKit uses right-hand coords with Y up; we flip Z (negate) on entry.
//
// See docs/implementation_v3.md Section 3.3 and Task 9

import SceneKit
import simd
import SwiftUI

// MARK: - SkeletonRenderer

struct SkeletonRenderer: UIViewRepresentable {
    let joints: [String: SIMD3<Float>]
    let dominantSide: DominantSide

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = SCNScene()
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = UIColor(red: 0.11, green: 0.11, blue: 0.12, alpha: 1.0)
        scnView.antialiasingMode = .multisampling2X

        let cameraNode = SCNNode()
        cameraNode.name = "defaultCamera"
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0.3, 0.7, 2.0)
        cameraNode.look(at: SCNVector3(0, 0.6, 0))
        scnView.scene?.rootNode.addChildNode(cameraNode)

        context.coordinator.scnView = scnView
        context.coordinator.rebuild(joints: joints, dominantSide: dominantSide)
        return scnView
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        // Efficient in-place update when skeleton already exists; full rebuild otherwise.
        if context.coordinator.hasBuiltSkeleton {
            context.coordinator.updatePositions(joints: joints, dominantSide: dominantSide)
        } else {
            context.coordinator.rebuild(joints: joints, dominantSide: dominantSide)
        }
    }

    // MARK: - Coordinator

    final class Coordinator {
        var scnView: SCNView?
        var hasBuiltSkeleton = false

        static let bones: [(String, String)] = [
            // Upper body
            ("left_shoulder",  "right_shoulder"),
            ("left_shoulder",  "left_elbow"),
            ("left_elbow",     "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow",    "right_wrist"),
            ("left_hip",       "right_hip"),
            ("left_shoulder",  "left_hip"),
            ("right_shoulder", "right_hip"),
            ("pelvis",         "left_hip"),
            ("pelvis",         "right_hip"),
            ("neck",           "left_shoulder"),
            ("neck",           "right_shoulder"),
            ("head",           "neck"),
            // Legs
            ("left_hip",       "left_knee"),
            ("right_hip",      "right_knee"),
            ("left_knee",      "left_ankle"),
            ("right_knee",     "right_ankle"),
        ]

        // MARK: Full rebuild

        func rebuild(joints: [String: SIMD3<Float>], dominantSide: DominantSide) {
            guard let scene = scnView?.scene else { return }

            scene.rootNode.childNodes
                .filter { $0.name?.hasPrefix("jt_") == true
                       || $0.name?.hasPrefix("bn_") == true
                       || $0.name == "xhair_x"
                       || $0.name == "xhair_z" }
                .forEach { $0.removeFromParentNode() }

            hasBuiltSkeleton = false
            guard !joints.isEmpty else { return }

            let wristKey = dominantSide == .right ? "right_wrist" : "left_wrist"

            for (key, pos) in joints {
                let sphere = SCNSphere(radius: 0.025)
                let mat = SCNMaterial()
                if key == wristKey {
                    mat.diffuse.contents  = UIColor(red: 1.0, green: 0.42, blue: 0.0, alpha: 1.0)
                    mat.emission.contents = UIColor(red: 1.0, green: 0.42, blue: 0.0, alpha: 0.7)
                } else {
                    mat.diffuse.contents = UIColor.white
                }
                sphere.materials = [mat]
                let node = SCNNode(geometry: sphere)
                node.name = "jt_\(key)"
                node.position = scnPos(pos)
                scene.rootNode.addChildNode(node)
            }

            for (a, b) in Coordinator.bones {
                guard let pa = joints[a], let pb = joints[b] else { continue }
                if let boneNode = makeBone(from: pa, to: pb, name: "bn_\(a)_\(b)") {
                    scene.rootNode.addChildNode(boneNode)
                }
            }

            if let pelvis = joints["pelvis"] {
                addCrosshair(to: scene, at: pelvis)
            }

            hasBuiltSkeleton = true
        }

        // MARK: In-place position update (called every video frame)

        func updatePositions(joints: [String: SIMD3<Float>], dominantSide: DominantSide) {
            guard let scene = scnView?.scene, !joints.isEmpty else { return }

            // Move joint spheres
            for (key, pos) in joints {
                scene.rootNode.childNode(withName: "jt_\(key)", recursively: false)?
                    .position = scnPos(pos)
            }

            // Update bone cylinders in-place
            for (a, b) in Coordinator.bones {
                guard let pa = joints[a], let pb = joints[b],
                      let node = scene.rootNode.childNode(
                          withName: "bn_\(a)_\(b)", recursively: false)
                else { continue }
                updateBone(node: node, from: pa, to: pb)
            }

            // Move crosshair
            if let pelvis = joints["pelvis"] {
                let p = scnPos(pelvis)
                scene.rootNode.childNode(withName: "xhair_x", recursively: false)?.position = p
                scene.rootNode.childNode(withName: "xhair_z", recursively: false)?.position = p
            }
        }

        // MARK: Helpers

        func scnPos(_ v: SIMD3<Float>) -> SCNVector3 {
            SCNVector3(v.x, v.y, -v.z)   // flip Z for SceneKit right-hand convention
        }

        private func makeBone(
            from a: SIMD3<Float>,
            to b: SIMD3<Float>,
            name: String
        ) -> SCNNode? {
            let diff   = b - a
            let length = simd_length(diff)
            guard length > 0.005 else { return nil }

            let cylinder = SCNCylinder(radius: 0.01, height: CGFloat(length))
            let mat = SCNMaterial()
            mat.diffuse.contents = UIColor(white: 0.65, alpha: 1.0)
            cylinder.materials = [mat]

            let node = SCNNode(geometry: cylinder)
            node.name = name
            node.position = scnPos((a + b) * 0.5)
            applyOrientation(to: node, from: a, to: b)
            return node
        }

        private func updateBone(node: SCNNode, from a: SIMD3<Float>, to b: SIMD3<Float>) {
            let diff   = b - a
            let length = simd_length(diff)
            guard length > 0.005 else { return }

            node.position = scnPos((a + b) * 0.5)
            if let cyl = node.geometry as? SCNCylinder {
                cyl.height = CGFloat(length)
            }
            applyOrientation(to: node, from: a, to: b)
        }

        private func applyOrientation(to node: SCNNode,
                                      from a: SIMD3<Float>, to b: SIMD3<Float>) {
            let diff   = b - a
            let length = simd_length(diff)
            guard length > 0.005 else { return }

            let dir    = simd_normalize(diff)
            // Apply same Z flip used in scnPos
            let scnDir = SIMD3<Float>(dir.x, dir.y, -dir.z)
            let yAxis  = SIMD3<Float>(0, 1, 0)
            let cross  = simd_cross(yAxis, scnDir)
            let crossLen = simd_length(cross)

            if crossLen < 1e-5 {
                node.eulerAngles = scnDir.y < 0
                    ? SCNVector3(Float.pi, 0, 0)
                    : SCNVector3(0, 0, 0)
            } else {
                let angle = acos(max(-1, min(1, simd_dot(yAxis, scnDir))))
                let axis  = simd_normalize(cross)
                node.rotation = SCNVector4(axis.x, axis.y, axis.z, angle)
            }
        }

        private func addCrosshair(to scene: SCNScene, at pos: SIMD3<Float>) {
            let white = UIColor.white

            let barX = SCNBox(width: 0.30, height: 0.005, length: 0.005, chamferRadius: 0)
            let matX = SCNMaterial(); matX.diffuse.contents = white
            barX.materials = [matX]
            let xNode = SCNNode(geometry: barX)
            xNode.name = "xhair_x"
            xNode.position = scnPos(pos)
            scene.rootNode.addChildNode(xNode)

            let barZ = SCNBox(width: 0.005, height: 0.005, length: 0.30, chamferRadius: 0)
            let matZ = SCNMaterial(); matZ.diffuse.contents = white
            barZ.materials = [matZ]
            let zNode = SCNNode(geometry: barZ)
            zNode.name = "xhair_z"
            zNode.position = scnPos(pos)
            scene.rootNode.addChildNode(zNode)
        }
    }
}

// MARK: - Preview

#Preview {
    SkeletonRenderer(
        joints: [
            "pelvis":          SIMD3( 0.00,  0.00,  0.00),
            "left_hip":        SIMD3(-0.10, -0.05,  0.00),
            "right_hip":       SIMD3( 0.10, -0.05,  0.00),
            "left_shoulder":   SIMD3(-0.20,  0.45,  0.00),
            "right_shoulder":  SIMD3( 0.20,  0.45,  0.00),
            "neck":            SIMD3( 0.00,  0.50,  0.00),
            "head":            SIMD3( 0.00,  0.62,  0.00),
            "left_elbow":      SIMD3(-0.32,  0.25,  0.00),
            "right_elbow":     SIMD3( 0.36,  0.20,  0.10),
            "left_wrist":      SIMD3(-0.38,  0.05,  0.00),
            "right_wrist":     SIMD3( 0.52,  0.10,  0.20),
        ],
        dominantSide: .right
    )
    .ignoresSafeArea()
}
