// SkeletonRenderer.swift
// TennisContact — iOS
//
// Python reference: python/src/visualization_3d.py
// SceneKit SCNScene with a stick-figure skeleton built from joint positions.
// Dominant wrist node glows orange at the contact frame.
// Pelvis shown as a white reference crosshair.
// User can drag to rotate the 3D view freely (allowsCameraControl = true).
//
// Coordinate system matches CoordinateTransform output:
//   X = lateral (positive = dominant hand side)
//   Y = vertical (positive = upward)
//   Z = depth (positive = toward camera)
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

        // Default camera: eye-level, slight offset to the side for 3/4 view
        let cameraNode = SCNNode()
        cameraNode.name = "defaultCamera"
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0.3, 0.7, 2.0)
        let lookAt = SCNLookAtConstraint(target: nil)
        lookAt.isGimbalLockEnabled = true
        // Point toward body centre (slightly above pelvis)
        cameraNode.look(at: SCNVector3(0, 0.6, 0))
        scnView.scene?.rootNode.addChildNode(cameraNode)

        context.coordinator.scnView = scnView
        context.coordinator.rebuild(joints: joints, dominantSide: dominantSide)
        return scnView
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        context.coordinator.rebuild(joints: joints, dominantSide: dominantSide)
    }

    // MARK: - Coordinator

    final class Coordinator {
        var scnView: SCNView?

        // Bone pairs — both endpoints must be present to draw the cylinder
        private static let bones: [(String, String)] = [
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
        ]

        func rebuild(joints: [String: SIMD3<Float>], dominantSide: DominantSide) {
            guard let scene = scnView?.scene else { return }

            // Remove previous skeleton geometry, keep camera
            scene.rootNode.childNodes
                .filter { $0.name?.hasPrefix("jt_") == true
                       || $0.name?.hasPrefix("bn_") == true
                       || $0.name == "xhair_x"
                       || $0.name == "xhair_z" }
                .forEach { $0.removeFromParentNode() }

            guard !joints.isEmpty else { return }

            let wristKey = dominantSide == .right ? "right_wrist" : "left_wrist"

            // Joint spheres
            for (key, pos) in joints {
                let sphere = SCNSphere(radius: 0.025)
                let mat = SCNMaterial()
                if key == wristKey {
                    // Dominant wrist: glowing orange
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

            // Bone cylinders
            for (a, b) in Coordinator.bones {
                guard let pa = joints[a], let pb = joints[b] else { continue }
                if let boneNode = boneCylinder(from: pa, to: pb, name: "bn_\(a)_\(b)") {
                    scene.rootNode.addChildNode(boneNode)
                }
            }

            // Pelvis crosshair
            if let pelvis = joints["pelvis"] {
                addCrosshair(to: scene, at: pelvis)
            }
        }

        // MARK: Helpers

        private func scnPos(_ v: SIMD3<Float>) -> SCNVector3 {
            SCNVector3(v.x, v.y, -v.z)   // flip Z: OpenGL convention (into screen = negative)
        }

        private func boneCylinder(
            from a: SIMD3<Float>,
            to b: SIMD3<Float>,
            name: String
        ) -> SCNNode? {
            let diff = b - a
            let length = simd_length(diff)
            guard length > 0.005 else { return nil }

            let cylinder = SCNCylinder(radius: 0.01, height: CGFloat(length))
            let mat = SCNMaterial()
            mat.diffuse.contents = UIColor(white: 0.65, alpha: 1.0)
            cylinder.materials = [mat]

            let node = SCNNode(geometry: cylinder)
            node.name = name

            // Midpoint
            let mid = (a + b) * 0.5
            node.position = scnPos(mid)

            // Orient cylinder (Y-axis by default) to point along diff
            let dir = simd_normalize(diff)
            // SCNNode Y axis in world space after applying scnPos flip on Z:
            // Since we flip Z in scnPos, we also need to flip the diff.z
            let scnDir = SIMD3<Float>(dir.x, dir.y, -dir.z)
            let yAxis  = SIMD3<Float>(0, 1, 0)
            let cross  = simd_cross(yAxis, scnDir)
            let crossLen = simd_length(cross)
            if crossLen < 1e-5 {
                if scnDir.y < 0 {
                    node.eulerAngles = SCNVector3(Float.pi, 0, 0)
                }
            } else {
                let angle = acos(max(-1, min(1, simd_dot(yAxis, scnDir))))
                let axis  = simd_normalize(cross)
                node.rotation = SCNVector4(axis.x, axis.y, axis.z, angle)
            }
            return node
        }

        private func addCrosshair(to scene: SCNScene, at pos: SIMD3<Float>) {
            let white = UIColor.white

            // X-axis bar
            let barX = SCNBox(width: 0.30, height: 0.005, length: 0.005, chamferRadius: 0)
            let matX = SCNMaterial()
            matX.diffuse.contents = white
            barX.materials = [matX]
            let xNode = SCNNode(geometry: barX)
            xNode.name = "xhair_x"
            xNode.position = scnPos(pos)
            scene.rootNode.addChildNode(xNode)

            // Z-axis bar (maps to depth; shown as Z in SceneKit after flip)
            let barZ = SCNBox(width: 0.005, height: 0.005, length: 0.30, chamferRadius: 0)
            let matZ = SCNMaterial()
            matZ.diffuse.contents = white
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
