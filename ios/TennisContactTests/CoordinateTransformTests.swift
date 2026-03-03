// CoordinateTransformTests.swift
// TennisContactTests
//
// Unit tests for CoordinateTransform: pelvis centering, handedness mirroring,
// ground-plane estimation, and ground-plane application.
//
// Key invariant verified by these tests:
//   A left-handed player's joints, when processed with dominantSide = .left,
//   produce measurements identical in magnitude to a mirror-symmetric
//   right-handed player's joints processed with dominantSide = .right.

import XCTest
import simd
@testable import TennisContact

final class CoordinateTransformTests: XCTestCase {

    // MARK: - pelvisOriginTransform

    func testPelvisBecomesOrigin() {
        let joints: [String: SIMD3<Float>] = [
            "pelvis":      SIMD3(1.0, 2.0, 3.0),
            "right_wrist": SIMD3(1.3, 2.0, 3.0),
        ]
        let result = CoordinateTransform.pelvisOriginTransform(joints, dominantSide: .right)

        XCTAssertEqual(result["pelvis"]!, .zero, accuracy: 1e-5)
    }

    func testJointsShiftedByPelvis() {
        let joints: [String: SIMD3<Float>] = [
            "pelvis":      SIMD3(1.0, 2.0, 3.0),
            "right_wrist": SIMD3(1.3, 2.1, 3.2),
        ]
        let result = CoordinateTransform.pelvisOriginTransform(joints, dominantSide: .right)
        let wrist = result["right_wrist"]!

        XCTAssertEqual(wrist.x,  0.3, accuracy: 1e-5)
        XCTAssertEqual(wrist.y,  0.1, accuracy: 1e-5)
        XCTAssertEqual(wrist.z,  0.2, accuracy: 1e-5)
    }

    func testNoPelvisReturnsUnchanged() {
        let joints: [String: SIMD3<Float>] = [
            "right_wrist": SIMD3(0.3, 0.1, 0.5),
        ]
        let result = CoordinateTransform.pelvisOriginTransform(joints, dominantSide: .right)
        XCTAssertEqual(result["right_wrist"]!, SIMD3(0.3, 0.1, 0.5), accuracy: 1e-5)
    }

    // MARK: - Handedness mirroring

    func testRightHandedNoXFlip() {
        // Right wrist is to the right of pelvis → positive X after centering.
        let joints: [String: SIMD3<Float>] = [
            "pelvis":      SIMD3(0, 0, 0),
            "right_wrist": SIMD3(0.3, 0, 0),
        ]
        let result = CoordinateTransform.pelvisOriginTransform(joints, dominantSide: .right)
        XCTAssertGreaterThan(result["right_wrist"]!.x, 0,
            "Right wrist should be on positive X for a right-handed player")
    }

    func testLeftHandedFlipsXAxis() {
        // Left wrist is to the LEFT of pelvis → negative X before flip.
        // After flip it should be positive, matching the right-handed convention.
        let joints: [String: SIMD3<Float>] = [
            "pelvis":     SIMD3(0, 0, 0),
            "left_wrist": SIMD3(-0.3, 0, 0),
        ]
        let result = CoordinateTransform.pelvisOriginTransform(joints, dominantSide: .left)
        XCTAssertGreaterThan(result["left_wrist"]!.x, 0,
            "Left wrist should be on positive X after left-handed mirroring")
    }

    func testMirrorSymmetry() {
        // Right-handed player with right wrist at (+0.3, 0.1, 0.5) relative to pelvis.
        // Mirror-symmetric left-handed player has left wrist at (−0.3, 0.1, 0.5).
        // After their respective transforms, lateral (X) magnitude should be equal.
        let rightJoints: [String: SIMD3<Float>] = [
            "pelvis":      .zero,
            "right_wrist": SIMD3( 0.3, 0.1, 0.5),
        ]
        let leftJoints: [String: SIMD3<Float>] = [
            "pelvis":     .zero,
            "left_wrist": SIMD3(-0.3, 0.1, 0.5),
        ]

        let rightResult = CoordinateTransform.pelvisOriginTransform(rightJoints, dominantSide: .right)
        let leftResult  = CoordinateTransform.pelvisOriginTransform(leftJoints,  dominantSide: .left)

        let rightX = rightResult["right_wrist"]!.x
        let leftX  = leftResult["left_wrist"]!.x

        XCTAssertEqual(leftX, rightX, accuracy: 1e-5,
            "Mirror-symmetric left-handed contact should produce the same lateral measurement")
    }

    func testYAndZNotAffectedByHandednessFlip() {
        // Only X is flipped; Y and Z should be unchanged.
        let joints: [String: SIMD3<Float>] = [
            "pelvis":     .zero,
            "left_wrist": SIMD3(-0.3, 0.1, 0.5),
        ]
        let result = CoordinateTransform.pelvisOriginTransform(joints, dominantSide: .left)
        let wrist = result["left_wrist"]!

        XCTAssertEqual(wrist.y, 0.1, accuracy: 1e-5, "Y should not be flipped")
        XCTAssertEqual(wrist.z, 0.5, accuracy: 1e-5, "Z should not be flipped")
    }

    // MARK: - estimateGroundPlane

    func testGroundPlaneAveragesAnkleY() {
        let joints: [String: SIMD3<Float>] = [
            "pelvis":       .zero,
            "left_ankle":   SIMD3(0, -1.0, 0),
            "right_ankle":  SIMD3(0, -1.2, 0),
        ]
        let ground = CoordinateTransform.estimateGroundPlane(joints: joints)
        XCTAssertEqual(ground, -1.1, accuracy: 1e-5)
    }

    func testGroundPlaneOneAnkle() {
        let joints: [String: SIMD3<Float>] = [
            "left_ankle": SIMD3(0, -0.9, 0),
        ]
        let ground = CoordinateTransform.estimateGroundPlane(joints: joints)
        XCTAssertEqual(ground, -0.9, accuracy: 1e-5)
    }

    func testGroundPlaneNoAnklesReturnsZero() {
        let joints: [String: SIMD3<Float>] = [
            "pelvis": .zero,
        ]
        XCTAssertEqual(CoordinateTransform.estimateGroundPlane(joints: joints), 0)
    }

    func testGroundPlaneEmptyReturnsZero() {
        XCTAssertEqual(CoordinateTransform.estimateGroundPlane(joints: [:]), 0)
    }

    // MARK: - applyGroundPlane

    func testApplyGroundPlaneShiftsY() {
        let joints: [String: SIMD3<Float>] = [
            "left_ankle": SIMD3(0, -1.1, 0),
            "pelvis":     SIMD3(0,  0.0, 0),
        ]
        let shifted = CoordinateTransform.applyGroundPlane(joints: joints, groundY: -1.1)

        XCTAssertEqual(shifted["left_ankle"]!.y, 0.0, accuracy: 1e-5,
            "Ankle should be at y=0 after ground plane applied")
        XCTAssertEqual(shifted["pelvis"]!.y, 1.1, accuracy: 1e-5,
            "Pelvis should be 1.1 units above ground")
    }

    func testApplyGroundPlaneDoesNotTouchXOrZ() {
        let joints: [String: SIMD3<Float>] = [
            "pelvis": SIMD3(0.5, 0.0, 0.3),
        ]
        let shifted = CoordinateTransform.applyGroundPlane(joints: joints, groundY: -1.0)
        XCTAssertEqual(shifted["pelvis"]!.x, 0.5, accuracy: 1e-5)
        XCTAssertEqual(shifted["pelvis"]!.z, 0.3, accuracy: 1e-5)
    }

    // MARK: - Full pipeline

    func testFullPipelineRightHanded() {
        // Simulate raw pelvis-centered data; verify pelvis ends at y = groundOffset above ground.
        var joints: [String: SIMD3<Float>] = [
            "pelvis":      SIMD3(0,  0.0, 0),
            "right_wrist": SIMD3(0.3, 0.5, 0),
            "left_ankle":  SIMD3(0, -1.0, 0),
            "right_ankle": SIMD3(0, -1.0, 0),
        ]

        joints = CoordinateTransform.pelvisOriginTransform(joints, dominantSide: .right)
        let groundY = CoordinateTransform.estimateGroundPlane(joints: joints)
        joints = CoordinateTransform.applyGroundPlane(joints: joints, groundY: groundY)

        XCTAssertEqual(joints["left_ankle"]!.y,  0.0, accuracy: 1e-5)
        XCTAssertEqual(joints["right_ankle"]!.y, 0.0, accuracy: 1e-5)
        XCTAssertGreaterThan(joints["pelvis"]!.y, 0,
            "Pelvis should be above ground after ground plane is applied")
    }
}

// MARK: - SIMD3 XCTest helpers

private func XCTAssertEqual(
    _ a: SIMD3<Float>,
    _ b: SIMD3<Float>,
    accuracy: Float,
    _ message: String = "",
    file: StaticString = #file,
    line: UInt = #line
) {
    XCTAssertEqual(a.x, b.x, accuracy: accuracy, "x: \(message)", file: file, line: line)
    XCTAssertEqual(a.y, b.y, accuracy: accuracy, "y: \(message)", file: file, line: line)
    XCTAssertEqual(a.z, b.z, accuracy: accuracy, "z: \(message)", file: file, line: line)
}
