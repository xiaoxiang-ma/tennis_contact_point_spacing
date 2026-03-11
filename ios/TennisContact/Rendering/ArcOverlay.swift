// ArcOverlay.swift
// TennisContact — iOS
//
// Python reference: python/src/visualization.py (2D annotation approach)
// CAShapeLayer subclass that draws the dominant wrist contact marker over the
// video panel.  A glowing orange ring marks the contact point position.
//
// V1: draws a single ring at the projected wrist position.
// Future: extend to draw the full swing arc from the 90-frame wrist path.
//
// See docs/implementation_v3.md Section 3.3 and Task 8

import QuartzCore
import UIKit
import simd

final class ArcOverlay: CAShapeLayer {

    // MARK: - Public API

    /// Update the overlay to show the contact ring at the dominant wrist position.
    ///
    /// - Parameters:
    ///   - wristPosition: Pelvis-centred, ground-adjusted dominant wrist (meters).
    ///   - viewSize: The size of the video view in points.
    func update(wristPosition: SIMD3<Float>, viewSize: CGSize) {
        // Clear previous drawings
        sublayers?.forEach { $0.removeFromSuperlayer() }
        path = nil
        frame = CGRect(origin: .zero, size: viewSize)

        guard viewSize.width > 0, viewSize.height > 0 else { return }

        let (px, py) = project(wristPosition, into: viewSize)
        let ringRadius: CGFloat = 20

        // Outer glow layer
        let glow = CAShapeLayer()
        glow.path = UIBezierPath(
            arcCenter:  CGPoint(x: px, y: py),
            radius:     ringRadius + 4,
            startAngle: 0,
            endAngle:   2 * .pi,
            clockwise:  true
        ).cgPath
        glow.fillColor   = UIColor.clear.cgColor
        glow.strokeColor = UIColor(red: 1.0, green: 0.42, blue: 0.0, alpha: 0.35).cgColor
        glow.lineWidth   = 8
        addSublayer(glow)

        // Inner ring
        let ring = CAShapeLayer()
        ring.path = UIBezierPath(
            arcCenter:  CGPoint(x: px, y: py),
            radius:     ringRadius,
            startAngle: 0,
            endAngle:   2 * .pi,
            clockwise:  true
        ).cgPath
        ring.fillColor    = UIColor.clear.cgColor
        ring.strokeColor  = UIColor(red: 1.0, green: 0.42, blue: 0.0, alpha: 0.95).cgColor
        ring.lineWidth    = 3
        ring.shadowColor  = UIColor.orange.cgColor
        ring.shadowRadius = 6
        ring.shadowOpacity = 0.9
        ring.shadowOffset  = .zero
        addSublayer(ring)

        // Crosshair dot at centre
        let dotRadius: CGFloat = 3
        let dot = CAShapeLayer()
        dot.path = UIBezierPath(
            arcCenter:  CGPoint(x: px, y: py),
            radius:     dotRadius,
            startAngle: 0,
            endAngle:   2 * .pi,
            clockwise:  true
        ).cgPath
        dot.fillColor = UIColor.orange.cgColor
        addSublayer(dot)
    }

    // MARK: - Projection

    /// Simple linear projection from pelvis-centred world coordinates to pixel space.
    ///
    /// Assumes the player occupies roughly ±0.6 m laterally and 0–2.0 m vertically
    /// from the pelvis, and maps this range onto the full video frame.
    private func project(_ pos: SIMD3<Float>, into size: CGSize) -> (CGFloat, CGFloat) {
        // X: lateral.  Map [-0.7, 0.7] → [0, width].
        let xRange: Float = 1.4
        let px = CGFloat((pos.x + 0.7) / xRange) * size.width

        // Y: vertical.  Map [2.0, -0.3] → [0, height]  (screen Y inverted vs. world Y).
        let yTop:    Float =  2.0
        let yBottom: Float = -0.3
        let yRange         = yTop - yBottom
        let py = CGFloat((yTop - pos.y) / yRange) * size.height

        return (px, py)
    }
}

// MARK: - ArcOverlayHostView

/// UIView that hosts an ArcOverlay as a sublayer, for use in UIViewRepresentable.
final class ArcOverlayHostView: UIView {
    let arcLayer = ArcOverlay()

    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        backgroundColor = .clear
        layer.addSublayer(arcLayer)
    }

    required init?(coder: NSCoder) { super.init(coder: coder) }

    override func layoutSubviews() {
        super.layoutSubviews()
        arcLayer.frame = bounds
    }
}
