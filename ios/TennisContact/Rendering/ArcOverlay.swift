// ArcOverlay.swift
// TennisContact — iOS
//
// Python reference: python/src/visualization.py (2D annotation approach)
// CAShapeLayer subclass that draws the dominant wrist contact marker over the
// video panel.  A glowing orange ring marks the contact point position.
//
// imagePoint is a normalized screen-space CGPoint (0–1, UIKit convention, Y=0 at
// top) obtained directly from VNHumanBodyPoseObservation.recognizedPoint() in
// PoseEstimator — no 3D→2D projection is needed.
//
// V1: draws a single ring at the contact wrist position.
// Future: extend to draw the full swing arc from the 90-frame wrist path.
//
// See docs/implementation_v3.md Section 3.3 and Task 8

import QuartzCore
import UIKit

final class ArcOverlay: CAShapeLayer {

    // MARK: - Public API

    /// Update the overlay to show the contact ring at the dominant wrist position.
    ///
    /// - Parameters:
    ///   - imagePoint: Normalized (0–1) screen position, UIKit convention (Y=0 at top).
    ///                 Comes directly from Vision 2D landmark — no projection required.
    ///   - viewSize: The size of the video view in points.
    func update(imagePoint: CGPoint, viewSize: CGSize) {
        sublayers?.forEach { $0.removeFromSuperlayer() }
        path = nil
        frame = CGRect(origin: .zero, size: viewSize)

        guard viewSize.width > 0, viewSize.height > 0 else { return }

        // Direct mapping: Vision normalized → pixel position
        let px = imagePoint.x * viewSize.width
        let py = imagePoint.y * viewSize.height
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

        // Centre dot
        let dot = CAShapeLayer()
        dot.path = UIBezierPath(
            arcCenter:  CGPoint(x: px, y: py),
            radius:     3,
            startAngle: 0,
            endAngle:   2 * .pi,
            clockwise:  true
        ).cgPath
        dot.fillColor = UIColor.orange.cgColor
        addSublayer(dot)
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
