// ArcOverlay.swift
// TennisContact — iOS
//
// Python reference: python/src/visualization.py (2D annotation approach)
// CAShapeLayer subclass that draws the dominant wrist arc over the video panel.
// Projects 3D pelvis-relative wrist path coordinates into 2D pixel space.
// A glowing ring marks the contact frame position on the arc.
//
// See docs/implementation_v3.md Section 3.3 and Task 8

import QuartzCore
import UIKit

final class ArcOverlay: CAShapeLayer {
    // TODO (Task 8): implement update(wristPath:contactIndex:videoSize:)
    // wristPath: projected 2D CGPoints for each frame in the swing window
    // contactIndex: which point in wristPath to highlight with the glowing ring
}
