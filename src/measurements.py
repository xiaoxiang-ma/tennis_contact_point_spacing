"""Compute contact point measurements relative to body landmarks."""

import numpy as np
from typing import Dict, Tuple


CM_PER_INCH = 2.54


def compute_measurements(landmarks: Dict[str, np.ndarray],
                         contact_point: np.ndarray) -> Dict[str, float]:
    """Compute spatial measurements of contact point relative to body.

    Assumes landmarks are already in pelvis-centered, ground-adjusted
    coordinates (meters). Contact point is the wrist position.

    Coordinate convention (MediaPipe world landmarks):
        x: lateral (positive = player's left)
        y: vertical (positive = down, but we flip so positive = up)
        z: depth (positive = toward camera / behind player)

    Args:
        landmarks: Dict of landmark name -> np.array([x, y, z]) in meters.
        contact_point: np.array([x, y, z]) of the contact wrist in meters.

    Returns:
        Dict with measurement names as keys and values in cm.
        Also includes '_inches' suffixed versions.
    """
    pelvis = landmarks.get("pelvis", np.zeros(3))
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")

    # Contact point relative to pelvis (should already be centered, but be safe)
    rel = contact_point - pelvis

    # Lateral offset (x-axis): positive = to player's left
    lateral_cm = rel[0] * 100.0

    # Forward/back (z-axis): negative z = in front of player in MediaPipe world coords
    forward_back_cm = -rel[2] * 100.0  # positive = in front

    # Height above ground: MediaPipe y is negative upward in world coords
    # After ground-plane adjustment, contact_point y should represent height
    height_cm = -contact_point[1] * 100.0

    results = {
        "lateral_offset_cm": lateral_cm,
        "forward_back_cm": forward_back_cm,
        "height_above_ground_cm": height_cm,
    }

    # Shoulder-line distance
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        shoulder_vec = right_shoulder - left_shoulder
        shoulder_len = np.linalg.norm(shoulder_vec)

        if shoulder_len > 1e-6:
            shoulder_dir = shoulder_vec / shoulder_len
            to_contact = contact_point - shoulder_mid
            # Perpendicular distance from shoulder line
            proj = np.dot(to_contact, shoulder_dir) * shoulder_dir
            perp = to_contact - proj
            shoulder_line_dist_cm = np.linalg.norm(perp) * 100.0
        else:
            shoulder_line_dist_cm = np.linalg.norm(contact_point - shoulder_mid) * 100.0

        results["shoulder_line_distance_cm"] = shoulder_line_dist_cm

        # Contact height relative to shoulder height
        shoulder_height = -shoulder_mid[1] * 100.0
        results["relative_to_shoulder_height_cm"] = height_cm - shoulder_height

    # Add inch conversions
    inch_results = {}
    for key, val in results.items():
        inch_key = key.replace("_cm", "_inches")
        inch_results[inch_key] = val / CM_PER_INCH
    results.update(inch_results)

    return results
