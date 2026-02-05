"""Compute contact point measurements relative to body landmarks.

Provides both continuous measurements (cm/inches) and categorical labels
for contact point spatial localization.
"""

import numpy as np
from typing import Dict, Tuple, Optional


CM_PER_INCH = 2.54

# Categorical thresholds (in cm, relative to body)
HEIGHT_THRESHOLDS = {
    'low': 60.0,        # Below 60cm from ground = low contact
    'hip_level': 90.0,  # 60-90cm = hip level
    'waist_level': 110.0,  # 90-110cm = waist level
    'chest_level': 140.0,  # 110-140cm = chest level
    # Above 140cm = high contact
}

FORWARD_THRESHOLDS = {
    'behind': -10.0,    # < -10cm = behind body
    'neutral': 10.0,    # -10 to 10cm = neutral
    # > 10cm = forward
}

LATERAL_THRESHOLDS = {
    'far_left': -50.0,   # < -50cm = far left
    'left': -20.0,       # -50 to -20cm = left
    'center': 20.0,      # -20 to 20cm = center
    'right': 50.0,       # 20 to 50cm = right
    # > 50cm = far right
}


def classify_height(height_cm: float) -> str:
    """Classify contact height into categorical label.

    Args:
        height_cm: Height above ground in cm.

    Returns:
        One of: 'low', 'hip_level', 'waist_level', 'chest_level', 'high'
    """
    if height_cm < HEIGHT_THRESHOLDS['low']:
        return 'low'
    elif height_cm < HEIGHT_THRESHOLDS['hip_level']:
        return 'hip_level'
    elif height_cm < HEIGHT_THRESHOLDS['waist_level']:
        return 'waist_level'
    elif height_cm < HEIGHT_THRESHOLDS['chest_level']:
        return 'chest_level'
    else:
        return 'high'


def classify_forward_back(forward_cm: float) -> str:
    """Classify forward/back position into categorical label.

    Args:
        forward_cm: Distance forward from pelvis in cm (positive = forward).

    Returns:
        One of: 'behind', 'neutral', 'forward'
    """
    if forward_cm < FORWARD_THRESHOLDS['behind']:
        return 'behind'
    elif forward_cm < FORWARD_THRESHOLDS['neutral']:
        return 'neutral'
    else:
        return 'forward'


def classify_lateral(lateral_cm: float) -> str:
    """Classify lateral position into categorical label.

    Args:
        lateral_cm: Lateral offset in cm (positive = player's left).

    Returns:
        One of: 'far_left', 'left', 'center', 'right', 'far_right'
    """
    if lateral_cm < LATERAL_THRESHOLDS['far_left']:
        return 'far_left'
    elif lateral_cm < LATERAL_THRESHOLDS['left']:
        return 'left'
    elif lateral_cm < LATERAL_THRESHOLDS['center']:
        return 'center'
    elif lateral_cm < LATERAL_THRESHOLDS['right']:
        return 'right'
    else:
        return 'far_right'


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

    # Add categorical labels
    results["height_category"] = classify_height(height_cm)
    results["forward_category"] = classify_forward_back(forward_back_cm)
    results["lateral_category"] = classify_lateral(lateral_cm)

    # Composite description
    results["contact_zone"] = f"{results['height_category']}_{results['forward_category']}"

    return results


def compute_relative_to_landmarks(
    landmarks: Dict[str, np.ndarray],
    contact_point: np.ndarray,
) -> Dict[str, float]:
    """Compute contact point position relative to specific body landmarks.

    Provides additional measurements relative to shoulder and hip lines.

    Args:
        landmarks: Dict of landmark name -> np.array([x, y, z]) in meters.
        contact_point: np.array([x, y, z]) of the contact point in meters.

    Returns:
        Dict with measurements relative to body landmarks.
    """
    results = {}

    # Get key landmarks
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")

    # Shoulder-relative measurements
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        shoulder_vec = right_shoulder - left_shoulder
        shoulder_width = np.linalg.norm(shoulder_vec)

        # Distance in front of shoulder line (z-axis)
        shoulder_forward_cm = -(contact_point[2] - shoulder_mid[2]) * 100.0
        results["cm_in_front_of_shoulder"] = shoulder_forward_cm
        results["inches_in_front_of_shoulder"] = shoulder_forward_cm / CM_PER_INCH

        # Height relative to shoulder
        shoulder_height_cm = -shoulder_mid[1] * 100.0
        contact_height_cm = -contact_point[1] * 100.0
        results["cm_above_shoulder"] = contact_height_cm - shoulder_height_cm
        results["inches_above_shoulder"] = (contact_height_cm - shoulder_height_cm) / CM_PER_INCH

    # Hip-relative measurements
    if left_hip is not None and right_hip is not None:
        hip_mid = (left_hip + right_hip) / 2.0

        # Distance in front of hip line (z-axis)
        hip_forward_cm = -(contact_point[2] - hip_mid[2]) * 100.0
        results["cm_in_front_of_hip"] = hip_forward_cm
        results["inches_in_front_of_hip"] = hip_forward_cm / CM_PER_INCH

        # Height relative to hip
        hip_height_cm = -hip_mid[1] * 100.0
        contact_height_cm = -contact_point[1] * 100.0
        results["cm_above_hip"] = contact_height_cm - hip_height_cm
        results["inches_above_hip"] = (contact_height_cm - hip_height_cm) / CM_PER_INCH

    return results
