"""Coordinate transformation utilities for pose data."""

import numpy as np
from typing import Dict, Tuple


def pelvis_origin_transform(landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Transform all landmarks so pelvis is at the origin.

    Args:
        landmarks: Dict mapping landmark name to (x, y, z) array.

    Returns:
        New dict with all landmarks shifted so pelvis = (0, 0, 0).
    """
    pelvis = landmarks.get("pelvis")
    if pelvis is None:
        return landmarks

    return {name: coords - pelvis for name, coords in landmarks.items()}


def estimate_ground_plane(landmarks: Dict[str, np.ndarray]) -> float:
    """Estimate ground plane z-value from ankle landmarks.

    Uses the average y-coordinate of ankle landmarks as the ground reference
    (in image/MediaPipe coordinates, y increases downward, so ankles have
    the largest y values among body landmarks).

    Args:
        landmarks: Dict mapping landmark name to (x, y, z) array.

    Returns:
        Ground plane z-value (the lowest point, used as z=0 reference).
    """
    ankle_keys = ["left_ankle", "right_ankle"]
    ankle_positions = [landmarks[k] for k in ankle_keys if k in landmarks]

    if not ankle_positions:
        return 0.0

    # Ground is at the minimum z (or maximum y in image coords).
    # In our pelvis-centered system, ground z is the ankle z-value.
    return float(np.mean([pos[2] for pos in ankle_positions]))


def apply_ground_plane(landmarks: Dict[str, np.ndarray], ground_z: float) -> Dict[str, np.ndarray]:
    """Shift landmarks so that the ground plane is at z=0.

    Args:
        landmarks: Pelvis-centered landmarks dict.
        ground_z: The z-value that should become 0.

    Returns:
        New dict with z-values adjusted so ground = 0.
    """
    offset = np.array([0.0, 0.0, ground_z])
    return {name: coords - offset for name, coords in landmarks.items()}
