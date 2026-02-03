"""Visualization: skeleton overlay, contact point marker, annotations."""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


# Skeleton connections with color coding (BGR)
SKELETON_CONNECTIONS = [
    # Torso (white)
    ("left_shoulder", "right_shoulder", (255, 255, 255)),
    ("left_hip", "right_hip", (255, 255, 255)),
    ("left_shoulder", "left_hip", (255, 255, 255)),
    ("right_shoulder", "right_hip", (255, 255, 255)),
    # Left arm (blue)
    ("left_shoulder", "left_elbow", (255, 150, 50)),
    ("left_elbow", "left_wrist", (255, 150, 50)),
    # Right arm (blue)
    ("right_shoulder", "right_elbow", (255, 150, 50)),
    ("right_elbow", "right_wrist", (255, 150, 50)),
    # Left leg (green)
    ("left_hip", "left_knee", (50, 220, 50)),
    ("left_knee", "left_ankle", (50, 220, 50)),
    # Right leg (green)
    ("right_hip", "right_knee", (50, 220, 50)),
    ("right_knee", "right_ankle", (50, 220, 50)),
    # Head
    ("nose", "left_shoulder", (200, 200, 200)),
    ("nose", "right_shoulder", (200, 200, 200)),
]


def draw_skeleton(frame: np.ndarray,
                  pixel_landmarks: Dict[str, Tuple[int, int]],
                  thickness: int = 3) -> np.ndarray:
    """Draw skeleton overlay on frame.

    Args:
        frame: BGR image (will be modified in-place and returned).
        pixel_landmarks: Dict of landmark name -> (px, py).
        thickness: Line thickness.

    Returns:
        Annotated frame.
    """
    overlay = frame.copy()

    # Draw connections
    for start, end, color in SKELETON_CONNECTIONS:
        if start in pixel_landmarks and end in pixel_landmarks:
            p1 = pixel_landmarks[start]
            p2 = pixel_landmarks[end]
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)

    # Draw joint circles
    for name, (px, py) in pixel_landmarks.items():
        cv2.circle(overlay, (px, py), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # Blend
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame


def draw_contact_point(frame: np.ndarray,
                       px: int, py: int,
                       radius: int = 10) -> np.ndarray:
    """Draw a red contact point marker."""
    cv2.circle(frame, (px, py), radius, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), radius + 2, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_measurements(frame: np.ndarray,
                      measurements: Dict[str, float],
                      frame_num: int,
                      fps: float,
                      position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Add measurement text annotations to frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    outline_color = (0, 0, 0)
    thickness = 2
    outline_thickness = 4

    timestamp = frame_num / fps if fps > 0 else 0
    lines = [
        f"Frame: {frame_num} | Time: {timestamp:.2f}s",
        f"Lateral: {measurements.get('lateral_offset_cm', 0):.1f}cm "
        f"({measurements.get('lateral_offset_inches', 0):.1f}in)",
        f"Forward: {measurements.get('forward_back_cm', 0):.1f}cm "
        f"({measurements.get('forward_back_inches', 0):.1f}in)",
        f"Height: {measurements.get('height_above_ground_cm', 0):.1f}cm "
        f"({measurements.get('height_above_ground_inches', 0):.1f}in)",
    ]

    if "shoulder_line_distance_cm" in measurements:
        lines.append(
            f"Shoulder dist: {measurements['shoulder_line_distance_cm']:.1f}cm "
            f"({measurements['shoulder_line_distance_inches']:.1f}in)"
        )
    if "relative_to_shoulder_height_cm" in measurements:
        lines.append(
            f"vs Shoulder: {measurements['relative_to_shoulder_height_cm']:.1f}cm "
            f"({measurements['relative_to_shoulder_height_inches']:.1f}in)"
        )

    x, y = position
    for line in lines:
        # Outline
        cv2.putText(frame, line, (x, y), font, font_scale, outline_color,
                     outline_thickness, cv2.LINE_AA)
        # Foreground
        cv2.putText(frame, line, (x, y), font, font_scale, color,
                     thickness, cv2.LINE_AA)
        y += 28

    return frame


def annotate_contact_frame(frame: np.ndarray,
                           pixel_landmarks: Dict[str, Tuple[int, int]],
                           contact_wrist: str,
                           measurements: Dict[str, float],
                           frame_num: int,
                           fps: float) -> np.ndarray:
    """Full annotation pipeline for a single contact frame.

    Args:
        frame: BGR image.
        pixel_landmarks: Landmark pixel positions.
        contact_wrist: "left_wrist" or "right_wrist".
        measurements: From measurements.compute_measurements().
        frame_num: Frame number.
        fps: Video FPS.

    Returns:
        Annotated BGR frame.
    """
    annotated = frame.copy()
    annotated = draw_skeleton(annotated, pixel_landmarks)

    if contact_wrist in pixel_landmarks:
        cx, cy = pixel_landmarks[contact_wrist]
        annotated = draw_contact_point(annotated, cx, cy)

    annotated = draw_measurements(annotated, measurements, frame_num, fps)
    return annotated


def save_annotated_frame(frame: np.ndarray, output_path: str) -> None:
    """Save annotated frame as PNG."""
    cv2.imwrite(output_path, frame)
