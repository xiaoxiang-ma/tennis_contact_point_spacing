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


def create_diagnostic_video(
    frames: List[np.ndarray],
    ball_detections: Dict[int, Tuple[float, float, float]],
    contacts: List[Tuple[int, float, str]],
    fps: float,
    output_path: str,
    known_contact_frame: Optional[int] = None,
    show_trajectory: bool = True,
    trajectory_tail: int = 30,
) -> None:
    """Create a diagnostic video with ball tracking and detection overlays.

    Args:
        frames: List of BGR frames.
        ball_detections: Dict of frame_num -> (x, y, confidence).
        contacts: List of (frame_num, confidence, source) detected contacts.
        fps: Video frame rate.
        output_path: Path to save output video.
        known_contact_frame: Optional frame number of known real contact (for comparison).
        show_trajectory: Whether to draw ball trajectory trail.
        trajectory_tail: Number of frames of trajectory to show.
    """
    from .contact_detection import compute_ball_velocity

    if not frames:
        raise ValueError("No frames provided")

    h, w = frames[0].shape[:2]

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Build trajectory and velocity data
    trajectory = [
        (f, x, y, c) for f, (x, y, c) in sorted(ball_detections.items())
    ]
    velocities = compute_ball_velocity(trajectory, fps)
    vel_dict = {v[0]: {'vx': v[1], 'vy': v[2], 'speed': v[3]} for v in velocities}

    # Compute reference speed for spike detection
    speeds = [v[3] for v in velocities]
    ref_speed = np.median(speeds) if speeds else 100
    spike_threshold = ref_speed * 2.0

    # Build contact frame set for quick lookup
    contact_frames = {c[0]: (c[1], c[2]) for c in contacts}

    # Process each frame
    for frame_idx, frame in enumerate(frames):
        annotated = frame.copy()
        time_sec = frame_idx / fps

        # --- Draw trajectory tail ---
        if show_trajectory:
            trail_frames = [
                f for f in sorted(ball_detections.keys())
                if frame_idx - trajectory_tail <= f <= frame_idx
            ]
            if len(trail_frames) > 1:
                for i in range(1, len(trail_frames)):
                    f_prev, f_curr = trail_frames[i-1], trail_frames[i]
                    x1, y1, _ = ball_detections[f_prev]
                    x2, y2, _ = ball_detections[f_curr]
                    # Fade color based on age
                    alpha = (i / len(trail_frames))
                    color = (int(100 + 155 * alpha), int(200 * alpha), 0)
                    cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # --- Draw ball position ---
        ball_detected = frame_idx in ball_detections
        if ball_detected:
            bx, by, bconf = ball_detections[frame_idx]
            # Ball circle (green if detected, size based on confidence)
            radius = int(8 + 12 * bconf)
            cv2.circle(annotated, (int(bx), int(by)), radius, (0, 255, 0), 2)
            cv2.circle(annotated, (int(bx), int(by)), 3, (0, 255, 0), -1)

            # Draw velocity vector
            if frame_idx in vel_dict:
                v = vel_dict[frame_idx]
                vx, vy, speed = v['vx'], v['vy'], v['speed']
                # Scale velocity for visualization (normalize to ~50px max)
                scale = 50 / max(ref_speed, 1)
                end_x = int(bx + vx * scale * 0.1)
                end_y = int(by + vy * scale * 0.1)
                cv2.arrowedLine(annotated, (int(bx), int(by)), (end_x, end_y),
                               (255, 255, 0), 2, tipLength=0.3)

        # --- Compute detection signals for this frame ---
        is_spike = False
        is_reversal = False
        is_decel = False
        confidence = 0.0

        if frame_idx in vel_dict:
            v = vel_dict[frame_idx]
            speed = v['speed']

            # Spike check
            if speed > spike_threshold:
                is_spike = True
                confidence += 0.2 * min((speed / ref_speed) / 2.0, 1.5)

            # Find previous velocity for reversal/decel check
            prev_frames = [f for f in vel_dict.keys() if f < frame_idx]
            if prev_frames:
                prev_f = max(prev_frames)
                pv = vel_dict[prev_f]

                # Reversal check
                dot = pv['vx'] * v['vx'] + pv['vy'] * v['vy']
                if dot < 0:
                    mag0 = np.sqrt(pv['vx']**2 + pv['vy']**2)
                    mag1 = np.sqrt(v['vx']**2 + v['vy']**2)
                    cos_angle = dot / (mag0 * mag1 + 1e-6)
                    reversal_strength = max(0, -cos_angle)
                    if reversal_strength > 0.1:
                        is_reversal = True
                        confidence += 0.3 * (0.5 + 0.5 * reversal_strength)

                # Decel check
                if pv['speed'] > ref_speed and speed < pv['speed'] * 0.5:
                    is_decel = True
                    decel_ratio = 1 - (speed / pv['speed'])
                    confidence += 0.2 * decel_ratio

        confidence = min(confidence, 0.7)

        # --- Draw info panel (top-left) ---
        panel_h = 140
        cv2.rectangle(annotated, (0, 0), (320, panel_h), (0, 0, 0), -1)
        cv2.rectangle(annotated, (0, 0), (320, panel_h), (100, 100, 100), 1)

        # Frame info
        _draw_text(annotated, f"Frame: {frame_idx} | {time_sec:.2f}s", (10, 22), scale=0.55)

        # Ball detection status
        ball_status = "DETECTED" if ball_detected else "MISSING"
        ball_color = (0, 255, 0) if ball_detected else (0, 0, 255)
        _draw_text(annotated, f"Ball: {ball_status}", (10, 44), color=ball_color, scale=0.55)

        # Velocity info
        if frame_idx in vel_dict:
            speed = vel_dict[frame_idx]['speed']
            speed_color = (0, 255, 255) if is_spike else (200, 200, 200)
            _draw_text(annotated, f"Speed: {speed:.0f} px/s (ref: {ref_speed:.0f})",
                      (10, 66), color=speed_color, scale=0.5)
        else:
            _draw_text(annotated, "Speed: ---", (10, 66), color=(128, 128, 128), scale=0.5)

        # Detection signals
        signals = []
        if is_spike:
            signals.append(("SPIKE", (0, 255, 255)))
        if is_reversal:
            signals.append(("REVERSAL", (255, 0, 255)))
        if is_decel:
            signals.append(("DECEL", (255, 165, 0)))

        signal_x = 10
        for sig_text, sig_color in signals:
            cv2.rectangle(annotated, (signal_x, 75), (signal_x + 70, 95), sig_color, -1)
            _draw_text(annotated, sig_text, (signal_x + 4, 90), color=(0, 0, 0), scale=0.4)
            signal_x += 75

        # Confidence bar
        _draw_text(annotated, f"Conf: {confidence:.2f}", (10, 115), scale=0.5)
        bar_width = int(200 * confidence / 0.7)
        cv2.rectangle(annotated, (70, 103), (270, 118), (50, 50, 50), -1)
        if bar_width > 0:
            bar_color = (0, 255, 0) if confidence > 0.4 else (0, 255, 255) if confidence > 0.2 else (0, 165, 255)
            cv2.rectangle(annotated, (70, 103), (70 + bar_width, 118), bar_color, -1)

        # --- Highlight contact frames ---
        is_detected_contact = frame_idx in contact_frames
        is_known_contact = known_contact_frame is not None and frame_idx == known_contact_frame

        if is_detected_contact:
            det_conf, det_source = contact_frames[frame_idx]
            # Red border
            cv2.rectangle(annotated, (2, 2), (w-3, h-3), (0, 0, 255), 4)
            _draw_text(annotated, f"DETECTED CONTACT (conf={det_conf:.2f}, {det_source})",
                      (10, 135), color=(0, 0, 255), scale=0.55)

        if is_known_contact:
            # Yellow border for known contact
            border_color = (0, 255, 255)
            cv2.rectangle(annotated, (6, 6), (w-7, h-7), border_color, 4)
            _draw_text(annotated, "<<< KNOWN REAL CONTACT >>>",
                      (w//2 - 130, 30), color=border_color, scale=0.7)

        # --- Reference info (bottom) ---
        _draw_text(annotated, f"Ref speed: {ref_speed:.0f} | Spike thresh: {spike_threshold:.0f}",
                  (10, h - 15), color=(150, 150, 150), scale=0.45)

        out.write(annotated)

    out.release()
    print(f"Diagnostic video saved to: {output_path}")


def _draw_text(frame: np.ndarray, text: str, pos: Tuple[int, int],
               color: Tuple[int, int, int] = (255, 255, 255),
               scale: float = 0.6, thickness: int = 1) -> None:
    """Draw text with black outline for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    # Outline
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
