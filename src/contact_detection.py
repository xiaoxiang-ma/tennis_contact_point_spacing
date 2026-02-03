"""Detect ball-racket contact frames from ball trajectory and pose data."""

import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_ball_velocity(detections: List[Tuple[int, float, float, float]],
                          fps: float) -> List[Tuple[int, float, float, float]]:
    """Compute ball velocity magnitude between consecutive detections.

    Args:
        detections: List of (frame_num, x, y, confidence).
        fps: Video frame rate.

    Returns:
        List of (frame_num, vx, vy, speed) in pixels/second.
    """
    if len(detections) < 2:
        return []

    velocities = []
    for i in range(1, len(detections)):
        f0, x0, y0, _ = detections[i - 1]
        f1, x1, y1, _ = detections[i]

        dt = (f1 - f0) / fps if fps > 0 else 1.0
        if dt <= 0:
            continue

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        speed = np.sqrt(vx ** 2 + vy ** 2)
        velocities.append((f1, vx, vy, speed))

    return velocities


def detect_contacts(
    ball_detections: List[Tuple[int, float, float, float]],
    fps: float,
    wrist_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    velocity_spike_threshold: float = 2.0,
    wrist_proximity_px: float = 150.0,
    min_frame_gap: int = 10,
) -> List[Tuple[int, float]]:
    """Detect contact frames from ball trajectory and optional wrist positions.

    Contact is detected when:
    1. Ball velocity has a sudden spike (direction reversal or speed change), AND
    2. Ball is near a player's wrist (if wrist positions provided).

    Args:
        ball_detections: From BallTracker.detect_all().
        fps: Video frame rate.
        wrist_positions: Optional dict mapping frame_num to (x, y) pixel
            coordinates of the dominant wrist.
        velocity_spike_threshold: Multiplier over median speed to count as spike.
        wrist_proximity_px: Max distance in pixels from ball to wrist for contact.
        min_frame_gap: Minimum frames between consecutive contacts.

    Returns:
        List of (frame_num, confidence) for detected contact frames,
        sorted by frame number.
    """
    velocities = compute_ball_velocity(ball_detections, fps)
    if len(velocities) < 3:
        return []

    speeds = np.array([v[3] for v in velocities])
    median_speed = np.median(speeds)
    if median_speed < 1e-6:
        return []

    # Build frame->position lookup
    ball_pos = {d[0]: (d[1], d[2]) for d in ball_detections}

    # Detect velocity spikes and direction changes
    candidates = []
    for i in range(1, len(velocities)):
        f_prev, vx0, vy0, s0 = velocities[i - 1]
        f_curr, vx1, vy1, s1 = velocities[i]

        # Direction reversal: dot product of consecutive velocity vectors
        dot = vx0 * vx1 + vy0 * vy1
        direction_reversal = dot < 0

        # Speed spike
        speed_spike = s1 > median_speed * velocity_spike_threshold

        if not (direction_reversal or speed_spike):
            continue

        # Confidence based on how strong the signal is
        conf = 0.5
        if direction_reversal:
            conf += 0.25
        if speed_spike:
            conf += 0.25 * min(s1 / (median_speed * velocity_spike_threshold), 2.0) / 2.0

        # Check wrist proximity if available
        if wrist_positions is not None and f_curr in ball_pos:
            bx, by = ball_pos[f_curr]
            if f_curr in wrist_positions:
                wx, wy = wrist_positions[f_curr]
                dist = np.sqrt((bx - wx) ** 2 + (by - wy) ** 2)
                if dist > wrist_proximity_px:
                    continue
                # Boost confidence if very close
                conf += 0.2 * max(0, 1.0 - dist / wrist_proximity_px)

        conf = min(conf, 1.0)
        candidates.append((f_curr, conf))

    # Suppress detections too close together (keep highest confidence)
    candidates.sort(key=lambda c: c[0])
    contacts = []
    for frame, conf in candidates:
        if contacts and frame - contacts[-1][0] < min_frame_gap:
            if conf > contacts[-1][1]:
                contacts[-1] = (frame, conf)
            continue
        contacts.append((frame, conf))

    return contacts
