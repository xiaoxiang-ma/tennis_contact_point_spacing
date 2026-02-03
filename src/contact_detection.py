"""Detect ball-racket contact frames from ball trajectory and pose data."""

import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_ball_velocity(detections: List[Tuple[int, float, float, float]],
                          fps: float) -> List[Tuple[int, float, float, float]]:
    """Compute ball velocity magnitude between consecutive detections.

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
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Detect contact frames from ball trajectory and optional wrist positions.

    Uses velocity direction reversal, speed spikes, and acceleration spikes.
    Wrist proximity is a soft confidence boost (not a hard filter).
    """
    velocities = compute_ball_velocity(ball_detections, fps)
    if len(velocities) < 3:
        return []

    speeds = np.array([v[3] for v in velocities])
    # Use 75th percentile of nonzero speeds as reference, since median
    # can be 0 when HSV locks onto a static object most of the time
    nonzero_speeds = speeds[speeds > 1e-3]
    if len(nonzero_speeds) < 3:
        if debug:
            print(f"  [debug] Only {len(nonzero_speeds)} nonzero velocity samples, not enough")
        return []

    ref_speed = np.percentile(nonzero_speeds, 75)

    if debug:
        print(f"  [debug] {len(velocities)} velocity samples")
        print(f"  [debug] {len(nonzero_speeds)} nonzero, ref speed (p75): {ref_speed:.1f} px/s")
        print(f"  [debug] speed range: {speeds.min():.1f} - {speeds.max():.1f} px/s")

    if ref_speed < 1e-3:
        return []

    # Compute acceleration
    accels = []
    for i in range(1, len(velocities)):
        f0, vx0, vy0, s0 = velocities[i - 1]
        f1, vx1, vy1, s1 = velocities[i]
        dt = (f1 - f0) / fps if fps > 0 else 1.0
        if dt <= 0:
            continue
        ax = (vx1 - vx0) / dt
        ay = (vy1 - vy0) / dt
        accels.append((f1, np.sqrt(ax**2 + ay**2)))

    accel_vals = np.array([a[1] for a in accels]) if accels else np.array([0.0])
    nonzero_accels = accel_vals[accel_vals > 1e-3]
    ref_accel = np.percentile(nonzero_accels, 75) if len(nonzero_accels) > 0 else 0.0
    accel_by_frame = {a[0]: a[1] for a in accels}

    if debug:
        print(f"  [debug] ref accel (p75): {ref_accel:.1f} px/s²")

    ball_pos = {d[0]: (d[1], d[2]) for d in ball_detections}

    candidates = []
    for i in range(1, len(velocities)):
        f_prev, vx0, vy0, s0 = velocities[i - 1]
        f_curr, vx1, vy1, s1 = velocities[i]

        # Signal 1: Direction reversal
        dot = vx0 * vx1 + vy0 * vy1
        direction_reversal = dot < 0

        # Signal 2: Speed spike (compared to ref_speed, not median)
        speed_spike = s1 > ref_speed * velocity_spike_threshold

        # Signal 3: Acceleration spike
        accel = accel_by_frame.get(f_curr, 0.0)
        accel_spike = (ref_accel > 1e-3 and
                       accel > ref_accel * velocity_spike_threshold)

        if not (direction_reversal or speed_spike or accel_spike):
            continue

        conf = 0.0
        if direction_reversal:
            conf += 0.35
        if speed_spike:
            conf += 0.25 * min(s1 / (ref_speed * velocity_spike_threshold), 2.0) / 2.0
        if accel_spike and ref_accel > 1e-3:
            conf += 0.25 * min(accel / (ref_accel * velocity_spike_threshold), 2.0) / 2.0

        # Wrist proximity as soft boost
        if wrist_positions is not None and f_curr in ball_pos and f_curr in wrist_positions:
            bx, by = ball_pos[f_curr]
            wx, wy = wrist_positions[f_curr]
            dist = np.sqrt((bx - wx) ** 2 + (by - wy) ** 2)
            if dist < wrist_proximity_px:
                conf += 0.2 * max(0, 1.0 - dist / wrist_proximity_px)

        if conf < 0.2:
            continue

        conf = min(conf, 1.0)
        candidates.append((f_curr, conf))

        if debug:
            print(f"  [debug] candidate frame {f_curr}: conf={conf:.2f}, "
                  f"reversal={direction_reversal}, speed_spike={speed_spike}, "
                  f"accel_spike={accel_spike}, speed={s1:.0f}")

    if debug:
        print(f"  [debug] {len(candidates)} candidates before suppression")

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
