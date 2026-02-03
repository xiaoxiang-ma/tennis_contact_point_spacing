"""Detect ball-racket contact frames from wrist kinematics and/or ball trajectory."""

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


def detect_contacts_wrist(
    wrist_positions: Dict[int, Tuple[float, float]],
    fps: float,
    min_frame_gap: int = 18,
    speed_percentile: float = 85.0,
    decel_ratio: float = 0.5,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Detect contact frames from wrist kinematics alone.

    During a groundstroke the wrist accelerates through the hitting zone
    then decelerates sharply at or just after contact.  We look for frames
    where wrist speed is high AND followed by a rapid deceleration.

    Args:
        wrist_positions: Dict frame_num -> (x, y) pixel coords.
        fps: Video frame rate.
        min_frame_gap: Minimum frames between contacts.
        speed_percentile: Wrist speed must exceed this percentile to be
            considered part of a swing.
        decel_ratio: Required speed drop ratio within the look-ahead window
            (0.5 = speed must drop to <50% of peak).
        debug: Print diagnostics.

    Returns:
        List of (frame_num, confidence) sorted by frame number.
    """
    sorted_frames = sorted(wrist_positions.keys())
    if len(sorted_frames) < 5:
        return []

    # Compute wrist speed
    wrist_speeds: Dict[int, float] = {}
    for i in range(1, len(sorted_frames)):
        f0, f1 = sorted_frames[i - 1], sorted_frames[i]
        dt = (f1 - f0) / fps
        if dt <= 0:
            continue
        x0, y0 = wrist_positions[f0]
        x1, y1 = wrist_positions[f1]
        speed = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / dt
        wrist_speeds[f1] = speed

    if len(wrist_speeds) < 5:
        return []

    speed_arr = np.array(list(wrist_speeds.values()))
    threshold = np.percentile(speed_arr, speed_percentile)

    if debug:
        print(f"  [debug-wrist] {len(wrist_speeds)} wrist speed samples")
        print(f"  [debug-wrist] speed range: {speed_arr.min():.0f} - {speed_arr.max():.0f} px/s")
        print(f"  [debug-wrist] p{speed_percentile:.0f} threshold: {threshold:.0f} px/s")

    # Look-ahead window in frames (check deceleration over ~100ms)
    look_ahead = max(3, int(fps * 0.1))

    speed_by_frame = wrist_speeds
    frames_with_speed = sorted(speed_by_frame.keys())

    candidates = []
    for i, f in enumerate(frames_with_speed):
        s = speed_by_frame[f]
        if s < threshold:
            continue

        # Check for deceleration in the look-ahead window
        future_speeds = []
        for j in range(i + 1, min(i + look_ahead + 1, len(frames_with_speed))):
            fj = frames_with_speed[j]
            if fj - f > look_ahead + 2:
                break
            future_speeds.append(speed_by_frame[fj])

        if not future_speeds:
            continue

        min_future = min(future_speeds)
        if min_future < s * decel_ratio:
            # Confidence: how sharp the deceleration is + how high the peak speed
            decel_strength = 1.0 - (min_future / s)
            speed_strength = min(s / threshold, 2.0) / 2.0
            conf = 0.4 * decel_strength + 0.4 * speed_strength + 0.2
            conf = min(conf, 1.0)
            candidates.append((f, conf))

            if debug:
                print(f"  [debug-wrist] candidate frame {f}: speed={s:.0f}, "
                      f"min_future={min_future:.0f}, conf={conf:.2f}")

    if debug:
        print(f"  [debug-wrist] {len(candidates)} candidates before suppression")

    # Non-maximum suppression: keep highest confidence per gap window
    candidates.sort(key=lambda c: c[0])
    contacts = []
    for frame, conf in candidates:
        if contacts and frame - contacts[-1][0] < min_frame_gap:
            if conf > contacts[-1][1]:
                contacts[-1] = (frame, conf)
            continue
        contacts.append((frame, conf))

    return contacts


def detect_contacts(
    ball_detections: List[Tuple[int, float, float, float]],
    fps: float,
    wrist_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    velocity_spike_threshold: float = 2.0,
    wrist_proximity_px: float = 150.0,
    min_frame_gap: int = 10,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Detect contact frames — tries wrist-based detection first, then
    falls back to ball trajectory if wrist data is unavailable.
    """
    # Primary: wrist kinematics (more reliable when ball detection is noisy)
    if wrist_positions and len(wrist_positions) > 10:
        contacts = detect_contacts_wrist(
            wrist_positions, fps,
            min_frame_gap=min_frame_gap,
            debug=debug,
        )
        if contacts:
            return contacts
        if debug:
            print("  [debug] Wrist-based detection found 0 contacts, trying ball trajectory...")

    # Fallback: ball trajectory analysis
    velocities = compute_ball_velocity(ball_detections, fps)
    if len(velocities) < 3:
        return []

    speeds = np.array([v[3] for v in velocities])
    nonzero_speeds = speeds[speeds > 1e-3]
    if len(nonzero_speeds) < 3:
        return []

    ref_speed = np.percentile(nonzero_speeds, 75)
    if debug:
        print(f"  [debug-ball] ref speed (p75): {ref_speed:.1f} px/s")

    if ref_speed < 1e-3:
        return []

    ball_pos = {d[0]: (d[1], d[2]) for d in ball_detections}

    candidates = []
    for i in range(1, len(velocities)):
        f_prev, vx0, vy0, s0 = velocities[i - 1]
        f_curr, vx1, vy1, s1 = velocities[i]

        dot = vx0 * vx1 + vy0 * vy1
        direction_reversal = dot < 0
        speed_spike = s1 > ref_speed * velocity_spike_threshold

        if not (direction_reversal or speed_spike):
            continue

        conf = 0.0
        if direction_reversal:
            conf += 0.35
        if speed_spike:
            conf += 0.3

        if wrist_positions and f_curr in ball_pos and f_curr in wrist_positions:
            bx, by = ball_pos[f_curr]
            wx, wy = wrist_positions[f_curr]
            dist = np.sqrt((bx - wx) ** 2 + (by - wy) ** 2)
            if dist < wrist_proximity_px:
                conf += 0.2

        if conf < 0.2:
            continue
        candidates.append((f_curr, min(conf, 1.0)))

    candidates.sort(key=lambda c: c[0])
    contacts = []
    for frame, conf in candidates:
        if contacts and frame - contacts[-1][0] < min_frame_gap:
            if conf > contacts[-1][1]:
                contacts[-1] = (frame, conf)
            continue
        contacts.append((frame, conf))

    return contacts
