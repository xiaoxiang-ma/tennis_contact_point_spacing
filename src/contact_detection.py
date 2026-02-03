"""Detect ball-racket contact frames using TrackNet ball tracking + audio analysis.

This module combines two signals for robust contact detection:
1. Ball trajectory analysis (velocity reversal/spike from TrackNet)
2. Audio impact detection (sound spike in 1-4kHz range)

When both signals agree, confidence is high. Either signal alone
provides moderate confidence.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .tracknet import (
    TrackNetDetector,
    extrapolate_ball_position,
    interpolate_ball_position,
)
from .audio_detection import detect_contacts_audio


def compute_ball_velocity(
    trajectory: List[Tuple[int, float, float, float]],
    fps: float,
) -> List[Tuple[int, float, float, float]]:
    """Compute ball velocity between consecutive detections.

    Args:
        trajectory: List of (frame_num, x, y, confidence).
        fps: Video frame rate.

    Returns:
        List of (frame_num, vx, vy, speed) in pixels/second.
    """
    if len(trajectory) < 2:
        return []

    velocities = []
    for i in range(1, len(trajectory)):
        f0, x0, y0, _ = trajectory[i - 1]
        f1, x1, y1, _ = trajectory[i]

        dt = (f1 - f0) / fps if fps > 0 else 1.0
        if dt <= 0:
            continue

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        speed = np.sqrt(vx ** 2 + vy ** 2)
        velocities.append((f1, vx, vy, speed))

    return velocities


def detect_trajectory_contacts(
    trajectory: List[Tuple[int, float, float, float]],
    fps: float,
    velocity_spike_factor: float = 2.0,
    min_frame_gap: int = 15,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Detect contacts from ball trajectory (velocity reversals and spikes).

    At contact, the ball changes direction (velocity reversal) and/or
    experiences a sudden speed change.

    Args:
        trajectory: Ball positions from TrackNet.
        fps: Video frame rate.
        velocity_spike_factor: Speed must exceed baseline by this factor.
        min_frame_gap: Minimum frames between contacts.
        debug: Print debug info.

    Returns:
        List of (frame_num, confidence) for detected contacts.
    """
    velocities = compute_ball_velocity(trajectory, fps)

    if len(velocities) < 3:
        if debug:
            print(f"  [trajectory] Too few velocity samples: {len(velocities)}")
        return []

    speeds = np.array([v[3] for v in velocities])
    nonzero = speeds[speeds > 1e-3]

    if len(nonzero) < 3:
        return []

    # Reference speed (median is more robust than percentile for short clips)
    ref_speed = np.median(nonzero)
    speed_threshold = ref_speed * velocity_spike_factor

    if debug:
        print(f"  [trajectory] {len(velocities)} velocity samples")
        print(f"  [trajectory] Speed range: {speeds.min():.0f} - {speeds.max():.0f} px/s")
        print(f"  [trajectory] Reference speed: {ref_speed:.0f} px/s")

    candidates = []

    for i in range(1, len(velocities)):
        f_prev, vx0, vy0, s0 = velocities[i - 1]
        f_curr, vx1, vy1, s1 = velocities[i]

        # Check for direction reversal (dot product < 0)
        dot = vx0 * vx1 + vy0 * vy1
        direction_reversal = dot < 0

        # Check for speed spike
        speed_spike = s1 > speed_threshold

        # Check for sudden deceleration
        deceleration = (s0 > ref_speed) and (s1 < s0 * 0.5)

        if not (direction_reversal or speed_spike or deceleration):
            continue

        # Compute confidence based on signal strength
        conf = 0.0
        reasons = []

        if direction_reversal:
            # Stronger reversal = higher confidence
            cos_angle = dot / (np.sqrt(vx0**2 + vy0**2) * np.sqrt(vx1**2 + vy1**2) + 1e-6)
            reversal_strength = max(0, -cos_angle)  # 0 to 1
            conf += 0.3 * (0.5 + 0.5 * reversal_strength)
            reasons.append(f"reversal({reversal_strength:.2f})")

        if speed_spike:
            spike_ratio = s1 / ref_speed
            conf += 0.2 * min(spike_ratio / velocity_spike_factor, 1.5)
            reasons.append(f"spike({spike_ratio:.1f}x)")

        if deceleration:
            decel_ratio = 1 - (s1 / s0)
            conf += 0.2 * decel_ratio
            reasons.append(f"decel({decel_ratio:.2f})")

        if conf < 0.15:
            continue

        conf = min(conf, 0.7)  # Cap at 0.7 for trajectory-only detection
        candidates.append((f_curr, conf, reasons))

        if debug:
            print(f"    Frame {f_curr}: {', '.join(reasons)} -> conf={conf:.2f}")

    # Non-maximum suppression
    candidates.sort(key=lambda c: c[0])
    contacts = []
    for frame, conf, _ in candidates:
        if contacts and frame - contacts[-1][0] < min_frame_gap:
            if conf > contacts[-1][1]:
                contacts[-1] = (frame, conf)
            continue
        contacts.append((frame, conf))

    if debug:
        print(f"  [trajectory] {len(contacts)} contacts after NMS")

    return contacts


def fuse_contact_signals(
    trajectory_contacts: List[Tuple[int, float]],
    audio_contacts: List[Tuple[int, float]],
    fps: float,
    tolerance_frames: int = 3,
    debug: bool = False,
) -> List[Tuple[int, float, str]]:
    """Fuse trajectory and audio contact detections.

    When both signals detect a contact within tolerance, confidence increases.
    Single-signal detections have lower confidence.

    Args:
        trajectory_contacts: From ball trajectory analysis.
        audio_contacts: From audio impact detection.
        fps: Video frame rate.
        tolerance_frames: Maximum frame difference for matching.
        debug: Print debug info.

    Returns:
        List of (frame_num, confidence, source) where source is
        'both', 'trajectory', or 'audio'.
    """
    if debug:
        print(f"  [fusion] Trajectory contacts: {len(trajectory_contacts)}")
        print(f"  [fusion] Audio contacts: {len(audio_contacts)}")

    # Convert to sets for easier matching
    traj_dict = {f: c for f, c in trajectory_contacts}
    audio_dict = {f: c for f, c in audio_contacts}

    used_audio = set()
    results = []

    # Process trajectory contacts, looking for audio matches
    for traj_frame, traj_conf in trajectory_contacts:
        # Look for matching audio contact
        best_audio_frame = None
        best_audio_conf = 0

        for audio_frame, audio_conf in audio_contacts:
            if abs(audio_frame - traj_frame) <= tolerance_frames:
                if audio_conf > best_audio_conf:
                    best_audio_frame = audio_frame
                    best_audio_conf = audio_conf

        if best_audio_frame is not None:
            # Both signals agree - high confidence
            used_audio.add(best_audio_frame)
            # Use weighted average for frame, boost confidence
            combined_frame = int((traj_frame + best_audio_frame) / 2)
            combined_conf = min(0.5 * traj_conf + 0.5 * best_audio_conf + 0.3, 1.0)
            results.append((combined_frame, combined_conf, 'both'))

            if debug:
                print(f"    MATCH: traj={traj_frame}, audio={best_audio_frame} "
                      f"-> frame={combined_frame}, conf={combined_conf:.2f}")
        else:
            # Trajectory only - moderate confidence
            results.append((traj_frame, traj_conf, 'trajectory'))

            if debug:
                print(f"    Trajectory only: frame={traj_frame}, conf={traj_conf:.2f}")

    # Add unmatched audio contacts
    for audio_frame, audio_conf in audio_contacts:
        if audio_frame not in used_audio:
            # Check it's not too close to existing results
            too_close = any(abs(audio_frame - r[0]) <= tolerance_frames for r in results)
            if not too_close:
                results.append((audio_frame, audio_conf * 0.8, 'audio'))

                if debug:
                    print(f"    Audio only: frame={audio_frame}, conf={audio_conf*0.8:.2f}")

    # Sort by frame number
    results.sort(key=lambda r: r[0])

    return results


def detect_contacts(
    video_path: str,
    frames: List,
    fps: float,
    tracknet_detector: Optional[TrackNetDetector] = None,
    min_frame_gap: int = 15,
    use_audio: bool = True,
    debug: bool = False,
    save_debug_frames: bool = False,
) -> Tuple[List[Tuple[int, float, str]], Dict[int, Tuple[float, float, float]]]:
    """Detect contact frames using TrackNet + audio fusion.

    Main entry point for contact detection.

    Args:
        video_path: Path to video file (needed for audio extraction).
        frames: List of BGR frames.
        fps: Video frame rate.
        tracknet_detector: Pre-initialized TrackNet detector (optional).
        min_frame_gap: Minimum frames between contacts.
        use_audio: Whether to use audio analysis.
        debug: Print debug info.
        save_debug_frames: Save TrackNet debug visualizations.

    Returns:
        Tuple of:
        - List of (frame_num, confidence, source) for contacts
        - Dict of frame_num -> (x, y, conf) ball positions
    """
    # Initialize TrackNet if not provided
    if tracknet_detector is None:
        if debug:
            print("Initializing TrackNet detector...")
        tracknet_detector = TrackNetDetector(save_debug_frames=save_debug_frames)

    # Run TrackNet ball detection
    if debug:
        print(f"Running TrackNet on {len(frames)} frames...")

    ball_detections = tracknet_detector.detect_all(frames, progress=True)
    trajectory = tracknet_detector.get_ball_trajectory(ball_detections)

    if debug:
        print(f"  TrackNet detected ball in {len(ball_detections)}/{len(frames)} frames")

    # Detect contacts from trajectory
    trajectory_contacts = detect_trajectory_contacts(
        trajectory, fps,
        min_frame_gap=min_frame_gap,
        debug=debug,
    )

    # Detect contacts from audio
    audio_contacts = []
    if use_audio:
        if debug:
            print("Running audio analysis...")
        try:
            audio_contacts = detect_contacts_audio(
                video_path, fps,
                min_gap_ms=min_frame_gap * 1000 / fps,
                debug=debug,
            )
        except Exception as e:
            if debug:
                print(f"  Audio analysis failed: {e}")

    # Fuse signals
    if debug:
        print("Fusing contact signals...")

    contacts = fuse_contact_signals(
        trajectory_contacts,
        audio_contacts,
        fps,
        tolerance_frames=max(3, int(fps * 0.05)),  # ~50ms tolerance
        debug=debug,
    )

    if debug:
        print(f"\nFinal contacts: {len(contacts)}")
        for frame, conf, source in contacts:
            time_sec = frame / fps
            print(f"  Frame {frame} ({time_sec:.2f}s): conf={conf:.2f}, source={source}")

    return contacts, ball_detections


def get_contact_ball_position(
    contact_frame: int,
    ball_detections: Dict[int, Tuple[float, float, float]],
    trajectory: Optional[List[Tuple[int, float, float, float]]] = None,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Get ball position at contact frame.

    If ball not directly detected at contact frame, uses extrapolation
    or interpolation from nearby detections.

    Args:
        contact_frame: Frame number of contact.
        ball_detections: Dict of frame -> (x, y, conf) from TrackNet.
        trajectory: Optional sorted trajectory list for extrapolation.

    Returns:
        Tuple of:
        - (x, y) pixel coordinates or None
        - method: 'detected', 'interpolated', 'extrapolated', or 'none'
    """
    # Direct detection
    if contact_frame in ball_detections:
        x, y, _ = ball_detections[contact_frame]
        return (x, y), 'detected'

    # Build trajectory if not provided
    if trajectory is None:
        trajectory = [
            (f, x, y, c) for f, (x, y, c) in sorted(ball_detections.items())
        ]

    # Try interpolation first (more accurate than extrapolation)
    interp = interpolate_ball_position(trajectory, contact_frame)
    if interp is not None:
        return (interp[0], interp[1]), 'interpolated'

    # Try extrapolation
    extrap = extrapolate_ball_position(trajectory, contact_frame)
    if extrap is not None:
        return (extrap[0], extrap[1]), 'extrapolated'

    return None, 'none'


# Legacy function for backwards compatibility
def detect_contacts_legacy(
    ball_detections: List[Tuple[int, float, float, float]],
    fps: float,
    wrist_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    velocity_spike_threshold: float = 2.0,
    wrist_proximity_px: float = 150.0,
    min_frame_gap: int = 10,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Legacy contact detection for backwards compatibility.

    Deprecated: Use detect_contacts() instead.
    """
    trajectory = [(f, x, y, c) for f, x, y, c in ball_detections]
    contacts = detect_trajectory_contacts(
        trajectory, fps,
        velocity_spike_factor=velocity_spike_threshold,
        min_frame_gap=min_frame_gap,
        debug=debug,
    )
    return [(f, c) for f, c in contacts]
