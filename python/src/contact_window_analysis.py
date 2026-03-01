"""Multi-frame window analysis around a detected contact frame.

Provides:
- Smoothed pose from ±N frames
- Swing velocity vector from wrist trajectory
- Racket geometry (head center, face normal, right/up_r axes)
- Ball contact point via HSV detection + trajectory fitting
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_contact_window(
    frames: list,
    contact_frame: int,
    fps: float,
    pose_estimator,
    window_frames: int = 5,
) -> dict:
    """Analyze a ±window_frames window around contact_frame.

    Args:
        frames:         All video frames (BGR numpy arrays).
        contact_frame:  Detected contact frame index.
        fps:            Video frame rate.
        pose_estimator: PoseEstimator instance (reuse from caller).
        window_frames:  Half-width of analysis window (default 5).

    Returns dict with keys:
        landmarks_3d      : smoothed world landmarks (dict name -> np.array(3,))
        raw_result        : mediapipe result from the contact frame
        swing_velocity    : np.array(3,) — wrist swing direction (unit vector)
        striking_side     : 'right' | 'left' — auto-detected from wrist extension
        racket_geometry   : dict with head_center, face_normal, right, up_r,
                            handle_start, handle_end, elbow
        ball_contact_3d   : np.array(3,) | None — 3D contact point on racket face
        ball_detections   : list of {frame, cx, cy, radius}
        detection_quality : 'good' | 'partial' | 'fallback'
    """
    num_frames = len(frames)

    start = max(0, contact_frame - window_frames)
    end   = min(num_frames - 1, contact_frame + window_frames)
    frame_indices = list(range(start, end + 1))

    # ------------------------------------------------------------------
    # 1. Run pose on all window frames; smooth landmarks
    # ------------------------------------------------------------------
    all_landmarks: Dict[int, dict] = {}
    contact_raw_result = None

    for fi in frame_indices:
        lm, raw = pose_estimator.process_frame(frames[fi])
        if lm is not None:
            all_landmarks[fi] = lm
        if fi == contact_frame and raw is not None:
            contact_raw_result = raw

    smoothed = _smooth_landmarks(all_landmarks, frame_indices)

    # ------------------------------------------------------------------
    # 2. Auto-detect striking side: whichever wrist is further from pelvis
    # ------------------------------------------------------------------
    if smoothed is None:
        return {
            "landmarks_3d": {},
            "raw_result": contact_raw_result,
            "swing_velocity": np.array([0.0, 0.0, 1.0]),
            "striking_side": "right",
            "racket_geometry": None,
            "ball_contact_3d": None,
            "ball_detections": [],
            "detection_quality": "fallback",
        }

    pelvis = smoothed.get("pelvis", np.zeros(3))
    rw_dist = np.linalg.norm(smoothed.get("right_wrist", pelvis) - pelvis)
    lw_dist = np.linalg.norm(smoothed.get("left_wrist",  pelvis) - pelvis)
    if rw_dist >= lw_dist:
        striking_side = "right"
        wrist_key, elbow_key = "right_wrist", "right_elbow"
    else:
        striking_side = "left"
        wrist_key, elbow_key = "left_wrist", "left_elbow"

    # ------------------------------------------------------------------
    # 3. Swing velocity: wrist displacement over a ±2 frame sub-window
    # ------------------------------------------------------------------
    swing_velocity = _compute_swing_velocity(all_landmarks, contact_frame, wrist_key)

    # ------------------------------------------------------------------
    # 4. Racket geometry
    # ------------------------------------------------------------------
    racket_geometry = _build_racket_geometry(smoothed, swing_velocity, wrist_key, elbow_key)

    # ------------------------------------------------------------------
    # 4. Ball detection over the window
    # ------------------------------------------------------------------
    ball_detections = []
    for fi in frame_indices:
        if fi == contact_frame:
            continue  # often blurred exactly at contact
        result = _detect_ball_hsv(frames[fi])
        if result is not None:
            cx, cy, r = result
            ball_detections.append({"frame": fi, "cx": cx, "cy": cy, "radius": r})

    # ------------------------------------------------------------------
    # 5. Contact point: trajectory fit → map onto racket face
    # ------------------------------------------------------------------
    ball_contact_3d, detection_quality = _estimate_ball_contact(
        ball_detections, contact_frame, racket_geometry, frames[contact_frame].shape
    )

    return {
        "landmarks_3d": smoothed,
        "raw_result": contact_raw_result,
        "swing_velocity": swing_velocity,
        "striking_side": striking_side,
        "racket_geometry": racket_geometry,
        "ball_contact_3d": ball_contact_3d,
        "ball_detections": ball_detections,
        "detection_quality": detection_quality,
    }


# ---------------------------------------------------------------------------
# Internal helpers — pose
# ---------------------------------------------------------------------------

def _smooth_landmarks(
    all_landmarks: Dict[int, dict],
    frame_indices: list,
) -> Optional[dict]:
    """Average world landmarks across all frames that produced a result."""
    if not all_landmarks:
        return None

    # Collect per-key accumulation
    sums: dict = {}
    counts: dict = {}
    for lm_dict in all_landmarks.values():
        for name, arr in lm_dict.items():
            if name not in sums:
                sums[name] = np.zeros(3)
                counts[name] = 0
            sums[name] += arr
            counts[name] += 1

    return {name: sums[name] / counts[name] for name in sums}


def _compute_swing_velocity(
    all_landmarks: Dict[int, dict],
    contact_frame: int,
    wrist_key: str,
) -> np.ndarray:
    """Compute unit swing velocity from wrist positions around contact."""
    # Try ±2 frame window; fall back to whatever is available
    before = all_landmarks.get(contact_frame - 2) or all_landmarks.get(contact_frame - 1)
    after  = all_landmarks.get(contact_frame + 2) or all_landmarks.get(contact_frame + 1)

    if before is not None and after is not None:
        v = after[wrist_key] - before[wrist_key] if wrist_key in after and wrist_key in before else None
    elif after is not None and contact_frame in all_landmarks:
        v = after.get(wrist_key, np.zeros(3)) - all_landmarks[contact_frame].get(wrist_key, np.zeros(3))
    elif before is not None and contact_frame in all_landmarks:
        v = all_landmarks[contact_frame].get(wrist_key, np.zeros(3)) - before.get(wrist_key, np.zeros(3))
    else:
        v = None

    if v is None or np.linalg.norm(v) < 1e-6:
        # Default: forward-ish swing in MediaPipe coords (negative z = toward camera)
        return np.array([0.0, 0.0, -1.0])

    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Internal helpers — racket geometry
# ---------------------------------------------------------------------------

RACKET_WRIST_TO_HEAD_M = 0.38   # wrist to head center
RACKET_SEMI_A_M = 0.125         # semi-axis horizontal
RACKET_SEMI_B_M = 0.145         # semi-axis vertical
HANDLE_LENGTH_M = 0.15          # extra handle beyond elbow


def _build_racket_geometry(
    landmarks: dict,
    swing_velocity: np.ndarray,
    wrist_key: str,
    elbow_key: str,
) -> Optional[dict]:
    """Build racket geometry using swing plane method.

    Returns dict with:
        head_center  : np.array(3,)
        face_normal  : np.array(3,)
        right        : np.array(3,)  — local horizontal axis of racket face
        up_r         : np.array(3,)  — local vertical axis of racket face
        handle_start : np.array(3,)  — elbow or behind-elbow
        handle_end   : np.array(3,)  — wrist
        elbow        : np.array(3,)
        wrist        : np.array(3,)
    """
    wrist = landmarks.get(wrist_key)
    elbow = landmarks.get(elbow_key)

    if wrist is None or elbow is None:
        return None

    # Forearm direction (elbow → wrist)
    v_forearm = wrist - elbow
    forearm_len = np.linalg.norm(v_forearm)
    if forearm_len < 1e-6:
        return None
    v_forearm = v_forearm / forearm_len

    # Racket face normal = cross(forearm, swing)
    face_normal = np.cross(v_forearm, swing_velocity)
    fn_len = np.linalg.norm(face_normal)
    if fn_len < 1e-4:
        # Degenerate: forearm and swing nearly parallel — use a fallback normal
        face_normal = np.array([1.0, 0.0, 0.0])
    else:
        face_normal = face_normal / fn_len

    # Racket face basis
    right = np.cross(v_forearm, face_normal)
    r_len = np.linalg.norm(right)
    if r_len < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / r_len

    up_r = np.cross(right, v_forearm)
    up_r_len = np.linalg.norm(up_r)
    if up_r_len < 1e-6:
        up_r = np.array([0.0, -1.0, 0.0])
    else:
        up_r = up_r / up_r_len

    head_center = wrist + RACKET_WRIST_TO_HEAD_M * v_forearm

    # Handle: extend slightly behind elbow for visual clarity
    handle_start = elbow - HANDLE_LENGTH_M * v_forearm

    return {
        "head_center": head_center,
        "face_normal": face_normal,
        "right": right,
        "up_r": up_r,
        "handle_start": handle_start,
        "handle_end": wrist,
        "elbow": elbow,
        "wrist": wrist,
    }


# ---------------------------------------------------------------------------
# Internal helpers — ball detection
# ---------------------------------------------------------------------------

# HSV range for yellow-green tennis balls
_BALL_H_LOW, _BALL_H_HIGH = 25, 70
_BALL_S_LOW, _BALL_S_HIGH = 50, 255
_BALL_V_LOW, _BALL_V_HIGH = 80, 255
_BALL_MIN_RADIUS_PX = 8
_BALL_MAX_RADIUS_PX = 80
_BALL_MIN_CIRCULARITY = 0.55


def _detect_ball_hsv(frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Detect a tennis ball in a BGR frame via HSV color thresholding.

    Returns (cx, cy, radius_px) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([_BALL_H_LOW,  _BALL_S_LOW,  _BALL_V_LOW]),
        np.array([_BALL_H_HIGH, _BALL_S_HIGH, _BALL_V_HIGH]),
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[Tuple[float, int, int, int]] = None  # (circularity, cx, cy, r)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < _BALL_MIN_CIRCULARITY:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        r = int(radius)
        if r < _BALL_MIN_RADIUS_PX or r > _BALL_MAX_RADIUS_PX:
            continue

        if best is None or circularity > best[0]:
            best = (circularity, int(cx), int(cy), r)

    if best is None:
        return None
    _, cx, cy, r = best
    return cx, cy, r


def _fit_line_2d(points: List[Tuple[float, float]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Least-squares line fit through 2D points.

    Returns (origin, direction) or None if fewer than 2 points.
    """
    if len(points) < 2:
        return None
    pts = np.array(points, dtype=float)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if np.linalg.norm(direction) < 1e-9:
        return None
    return mean, direction


# ---------------------------------------------------------------------------
# Internal helpers — 2D ball → 3D racket face
# ---------------------------------------------------------------------------

def _project_3d_to_2d(
    point_3d: np.ndarray,
    frame_shape: tuple,
    focal_fraction: float = 1.0,
) -> Tuple[float, float]:
    """Approximate perspective projection from MediaPipe world coords to pixel space.

    MediaPipe world coords are in metres, origin near the body.
    We use a simple pinhole model with focal_fraction * max(width, height) as focal length.
    """
    h, w = frame_shape[:2]
    f = focal_fraction * max(h, w)
    cx_img, cy_img = w / 2.0, h / 2.0

    x, y, z = point_3d
    # MediaPipe: z is depth (positive = behind player, negative = in front).
    # Add a scene depth offset so z_cam > 0.
    z_cam = z + 3.0
    if abs(z_cam) < 1e-4:
        z_cam = 1e-4

    px = cx_img + f * x / z_cam
    py = cy_img + f * y / z_cam
    return px, py


def _map_2d_to_racket_face(
    ball_2d: Tuple[float, float],
    racket_geometry: dict,
    frame_shape: tuple,
) -> np.ndarray:
    """Map a 2D ball pixel position onto the 3D racket face plane.

    Projects the racket ellipse axes into 2D, then expresses ball_2d as
    local (u, v) on the face, then reconstructs in 3D.
    """
    head_center = racket_geometry["head_center"]
    right       = racket_geometry["right"]
    up_r        = racket_geometry["up_r"]

    # Project key 3D points to 2D
    center_2d = np.array(_project_3d_to_2d(head_center,              frame_shape))
    right_2d  = np.array(_project_3d_to_2d(head_center + right,      frame_shape))
    up_2d     = np.array(_project_3d_to_2d(head_center + up_r,       frame_shape))

    # Basis in 2D image space (could be scaled by perspective)
    e1_2d = right_2d - center_2d
    e2_2d = up_2d    - center_2d

    e1_len = np.linalg.norm(e1_2d)
    e2_len = np.linalg.norm(e2_2d)
    if e1_len < 1e-4 or e2_len < 1e-4:
        return head_center.copy()

    # Solve: ball_2d = center_2d + u * e1_2d + v * e2_2d
    # Least-squares: A @ [u, v] = b
    A = np.column_stack([e1_2d, e2_2d])
    b = np.array(ball_2d) - center_2d
    try:
        uv, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return head_center.copy()

    u, v = uv
    # Clamp to roughly the racket face ellipse
    u = np.clip(u, -RACKET_SEMI_A_M, RACKET_SEMI_A_M)
    v = np.clip(v, -RACKET_SEMI_B_M, RACKET_SEMI_B_M)

    return head_center + u * right + v * up_r


# ---------------------------------------------------------------------------
# Contact point estimation
# ---------------------------------------------------------------------------

def _estimate_ball_contact(
    ball_detections: list,
    contact_frame: int,
    racket_geometry: Optional[dict],
    frame_shape: tuple,
) -> Tuple[Optional[np.ndarray], str]:
    """Fit ball trajectory and map 2D contact point onto 3D racket face.

    Returns (ball_contact_3d, detection_quality).
    """
    if racket_geometry is None:
        return None, "fallback"

    pre  = [(d["cx"], d["cy"], d["frame"]) for d in ball_detections if d["frame"] < contact_frame]
    post = [(d["cx"], d["cy"], d["frame"]) for d in ball_detections if d["frame"] > contact_frame]

    pre_pts  = [(cx, cy) for cx, cy, _ in pre]
    post_pts = [(cx, cy) for cx, cy, _ in post]

    contact_2d: Optional[Tuple[float, float]] = None

    if len(pre_pts) >= 2 and len(post_pts) >= 2:
        pre_line  = _fit_line_2d(pre_pts)
        post_line = _fit_line_2d(post_pts)
        if pre_line is not None and post_line is not None:
            # Extrapolate both lines to the contact frame index
            pre_t_vals  = [f for _, _, f in pre]
            post_t_vals = [f for _, _, f in post]
            pre_origin,  pre_dir  = pre_line
            post_origin, post_dir = post_line

            # t parameter: fraction of line span to extrapolate
            if pre_t_vals:
                dt_pre  = contact_frame - np.mean(pre_t_vals)
                pre_span  = max(np.ptp(pre_t_vals), 1)
                p_pre  = pre_origin  + pre_dir  * (dt_pre  / pre_span  * np.linalg.norm(
                    np.array(pre_pts[-1]) - np.array(pre_pts[0])))
            else:
                p_pre = np.array(pre_pts[-1], dtype=float)

            if post_t_vals:
                dt_post = np.mean(post_t_vals) - contact_frame
                post_span = max(np.ptp(post_t_vals), 1)
                p_post = post_origin + post_dir * (-dt_post / post_span * np.linalg.norm(
                    np.array(post_pts[-1]) - np.array(post_pts[0])))
            else:
                p_post = np.array(post_pts[0], dtype=float)

            contact_2d = ((p_pre[0] + p_post[0]) / 2, (p_pre[1] + p_post[1]) / 2)
            quality = "good"

    elif len(pre_pts) >= 2:
        line = _fit_line_2d(pre_pts)
        if line is not None:
            o, d = line
            pre_t_vals = [f for _, _, f in pre]
            dt = contact_frame - np.mean(pre_t_vals)
            span = max(np.ptp(pre_t_vals), 1)
            travel = np.linalg.norm(np.array(pre_pts[-1]) - np.array(pre_pts[0]))
            pt = o + d * (dt / span * travel)
            contact_2d = (float(pt[0]), float(pt[1]))
        quality = "partial"

    elif len(post_pts) >= 2:
        line = _fit_line_2d(post_pts)
        if line is not None:
            o, d = line
            post_t_vals = [f for _, _, f in post]
            dt = np.mean(post_t_vals) - contact_frame
            span = max(np.ptp(post_t_vals), 1)
            travel = np.linalg.norm(np.array(post_pts[-1]) - np.array(post_pts[0]))
            pt = o + d * (-dt / span * travel)
            contact_2d = (float(pt[0]), float(pt[1]))
        quality = "partial"

    else:
        quality = "fallback"

    if contact_2d is not None:
        contact_3d = _map_2d_to_racket_face(contact_2d, racket_geometry, frame_shape)
    else:
        # Fallback: center of racket head
        contact_3d = racket_geometry["head_center"].copy()
        quality = "fallback"

    return contact_3d, quality
