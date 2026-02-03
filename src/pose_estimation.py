"""3D pose estimation using MediaPipe Pose."""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional


# MediaPipe landmark indices we care about
LANDMARK_MAP = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class PoseEstimator:
    """MediaPipe Pose wrapper that returns 3D landmarks."""

    def __init__(self, static_image_mode: bool = True,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Estimate 3D pose from a single BGR frame.

        Args:
            frame: BGR numpy array.

        Returns:
            Dict mapping landmark name to np.array([x, y, z]) in
            MediaPipe's normalized coordinate space, or None if no
            pose detected. x/y are in [0,1] image coords, z is
            relative depth.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_world_landmarks is None:
            return None

        landmarks = {}
        wl = result.pose_world_landmarks.landmark

        for name, idx in LANDMARK_MAP.items():
            lm = wl[idx]
            landmarks[name] = np.array([lm.x, lm.y, lm.z])

        # Synthesize pelvis as midpoint of hips
        if "left_hip" in landmarks and "right_hip" in landmarks:
            landmarks["pelvis"] = (landmarks["left_hip"] + landmarks["right_hip"]) / 2.0

        # Use nose as head proxy
        if "nose" in landmarks:
            landmarks["head"] = landmarks["nose"]

        return landmarks

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
