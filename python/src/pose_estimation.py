"""3D pose estimation using MediaPipe Pose."""

import cv2
import numpy as np
from typing import Dict, Optional, Any


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


def _create_pose_legacy(static_image_mode, model_complexity, min_detection_confidence):
    """Create pose using legacy mp.solutions API."""
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
    ), "legacy"


def _create_pose_tasks(static_image_mode, model_complexity, min_detection_confidence):
    """Create pose using newer mediapipe.tasks API."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import urllib.request
    import os

    # Download model if needed
    model_path = "/tmp/pose_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        if model_complexity <= 1:
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=min_detection_confidence,
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector, "tasks"


def create_pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5):
    """Create a MediaPipe pose detector, trying legacy API first then tasks API."""
    try:
        return _create_pose_legacy(static_image_mode, model_complexity, min_detection_confidence)
    except (AttributeError, ImportError):
        return _create_pose_tasks(static_image_mode, model_complexity, min_detection_confidence)


class PoseEstimator:
    """MediaPipe Pose wrapper that returns 3D landmarks."""

    def __init__(self, static_image_mode: bool = True,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5):
        self.pose, self.api_mode = create_pose(
            static_image_mode, model_complexity, min_detection_confidence
        )

    def process_frame(self, frame: np.ndarray):
        """Estimate 3D pose from a single BGR frame.

        Returns:
            Tuple of (landmarks_dict, raw_result) or (None, None).
            landmarks_dict maps name to np.array([x, y, z]).
            raw_result is the mediapipe result object for pixel-coord access.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.api_mode == "legacy":
            return self._process_legacy(rgb)
        else:
            return self._process_tasks(rgb)

    def _process_legacy(self, rgb: np.ndarray):
        result = self.pose.process(rgb)
        if result.pose_world_landmarks is None:
            return None, None

        landmarks = {}
        wl = result.pose_world_landmarks.landmark
        for name, idx in LANDMARK_MAP.items():
            lm = wl[idx]
            landmarks[name] = np.array([lm.x, lm.y, lm.z])

        self._add_synthetic(landmarks)
        return landmarks, result

    def _process_tasks(self, rgb: np.ndarray):
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(mp_image)

        if not result.pose_world_landmarks or len(result.pose_world_landmarks) == 0:
            return None, None

        world_lm = result.pose_world_landmarks[0]
        landmarks = {}
        for name, idx in LANDMARK_MAP.items():
            lm = world_lm[idx]
            landmarks[name] = np.array([lm.x, lm.y, lm.z])

        self._add_synthetic(landmarks)
        return landmarks, result

    @staticmethod
    def _add_synthetic(landmarks):
        if "left_hip" in landmarks and "right_hip" in landmarks:
            landmarks["pelvis"] = (landmarks["left_hip"] + landmarks["right_hip"]) / 2.0
        if "nose" in landmarks:
            landmarks["head"] = landmarks["nose"]

    def get_pixel_landmarks(self, result, frame_shape) -> Optional[Dict[str, tuple]]:
        """Extract pixel-space landmarks from a raw result.

        Returns dict of name -> (px, py) or None.
        """
        h, w = frame_shape[:2]

        if self.api_mode == "legacy":
            if result is None or result.pose_landmarks is None:
                return None
            pixel = {}
            for name, idx in LANDMARK_MAP.items():
                lm = result.pose_landmarks.landmark[idx]
                pixel[name] = (int(lm.x * w), int(lm.y * h))
        else:
            if result is None or not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return None
            img_lm = result.pose_landmarks[0]
            pixel = {}
            for name, idx in LANDMARK_MAP.items():
                lm = img_lm[idx]
                pixel[name] = (int(lm.x * w), int(lm.y * h))

        # Synthetic landmarks
        if "left_hip" in pixel and "right_hip" in pixel:
            lh, rh = pixel["left_hip"], pixel["right_hip"]
            pixel["pelvis"] = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        if "nose" in pixel:
            pixel["head"] = pixel["nose"]

        return pixel

    def close(self):
        if self.api_mode == "legacy":
            self.pose.close()
        else:
            self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
