"""Video I/O utilities for loading and extracting frames."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


def load_video(path: str) -> Tuple[List[np.ndarray], Dict]:
    """Load a video file and return all frames plus metadata.

    Args:
        path: Path to the video file.

    Returns:
        Tuple of (frames list as BGR numpy arrays, metadata dict with
        'fps', 'width', 'height', 'frame_count', 'duration_sec').

    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    metadata["duration_sec"] = metadata["frame_count"] / metadata["fps"] if metadata["fps"] > 0 else 0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames, metadata


def extract_frame(video_path: str, frame_num: int) -> Optional[np.ndarray]:
    """Extract a single frame from a video file.

    Args:
        video_path: Path to the video file.
        frame_num: Zero-indexed frame number to extract.

    Returns:
        BGR numpy array of the frame, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None
