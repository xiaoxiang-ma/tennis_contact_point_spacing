"""Tennis ball detection using TrackNet with HSV color fallback."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm


class BallTracker:
    """Detects tennis balls in video frames.

    Attempts to load a TrackNet model for inference. If weights are
    unavailable, falls back to HSV color-based detection.
    """

    def __init__(self, tracknet_weights: Optional[str] = None):
        """Initialize the ball tracker.

        Args:
            tracknet_weights: Path to TrackNet .pth weights file.
                If None or file not found, uses HSV fallback.
        """
        self.model = None
        self.use_tracknet = False

        if tracknet_weights:
            try:
                self.model = self._load_tracknet(tracknet_weights)
                self.use_tracknet = True
            except Exception:
                pass

    def _load_tracknet(self, weights_path: str):
        """Load TrackNet model from weights file.

        TrackNet expects 3 consecutive frames (9-channel input) and outputs
        a heatmap of ball location.
        """
        import torch
        import torch.nn as nn

        class TrackNetSimple(nn.Module):
            """Simplified TrackNet-like architecture."""

            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(9, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                    nn.Conv2d(64, 1, 1), nn.Sigmoid(),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = TrackNetSimple()
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def detect_all(self, frames: List[np.ndarray],
                   progress: bool = True) -> List[Tuple[int, float, float, float]]:
        """Detect the ball in all frames.

        Args:
            frames: List of BGR frames.
            progress: Whether to show a progress bar.

        Returns:
            List of (frame_num, x, y, confidence) for frames where
            the ball was detected. x, y are in pixel coordinates.
        """
        if self.use_tracknet and self.model is not None:
            return self._detect_tracknet(frames, progress)
        return self._detect_hsv(frames, progress)

    def _detect_tracknet(self, frames: List[np.ndarray],
                         progress: bool) -> List[Tuple[int, float, float, float]]:
        """Run TrackNet inference on consecutive frame triplets."""
        import torch

        results = []
        h, w = 360, 640  # TrackNet input size
        iterator = range(2, len(frames))
        if progress:
            iterator = tqdm(iterator, desc="TrackNet ball detection")

        for i in iterator:
            triplet = []
            for j in (i - 2, i - 1, i):
                resized = cv2.resize(frames[j], (w, h)) / 255.0
                triplet.append(resized)

            inp = np.concatenate(triplet, axis=2).transpose(2, 0, 1)
            inp_tensor = torch.FloatTensor(inp).unsqueeze(0)

            with torch.no_grad():
                heatmap = self.model(inp_tensor).squeeze().numpy()

            max_val = heatmap.max()
            if max_val > 0.5:
                y_pred, x_pred = np.unravel_index(heatmap.argmax(), heatmap.shape)
                # Scale back to original resolution
                orig_h, orig_w = frames[i].shape[:2]
                x_orig = x_pred * orig_w / w
                y_orig = y_pred * orig_h / h
                results.append((i, float(x_orig), float(y_orig), float(max_val)))

        return results

    def _detect_hsv(self, frames: List[np.ndarray],
                    progress: bool) -> List[Tuple[int, float, float, float]]:
        """Fallback: detect tennis ball using HSV color thresholding.

        Tennis balls are bright yellow-green. We threshold in HSV space,
        find contours, and pick the best circular candidate.
        """
        results = []
        iterator = range(len(frames))
        if progress:
            iterator = tqdm(iterator, desc="HSV ball detection")

        for i in iterator:
            detection = self._detect_hsv_single(frames[i])
            if detection is not None:
                x, y, conf = detection
                results.append((i, x, y, conf))

        return results

    @staticmethod
    def _detect_hsv_single(frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect tennis ball in a single frame using HSV color filtering.

        Returns (x, y, confidence) or None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Tennis ball yellow-green range (two ranges to handle hue wrapping)
        lower1 = np.array([25, 80, 80])
        upper1 = np.array([45, 255, 255])
        lower2 = np.array([20, 100, 100])
        upper2 = np.array([50, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 5000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            score = circularity * min(area / 200.0, 1.0)
            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(min(score, 1.0)))

        return best
