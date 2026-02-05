"""Spatial localization for tennis contact point detection.

Stage 2 of the contact detection pipeline: Determine WHERE the contact occurred.
Uses CV techniques to detect racket position and estimate contact point location.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class RacketDetector:
    """Detect tennis racket in video frames using CV techniques.

    Uses a combination of:
    1. Color-based detection (racket strings/frame often have distinct colors)
    2. Edge detection and contour analysis
    3. Shape matching (looking for oval/ellipse shapes)
    4. Player pose integration (racket should be near wrist)
    """

    def __init__(
        self,
        wrist_search_radius: int = 150,
        min_racket_area: int = 500,
        max_racket_area: int = 15000,
        edge_threshold1: int = 50,
        edge_threshold2: int = 150,
    ):
        """Initialize racket detector.

        Args:
            wrist_search_radius: Radius (pixels) around wrist to search for racket.
            min_racket_area: Minimum contour area for racket detection.
            max_racket_area: Maximum contour area for racket detection.
            edge_threshold1: Lower threshold for Canny edge detection.
            edge_threshold2: Upper threshold for Canny edge detection.
        """
        self.wrist_search_radius = wrist_search_radius
        self.min_racket_area = min_racket_area
        self.max_racket_area = max_racket_area
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2

    def detect_racket_region(
        self,
        frame: np.ndarray,
        wrist_position: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict]:
        """Detect racket region in frame.

        Args:
            frame: BGR image.
            wrist_position: Optional (x, y) pixel coordinates of wrist to guide search.

        Returns:
            Dict with 'center', 'bbox', 'contour', 'confidence' or None if not detected.
        """
        h, w = frame.shape[:2]

        # Create region of interest around wrist if provided
        if wrist_position is not None:
            wx, wy = wrist_position
            x1 = max(0, wx - self.wrist_search_radius)
            y1 = max(0, wy - self.wrist_search_radius)
            x2 = min(w, wx + self.wrist_search_radius)
            y2 = min(h, wy + self.wrist_search_radius)
            roi = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            roi = frame
            offset = (0, 0)

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)

        # Dilate edges to connect broken contours
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter and score contours
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_racket_area or area > self.max_racket_area:
                continue

            # Fit ellipse if enough points
            if len(contour) < 5:
                continue

            ellipse = cv2.fitEllipse(contour)
            center, (ma, mi), angle = ellipse

            # Rackets are typically elongated (aspect ratio check)
            aspect_ratio = max(ma, mi) / (min(ma, mi) + 1e-6)
            if aspect_ratio < 1.5 or aspect_ratio > 5.0:
                continue

            # Score based on how ellipse-like the contour is
            ellipse_area = np.pi * ma * mi / 4
            ellipse_fit_score = min(area, ellipse_area) / (max(area, ellipse_area) + 1e-6)

            # Boost score if near wrist
            distance_score = 1.0
            if wrist_position is not None:
                dist = np.sqrt((center[0] + offset[0] - wrist_position[0])**2 +
                              (center[1] + offset[1] - wrist_position[1])**2)
                distance_score = max(0, 1 - dist / self.wrist_search_radius)

            confidence = ellipse_fit_score * 0.6 + distance_score * 0.4

            bbox = cv2.boundingRect(contour)
            bbox = (bbox[0] + offset[0], bbox[1] + offset[1], bbox[2], bbox[3])

            candidates.append({
                'center': (int(center[0] + offset[0]), int(center[1] + offset[1])),
                'bbox': bbox,
                'contour': contour + np.array([[offset]]),
                'ellipse': (
                    (center[0] + offset[0], center[1] + offset[1]),
                    (ma, mi),
                    angle
                ),
                'confidence': confidence,
                'area': area,
            })

        if not candidates:
            return None

        # Return highest confidence candidate
        best = max(candidates, key=lambda c: c['confidence'])
        return best

    def estimate_racket_head_center(
        self,
        frame: np.ndarray,
        wrist_position: Tuple[int, int],
        elbow_position: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """Estimate the center of the racket head based on arm position.

        If racket detection fails, uses geometric estimation based on
        wrist and elbow positions.

        Args:
            frame: BGR image.
            wrist_position: (x, y) pixel coordinates of wrist.
            elbow_position: Optional (x, y) pixel coordinates of elbow.

        Returns:
            Tuple of ((x, y) racket head center or None, confidence).
        """
        # Try CV-based detection first
        detection = self.detect_racket_region(frame, wrist_position)

        if detection is not None and detection['confidence'] > 0.5:
            return detection['center'], detection['confidence']

        # Fallback: Geometric estimation
        if elbow_position is None:
            # Without elbow, estimate racket is ~30 pixels beyond wrist
            # in the direction of the frame center (rough heuristic)
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            wx, wy = wrist_position

            # Direction from wrist toward frame center
            dx = cx - wx
            dy = cy - wy
            norm = np.sqrt(dx**2 + dy**2) + 1e-6

            # Racket head is typically 25-40cm from wrist, estimate ~40 pixels
            racket_offset = 40
            rx = int(wx + (dx / norm) * racket_offset)
            ry = int(wy + (dy / norm) * racket_offset)

            return (rx, ry), 0.3

        # Use elbow-wrist direction to estimate racket head
        wx, wy = wrist_position
        ex, ey = elbow_position

        # Direction from elbow to wrist (forearm direction)
        dx = wx - ex
        dy = wy - ey
        norm = np.sqrt(dx**2 + dy**2) + 1e-6

        # Racket head extends beyond wrist in forearm direction
        # Typical racket length is ~27 inches, head center ~15 inches from wrist
        # At typical video resolution, this is roughly 30-60 pixels
        racket_offset = 50

        rx = int(wx + (dx / norm) * racket_offset)
        ry = int(wy + (dy / norm) * racket_offset)

        # Clamp to frame bounds
        h, w = frame.shape[:2]
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))

        return (rx, ry), 0.4


class ContactPointLocalizer:
    """Localize the exact contact point between ball and racket.

    Combines ball detection, racket detection, and pose estimation
    to determine the 3D contact point in body-relative coordinates.
    """

    def __init__(self):
        self.racket_detector = RacketDetector()

    def localize_contact(
        self,
        frame: np.ndarray,
        ball_position: Optional[Tuple[float, float]],
        pixel_landmarks: Dict[str, Tuple[int, int]],
        contact_wrist: str = "right_wrist",
    ) -> Dict:
        """Localize contact point in the frame.

        Args:
            frame: BGR image.
            ball_position: (x, y) pixel coordinates of ball or None.
            pixel_landmarks: Dict of landmark name -> (px, py).
            contact_wrist: Which wrist to use ("left_wrist" or "right_wrist").

        Returns:
            Dict with:
            - 'contact_pixel': (x, y) pixel coordinates of estimated contact
            - 'method': How contact was determined
            - 'confidence': Confidence in localization
            - 'racket_center': Estimated racket head center if detected
            - 'ball_position': Ball position if available
        """
        result = {
            'contact_pixel': None,
            'method': 'none',
            'confidence': 0.0,
            'racket_center': None,
            'ball_position': ball_position,
        }

        # Get wrist position
        wrist_pos = pixel_landmarks.get(contact_wrist)
        if wrist_pos is None:
            return result

        # Get elbow position for better racket estimation
        elbow_name = contact_wrist.replace("wrist", "elbow")
        elbow_pos = pixel_landmarks.get(elbow_name)

        # Estimate racket head center
        racket_center, racket_conf = self.racket_detector.estimate_racket_head_center(
            frame, wrist_pos, elbow_pos
        )
        result['racket_center'] = racket_center

        # Determine contact point based on available information
        if ball_position is not None and racket_center is not None:
            # Best case: Both ball and racket detected
            # Contact is at the ball position (ball hits racket)
            bx, by = ball_position

            # Verify ball is near racket (within reasonable distance)
            if racket_center is not None:
                dist_to_racket = np.sqrt(
                    (bx - racket_center[0])**2 + (by - racket_center[1])**2
                )
                if dist_to_racket < 100:  # Ball should be close to racket at contact
                    result['contact_pixel'] = (int(bx), int(by))
                    result['method'] = 'ball_detected'
                    result['confidence'] = min(0.9, 0.5 + racket_conf * 0.4)
                else:
                    # Ball far from racket, use midpoint
                    cx = int((bx + racket_center[0]) / 2)
                    cy = int((by + racket_center[1]) / 2)
                    result['contact_pixel'] = (cx, cy)
                    result['method'] = 'ball_racket_midpoint'
                    result['confidence'] = 0.6
            else:
                result['contact_pixel'] = (int(bx), int(by))
                result['method'] = 'ball_only'
                result['confidence'] = 0.7

        elif ball_position is not None:
            # Ball detected but no racket
            result['contact_pixel'] = (int(ball_position[0]), int(ball_position[1]))
            result['method'] = 'ball_only'
            result['confidence'] = 0.7

        elif racket_center is not None:
            # Racket detected but no ball
            result['contact_pixel'] = racket_center
            result['method'] = 'racket_center'
            result['confidence'] = racket_conf

        else:
            # Fallback to wrist position
            result['contact_pixel'] = wrist_pos
            result['method'] = 'wrist_fallback'
            result['confidence'] = 0.3

        return result

    def compute_body_relative_3d(
        self,
        contact_pixel: Tuple[int, int],
        pixel_landmarks: Dict[str, Tuple[int, int]],
        landmarks_3d: Dict[str, np.ndarray],
        contact_wrist: str = "right_wrist",
    ) -> Optional[np.ndarray]:
        """Convert pixel contact point to body-relative 3D coordinates.

        Uses the 3D pose to estimate depth and transform the contact
        point into body-relative coordinates.

        Args:
            contact_pixel: (x, y) pixel coordinates of contact.
            pixel_landmarks: Dict of landmark name -> (px, py).
            landmarks_3d: Dict of landmark name -> np.array([x, y, z]) in meters.
            contact_wrist: Which wrist to use for depth estimation.

        Returns:
            np.array([x, y, z]) in body-relative coordinates (meters) or None.
        """
        wrist_pixel = pixel_landmarks.get(contact_wrist)
        wrist_3d = landmarks_3d.get(contact_wrist)
        pelvis_3d = landmarks_3d.get("pelvis")

        if wrist_pixel is None or wrist_3d is None or pelvis_3d is None:
            return None

        # Calculate pixel offset from wrist
        px_offset_x = contact_pixel[0] - wrist_pixel[0]
        px_offset_y = contact_pixel[1] - wrist_pixel[1]

        # Estimate scale factor (pixels to meters)
        # This is approximate and depends on camera/distance
        # Using shoulder width as reference when available
        left_shoulder = pixel_landmarks.get("left_shoulder")
        right_shoulder = pixel_landmarks.get("right_shoulder")
        left_shoulder_3d = landmarks_3d.get("left_shoulder")
        right_shoulder_3d = landmarks_3d.get("right_shoulder")

        if all([left_shoulder, right_shoulder, left_shoulder_3d is not None, right_shoulder_3d is not None]):
            # Compute scale from shoulder width
            pixel_shoulder_width = np.sqrt(
                (right_shoulder[0] - left_shoulder[0])**2 +
                (right_shoulder[1] - left_shoulder[1])**2
            )
            real_shoulder_width = np.linalg.norm(right_shoulder_3d - left_shoulder_3d)
            scale = real_shoulder_width / (pixel_shoulder_width + 1e-6)
        else:
            # Default scale factor (approximate)
            scale = 0.001  # 1 pixel ≈ 1mm at typical distances

        # Apply offset in 3D (assuming camera faces player, z is depth)
        contact_3d = wrist_3d.copy()
        contact_3d[0] += px_offset_x * scale  # Lateral
        contact_3d[1] += px_offset_y * scale  # Vertical

        # Return pelvis-relative coordinates
        return contact_3d - pelvis_3d


def localize_contact_point(
    frame: np.ndarray,
    ball_position: Optional[Tuple[float, float]],
    pixel_landmarks: Dict[str, Tuple[int, int]],
    landmarks_3d: Dict[str, np.ndarray],
    shot_type: str = "right_forehand",
) -> Dict:
    """High-level function to localize contact point.

    Convenience function that creates a localizer and computes all
    contact point information.

    Args:
        frame: BGR image at contact moment.
        ball_position: (x, y) ball pixel position or None.
        pixel_landmarks: 2D pixel landmarks from pose estimation.
        landmarks_3d: 3D landmarks from pose estimation.
        shot_type: Shot type to determine which arm ("right_forehand", etc.).

    Returns:
        Dict with comprehensive contact localization results.
    """
    # Determine contact wrist based on shot type
    if shot_type in ["right_forehand", "right_backhand"]:
        contact_wrist = "right_wrist"
    else:
        contact_wrist = "left_wrist"

    localizer = ContactPointLocalizer()

    # Get 2D localization
    loc_result = localizer.localize_contact(
        frame, ball_position, pixel_landmarks, contact_wrist
    )

    # Get 3D body-relative coordinates
    contact_3d = None
    if loc_result['contact_pixel'] is not None:
        contact_3d = localizer.compute_body_relative_3d(
            loc_result['contact_pixel'],
            pixel_landmarks,
            landmarks_3d,
            contact_wrist,
        )

    loc_result['contact_3d'] = contact_3d
    loc_result['contact_wrist'] = contact_wrist

    return loc_result
