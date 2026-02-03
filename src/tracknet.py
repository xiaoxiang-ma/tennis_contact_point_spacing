"""TrackNet model for tennis ball detection.

Based on the architecture from https://github.com/yastrebksv/TrackNet
Pre-trained weights available for download.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm


# Google Drive file ID for pre-trained weights
WEIGHTS_GDRIVE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
WEIGHTS_FILENAME = "tracknet_weights.pth"


def download_weights(save_dir: str = "weights") -> str:
    """Download pre-trained TrackNet weights from Google Drive.

    Args:
        save_dir: Directory to save weights to.

    Returns:
        Path to the downloaded weights file.
    """
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, WEIGHTS_FILENAME)

    if os.path.exists(weights_path):
        return weights_path

    print(f"Downloading TrackNet weights to {weights_path}...")

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={WEIGHTS_GDRIVE_ID}"
        gdown.download(url, weights_path, quiet=False)
    except ImportError:
        # Fallback: use requests with Drive direct download
        import requests

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={WEIGHTS_GDRIVE_ID}"
        response = session.get(url, stream=True)
        token = get_confirm_token(response)

        if token:
            url = f"{url}&confirm={token}"
            response = session.get(url, stream=True)

        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    print(f"Weights downloaded to {weights_path}")
    return weights_path


class ConvBlock(nn.Module):
    """Convolutional block: Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TrackNet(nn.Module):
    """TrackNet architecture for ball detection.

    Takes 3 consecutive frames (9 channels) as input and outputs a heatmap
    of ball location probability.

    Architecture based on VGG16 encoder with upsampling decoder.
    Input: (batch, 9, 360, 640)
    Output: (batch, 1, 360, 640)
    """

    def __init__(self, input_channels: int = 9, output_channels: int = 1):
        super().__init__()

        # Encoder (VGG-style)
        self.conv1 = nn.Sequential(
            ConvBlock(input_channels, 64),
            ConvBlock(64, 64),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv1 = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv2 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 128),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv3 = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
        )

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)

        # Decoder
        x = self.up1(x)
        x = self.deconv1(x)

        x = self.up2(x)
        x = self.deconv2(x)

        x = self.up3(x)
        x = self.deconv3(x)

        x = self.final(x)
        x = self.sigmoid(x)

        return x


class TrackNetDetector:
    """High-level interface for TrackNet ball detection."""

    # TrackNet input dimensions
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 360

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize TrackNet detector.

        Args:
            weights_path: Path to weights file. If None, downloads automatically.
            device: 'cuda', 'cpu', or None for auto-detect.
            confidence_threshold: Minimum heatmap value to consider a detection.
        """
        self.confidence_threshold = confidence_threshold

        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Download weights if needed
        if weights_path is None:
            weights_path = download_weights()

        # Load model
        self.model = TrackNet()

        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)
            print(f"Loaded TrackNet weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights ({e}). Using random initialization.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_frames(
        self,
        frames: List[np.ndarray],
    ) -> torch.Tensor:
        """Preprocess 3 consecutive frames for TrackNet input.

        Args:
            frames: List of 3 BGR frames.

        Returns:
            Tensor of shape (1, 9, 360, 640).
        """
        processed = []
        for frame in frames:
            # Resize to TrackNet input size
            resized = cv2.resize(frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
            # Convert BGR to RGB and normalize
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            processed.append(normalized)

        # Stack frames: (3, H, W, 3) -> (H, W, 9)
        stacked = np.concatenate(processed, axis=2)
        # Transpose to (9, H, W)
        tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).float()
        # Add batch dimension
        return tensor.unsqueeze(0)

    def postprocess_heatmap(
        self,
        heatmap: np.ndarray,
        original_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float]]:
        """Extract ball position from heatmap.

        Args:
            heatmap: Output heatmap from model (H, W).
            original_size: (width, height) of original frame.

        Returns:
            (x, y, confidence) in original frame coordinates, or None.
        """
        max_val = heatmap.max()

        if max_val < self.confidence_threshold:
            return None

        # Find peak location
        y_pred, x_pred = np.unravel_index(heatmap.argmax(), heatmap.shape)

        # Scale to original resolution
        orig_w, orig_h = original_size
        x_orig = x_pred * orig_w / self.INPUT_WIDTH
        y_orig = y_pred * orig_h / self.INPUT_HEIGHT

        return (float(x_orig), float(y_orig), float(max_val))

    def detect_single(
        self,
        frames: List[np.ndarray],
    ) -> Optional[Tuple[float, float, float]]:
        """Detect ball in a triplet of frames.

        Args:
            frames: List of exactly 3 consecutive BGR frames.

        Returns:
            (x, y, confidence) or None if no detection.
        """
        if len(frames) != 3:
            raise ValueError("TrackNet requires exactly 3 frames")

        original_size = (frames[2].shape[1], frames[2].shape[0])

        # Preprocess
        input_tensor = self.preprocess_frames(frames).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocess
        heatmap = output.squeeze().cpu().numpy()
        return self.postprocess_heatmap(heatmap, original_size)

    def detect_all(
        self,
        frames: List[np.ndarray],
        progress: bool = True,
        frame_skip: int = 1,
    ) -> Dict[int, Tuple[float, float, float]]:
        """Detect ball in all frames of a video.

        Args:
            frames: List of BGR frames.
            progress: Show progress bar.
            frame_skip: Process every Nth frame (1 = all frames).

        Returns:
            Dict mapping frame_num -> (x, y, confidence).
        """
        if len(frames) < 3:
            return {}

        results = {}
        original_size = (frames[0].shape[1], frames[0].shape[0])

        # Build frame indices to process
        indices = list(range(2, len(frames), frame_skip))

        iterator = indices
        if progress:
            iterator = tqdm(indices, desc="TrackNet ball detection")

        for i in iterator:
            triplet = [frames[i-2], frames[i-1], frames[i]]
            detection = self.detect_single(triplet)

            if detection is not None:
                results[i] = detection

        return results

    def get_ball_trajectory(
        self,
        detections: Dict[int, Tuple[float, float, float]],
    ) -> List[Tuple[int, float, float, float]]:
        """Convert detection dict to sorted trajectory list.

        Returns:
            List of (frame_num, x, y, confidence) sorted by frame.
        """
        trajectory = [
            (frame, x, y, conf)
            for frame, (x, y, conf) in detections.items()
        ]
        trajectory.sort(key=lambda t: t[0])
        return trajectory


def extrapolate_ball_position(
    trajectory: List[Tuple[int, float, float, float]],
    target_frame: int,
    window: int = 10,
    min_points: int = 3,
) -> Optional[Tuple[float, float, float]]:
    """Extrapolate ball position for frames where detection failed.

    Uses quadratic (parabolic) fit to recent positions to predict
    ball location at target frame. This handles the parabolic
    trajectory of a tennis ball in flight.

    Args:
        trajectory: List of (frame_num, x, y, confidence) detections.
        target_frame: Frame number to extrapolate to.
        window: Number of preceding frames to use for fitting.
        min_points: Minimum detections required for extrapolation.

    Returns:
        (x, y, confidence) or None if extrapolation not possible.
        Confidence is reduced (0.3-0.5) for extrapolated positions.
    """
    # Get recent detections before target frame
    recent = [t for t in trajectory if t[0] < target_frame]
    recent = recent[-window:]  # Keep only last N

    if len(recent) < min_points:
        return None

    frames = np.array([t[0] for t in recent])
    xs = np.array([t[1] for t in recent])
    ys = np.array([t[2] for t in recent])

    try:
        # Fit quadratic polynomials
        # x(t) = at^2 + bt + c (often linear for x)
        # y(t) = at^2 + bt + c (parabolic due to gravity)

        # Use linear for x (horizontal motion is roughly constant velocity)
        x_coeffs = np.polyfit(frames, xs, 1)
        x_pred = np.polyval(x_coeffs, target_frame)

        # Use quadratic for y (accounts for gravity)
        y_coeffs = np.polyfit(frames, ys, 2)
        y_pred = np.polyval(y_coeffs, target_frame)

        # Confidence decreases with extrapolation distance
        max_frame = frames.max()
        distance = target_frame - max_frame
        base_confidence = 0.5
        decay = 0.05 * distance  # Lose 5% confidence per frame
        confidence = max(0.3, base_confidence - decay)

        return (float(x_pred), float(y_pred), confidence)

    except Exception:
        return None


def interpolate_ball_position(
    trajectory: List[Tuple[int, float, float, float]],
    target_frame: int,
) -> Optional[Tuple[float, float, float]]:
    """Interpolate ball position between two known detections.

    Uses linear interpolation for frames between detections.

    Args:
        trajectory: List of (frame_num, x, y, confidence) detections.
        target_frame: Frame number to interpolate.

    Returns:
        (x, y, confidence) or None if interpolation not possible.
    """
    # Find surrounding detections
    before = [t for t in trajectory if t[0] < target_frame]
    after = [t for t in trajectory if t[0] > target_frame]

    if not before or not after:
        return None

    prev_det = before[-1]  # Closest before
    next_det = after[0]    # Closest after

    f0, x0, y0, c0 = prev_det
    f1, x1, y1, c1 = next_det

    # Linear interpolation
    t = (target_frame - f0) / (f1 - f0)
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)

    # Confidence is average of surrounding, slightly reduced
    confidence = 0.8 * (c0 + c1) / 2

    return (float(x), float(y), float(confidence))
