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
    """Download pre-trained TrackNet weights from Google Drive."""
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
    """Convolutional block: Conv -> ReLU -> BatchNorm (matches yastrebksv/TrackNet)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pad: int = 1, stride: int = 1, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    """BallTrackerNet architecture matching yastrebksv/TrackNet exactly.

    Input: (batch, 9, H, W) - 3 consecutive RGB frames concatenated
    Output: (batch, 256, H, W) - heatmap with 256 classes
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # Encoder
        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)

        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)

        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)

        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=out_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        # Decoder
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)

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
        save_debug_frames: bool = False,
        debug_output_dir: Optional[str] = None,
    ):
        """Initialize TrackNet detector.

        Args:
            weights_path: Path to weights file. If None, downloads automatically.
            device: 'cuda', 'cpu', or None for auto-detect.
            confidence_threshold: Minimum confidence to consider a detection.
            save_debug_frames: Whether to save debug visualizations.
            debug_output_dir: Directory to save debug frames.
        """
        self.confidence_threshold = confidence_threshold
        self.save_debug_frames = save_debug_frames
        self.debug_output_dir = debug_output_dir or "/content/output/tracknet_debug"

        if save_debug_frames:
            os.makedirs(self.debug_output_dir, exist_ok=True)

        # Set device with proper detection
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("  Using CPU (GPU not available)")
        self.device = torch.device(device)

        # Download weights if needed
        if weights_path is None:
            weights_path = download_weights()

        # Load model
        self.model = BallTrackerNet(out_channels=256)
        self.weights_loaded = False

        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)
            self.weights_loaded = True
            print(f"  TrackNet weights loaded successfully from {weights_path}")
        except Exception as e:
            print(f"  ERROR: Could not load weights: {e}")
            print("  Model will use random weights (detection will not work!)")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess 3 consecutive frames for TrackNet input.

        Args:
            frames: List of 3 BGR frames.

        Returns:
            Tensor of shape (1, 9, 360, 640).
        """
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            # Transpose to (C, H, W)
            transposed = normalized.transpose(2, 0, 1)
            processed.append(transposed)

        # Stack along channel dimension: (9, H, W)
        stacked = np.concatenate(processed, axis=0)
        tensor = torch.from_numpy(stacked).float()
        return tensor.unsqueeze(0)  # Add batch dimension

    def postprocess_output(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int],
        frame_idx: Optional[int] = None,
        input_frame: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract ball position from model output.

        The model outputs 256 channels. We take argmax across channels to get
        a class prediction per pixel, then find pixels with class > 0 (class 0 = no ball).

        Args:
            output: Model output tensor (1, 256, H, W).
            original_size: (width, height) of original frame.
            frame_idx: Frame index for debug output.
            input_frame: Original frame for debug visualization.

        Returns:
            (x, y, confidence) in original frame coordinates, or None.
        """
        # Get predictions: (H, W) with values 0-255
        output_np = output.squeeze(0).cpu().numpy()  # (256, H, W)

        # Sum across all non-zero classes to get ball probability heatmap
        # Class 0 is background, classes 1-255 indicate ball presence
        ball_prob = output_np[1:].sum(axis=0)  # (H, W)

        # Alternatively, use argmax and check for non-zero class
        class_pred = output_np.argmax(axis=0)  # (H, W)
        ball_mask = class_pred > 0

        # Save debug visualization
        if self.save_debug_frames and frame_idx is not None and input_frame is not None:
            self._save_debug_frame(input_frame, ball_prob, class_pred, frame_idx)

        # Find ball position from probability map
        max_prob = ball_prob.max()

        if max_prob < self.confidence_threshold or not ball_mask.any():
            return None

        # Find centroid of detected ball region
        y_coords, x_coords = np.where(ball_mask)

        if len(x_coords) == 0:
            return None

        x_center = x_coords.mean()
        y_center = y_coords.mean()

        # Scale to original resolution
        orig_w, orig_h = original_size
        x_orig = x_center * orig_w / self.INPUT_WIDTH
        y_orig = y_center * orig_h / self.INPUT_HEIGHT

        # Confidence based on detection area and max probability
        confidence = min(max_prob / 10.0, 1.0)  # Normalize

        return (float(x_orig), float(y_orig), float(confidence))

    def _save_debug_frame(
        self,
        input_frame: np.ndarray,
        ball_prob: np.ndarray,
        class_pred: np.ndarray,
        frame_idx: int,
    ):
        """Save debug visualization showing detection heatmap."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original frame
        frame_rgb = cv2.cvtColor(
            cv2.resize(input_frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT)),
            cv2.COLOR_BGR2RGB
        )
        axes[0].imshow(frame_rgb)
        axes[0].set_title(f"Frame {frame_idx}")
        axes[0].axis('off')

        # Ball probability heatmap
        axes[1].imshow(ball_prob, cmap='hot')
        axes[1].set_title(f"Ball Probability (max={ball_prob.max():.2f})")
        axes[1].axis('off')

        # Class predictions
        axes[2].imshow(class_pred, cmap='viridis')
        axes[2].set_title(f"Class Prediction (unique={len(np.unique(class_pred))})")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_output_dir, f"frame_{frame_idx:04d}.png"), dpi=100)
        plt.close()

    def detect_single(
        self,
        frames: List[np.ndarray],
        frame_idx: Optional[int] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """Detect ball in a triplet of frames.

        Args:
            frames: List of exactly 3 consecutive BGR frames.
            frame_idx: Frame index for debug output.

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
        return self.postprocess_output(
            output, original_size,
            frame_idx=frame_idx,
            input_frame=frames[2] if self.save_debug_frames else None
        )

    def detect_all(
        self,
        frames: List[np.ndarray],
        progress: bool = True,
        frame_skip: int = 1,
        batch_size: int = 8,
    ) -> Dict[int, Tuple[float, float, float]]:
        """Detect ball in all frames of a video.

        Args:
            frames: List of BGR frames.
            progress: Show progress bar.
            frame_skip: Process every Nth frame (1 = all frames).
            batch_size: Number of frame triplets to process at once.

        Returns:
            Dict mapping frame_num -> (x, y, confidence).
        """
        if len(frames) < 3:
            return {}

        if not self.weights_loaded:
            print("WARNING: Weights not loaded, detection will not work!")

        results = {}
        original_size = (frames[0].shape[1], frames[0].shape[0])

        # Build frame indices to process
        indices = list(range(2, len(frames), frame_skip))

        # Process in batches for efficiency
        iterator = range(0, len(indices), batch_size)
        if progress:
            iterator = tqdm(iterator, desc="TrackNet ball detection",
                          total=(len(indices) + batch_size - 1) // batch_size)

        for batch_start in iterator:
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Prepare batch
            batch_tensors = []
            for i in batch_indices:
                triplet = [frames[i-2], frames[i-1], frames[i]]
                tensor = self.preprocess_frames(triplet)
                batch_tensors.append(tensor)

            if not batch_tensors:
                continue

            # Stack into single batch
            batch = torch.cat(batch_tensors, dim=0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(batch)

            # Process each output
            for j, i in enumerate(batch_indices):
                output = outputs[j:j+1]
                result = self.postprocess_output(
                    output, original_size,
                    frame_idx=i if self.save_debug_frames else None,
                    input_frame=frames[i] if self.save_debug_frames else None
                )
                if result is not None:
                    results[i] = result

                # Save first few debug frames regardless
                if self.save_debug_frames and i < 10:
                    output_np = output.squeeze(0).cpu().numpy()
                    ball_prob = output_np[1:].sum(axis=0)
                    class_pred = output_np.argmax(axis=0)
                    self._save_debug_frame(frames[i], ball_prob, class_pred, i)

        return results

    def get_ball_trajectory(
        self,
        detections: Dict[int, Tuple[float, float, float]],
    ) -> List[Tuple[int, float, float, float]]:
        """Convert detection dict to sorted trajectory list."""
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
    """Extrapolate ball position using parabolic fit."""
    recent = [t for t in trajectory if t[0] < target_frame]
    recent = recent[-window:]

    if len(recent) < min_points:
        return None

    frames = np.array([t[0] for t in recent])
    xs = np.array([t[1] for t in recent])
    ys = np.array([t[2] for t in recent])

    try:
        x_coeffs = np.polyfit(frames, xs, 1)
        x_pred = np.polyval(x_coeffs, target_frame)

        y_coeffs = np.polyfit(frames, ys, 2)
        y_pred = np.polyval(y_coeffs, target_frame)

        max_frame = frames.max()
        distance = target_frame - max_frame
        confidence = max(0.3, 0.5 - 0.05 * distance)

        return (float(x_pred), float(y_pred), confidence)
    except Exception:
        return None


def interpolate_ball_position(
    trajectory: List[Tuple[int, float, float, float]],
    target_frame: int,
) -> Optional[Tuple[float, float, float]]:
    """Interpolate ball position between two known detections."""
    before = [t for t in trajectory if t[0] < target_frame]
    after = [t for t in trajectory if t[0] > target_frame]

    if not before or not after:
        return None

    prev_det = before[-1]
    next_det = after[0]

    f0, x0, y0, c0 = prev_det
    f1, x1, y1, c1 = next_det

    t = (target_frame - f0) / (f1 - f0)
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    confidence = 0.8 * (c0 + c1) / 2

    return (float(x), float(y), float(confidence))
