"""Audio-based contact detection for tennis videos.

Detects ball-racket impact sounds to identify contact frames.
Tennis ball impacts produce a distinctive sharp sound in the 1-4kHz range.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


def extract_audio_from_video(video_path: str, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """Extract audio track from video file.

    Args:
        video_path: Path to video file.
        sample_rate: Target sample rate for audio.

    Returns:
        Tuple of (audio_samples as 1D numpy array, sample_rate).

    Raises:
        ImportError: If moviepy is not installed.
        ValueError: If video has no audio track.
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "moviepy is required for audio extraction. "
            "Install with: pip install moviepy"
        )

    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        raise ValueError(f"Video {video_path} has no audio track")

    # Extract audio as numpy array
    audio = clip.audio.to_soundarray(fps=sample_rate)

    # Convert to mono if stereo
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    clip.close()

    return audio.astype(np.float32), sample_rate


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_freq: float = 1000.0,
    high_freq: float = 4000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply bandpass filter to isolate tennis ball impact frequencies.

    Tennis ball-racket impacts produce sounds primarily in 1-4kHz range.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        low_freq: Low cutoff frequency.
        high_freq: High cutoff frequency.
        order: Filter order.

    Returns:
        Filtered audio samples.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        warnings.warn("scipy not available, skipping bandpass filter")
        return audio

    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Ensure frequencies are in valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)

    return filtered.astype(np.float32)


def compute_envelope(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: float = 10.0,
) -> np.ndarray:
    """Compute amplitude envelope of audio signal.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        window_ms: Smoothing window in milliseconds.

    Returns:
        Amplitude envelope (same length as input).
    """
    # Rectify (absolute value)
    rectified = np.abs(audio)

    # Smoothing window
    window_samples = int(sample_rate * window_ms / 1000)
    window_samples = max(1, window_samples)

    # Simple moving average for smoothing
    kernel = np.ones(window_samples) / window_samples
    envelope = np.convolve(rectified, kernel, mode='same')

    return envelope


def find_impact_peaks(
    envelope: np.ndarray,
    sample_rate: int,
    video_fps: float,
    min_peak_height: Optional[float] = None,
    noise_percentile: float = 75.0,
    peak_threshold_factor: float = 3.0,
    min_gap_ms: float = 200.0,
) -> List[Tuple[int, float]]:
    """Find impact sound peaks in audio envelope.

    Args:
        envelope: Amplitude envelope.
        sample_rate: Audio sample rate.
        video_fps: Video frame rate (for frame number conversion).
        min_peak_height: Absolute minimum peak height. If None, computed adaptively.
        noise_percentile: Percentile of envelope to use as noise floor.
        peak_threshold_factor: Peak must exceed noise floor by this factor.
        min_gap_ms: Minimum gap between peaks in milliseconds.

    Returns:
        List of (frame_number, confidence) for detected impacts.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        # Fallback to simple peak detection
        return _find_peaks_simple(
            envelope, sample_rate, video_fps,
            min_peak_height, noise_percentile, peak_threshold_factor, min_gap_ms
        )

    # Compute adaptive threshold based on noise floor
    noise_floor = np.percentile(envelope, noise_percentile)
    threshold = noise_floor * peak_threshold_factor

    if min_peak_height is not None:
        threshold = max(threshold, min_peak_height)

    # Minimum distance between peaks (in samples)
    min_distance = int(sample_rate * min_gap_ms / 1000)

    # Find peaks
    peaks, properties = find_peaks(
        envelope,
        height=threshold,
        distance=min_distance,
        prominence=threshold * 0.5,
    )

    if len(peaks) == 0:
        return []

    # Convert to frame numbers and compute confidence
    results = []
    peak_heights = properties['peak_heights']
    max_height = peak_heights.max()

    for peak_idx, height in zip(peaks, peak_heights):
        # Convert sample index to time, then to frame number
        time_sec = peak_idx / sample_rate
        frame_num = int(time_sec * video_fps)

        # Confidence based on peak prominence
        confidence = min(0.4 + 0.6 * (height / max_height), 1.0)
        results.append((frame_num, confidence))

    return results


def _find_peaks_simple(
    envelope: np.ndarray,
    sample_rate: int,
    video_fps: float,
    min_peak_height: Optional[float],
    noise_percentile: float,
    peak_threshold_factor: float,
    min_gap_ms: float,
) -> List[Tuple[int, float]]:
    """Simple peak detection fallback when scipy is not available."""
    noise_floor = np.percentile(envelope, noise_percentile)
    threshold = noise_floor * peak_threshold_factor

    if min_peak_height is not None:
        threshold = max(threshold, min_peak_height)

    min_gap_samples = int(sample_rate * min_gap_ms / 1000)

    peaks = []
    i = 1
    while i < len(envelope) - 1:
        if envelope[i] > threshold:
            # Check if local maximum
            if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
                peaks.append((i, envelope[i]))
                i += min_gap_samples  # Skip ahead to enforce minimum gap
                continue
        i += 1

    if not peaks:
        return []

    max_height = max(h for _, h in peaks)
    results = []
    for peak_idx, height in peaks:
        time_sec = peak_idx / sample_rate
        frame_num = int(time_sec * video_fps)
        confidence = min(0.4 + 0.6 * (height / max_height), 1.0)
        results.append((frame_num, confidence))

    return results


def detect_contacts_audio(
    video_path: str,
    video_fps: float,
    sample_rate: int = 22050,
    low_freq: float = 1000.0,
    high_freq: float = 4000.0,
    min_gap_ms: float = 300.0,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Detect contact frames from audio track.

    Main entry point for audio-based contact detection.

    Args:
        video_path: Path to video file.
        video_fps: Video frame rate.
        sample_rate: Audio sample rate for processing.
        low_freq: Bandpass filter low cutoff.
        high_freq: Bandpass filter high cutoff.
        min_gap_ms: Minimum gap between contacts (ms).
        debug: Print debug information.

    Returns:
        List of (frame_number, confidence) for detected contacts.
    """
    # Extract audio
    try:
        audio, sr = extract_audio_from_video(video_path, sample_rate)
    except (ImportError, ValueError) as e:
        if debug:
            print(f"  [audio] Could not extract audio: {e}")
        return []

    if debug:
        duration = len(audio) / sr
        print(f"  [audio] Extracted {duration:.2f}s of audio at {sr}Hz")

    # Bandpass filter to isolate impact frequencies
    filtered = bandpass_filter(audio, sr, low_freq, high_freq)

    # Compute envelope
    envelope = compute_envelope(filtered, sr, window_ms=5.0)

    if debug:
        print(f"  [audio] Envelope range: {envelope.min():.4f} - {envelope.max():.4f}")

    # Find peaks
    peaks = find_impact_peaks(
        envelope, sr, video_fps,
        min_gap_ms=min_gap_ms,
    )

    if debug:
        print(f"  [audio] Found {len(peaks)} impact peaks")
        for frame, conf in peaks:
            time_sec = frame / video_fps
            print(f"    Frame {frame} ({time_sec:.2f}s): confidence {conf:.2f}")

    return peaks


def get_audio_envelope_for_debug(
    video_path: str,
    sample_rate: int = 22050,
    low_freq: float = 1000.0,
    high_freq: float = 4000.0,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Get audio envelope for visualization/debugging.

    Returns:
        Tuple of (raw_audio, sample_rate, filtered_envelope).
    """
    audio, sr = extract_audio_from_video(video_path, sample_rate)
    filtered = bandpass_filter(audio, sr, low_freq, high_freq)
    envelope = compute_envelope(filtered, sr, window_ms=5.0)
    return audio, sr, envelope
