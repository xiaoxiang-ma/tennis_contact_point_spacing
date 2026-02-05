"""Audio-based contact detection for tennis videos.

Detects ball-racket impact sounds to identify contact frames.

Contact sound characteristics:
- Sharp transient impulse (5-10ms duration)
- Frequency range: 1-4 kHz (string vibration)
- Distinct from:
  - Shoe squeaks: longer duration, 500Hz-2kHz
  - Ball bounces: lower freq, 200-800Hz
  - Grunts: lower freq, happens before contact

Processing approach:
1. Extract audio at high sample rate (48kHz)
2. High-pass filter at 800Hz (remove bounces, grunts)
3. Bandpass 1-4kHz for string impact frequencies
4. Detect sharp transients (<20ms duration)
5. Pick strongest transient within swing window
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


# Default sample rate for high-fidelity transient detection
DEFAULT_SAMPLE_RATE = 48000

# Contact sound characteristics
CONTACT_FREQ_LOW = 1000.0  # Hz - lower bound of string vibration
CONTACT_FREQ_HIGH = 4000.0  # Hz - upper bound
HIGHPASS_CUTOFF = 800.0  # Hz - filter out bounces/grunts
TRANSIENT_MAX_DURATION_MS = 20.0  # Contact transients are <20ms
TRANSIENT_IMPULSE_MS = 5.0  # Typical impact is 5-10ms


def extract_audio_from_video(video_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Extract audio track from video file.

    Args:
        video_path: Path to video file.
        sample_rate: Target sample rate for audio.

    Returns:
        Tuple of (audio_samples as 1D numpy array, sample_rate).

    Raises:
        ImportError: If neither moviepy nor ffmpeg extraction works.
        ValueError: If video has no audio track.
    """
    # Try ffmpeg directly first (more reliable than moviepy)
    audio = _extract_audio_ffmpeg(video_path, sample_rate)
    if audio is not None:
        return audio, sample_rate

    # Fallback to moviepy
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError(
            "Audio extraction requires either ffmpeg in PATH or moviepy. "
            "Install with: pip install moviepy"
        )

    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        raise ValueError(f"Video {video_path} has no audio track")

    try:
        # Extract audio as numpy array
        # Use buffersize to avoid moviepy's chunking issues with numpy.stack
        audio = clip.audio.to_soundarray(fps=sample_rate, buffersize=50000)

        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
    finally:
        clip.close()

    return audio.astype(np.float32), sample_rate


def _extract_audio_ffmpeg(video_path: str, sample_rate: int) -> Optional[np.ndarray]:
    """Extract audio using ffmpeg directly (bypasses moviepy issues).

    Returns:
        Audio samples as 1D numpy array, or None if ffmpeg not available.
    """
    import subprocess
    import tempfile
    import os

    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    # Create temp file for raw audio
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Extract audio as raw PCM (signed 16-bit little-endian, mono)
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Raw PCM
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-f', 's16le',  # Raw format
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            return None

        # Read raw audio
        with open(tmp_path, 'rb') as f:
            raw_data = f.read()

        if len(raw_data) == 0:
            return None

        # Convert to numpy array (16-bit signed int -> float32 normalized)
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0  # Normalize to [-1, 1]

        return audio
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def highpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_freq: float = HIGHPASS_CUTOFF,
    order: int = 4,
) -> np.ndarray:
    """Apply high-pass filter to remove low-frequency noise.

    Removes ball bounces (200-800Hz) and grunts (low freq) from audio.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order.

    Returns:
        High-pass filtered audio samples.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        warnings.warn("scipy not available, skipping high-pass filter")
        return audio

    nyquist = sample_rate / 2
    cutoff = cutoff_freq / nyquist

    # Ensure cutoff is in valid range
    cutoff = max(0.01, min(cutoff, 0.99))

    b, a = butter(order, cutoff, btype='high')
    filtered = filtfilt(b, a, audio)

    return filtered.astype(np.float32)


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_freq: float = CONTACT_FREQ_LOW,
    high_freq: float = CONTACT_FREQ_HIGH,
    order: int = 4,
) -> np.ndarray:
    """Apply bandpass filter to isolate tennis ball impact frequencies.

    Tennis ball-racket impacts produce sounds primarily in 1-4kHz range
    (string vibration frequencies).

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
    window_ms: float = TRANSIENT_IMPULSE_MS,
) -> np.ndarray:
    """Compute amplitude envelope of audio signal.

    Uses a short window (5ms default) to capture sharp transients.

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


def measure_transient_duration(
    envelope: np.ndarray,
    peak_idx: int,
    sample_rate: int,
    threshold_ratio: float = 0.5,
) -> float:
    """Measure the duration of a transient at a peak.

    Contact transients are very short (5-10ms). Shoe squeaks and other
    sounds are longer duration.

    Args:
        envelope: Amplitude envelope.
        peak_idx: Index of the peak in envelope.
        sample_rate: Sample rate in Hz.
        threshold_ratio: Fraction of peak height to define transient bounds.

    Returns:
        Duration of transient in milliseconds.
    """
    peak_height = envelope[peak_idx]
    threshold = peak_height * threshold_ratio

    # Find start of transient (search backwards)
    start_idx = peak_idx
    while start_idx > 0 and envelope[start_idx] > threshold:
        start_idx -= 1

    # Find end of transient (search forwards)
    end_idx = peak_idx
    while end_idx < len(envelope) - 1 and envelope[end_idx] > threshold:
        end_idx += 1

    duration_samples = end_idx - start_idx
    duration_ms = (duration_samples / sample_rate) * 1000

    return duration_ms


def is_valid_contact_transient(
    envelope: np.ndarray,
    peak_idx: int,
    sample_rate: int,
    max_duration_ms: float = TRANSIENT_MAX_DURATION_MS,
) -> Tuple[bool, float]:
    """Check if a peak is a valid contact transient based on duration.

    Contact sounds are sharp transients (<20ms). Longer sounds are likely
    shoe squeaks or other noise.

    Args:
        envelope: Amplitude envelope.
        peak_idx: Index of the peak.
        sample_rate: Sample rate in Hz.
        max_duration_ms: Maximum duration for valid contact transient.

    Returns:
        Tuple of (is_valid, duration_ms).
    """
    duration_ms = measure_transient_duration(envelope, peak_idx, sample_rate)
    is_valid = duration_ms <= max_duration_ms
    return is_valid, duration_ms


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
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    low_freq: float = CONTACT_FREQ_LOW,
    high_freq: float = CONTACT_FREQ_HIGH,
    min_gap_ms: float = 300.0,
    use_transient_validation: bool = True,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Detect contact frames from audio track.

    Main entry point for audio-based contact detection.

    Processing pipeline:
    1. Extract audio at high sample rate (48kHz default)
    2. High-pass filter at 800Hz to remove bounces/grunts
    3. Bandpass filter 1-4kHz for string impact frequencies
    4. Compute amplitude envelope with short window (5ms)
    5. Find peaks and validate transient duration (<20ms)
    6. Return frame numbers with confidence scores

    Args:
        video_path: Path to video file.
        video_fps: Video frame rate.
        sample_rate: Audio sample rate for processing (48kHz recommended).
        low_freq: Bandpass filter low cutoff (1kHz default).
        high_freq: Bandpass filter high cutoff (4kHz default).
        min_gap_ms: Minimum gap between contacts (ms).
        use_transient_validation: If True, filter out non-transient sounds.
        debug: Print debug information.

    Returns:
        List of (frame_number, confidence) for detected contacts.
    """
    # Extract audio at high sample rate for transient detection
    try:
        audio, sr = extract_audio_from_video(video_path, sample_rate)
    except (ImportError, ValueError) as e:
        if debug:
            print(f"  [audio] Could not extract audio: {e}")
        return []

    if debug:
        duration = len(audio) / sr
        print(f"  [audio] Extracted {duration:.2f}s of audio at {sr}Hz")

    # Step 1: High-pass filter to remove bounces (200-800Hz) and grunts
    hp_filtered = highpass_filter(audio, sr, cutoff_freq=HIGHPASS_CUTOFF)

    if debug:
        print(f"  [audio] Applied high-pass filter at {HIGHPASS_CUTOFF}Hz")

    # Step 2: Bandpass filter to isolate string impact frequencies (1-4kHz)
    filtered = bandpass_filter(hp_filtered, sr, low_freq, high_freq)

    if debug:
        print(f"  [audio] Applied bandpass filter {low_freq}-{high_freq}Hz")

    # Step 3: Compute envelope with short window for transient detection
    envelope = compute_envelope(filtered, sr, window_ms=TRANSIENT_IMPULSE_MS)

    if debug:
        print(f"  [audio] Envelope range: {envelope.min():.4f} - {envelope.max():.4f}")

    # Step 4: Find peaks (candidate impacts)
    peaks = find_impact_peaks(
        envelope, sr, video_fps,
        min_gap_ms=min_gap_ms,
    )

    if debug:
        print(f"  [audio] Found {len(peaks)} candidate peaks")

    # Step 5: Validate transient duration (contact sounds are <20ms)
    if use_transient_validation and peaks:
        validated_peaks = []
        for frame_num, confidence in peaks:
            # Convert frame back to sample index
            time_sec = frame_num / video_fps
            peak_idx = int(time_sec * sr)
            peak_idx = max(0, min(len(envelope) - 1, peak_idx))

            is_valid, duration_ms = is_valid_contact_transient(
                envelope, peak_idx, sr, max_duration_ms=TRANSIENT_MAX_DURATION_MS
            )

            if is_valid:
                # Boost confidence for very short transients (more likely contact)
                if duration_ms <= 10.0:
                    confidence = min(confidence + 0.1, 1.0)
                validated_peaks.append((frame_num, confidence))
                if debug:
                    print(f"    Frame {frame_num}: VALID transient ({duration_ms:.1f}ms)")
            elif debug:
                print(f"    Frame {frame_num}: REJECTED - too long ({duration_ms:.1f}ms)")

        peaks = validated_peaks

    if debug:
        print(f"  [audio] Final: {len(peaks)} valid contact sounds")
        for frame, conf in peaks:
            time_sec = frame / video_fps
            print(f"    Frame {frame} ({time_sec:.2f}s): confidence {conf:.2f}")

    return peaks


def get_audio_envelope_for_debug(
    video_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    low_freq: float = CONTACT_FREQ_LOW,
    high_freq: float = CONTACT_FREQ_HIGH,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Get audio envelope for visualization/debugging.

    Returns:
        Tuple of (raw_audio, sample_rate, filtered_envelope).
    """
    audio, sr = extract_audio_from_video(video_path, sample_rate)
    hp_filtered = highpass_filter(audio, sr, cutoff_freq=HIGHPASS_CUTOFF)
    filtered = bandpass_filter(hp_filtered, sr, low_freq, high_freq)
    envelope = compute_envelope(filtered, sr, window_ms=TRANSIENT_IMPULSE_MS)
    return audio, sr, envelope


def detect_contacts_audio_advanced(
    video_path: str,
    video_fps: float,
    swing_window: Optional[Tuple[int, int]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    debug: bool = False,
) -> List[Tuple[int, float, dict]]:
    """Advanced audio contact detection with detailed analysis.

    Returns detailed information about each detected contact including
    transient characteristics for validation.

    Args:
        video_path: Path to video file.
        video_fps: Video frame rate.
        swing_window: Optional (start_frame, end_frame) to constrain search.
        sample_rate: Audio sample rate.
        debug: Print debug information.

    Returns:
        List of (frame_number, confidence, details_dict) for detected contacts.
        details_dict contains: duration_ms, peak_amplitude, frequency_content.
    """
    # Extract audio
    try:
        audio, sr = extract_audio_from_video(video_path, sample_rate)
    except (ImportError, ValueError) as e:
        if debug:
            print(f"  [audio] Could not extract audio: {e}")
        return []

    duration_sec = len(audio) / sr

    # Apply filtering
    hp_filtered = highpass_filter(audio, sr, cutoff_freq=HIGHPASS_CUTOFF)
    filtered = bandpass_filter(hp_filtered, sr, CONTACT_FREQ_LOW, CONTACT_FREQ_HIGH)
    envelope = compute_envelope(filtered, sr, window_ms=TRANSIENT_IMPULSE_MS)

    # Constrain to swing window if provided
    if swing_window is not None:
        start_frame, end_frame = swing_window
        start_sample = int((start_frame / video_fps) * sr)
        end_sample = int((end_frame / video_fps) * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(envelope), end_sample)
    else:
        start_sample = 0
        end_sample = len(envelope)

    # Find peaks in window
    window_envelope = envelope[start_sample:end_sample]

    try:
        from scipy.signal import find_peaks

        noise_floor = np.percentile(window_envelope, 75)
        threshold = noise_floor * 3.0
        min_distance = int(sr * 0.2)  # 200ms between peaks

        peaks, properties = find_peaks(
            window_envelope,
            height=threshold,
            distance=min_distance,
            prominence=threshold * 0.5,
        )
    except ImportError:
        # Simple fallback
        threshold = np.percentile(window_envelope, 90)
        peaks = []
        for i in range(1, len(window_envelope) - 1):
            if window_envelope[i] > threshold:
                if window_envelope[i] > window_envelope[i-1] and window_envelope[i] > window_envelope[i+1]:
                    peaks.append(i)
        peaks = np.array(peaks)
        properties = {'peak_heights': window_envelope[peaks] if len(peaks) > 0 else np.array([])}

    if len(peaks) == 0:
        return []

    # Analyze each peak
    results = []
    max_height = properties['peak_heights'].max() if len(properties['peak_heights']) > 0 else 1.0

    for i, peak_idx in enumerate(peaks):
        global_peak_idx = peak_idx + start_sample

        # Measure transient duration
        is_valid, duration_ms = is_valid_contact_transient(
            envelope, global_peak_idx, sr, max_duration_ms=TRANSIENT_MAX_DURATION_MS
        )

        if not is_valid:
            continue

        # Convert to frame number
        time_sec = global_peak_idx / sr
        frame_num = int(time_sec * video_fps)

        # Compute confidence
        height = properties['peak_heights'][i] if i < len(properties['peak_heights']) else envelope[global_peak_idx]
        base_confidence = 0.4 + 0.6 * (height / max_height)

        # Boost for short transients (more characteristic of contact)
        if duration_ms <= 10.0:
            base_confidence = min(base_confidence + 0.15, 1.0)
        elif duration_ms <= 15.0:
            base_confidence = min(base_confidence + 0.05, 1.0)

        confidence = min(base_confidence, 1.0)

        details = {
            'duration_ms': duration_ms,
            'peak_amplitude': float(height),
            'time_sec': time_sec,
            'is_sharp_transient': duration_ms <= 10.0,
        }

        results.append((frame_num, confidence, details))

        if debug:
            print(f"    Frame {frame_num} ({time_sec:.2f}s): conf={confidence:.2f}, "
                  f"duration={duration_ms:.1f}ms, amp={height:.4f}")

    # Sort by confidence (pick strongest in swing window)
    results.sort(key=lambda x: -x[1])

    return results
