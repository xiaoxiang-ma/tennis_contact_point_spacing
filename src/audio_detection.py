"""Audio-based contact detection for tennis videos.

Detects ball-racket impact sounds to identify contact frames.
Tennis ball impacts produce a distinctive short, symmetric thump in the
1-4kHz range (~5-20ms duration). This module distinguishes true impacts
from longer, asymmetric sounds like shoe screeches by analyzing peak shape
(FWHM duration and rise/fall symmetry).
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
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-f', 's16le',
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            return None

        with open(tmp_path, 'rb') as f:
            raw_data = f.read()

        if len(raw_data) == 0:
            return None

        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0

        return audio
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
    rectified = np.abs(audio)

    window_samples = int(sample_rate * window_ms / 1000)
    window_samples = max(1, window_samples)

    kernel = np.ones(window_samples) / window_samples
    envelope = np.convolve(rectified, kernel, mode='same')

    return envelope


def _measure_peak_shape(
    envelope: np.ndarray,
    peak_idx: int,
    sample_rate: int,
) -> Tuple[float, float]:
    """Measure shape properties of a peak in the envelope.

    Tennis ball contacts are short (~5-20ms) and symmetric (sharp attack,
    sharp decay). Shoe screeches are longer (~50-200ms) and asymmetric.

    Args:
        envelope: Amplitude envelope.
        peak_idx: Index of the peak sample.
        sample_rate: Audio sample rate.

    Returns:
        (fwhm_ms, symmetry) where:
        - fwhm_ms: Full width at half maximum in milliseconds.
        - symmetry: rise_time / fall_time ratio, clamped to [0, 1].
          1.0 = perfectly symmetric, 0.0 = extremely asymmetric.
    """
    peak_val = envelope[peak_idx]
    half_max = peak_val / 2.0

    # Walk left until below half max
    left = peak_idx
    while left > 0 and envelope[left] > half_max:
        left -= 1

    # Walk right until below half max
    right = peak_idx
    while right < len(envelope) - 1 and envelope[right] > half_max:
        right += 1

    fwhm_samples = right - left
    fwhm_ms = fwhm_samples * 1000.0 / sample_rate

    rise_samples = peak_idx - left
    fall_samples = right - peak_idx

    if rise_samples > 0 and fall_samples > 0:
        symmetry = min(rise_samples, fall_samples) / max(rise_samples, fall_samples)
    else:
        symmetry = 0.0

    return fwhm_ms, symmetry


def _impact_score(
    height: float,
    max_height: float,
    fwhm_ms: float,
    symmetry: float,
    max_fwhm_ms: float = 40.0,
) -> float:
    """Compute composite score that favors short, symmetric impulses.

    Ball contacts are short and symmetric. Shoe screeches are long and
    asymmetric. This score penalizes wide and asymmetric peaks so that
    the true contact wins NMS even if the screech is louder.

    Narrowness uses a continuous penalty (ideal_fwhm / actual_fwhm)
    rather than a binary threshold. A 12ms peak scores ~1.0 while a
    24ms peak scores ~0.5, even though both are under max_fwhm_ms.

    Args:
        height: Peak amplitude.
        max_height: Maximum peak amplitude across all candidates.
        fwhm_ms: Full width at half maximum.
        symmetry: Rise/fall symmetry ratio (0-1).
        max_fwhm_ms: Hard reject threshold. Also controls the ideal
            FWHM reference (ideal = max_fwhm_ms * 0.3, default ~12ms).

    Returns:
        Composite score in [0, 1].
    """
    # Amplitude component (low weight — loudness is unreliable)
    amp_score = height / max_height if max_height > 0 else 0.0

    # Narrowness: continuous penalty using ideal FWHM as reference.
    # ideal_fwhm ~12ms (a real ball contact after 5ms envelope smoothing).
    # A 12ms peak scores ~1.0, a 24ms peak scores ~0.5, a 60ms peak ~0.2.
    # Peaks beyond max_fwhm_ms are hard-capped at a low score.
    ideal_fwhm_ms = max_fwhm_ms * 0.3  # ~12ms when max_fwhm_ms=40
    if fwhm_ms <= ideal_fwhm_ms:
        narrow_score = 1.0
    elif fwhm_ms <= max_fwhm_ms:
        narrow_score = ideal_fwhm_ms / fwhm_ms
    else:
        narrow_score = ideal_fwhm_ms / fwhm_ms * 0.5  # extra penalty beyond hard limit

    # Symmetry score: direct ratio (already 0-1)
    sym_score = symmetry

    # Weighted composite: shape dominates, amplitude is minor
    # 20% amplitude, 40% narrowness, 40% symmetry
    score = 0.2 * amp_score + 0.4 * narrow_score + 0.4 * sym_score

    return score


def find_impact_peaks(
    envelope: np.ndarray,
    sample_rate: int,
    video_fps: float,
    min_peak_height: Optional[float] = None,
    noise_percentile: float = 75.0,
    peak_threshold_factor: float = 3.0,
    min_gap_ms: float = 200.0,
    max_impact_fwhm_ms: float = 40.0,
    debug: bool = False,
) -> List[Tuple[int, float]]:
    """Find impact sound peaks in audio envelope using shape analysis.

    Two-stage approach:
    1. Find ALL candidate peaks above threshold (small min-distance to
       avoid duplicates of the same event, not the full min_gap).
    2. Measure each peak's shape (FWHM, symmetry) and compute a composite
       score that favors short, symmetric impacts over long screeches.
    3. NMS with min_gap using composite score (not raw amplitude).

    Args:
        envelope: Amplitude envelope.
        sample_rate: Audio sample rate.
        video_fps: Video frame rate (for frame number conversion).
        min_peak_height: Absolute minimum peak height. If None, computed adaptively.
        noise_percentile: Percentile of envelope to use as noise floor.
        peak_threshold_factor: Peak must exceed noise floor by this factor.
        min_gap_ms: Minimum gap between peaks in milliseconds.
        max_impact_fwhm_ms: Expected max FWHM of a true ball impact. Peaks
            narrower than this get full narrowness credit; wider peaks are
            penalized. Default 40ms accounts for ~20ms impact + 5ms envelope
            smoothing.
        debug: Print debug info for each candidate peak.

    Returns:
        List of (frame_number, confidence) for detected impacts.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return _find_peaks_simple(
            envelope, sample_rate, video_fps,
            min_peak_height, noise_percentile, peak_threshold_factor, min_gap_ms
        )

    noise_floor = np.percentile(envelope, noise_percentile)
    threshold = noise_floor * peak_threshold_factor

    if min_peak_height is not None:
        threshold = max(threshold, min_peak_height)

    # Stage 1: Find ALL candidate peaks with a small distance (20ms) to
    # avoid merging distinct events into one. We do NOT use min_gap here
    # because that would force scipy to pick by amplitude alone.
    dedup_distance = int(sample_rate * 0.020)  # 20ms dedup
    dedup_distance = max(dedup_distance, 1)

    peaks, properties = find_peaks(
        envelope,
        height=threshold,
        distance=dedup_distance,
        prominence=threshold * 0.3,
    )

    if len(peaks) == 0:
        return []

    # Stage 2: Measure shape of each candidate and compute composite score
    peak_heights = properties['peak_heights']
    max_height = peak_heights.max()

    candidates = []
    for peak_idx, height in zip(peaks, peak_heights):
        fwhm_ms, symmetry = _measure_peak_shape(envelope, peak_idx, sample_rate)
        score = _impact_score(height, max_height, fwhm_ms, symmetry, max_impact_fwhm_ms)

        if debug:
            time_sec = peak_idx / sample_rate
            print(f"    candidate t={time_sec:.3f}s amp={height:.4f} "
                  f"fwhm={fwhm_ms:.1f}ms sym={symmetry:.2f} score={score:.3f}")

        candidates.append((peak_idx, height, fwhm_ms, symmetry, score))

    # Stage 3: NMS with min_gap — keep the candidate with the highest
    # composite score (not just tallest) in each window.
    min_gap_samples = int(sample_rate * min_gap_ms / 1000)
    candidates.sort(key=lambda c: c[0])  # sort by time

    selected = []
    for peak_idx, height, fwhm_ms, symmetry, score in candidates:
        if selected and (peak_idx - selected[-1][0]) < min_gap_samples:
            # Within min_gap of the previous selected peak — keep better score
            if score > selected[-1][4]:
                selected[-1] = (peak_idx, height, fwhm_ms, symmetry, score)
            continue
        selected.append((peak_idx, height, fwhm_ms, symmetry, score))

    # Convert to frame numbers with confidence based on composite score
    max_score = max(s[4] for s in selected) if selected else 1.0
    results = []
    for peak_idx, height, fwhm_ms, symmetry, score in selected:
        time_sec = peak_idx / sample_rate
        frame_num = int(time_sec * video_fps)
        confidence = min(0.4 + 0.6 * (score / max_score), 1.0)
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

    dedup_samples = int(sample_rate * 0.020)  # 20ms dedup

    # Find all local maxima above threshold
    raw_peaks = []
    i = 1
    while i < len(envelope) - 1:
        if envelope[i] > threshold:
            if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
                raw_peaks.append((i, envelope[i]))
                i += dedup_samples
                continue
        i += 1

    if not raw_peaks:
        return []

    # Measure shape and score each candidate
    max_height = max(h for _, h in raw_peaks)
    candidates = []
    for peak_idx, height in raw_peaks:
        fwhm_ms, symmetry = _measure_peak_shape(envelope, peak_idx, sample_rate)
        score = _impact_score(height, max_height, fwhm_ms, symmetry)
        candidates.append((peak_idx, height, fwhm_ms, symmetry, score))

    # NMS with min_gap using composite score
    min_gap_samples = int(sample_rate * min_gap_ms / 1000)
    candidates.sort(key=lambda c: c[0])

    selected = []
    for peak_idx, height, fwhm_ms, symmetry, score in candidates:
        if selected and (peak_idx - selected[-1][0]) < min_gap_samples:
            if score > selected[-1][4]:
                selected[-1] = (peak_idx, height, fwhm_ms, symmetry, score)
            continue
        selected.append((peak_idx, height, fwhm_ms, symmetry, score))

    max_score = max(s[4] for s in selected) if selected else 1.0
    results = []
    for peak_idx, height, fwhm_ms, symmetry, score in selected:
        time_sec = peak_idx / sample_rate
        frame_num = int(time_sec * video_fps)
        confidence = min(0.4 + 0.6 * (score / max_score), 1.0)
        results.append((frame_num, confidence))

    return results


def detect_contacts_audio(
    video_path: str,
    video_fps: float,
    sample_rate: int = 22050,
    low_freq: float = 1000.0,
    high_freq: float = 4000.0,
    min_gap_ms: float = 300.0,
    noise_percentile: float = 75.0,
    peak_threshold_factor: float = 3.0,
    max_impact_fwhm_ms: float = 40.0,
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
        noise_percentile: Percentile of envelope to use as noise floor.
        peak_threshold_factor: Peak must exceed noise floor by this factor.
        max_impact_fwhm_ms: Expected max FWHM of a true ball impact (ms).
        debug: Print debug information.

    Returns:
        List of (frame_number, confidence) for detected contacts.
    """
    try:
        audio, sr = extract_audio_from_video(video_path, sample_rate)
    except (ImportError, ValueError) as e:
        if debug:
            print(f"  [audio] Could not extract audio: {e}")
        return []

    if debug:
        duration = len(audio) / sr
        print(f"  [audio] Extracted {duration:.2f}s of audio at {sr}Hz")

    filtered = bandpass_filter(audio, sr, low_freq, high_freq)
    envelope = compute_envelope(filtered, sr, window_ms=5.0)

    if debug:
        print(f"  [audio] Envelope range: {envelope.min():.4f} - {envelope.max():.4f}")
        print(f"  [audio] Peak shape analysis (max impact FWHM: {max_impact_fwhm_ms:.0f}ms):")

    peaks = find_impact_peaks(
        envelope, sr, video_fps,
        min_gap_ms=min_gap_ms,
        noise_percentile=noise_percentile,
        peak_threshold_factor=peak_threshold_factor,
        max_impact_fwhm_ms=max_impact_fwhm_ms,
        debug=debug,
    )

    if debug:
        print(f"  [audio] Selected {len(peaks)} impact peaks")
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


def get_all_candidates_for_debug(
    video_path: str,
    video_fps: float,
    sample_rate: int = 22050,
    low_freq: float = 1000.0,
    high_freq: float = 4000.0,
    noise_percentile: float = 75.0,
    peak_threshold_factor: float = 3.0,
    max_impact_fwhm_ms: float = 40.0,
) -> List[dict]:
    """Get all candidate peaks with shape metrics for debug visualization.

    Returns:
        List of dicts with keys: sample_idx, time_sec, frame, amplitude,
        fwhm_ms, symmetry, score.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return []

    audio, sr = extract_audio_from_video(video_path, sample_rate)
    filtered = bandpass_filter(audio, sr, low_freq, high_freq)
    envelope = compute_envelope(filtered, sr, window_ms=5.0)

    noise_floor = np.percentile(envelope, noise_percentile)
    threshold = noise_floor * peak_threshold_factor

    dedup_distance = int(sr * 0.020)
    dedup_distance = max(dedup_distance, 1)

    peaks, properties = find_peaks(
        envelope,
        height=threshold,
        distance=dedup_distance,
        prominence=threshold * 0.3,
    )

    if len(peaks) == 0:
        return []

    peak_heights = properties['peak_heights']
    max_height = peak_heights.max()

    candidates = []
    for peak_idx, height in zip(peaks, peak_heights):
        fwhm_ms, symmetry = _measure_peak_shape(envelope, peak_idx, sr)
        score = _impact_score(height, max_height, fwhm_ms, symmetry, max_impact_fwhm_ms)
        time_sec = peak_idx / sr
        candidates.append({
            'sample_idx': int(peak_idx),
            'time_sec': time_sec,
            'frame': int(time_sec * video_fps),
            'amplitude': float(height),
            'fwhm_ms': fwhm_ms,
            'symmetry': symmetry,
            'score': score,
        })

    return candidates
