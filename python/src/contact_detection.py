"""Detect ball-racket contact frames using audio analysis.

Tennis ball impacts produce a distinctive short thump (5-20ms) in the 1-4kHz
range. This module uses pure audio signal processing to identify contact frames
without any visual ball tracking. Peak shape analysis (FWHM and symmetry)
distinguishes true impacts from shoe screeches and other court noise.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .audio_detection import (
    detect_contacts_audio,
    get_audio_envelope_for_debug,
    get_all_candidates_for_debug,
)


def detect_contacts(
    video_path: str,
    fps: float,
    sample_rate: int = 22050,
    low_freq: float = 1000.0,
    high_freq: float = 4000.0,
    min_gap_ms: float = 300.0,
    noise_percentile: float = 75.0,
    peak_threshold_factor: float = 3.0,
    max_impact_fwhm_ms: float = 40.0,
    debug: bool = False,
) -> List[Tuple[int, float, str]]:
    """Detect contact frames from audio track.

    Args:
        video_path: Path to video file.
        fps: Video frame rate.
        sample_rate: Audio sample rate for processing.
        low_freq: Bandpass filter low cutoff (Hz).
        high_freq: Bandpass filter high cutoff (Hz).
        min_gap_ms: Minimum gap between contacts (ms).
        noise_percentile: Percentile of envelope used as noise floor.
        peak_threshold_factor: Peak must exceed noise floor by this factor.
        max_impact_fwhm_ms: Expected max FWHM of a true ball impact (ms).
            Peaks wider than this are penalized (likely shoe screeches).
        debug: Print debug information.

    Returns:
        List of (frame_number, confidence, source) for detected contacts.
        source is always 'audio'.
    """
    if debug:
        print("Running audio-based contact detection...")
        print(f"  Bandpass: {low_freq:.0f}-{high_freq:.0f} Hz")
        print(f"  Min gap: {min_gap_ms:.0f} ms")
        print(f"  Threshold: {noise_percentile}th percentile x {peak_threshold_factor}")
        print(f"  Max impact FWHM: {max_impact_fwhm_ms:.0f} ms")

    try:
        peaks = detect_contacts_audio(
            video_path=video_path,
            video_fps=fps,
            sample_rate=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
            min_gap_ms=min_gap_ms,
            noise_percentile=noise_percentile,
            peak_threshold_factor=peak_threshold_factor,
            max_impact_fwhm_ms=max_impact_fwhm_ms,
            debug=debug,
        )
    except Exception as e:
        if debug:
            print(f"  Audio analysis failed: {e}")
        return []

    contacts = [(frame, conf, 'audio') for frame, conf in peaks]

    if debug:
        print(f"\nDetected {len(contacts)} contact(s):")
        for frame, conf, source in contacts:
            time_sec = frame / fps
            print(f"  Frame {frame} ({time_sec:.2f}s): confidence {conf:.2f}")

    return contacts


def get_debug_audio_data(
    video_path: str,
    sample_rate: int = 22050,
    low_freq: float = 1000.0,
    high_freq: float = 4000.0,
) -> Dict:
    """Get audio data for visualization/debugging.

    Returns:
        Dict with keys: raw_audio, sample_rate, envelope, duration_sec
    """
    raw_audio, sr, envelope = get_audio_envelope_for_debug(
        video_path, sample_rate, low_freq, high_freq,
    )
    return {
        'raw_audio': raw_audio,
        'sample_rate': sr,
        'envelope': envelope,
        'duration_sec': len(raw_audio) / sr,
    }


def get_debug_candidates(
    video_path: str,
    fps: float,
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
    return get_all_candidates_for_debug(
        video_path=video_path,
        video_fps=fps,
        sample_rate=sample_rate,
        low_freq=low_freq,
        high_freq=high_freq,
        noise_percentile=noise_percentile,
        peak_threshold_factor=peak_threshold_factor,
        max_impact_fwhm_ms=max_impact_fwhm_ms,
    )
