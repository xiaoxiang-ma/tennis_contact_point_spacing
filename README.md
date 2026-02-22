# Tennis Contact Point Analysis

Detects ball-racket contact frames in tennis videos using **audio signal processing** — no visual ball tracking required.

## How It Works

Tennis ball-racket impacts produce a sharp, distinctive thump lasting ~5–20ms in the 1–4 kHz frequency range. The algorithm exploits this:

1. **Extract audio** from the video (via ffmpeg or moviepy)
2. **Bandpass filter** (1–4 kHz) to isolate the impact frequency band
3. **Compute amplitude envelope** with a short smoothing window (~5ms)
4. **Find candidate peaks** above an adaptive noise threshold
5. **Shape analysis**: measure each peak's FWHM duration and rise/fall symmetry
6. **Score and select**: composite score (20% amplitude + 40% narrowness + 40% symmetry) favors short symmetric impacts over long asymmetric shoe screeches
7. **Non-maximum suppression** with a configurable minimum gap between contacts

This approach is more robust than visual ball tracking, which fails on fast, blurry, or partially occluded balls.

## Repository Structure

```
notebooks/
    contact_detection_by_sound.ipynb   # Main Colab notebook — run this
src/
    audio_detection.py                 # Core signal processing pipeline
    contact_detection.py               # High-level detection API
    visualization.py                   # Frame annotation utilities
    measurements.py                    # 3D contact point measurements
    pose_estimation.py                 # MediaPipe pose wrapper
utils/
    video_io.py                        # Video loading
    coordinate_transforms.py           # Pelvis-centered 3D transforms
```

## Usage

Open `notebooks/contact_detection_by_sound.ipynb` in Google Colab and run the cells in order:

| Cell | Purpose |
|------|---------|
| 1. Setup | Clone repo and install dependencies |
| 2. Upload & Detect | Upload video, run audio contact detection |
| 3. Audio Debug | Waveform, envelope, candidate peak shapes |
| 4. Visual Inspection | Annotated frame strips at each contact |
| 5. Pose Analysis | (Optional) 3D body pose + measurements at a contact |
| 6. Download | Download all output images and CSVs |

No GPU required — audio detection runs on CPU.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOW_FREQ` / `HIGH_FREQ` | 1000 / 4000 Hz | Bandpass filter range for impact sounds |
| `PEAK_THRESHOLD_FACTOR` | 3.0 | How many times above noise floor a peak must be |
| `NOISE_PERCENTILE` | 75 | Percentile used to estimate the noise floor |
| `MIN_GAP_MS` | 300 ms | Minimum time between successive contacts |
| `MAX_IMPACT_FWHM_MS` | 40 ms | Peaks wider than this are penalized (shoe screeches are typically 50–200ms) |

## Impact Scoring

Each candidate peak is scored on three components:

- **Amplitude** (20%): relative peak height — louder is slightly better, but this alone is unreliable
- **Narrowness** (40%): `ideal_fwhm / actual_fwhm` — a 12ms peak scores ~1.0; a 24ms peak scores ~0.5
- **Symmetry** (40%): `min(rise, fall) / max(rise, fall)` — symmetric spikes score near 1.0; gradual-onset screeches score near 0.0

Within each `MIN_GAP_MS` window, the candidate with the highest composite score is selected, even if it is quieter than a co-occurring screech.

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
mediapipe>=0.10.0
moviepy>=1.0.3
scipy>=1.10.0
tqdm
Pillow>=9.0.0
```

ffmpeg in PATH is preferred over moviepy for audio extraction (faster and more reliable). Install via your system package manager or [ffmpeg.org](https://ffmpeg.org).

## Limitations

- Requires a video with a clear audio track recorded at a reasonable distance from the court
- Background music, crowd noise, or very distant recording can reduce accuracy
- Very rapid exchanges (<300ms between contacts) may need `MIN_GAP_MS` reduced
- Pose analysis (cell 5) is intended for single-player, stationary-tripod, behind-baseline footage
