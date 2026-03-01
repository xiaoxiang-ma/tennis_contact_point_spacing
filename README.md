# Tennis Contact Analysis

Measures where a tennis player contacts the ball relative to their body — answering the question coaches can't easily quantify: *"Are you hitting the ball in front of you, or are you late?"*

## How It Works

1. **Audio detection** finds ball-racket contact moments from the video's sound track — no visual ball tracking needed
2. **Body pose estimation** reconstructs the player's 3D skeleton at each contact frame
3. **Contact measurement** calculates the dominant wrist position relative to the pelvis in 3D space
4. **Consistency scoring** tracks how repeatable the contact point is across an entire session

---

## Repository Structure

```
tennis-contact-analysis/
├── python/                     # Reference implementation (Google Colab / Python)
│   ├── src/                    # Core processing modules
│   │   ├── audio_detection.py  # Audio-based contact detection pipeline
│   │   ├── contact_detection.py
│   │   ├── pose_estimation.py  # MediaPipe pose wrapper
│   │   ├── measurements.py     # Contact point calculations
│   │   └── visualization.py    # Frame annotation utilities
│   ├── utils/
│   │   ├── video_io.py
│   │   └── coordinate_transforms.py
│   ├── notebooks/
│   │   └── contact_detection_by_sound.ipynb   # Main Colab notebook
│   └── requirements.txt
├── ios/                        # iPhone app (Swift / Xcode)
│   ├── TennisContact.xcodeproj
│   └── TennisContact/          # Swift source files
├── docs/
│   ├── architecture.md         # Quick-reference for Claude Code sessions
│   └── implementation_v3.md    # Full V3 implementation strategy
└── .github/
    └── workflows/
        └── ios.yml             # Automated build + test on push
```

---

## Python (Google Colab) — Proof of Concept

The `python/` folder contains a validated proof of concept that runs in Google Colab. It demonstrates the full contact detection and measurement pipeline.

### How to run

1. Open `python/notebooks/contact_detection_by_sound.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run Cell 1 (Setup) — clones the repo and installs dependencies
3. Run Cell 2 (Upload & Detect) — upload a tennis video, runs audio contact detection
4. Run Cell 3 (Audio Debug) — visualize the waveform and detected peaks
5. Run Cell 4 (Visual Inspection) — annotated frames at each detected contact
6. Run Cell 5 (Pose Analysis) — optional 3D body pose at a selected contact
7. Run Cell 6 (Download) — download all output images and CSVs

No GPU required. Audio detection runs on CPU in seconds.

### Audio detection approach

Tennis ball-racket impacts produce a sharp, distinctive thump (~5–20ms) in the 1–4 kHz frequency range. The algorithm:

- Applies a bandpass filter (1–4 kHz) to isolate the impact frequency band
- Computes an amplitude envelope with a short smoothing window
- Detects peaks above an adaptive noise threshold (75th percentile of the envelope)
- Scores each peak by shape: **20% amplitude + 40% narrowness + 40% symmetry**
- Runs non-maximum suppression with a 300ms minimum gap

Short, symmetric peaks (ball impacts) beat loud, wide peaks (shoe screeches).

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `LOW_FREQ` / `HIGH_FREQ` | 1000 / 4000 Hz | Bandpass filter range |
| `PEAK_THRESHOLD_FACTOR` | 3.0 | Multiples above noise floor |
| `NOISE_PERCENTILE` | 75 | Percentile used as noise floor |
| `MIN_GAP_MS` | 300 ms | Minimum time between contacts |
| `MAX_IMPACT_FWHM_MS` | 40 ms | Max width of a true ball impact |

---

## iOS App — V1 Target

The `ios/` folder contains the native iPhone app. It replicates the Python pipeline entirely on-device with no cloud dependency.

### Target experience

1. User records a tennis practice session on their iPhone (tripod, behind baseline, 60fps)
2. Opens the app, uploads the video
3. Waits ~2 minutes while the pipeline runs (audio detection → pose estimation → measurements)
4. Reviews a split-screen view for each shot:
   - **Left:** original video with wrist arc overlay
   - **Right:** draggable 3D skeleton frozen at contact
5. Reviews session statistics: mean contact position, variance, consistency score

### Technical approach

| Component | Technology |
|---|---|
| Audio contact detection | AVFoundation + Accelerate vDSP |
| Body pose estimation | Apple Vision `VNDetectHumanBodyPose3DRequest` |
| 3D rendering | SceneKit |
| Video overlay | CALayer / CAShapeLayer |
| Analytics charts | Swift Charts |
| Persistence | CoreData |

All processing runs on-device. No video is transmitted. App targets <50MB install size.

### Device requirements

- **Minimum:** iPhone with iOS 17.0+
- **Recommended:** iPhone 13+ for Neural Engine acceleration
- **Best:** iPhone 12 Pro+ (LiDAR) for metric 3D coordinates

---

## Development Setup (iOS)

See `docs/implementation_v3.md` for the full guide. Quick summary:

**Requirements:**
- Mac running macOS Ventura 13+
- Xcode 16+ (free from Mac App Store)
- Free Apple Developer account at developer.apple.com
- iPhone with iOS 17+ and a USB cable

**To open the project:**
```bash
open ios/TennisContact.xcodeproj
```

**To run tests:**
```bash
xcodebuild \
  -project ios/TennisContact.xcodeproj \
  -scheme TennisContact \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  test
```

---

## For Claude Code Sessions

Before starting any session, read these two files:
1. `docs/architecture.md` — quick-reference for all key decisions
2. `docs/implementation_v3.md` — full strategy and 14-task sequence

The `python/` folder is the **reference implementation**. When porting a component to Swift, always read the corresponding Python file first. The Python code is the specification; the Swift code is the implementation.

**Python → Swift file mapping:**

| Python | Swift |
|---|---|
| `python/src/audio_detection.py` | `ios/.../Processing/AudioDetector.swift` |
| `python/src/pose_estimation.py` | `ios/.../Processing/PoseEstimator.swift` |
| `python/utils/coordinate_transforms.py` | `ios/.../Processing/CoordinateTransform.swift` |
| `python/src/measurements.py` | `ios/.../Processing/StatisticsEngine.swift` |

---

## Requirements (Python)

```
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
mediapipe>=0.10.0
moviepy>=1.0.3
scipy>=1.10.0
plotly>=5.0.0
tqdm
Pillow>=9.0.0
```

ffmpeg in PATH is preferred over moviepy for audio extraction. Install via [ffmpeg.org](https://ffmpeg.org).

---

## Limitations (Python prototype)

- Requires video with a clear audio track recorded close to the court
- Background music, loud crowd noise, or very distant recording reduces accuracy
- Fast exchanges (<300ms between contacts) may need `MIN_GAP_MS` reduced
- Pose analysis is designed for single-player, stationary-tripod, behind-baseline footage
