# Tennis Contact Analysis — Architecture Reference

> **Purpose:** This file is the quick-reference context document for Claude Code.  
> Read this at the start of every session before making any changes.  
> For full detail on any section, see `docs/implementation_v3.md`.

---

## What This Project Is

An iPhone app that analyzes tennis practice videos to measure where a player contacts the ball relative to their body. It answers the question: *"Am I hitting the ball in front of my body, or am I late?"*

The app:
1. Detects ball-racket contact moments from audio (the distinctive thump sound)
2. Estimates the player's 3D body pose at each contact
3. Measures the dominant wrist position relative to the pelvis
4. Shows a split-screen: original video on the left, draggable 3D skeleton on the right
5. Tracks consistency across all shots in a session

**Everything runs on-device. No cloud. No internet required after install.**

---

## Repo Structure

```
tennis-contact-analysis/
├── python/                        # Reference implementation (READ-ONLY for iOS work)
│   ├── src/
│   │   ├── audio_detection.py     # THE reference for AudioDetector.swift
│   │   ├── contact_detection.py   # High-level detection API
│   │   ├── pose_estimation.py     # THE reference for PoseEstimator.swift
│   │   ├── measurements.py        # THE reference for StatisticsEngine.swift
│   │   └── visualization.py       # Reference for rendering approach
│   ├── utils/
│   │   ├── coordinate_transforms.py  # THE reference for CoordinateTransform.swift
│   │   └── video_io.py
│   └── notebooks/
│       └── contact_detection_by_sound.ipynb
├── ios/                           # Xcode project (iOS app)
│   ├── TennisContact.xcodeproj
│   └── TennisContact/
│       ├── Processing/
│       │   ├── AudioDetector.swift       # Port of python/src/audio_detection.py
│       │   ├── PoseEstimator.swift       # Apple Vision replacement for MediaPipe
│       │   ├── CoordinateTransform.swift # Port of python/utils/coordinate_transforms.py
│       │   ├── StatisticsEngine.swift    # Port of python/src/measurements.py
│       │   └── ProcessingPipeline.swift  # Orchestrator
│       ├── Rendering/
│       │   ├── SkeletonRenderer.swift    # SceneKit 3D skeleton
│       │   └── ArcOverlay.swift          # CALayer wrist path on video
│       ├── Session/
│       │   ├── HomeView.swift
│       │   ├── SessionView.swift
│       │   ├── ShotDetailView.swift      # Primary UX: split-screen
│       │   └── StatisticsView.swift
│       ├── Onboarding/
│       │   └── OnboardingView.swift
│       └── Models/
│           ├── Session.swift
│           ├── Shot.swift
│           └── ContactCandidate.swift
└── docs/
    ├── implementation_v3.md       # Full strategy document
    └── architecture.md            # This file
```

---

## The Four Components

### Component 1: Audio Contact Detection
**Swift file:** `ios/TennisContact/TennisContact/Processing/AudioDetector.swift`  
**Python reference:** `python/src/audio_detection.py`  
**Function to port:** `detect_contacts_audio()`

Tennis ball-racket impacts produce a sharp thump (~5–20ms) in the 1–4 kHz range. The algorithm detects these by shape — not just by loudness — so it ignores shoe screeches and background noise.

Pipeline:
```
video file → AVAssetReader (PCM Float32) → vDSP bandpass (1–4kHz)
→ amplitude envelope (5ms window) → adaptive threshold (75th percentile × 3.0)
→ candidate peaks → shape scoring → NMS (300ms gap) → [ContactCandidate]
```

Scoring weights (must match Python exactly):
- Amplitude: 20%
- Narrowness (FWHM): 40%
- Symmetry: 40%

Validation: Swift and Python must agree within ±50ms on the same video.

---

### Component 2: Body Pose Estimation
**Swift file:** `ios/TennisContact/TennisContact/Processing/PoseEstimator.swift`  
**Python reference:** `python/src/pose_estimation.py`  
**Replaces:** MediaPipe → Apple Vision `VNDetectHumanBodyPose3DRequest`

Processing window: ±5 frames around each audio-detected contact (11 frames total). Select the frame with highest joint confidence as the canonical contact frame.

Key joints (maps to Python's `LANDMARK_MAP`):
```swift
// Apple Vision joint names → our names
.centerShoulder    // not used directly
.leftShoulder      // "left_shoulder"
.rightShoulder     // "right_shoulder"
.leftElbow         // "left_elbow"
.rightElbow        // "right_elbow"
.leftWrist         // "left_wrist"
.rightWrist        // "right_wrist"
.leftHip           // "left_hip"
.rightHip          // "right_hip"
.root              // used to derive "pelvis"
```

Pelvis = midpoint of leftHip and rightHip (same as Python `_add_synthetic()`).

---

### Component 3: Coordinate Transform
**Swift file:** `ios/TennisContact/TennisContact/Processing/CoordinateTransform.swift`  
**Python reference:** `python/utils/coordinate_transforms.py`  
**Functions to port:** `pelvis_origin_transform()`, `estimate_ground_plane()`, `apply_ground_plane()`

Output coordinate system (pelvis-centered, camera-angle invariant):
```
X = forward  (positive = in front of player's chest)
Y = lateral  (positive = toward dominant hand side)
Z = vertical (positive = upward)
```

All measurements are reported in this coordinate system so that a "30cm forward contact" means the same thing regardless of camera distance or angle.

---

### Component 4: Statistics Engine
**Swift file:** `ios/TennisContact/TennisContact/Processing/StatisticsEngine.swift`  
**Python reference:** `python/src/measurements.py`  
**Function to port:** `compute_measurements()`

Core measurements per shot:
```
contactForward   = wrist.X - pelvis.X   (positive = in front)
contactLateral   = wrist.Y - pelvis.Y   (positive = toward dominant side)
contactHeight    = wrist.Z              (absolute height from ground)
wristVelocity    = |Δwrist| / Δtime    (across ±3 frames)
```

Session aggregates:
```
consistencyScore = 100 × exp(−0.5 × (stdForward² + stdLateral² + stdHeight²))
```

Score of 100 = perfectly consistent. Falls exponentially as variance increases.

---

## Key Architecture Decisions

### Why hand tracking, not racket tracking
The wrist position IS the contact point proxy. A right-handed forehand where the dominant wrist is behind the hip = late contact. No racket needed to tell that story. Golf apps (DeepSwing, Swing Profile) deliver full coaching value using hand path alone. Racket tracking is V2.

### Why audio, not visual ball tracking
The Python prototype validated this. Ball-racket impacts produce a distinctive 5–20ms thump in 1–4kHz. This is more reliable than visual tracking because:
- The ball is tiny and motion-blurred at contact
- Audio detection works regardless of lighting, court color, or background
- The Python implementation achieves ~90% accuracy on clean recordings

### Why Apple Vision, not MediaPipe
- Zero additional model size (no .tflite bundled)
- Native iOS framework — no third-party dependency
- Returns 3D coordinates directly in metric meters (on LiDAR devices)
- On non-LiDAR devices: relative 3D still sufficient for consistency tracking

### Why on-device, not cloud
- XView AI (the closest golf competitor) is ~1GB and cloud-dependent
- Our target: <50MB app size, works offline, instant processing
- Privacy: video never leaves the device

### Dominant hand logic
Set once during onboarding. Stored in `UserDefaults`. For all processing:
- Right-handed player → track `rightWrist`
- Left-handed player → track `leftWrist`
- Two-handed backhand → still track dominant wrist (it controls the racket face)

---

## Data Models

### ContactCandidate
```swift
struct ContactCandidate {
    let timestamp: TimeInterval      // seconds from video start
    let frameIndex: Int              // video frame number
    let confidence: Float            // 0.0–1.0 from scoring
}
```

### Shot
```swift
// CoreData entity
// timestamp: Double
// frameIndex: Int64
// forwardCm: Double      // wrist.X - pelvis.X in cm
// lateralCm: Double      // wrist.Y - pelvis.Y in cm
// heightCm: Double       // wrist.Z in cm
// velocityCmS: Double    // wrist velocity at contact
// classification: String // "early" | "optimal" | "late"
// jointData: Data        // archived [String: simd_float3] joint positions
// session: Session       // CoreData relationship
```

### Session
```swift
// CoreData entity
// date: Date
// videoFilename: String
// durationSeconds: Double
// consistencyScore: Double
// shots: Set<Shot>       // CoreData relationship
```

---

## iOS Framework Usage

| Task | Framework | Key API |
|---|---|---|
| Video import | PhotosUI | `PhotosPicker` |
| Video playback | AVFoundation | `AVPlayer`, `AVPlayerItem` |
| Audio extraction | AVFoundation | `AVAssetReader`, `AVAssetReaderTrackOutput` |
| Signal processing | Accelerate | `vDSP.bandpassFilter`, `vDSP.envelope` |
| Body pose | Vision | `VNDetectHumanBodyPose3DRequest` |
| 3D rendering | SceneKit | `SCNScene`, `SCNNode`, `SCNGeometry` |
| 2D overlay | QuartzCore | `CAShapeLayer` for arc overlay |
| Charts | Swift Charts | `Chart`, `PointMark` for scatter plot |
| Persistence | CoreData | `NSPersistentContainer` |
| Settings | Foundation | `UserDefaults` for dominant hand |

---

## Testing Approach

### Unit tests required for every Processing/ file
Each processing component must have tests in `TennisContactTests/` that verify:
1. Correct output on a known synthetic input
2. Graceful handling of edge cases (empty input, no person detected, etc.)

### Cross-validation against Python
The definitive test: run Python and Swift on the same tennis video, compare outputs.

```
python/data/sample_videos/test_01.mp4
  → Python detect_contacts_audio() → timestamps_python.txt
  → Swift AudioDetector → timestamps_swift.txt
  → diff must be < 50ms for every contact
```

### Real video regression test
Before every TestFlight build, manually process one test video end-to-end and verify:
- Shot count is correct (compare to manual count)
- Skeleton renders correctly in split-screen
- Statistics are plausible (forward contact ~20–40cm for good shots)

---

## What NOT to Do

- **Do not add racket tracking in V1.** This is saved for V2.
- **Do not add cloud processing.** Everything must run on-device.
- **Do not modify files in `python/`.** That folder is the reference implementation, not active development.
- **Do not use MediaPipe in the iOS app.** Use Apple Vision. MediaPipe adds ~50MB to app size.
- **Do not use WidgetKit, WatchKit, or any extension targets in V1.** Keep the project simple.
- **Do not import any tracking SDKs or analytics frameworks.** App Store privacy labels must stay clean.
