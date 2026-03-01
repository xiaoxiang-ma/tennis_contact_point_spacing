# Tennis Contact Analysis — V3 Implementation Strategy

**Version:** 3.0  
**Date:** February 2026  
**Platform:** iOS 17+ (iPhone, native Swift)  
**Core Simplification:** Hand tracking + body pose only — no racket tracking in V1

---

## 1. Strategic Premise

V3 is a deliberate simplification. Research into golf apps (DeepSwing, XView AI) confirmed that hand path tracking alone delivers high-value coaching insight. The hand IS the primary signal. The implement is derived from the hand.

**V3 Core Thesis:** Body pose + dominant hand position at the audio-detected contact frame tells a player everything they need to know about contact point quality. Lateral offset, forward distance, height, wrist velocity — all derivable from the skeleton alone. No racket tracking required in V1.

### What V1 does NOT build
- Racket head tracking (V2)
- Ball detection or ball tracking
- Cloud processing — everything runs on-device
- SMPL personalized body mesh (V1.1)
- Multiplayer or coach-facing tools

### What V1 delivers
- Audio-detected contact frames from uploaded video
- Full 3D body skeleton at each contact frame via Apple Vision
- Dominant wrist position in 3D pelvis-relative coordinates (the contact point)
- Wrist path trajectory drawn as a swing arc overlay on the video
- Split-screen UX: real video left, 3D skeleton right, synchronized playback
- Per-shot statistics: contact position, wrist velocity, height
- Session-level statistics: mean contact position, variance, consistency score

---

## 2. Complete User Flow

### 2.1 Onboarding (one-time)
1. User opens app for the first time
2. Dominant hand selection: Left / Right (stored in UserDefaults, never asked again)
3. Brief camera setup guide shown: tripod behind baseline, 4–6ft height, player fills 40–60% of frame
4. User lands on Home screen

### 2.2 Session flow
1. User taps "+" to start a new session
2. User selects video from Photo Library (up to 10 minutes, any common format)
3. App validates video: checks for audio track, minimum 720p resolution, frame rate
4. Processing begins — four-stage progress screen:
   - Stage 1/4: Extracting audio track
   - Stage 2/4: Detecting contact frames
   - Stage 3/4: Running body pose estimation
   - Stage 4/4: Computing statistics and swing arcs
5. Results screen: session summary with shot count and consistency score
6. User taps any shot → split-screen shot detail view

### 2.3 Shot detail view (primary UX)

Split-screen, horizontal:

| Left Panel — Video | Right Panel — 3D Skeleton |
|---|---|
| Original video plays | SceneKit 3D skeleton |
| Wrist arc drawn as colored CALayer overlay | Animates in sync with video timeline |
| Contact frame: glowing ring on arc | User can drag to rotate freely |
| Tap to pause/play; scrub timeline | Dominant wrist glows red/orange at contact |
| Shot number indicator (Shot 3 of 12) | Pelvis shown as white crosshair reference |
| | Distance annotations: forward, lateral, height |

### 2.4 Statistics view
**Per-shot metrics:**
- Contact frame timestamp and shot number
- Dominant wrist position relative to pelvis: Forward (cm), Lateral (cm), Height (cm)
- Dominant wrist velocity at contact (cm/s) — delta position across ±3 frames
- Contact classification: Early / Optimal / Late (rule-based from forward distance)

**Session-level metrics:**
- Total shots detected
- Mean forward contact distance ± standard deviation
- Mean lateral offset ± standard deviation
- Mean contact height ± standard deviation
- Consistency Score: 0–100 (inverse of contact position variance)
- Wrist velocity: min, max, mean across all contacts
- Contact scatter plot: all contact positions on overhead body silhouette

---

## 3. Technical Architecture

### 3.1 The four-component pipeline

All components run sequentially on-device after video submission. No cloud calls.

| Component | Technology | Input | Output |
|---|---|---|---|
| 1. Audio Extraction | AVFoundation AVAssetReader | Video file URL | PCM audio buffer (Float32 array) |
| 2. Contact Detection | Accelerate vDSP (Swift) | PCM audio buffer | Array of contact timestamps (seconds) |
| 3. Body Pose Estimation | Apple Vision VNDetectHumanBodyPose3DRequest | Video frames at contact windows | 3D joint positions per frame |
| 4. Measurement & Analytics | Swift / simd math | Joint positions + timestamps | Contact point stats, swing arc, scores |

### 3.2 Component 1: Audio contact detection

**Reference implementation:** `python/src/audio_detection.py`

The Swift port must replicate this exact pipeline:

```
AVAssetReader → PCM buffer (44.1kHz Float32)
→ vDSP bandpass filter (1–4 kHz)
→ amplitude envelope (5ms smoothing window)
→ adaptive noise floor (75th percentile)
→ peak detection with shape analysis
→ composite scoring: 20% amplitude + 40% narrowness + 40% symmetry
→ NMS (300ms minimum gap)
→ output: [ContactCandidate] with timestamp and confidence
```

Key parameters (must match Python defaults):
- `LOW_FREQ` = 1000 Hz
- `HIGH_FREQ` = 4000 Hz
- `PEAK_THRESHOLD_FACTOR` = 3.0
- `NOISE_PERCENTILE` = 75th percentile
- `MIN_GAP_MS` = 300ms
- `MAX_IMPACT_FWHM_MS` = 40ms
- Scoring weights: amplitude 0.2, narrowness 0.4, symmetry 0.4

Swift framework mapping:
- `scipy.signal.butter` + `filtfilt` → `vDSP` bandpass filter via Accelerate
- `scipy.signal.find_peaks` → Swift loop over envelope array
- `numpy` arrays → Swift `[Float]` or `vDSP.floatVector`

**Validation criterion:** Running the Swift AudioDetector and the Python `detect_contacts_audio()` on the same video must produce contact timestamps that agree within ±50ms.

### 3.3 Component 2: Body pose estimation

**Reference implementation:** `python/src/pose_estimation.py`

Apple Vision replaces MediaPipe. Key differences:

| Python (MediaPipe) | Swift (Apple Vision) |
|---|---|
| `mp.solutions.pose.Pose` | `VNDetectHumanBodyPose3DRequest` |
| 33 joints | 17 joints |
| Relative Z depth | Metric meters (LiDAR) or relative (non-LiDAR) |
| `pose_world_landmarks` | `recognizedPoints3D` returns `simd_float4x4` matrices |

**Joints used** (subset of Apple Vision's 17):
- `pelvis` — reference origin (0,0,0) for all measurements
- `leftShoulder`, `rightShoulder` — shoulder line reference
- `leftWrist`, `rightWrist` — contact point (dominant wrist is primary)
- `leftHip`, `rightHip` — body width reference
- `head` — posture reference

**Coordinate system** (pelvis-centered, camera-angle invariant):
```
X = forward (positive = in front of player's chest)
Y = lateral (positive = toward hitting side)
Z = vertical (positive = upward)
```

**Processing window strategy:** For each audio contact timestamp, process the ±5 frame window (11 frames at 60fps = 183ms). Select the frame with highest joint confidence as the canonical contact frame. This handles audio/video sync drift.

**Device tiers:**
- iPhone 12 Pro+ (LiDAR): metric 3D coordinates in meters
- iPhone 12 (non-LiDAR): relative 3D, still accurate for consistency tracking
- Label metric vs. relative clearly in UI

### 3.4 Component 3: Wrist path & swing arc

For each detected contact, expand to full swing window: ~60 frames before contact through 30 frames after (90 frames total at 60fps ≈ 1.5 seconds).

Run body pose on this extended window → extract dominant wrist position at every frame → wrist trajectory in pelvis-centered 3D space.

This trajectory is:
- Drawn as an arc overlay on the video panel (projected to 2D pixel space via CALayer)
- Rendered as a 3D curve in the SceneKit skeleton panel (SCNGeometry tube)
- Used to compute wrist velocity at contact: `|Δwrist_position| / Δtime` across ±3 frames

### 3.5 Component 4: Statistics engine

**Reference implementation:** `python/src/measurements.py`

Pure Swift math — no model. Replicates `compute_measurements()` exactly:

```swift
// Per shot
contactForward[i]  = wrist.X - pelvis.X  // at contact frame i
contactLateral[i]  = wrist.Y - pelvis.Y
contactHeight[i]   = wrist.Z             // absolute height
wristVelocity[i]   = |Δwrist| / Δtime   // across frames i-3 to i+3

// Session aggregates  
meanForward  = average(contactForward)
stdForward   = stddev(contactForward)
consistencyScore = 100 × exp(−0.5 × (stdForward² + stdLateral² + stdHeight²))
```

---

## 4. iOS Project Structure

### 4.1 Xcode project directory layout

```
ios/
├── TennisContact.xcodeproj
└── TennisContact/
    ├── App/
    │   ├── TennisContactApp.swift        # @main entry point
    │   └── ContentView.swift             # Root navigation (TabView or NavigationStack)
    ├── Onboarding/
    │   └── OnboardingView.swift          # Dominant hand selection, camera setup guide
    ├── Session/
    │   ├── HomeView.swift                # Session list, "+" button
    │   ├── SessionView.swift             # Video picker + processing progress
    │   ├── ShotDetailView.swift          # Split-screen: video left, skeleton right
    │   └── StatisticsView.swift          # Per-shot and session analytics
    ├── Processing/
    │   ├── AudioDetector.swift           # Port of python/src/audio_detection.py
    │   ├── PoseEstimator.swift           # Apple Vision VNDetectHumanBodyPose3DRequest
    │   ├── CoordinateTransform.swift     # Port of python/utils/coordinate_transforms.py
    │   ├── StatisticsEngine.swift        # Port of python/src/measurements.py
    │   └── ProcessingPipeline.swift      # Orchestrates all four components
    ├── Rendering/
    │   ├── SkeletonRenderer.swift        # SceneKit SCNScene with 17-joint stick figure
    │   └── ArcOverlay.swift             # CALayer wrist path arc drawn on video
    ├── Models/
    │   ├── Session.swift                 # CoreData Session entity
    │   ├── Shot.swift                    # CoreData Shot entity
    │   └── ContactCandidate.swift        # Value type: timestamp, frameIndex, confidence
    └── Resources/
        └── TennisContact.xcdatamodeld   # CoreData schema
```

### 4.2 Key Swift files — purpose and Python reference

| Swift File | Python Reference | Notes |
|---|---|---|
| `AudioDetector.swift` | `python/src/audio_detection.py` | Line-for-line port of `detect_contacts_audio()` |
| `PoseEstimator.swift` | `python/src/pose_estimation.py` | Replace MediaPipe with Apple Vision |
| `CoordinateTransform.swift` | `python/utils/coordinate_transforms.py` | `pelvis_origin_transform()` → Swift |
| `StatisticsEngine.swift` | `python/src/measurements.py` | `compute_measurements()` → Swift |
| `SkeletonRenderer.swift` | `python/src/visualization_3d.py` | SceneKit instead of Plotly |

---

## 5. Development Environment

### 5.1 Required hardware and software

| Requirement | Details |
|---|---|
| Mac | macOS Ventura 13+ required. MacBook Air M2/M3 recommended. |
| Xcode | Version 16+, free from Mac App Store. ~15GB download. |
| Apple ID | Free account at developer.apple.com. Allows running on own iPhone. |
| iPhone | iOS 17.0+ for VNDetectHumanBodyPose3DRequest. |
| USB cable | Lightning or USB-C to connect iPhone to Mac for testing. |
| Paid developer account | $99/year. Only needed for TestFlight and App Store. |

### 5.2 Installation order
1. Download Xcode from Mac App Store
2. Open Xcode once → accept prompt to install additional components
3. Sign in at developer.apple.com with Apple ID (free)
4. Connect iPhone via cable → trust the connection on both devices
5. In Xcode → Settings → Accounts → add Apple ID

### 5.3 Simulator vs. physical device

Use **simulator** for: UI layout, navigation, SwiftUI previews, CoreData  
Use **physical iPhone** for: AVFoundation video/audio, Apple Vision pose estimation, performance testing

---

## 6. CI/CD Pipeline

### 6.1 Local development loop (daily)
1. Claude Code edits a Swift file
2. Xcode detects the change automatically
3. Press ▶ (Cmd+R) → compiles and installs on iPhone in ~30 seconds
4. Test on device → describe issues to Claude Code → repeat

### 6.2 GitHub Actions (automated build checks)

File: `.github/workflows/ios.yml`

```yaml
name: iOS Build and Test
on: [push, pull_request]
jobs:
  build:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: |
          xcodebuild \
            -project ios/TennisContact.xcodeproj \
            -scheme TennisContact \
            -destination 'platform=iOS Simulator,name=iPhone 16' \
            build
      - name: Test
        run: |
          xcodebuild \
            -project ios/TennisContact.xcodeproj \
            -scheme TennisContact \
            -destination 'platform=iOS Simulator,name=iPhone 16' \
            test
```

### 6.3 TestFlight distribution
Requires paid $99/year Apple Developer Program.  
Steps: Xcode → Product → Archive → Distribute App → TestFlight → invite testers by email.

---

## 7. Claude Code Task Sequence

Give these tasks to Claude Code in order. Each depends on the previous.

**Task 1 — Repo restructure**
> Move `src/`, `utils/`, `notebooks/`, `requirements.txt` into a new `python/` folder. Update `.gitignore` to add Xcode ignores. Update `README.md`. Do not modify any files inside `python/`.

**Task 2 — Xcode project scaffold**
> Create the Xcode project at `ios/TennisContact/` with the directory structure in Section 4.1. Each Swift file gets a comment header referencing its Python equivalent. `ContentView.swift` shows "Tennis Contact" and the app version. Add `.github/workflows/ios.yml` from Section 6.2.

**Task 3 — AudioDetector.swift**
> Read `python/src/audio_detection.py` in full. Port `detect_contacts_audio()` to Swift using AVFoundation for audio extraction and Accelerate.vDSP for signal processing. Input: video file URL. Output: `[ContactCandidate]`. Write a unit test with a synthetic 1kHz tone burst that asserts correct detection within ±50ms.

**Task 4 — Video import UI**
> Implement video picker using PhotosUI.PhotosPicker. User taps "+", picks a video, sees filename and duration displayed. No processing yet.

**Task 5 — ProcessingPipeline.swift**
> Wire AudioDetector → Apple Vision pose estimation on the ±5 frame window around each contact. Output: `[Shot]` array with timestamp, frame index, and 17 joint positions. Show four-stage progress updates.

**Task 6 — CoordinateTransform.swift**
> Read `python/utils/coordinate_transforms.py`. Port `pelvis_origin_transform()` and `estimate_ground_plane()` to Swift using simd. Unit test: left-handed player input produces mirrored output.

**Task 7 — Processing progress screen**
> Four-stage progress indicator UI. Updates in real time as the pipeline runs. Async, does not block the main thread.

**Task 8 — ShotDetailView split screen**
> Left: AVPlayer with wrist arc drawn as CALayer. Right: SceneKit SCNScene with stick-figure skeleton. Both sides scrub in sync via a shared timeline binding.

**Task 9 — SkeletonRenderer.swift**
> Animate SCNNode positions from stored joint arrays at each frame. Dominant wrist node glows at contact frame. Pelvis shown as white reference crosshair. Draggable 3D view.

**Task 10 — StatisticsEngine.swift**
> Read `python/src/measurements.py`. Port `compute_measurements()` to Swift. Input: `[Shot]`. Output: `SessionStats` struct. Unit tests with known inputs.

**Task 11 — StatisticsView**
> Scrollable session screen: shot list, aggregate metrics, 2D contact scatter plot using Swift Charts. Consistency score displayed prominently.

**Task 12 — CoreData persistence**
> Session and Shot CoreData entities. Sessions load on app launch, display in reverse chronological order on HomeView.

**Task 13 — Error handling**
> Handle: no audio track, no person detected in frame, video <5 seconds, video >10 minutes (offer to trim). User-friendly messages, no raw stack traces.

**Task 14 — TestFlight prep**
> App icon (1024×1024 PNG required), launch screen, NSPhotoLibraryUsageDescription, privacy manifest. Verify build archives cleanly.

---

## 8. Validation Strategy

### 8.1 Test video dataset
Record 3 tennis videos before starting iOS development:
- 60fps, behind-baseline, tripod, 5 minutes each
- Store in `python/data/sample_videos/` (gitignored)
- These are the ground truth for every component

### 8.2 Cross-validation (Python vs. Swift)
For each ported component, run both implementations on the same input and compare outputs:

| Component | Acceptance criterion |
|---|---|
| AudioDetector | Contact timestamps agree within ±50ms |
| CoordinateTransform | Pelvis-relative coordinates match within ±1mm |
| StatisticsEngine | Mean/stddev values match within ±0.1cm |

### 8.3 Definition of done for each task
A task is complete when:
1. Code compiles without warnings
2. Unit tests pass
3. Component runs correctly on at least one real tennis video
4. Output matches Python reference implementation within tolerance

---

## 9. Risk Register

| Risk | Impact | Probability | Mitigation |
|---|---|---|---|
| Apple Vision 3D accuracy poor from behind-baseline | High | Medium | Test in Phase 0. Fall back to 2D pose + estimated depth. |
| Audio detection misses contacts (wind, background noise) | High | Medium | Expose threshold parameters as settings. Add manual contact picker as fallback. |
| Video >10min causes memory pressure | Medium | Low | Process in 60-second chunks using AVAssetReader timeRange. |
| Swift audio port produces different results than Python | High | Low | Keep Python reference. Build comparison test against same audio file. |
| SceneKit skeleton jitter from pose noise | Medium | High | Apply 5-frame Gaussian smoothing to joint positions before rendering. |
| App Store rejection (video processing privacy) | Medium | Low | All on-device. No video transmitted. Include privacy manifest. |

---

## 10. Competitive Context

**The XView AI clock:** XView's founder has stated tennis is their next target sport. Their golf model generalizes to racket sports at ~70% accuracy without retraining. Estimated time to a shipped tennis product: 12–18 months.

**Our differentiation:**
- Body-relative contact point measurement (no existing tennis app does this)
- Contact point variance as the primary consistency metric
- On-device only, <50MB app size vs. XView's ~1GB
- Tennis-native from day one — not a golf app ported to tennis
