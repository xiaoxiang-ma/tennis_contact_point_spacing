# Product Requirements Document (PRD)
# Tennis Contact Point Analysis Tool - MVP

**Version:** 1.0  
**Date:** February 2, 2026  
**Author:** Product Owner  
**Target Platform:** Google Colab Notebook  
**Development Approach:** Claude Code + GitHub Repository

---

## 1. Executive Summary

### 1.1 Product Vision
A computer vision-based tool that analyzes tennis rally videos to detect ball-racket contact moments, reconstructs the player's 3D body pose at contact, and measures the spatial relationship between the contact point and key body landmarks. This enables players to self-identify late or optimal contact points and develop consistent technique through quantified feedback.

### 1.2 Problem Statement
Tennis players often mishit due to suboptimal contact points (too late, too crushed sideways, wrong height). Without objective measurement, players struggle to:
- Identify exactly when and where contact issues occur
- Quantify the difference between good and bad contact
- Lock in the "feel" of proper contact through data-driven self-analysis

### 1.3 Success Criteria (MVP)
- Successfully processes a tennis rally video uploaded by user
- Detects ball-racket contact moments with reasonable accuracy (±2-3 frames acceptable)
- Generates 3D pose reconstruction at contact frames
- Outputs contact point measurements relative to body landmarks
- Provides visual feedback via 3D skeleton overlay on contact frames
- Completes processing in <5 minutes per video on free Colab GPU

---

## 2. MVP Scope

### 2.1 In Scope
**Core Features:**
1. Video upload functionality in Colab notebook
2. Automated ball-racket contact detection
3. 3D pose estimation at detected contact frames
4. Contact point measurement calculation
5. 3D skeleton overlay visualization on contact frames
6. Exportable results (annotated images + CSV with measurements)

**Technical Constraints:**
- Single camera, stationary tripod setup
- Behind-baseline court perspective
- Practice session footage (controlled environment)
- Groundstrokes only (forehands and backhands)
- Video format: MP4, MOV (common phone camera formats)
- Resolution: 720p minimum, 1080p recommended

### 2.2 Out of Scope (Future Versions)
- Multiple camera angles
- Match footage analysis
- Serves, volleys, overheads
- Real-time processing
- Web application deployment
- Mobile app
- Historical tracking/database
- Side-by-side comparison of multiple rallies
- Automatic shot type classification (forehand vs backhand)
- Racket head speed/swing path analysis

---

## 3. User Workflow

### 3.1 Primary User Journey
```
1. User records tennis rally video
   ↓
2. User opens Google Colab notebook (shared URL)
   ↓
3. User runs "Setup" cell (installs dependencies)
   ↓
4. User uploads video file via Colab file upload widget
   ↓
5. User runs "Process Video" cell
   ↓
6. System displays progress:
   - Ball tracking progress
   - Contact detection results (N contacts found)
   - Pose estimation progress
   ↓
7. System displays results for each contact:
   - Original frame with 3D skeleton overlay
   - Measurement annotations
   - Numerical data table
   ↓
8. User reviews visualizations
   ↓
9. User downloads:
   - Annotated images (PNG files)
   - Measurements CSV
```

### 3.2 User Personas
**Primary:** Intermediate to advanced tennis players (3.5-5.0 NTRP) seeking technique improvement through self-analysis

**Characteristics:**
- Has access to smartphone/tablet for video recording
- Basic familiarity with uploading files and running code cells
- Motivated to improve contact consistency
- Training 2-4x per week with access to practice courts

---

## 4. Functional Requirements

### 4.1 Video Input
**FR-1.1:** System shall accept video upload via Colab file widget  
**FR-1.2:** Supported formats: MP4, MOV, AVI  
**FR-1.3:** Recommended resolution: 1080p (minimum 720p)  
**FR-1.4:** Maximum file size: 500MB (Colab constraint)  
**FR-1.5:** Frame rate: 30fps minimum (60fps preferred for better contact detection)  

### 4.2 Ball Contact Detection
**FR-2.1:** System shall track tennis ball throughout video using existing CV models  
**FR-2.2:** System shall identify frames where ball-racket contact occurs  
**FR-2.3:** Detection approach: Ball trajectory tracking + velocity change analysis + proximity to player  
**FR-2.4:** Acceptable accuracy: ±2-3 frames from true contact moment  
**FR-2.5:** System shall extract individual frames at detected contact moments  
**FR-2.6:** System shall display confidence scores for each detected contact (optional for MVP, recommended)

**Technical Implementation Guidance:**
- Option A: TrackNet or TennisTracker (pre-trained tennis models)
- Option B: YOLOv8 fine-tuned on tennis ball detection
- Fallback: Simple ball tracking via color/motion detection (faster but less robust)

### 4.3 3D Pose Reconstruction
**FR-3.1:** System shall estimate 3D pose from contact frame images  
**FR-3.2:** Required body landmarks:
- Pelvis (center, used as reference origin)
- Left and right shoulders
- Left and right elbows
- Left and right wrists
- Head (top or center)
- Left and right hips (if available from model)

**FR-3.3:** System shall output 3D coordinates (x, y, z) for each landmark in meters or normalized units  
**FR-3.4:** Coordinate system: Pelvis as origin (0, 0, 0), ground plane as z=0 reference

**Technical Implementation Guidance:**
- Option A: MediaPipe Pose (fast, real-time capable, moderate 3D accuracy)
- Option B: MMPose + VideoPose3D (higher accuracy, slower)
- Option C: HybrIK (highest accuracy, slowest - save for v2)
- **Recommendation for MVP:** Start with MediaPipe for speed, validate accuracy

### 4.4 Contact Point Measurements
**FR-4.1:** System shall calculate the following measurements at each contact frame:

**Spatial Measurements (relative to pelvis):**
- Lateral offset (left/right, in inches or cm)
- Forward/backward distance (in front of or behind pelvis, in inches or cm)
- Vertical height above ground (in inches or cm)

**Additional Measurements:**
- Distance from shoulder line (perpendicular distance)
- Contact height relative to shoulder height
- Contact point position relative to body centerline

**FR-4.2:** All measurements shall be displayed in both metric (cm) and imperial (inches) units  
**FR-4.3:** Measurements shall be exported to CSV with the following columns:
```
frame_number, timestamp, contact_x, contact_y, contact_z, 
lateral_offset_cm, forward_back_cm, height_above_ground_cm,
shoulder_line_distance_cm, relative_to_shoulder_height_cm
```

### 4.5 Visualization
**FR-5.1:** System shall generate annotated images for each detected contact  
**FR-5.2:** Annotations shall include:
- 3D skeleton overlay (lines connecting body landmarks)
- Contact point marker (highlighted circle or cross)
- Key measurement text overlays (lateral offset, height, distance from pelvis)
- Frame number and timestamp

**FR-5.3:** Visual style requirements:
- Skeleton lines: thickness 2-3px, color-coded (e.g., arms in blue, legs in green)
- Contact point: red circle, radius 5-10px
- Text annotations: font size 14-18pt, contrasting color (white with black outline or vice versa)
- Original frame visible underneath overlay

**FR-5.4:** System shall display visualizations inline in Colab notebook  
**FR-5.5:** System shall save annotated images as PNG files to Colab file system for download

### 4.6 Results Export
**FR-6.1:** System shall generate downloadable outputs:
- Individual annotated PNG images (one per contact)
- Single CSV file with all measurements
- Optional: Summary statistics (average contact position, standard deviation)

**FR-6.2:** File naming convention:
```
contact_frame_<frame_number>.png
measurements_<video_filename>.csv
```

---

## 5. Non-Functional Requirements

### 5.1 Performance
**NFR-1.1:** Processing time: <5 minutes per 30-second video on Colab free tier GPU  
**NFR-1.2:** Memory usage: <12GB RAM (Colab constraint)  
**NFR-1.3:** Contact detection frame rate: Process at least 15fps during analysis phase

### 5.2 Accuracy
**NFR-2.1:** Ball contact detection: >80% true positive rate on practice rally footage  
**NFR-2.2:** 3D pose estimation: <5cm error in landmark localization (relative to ground truth, if measurable)  
**NFR-2.3:** Contact point measurements: ±2 inches acceptable error margin for MVP validation

### 5.3 Usability
**NFR-3.1:** Notebook shall include clear markdown documentation for each step  
**NFR-3.2:** Error messages shall be user-friendly (avoid raw stack traces where possible)  
**NFR-3.3:** Progress indicators shall update during long-running operations  
**NFR-3.4:** Total user clicks/interactions: <5 to go from upload to results

### 5.4 Compatibility
**NFR-4.1:** Runs on Google Colab free tier (no paid subscription required)  
**NFR-4.2:** Compatible with videos from common smartphones (iPhone, Android)  
**NFR-4.3:** No local installation required for end user

---

## 6. Technical Requirements

### 6.1 Development Environment
**Platform:** Google Colab notebook  
**Python Version:** 3.10+  
**GPU:** Colab free tier GPU (T4 or similar)  
**Storage:** Colab temporary storage (resets on session end)

### 6.2 Required Libraries (Preliminary)
```python
# Core
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0

# Computer Vision
# Option 1: MediaPipe
mediapipe>=0.10.0

# Option 2: MMPose path
# torch>=2.0.0
# mmcv>=2.0.0
# mmpose>=1.0.0

# Ball detection (choose one)
# ultralytics>=8.0.0  # YOLOv8
# OR custom TrackNet implementation

# Utilities
tqdm  # Progress bars
Pillow  # Image handling
```

### 6.3 Input Specifications
**Video Requirements:**
- Format: MP4 (H.264), MOV, AVI
- Resolution: 1280x720 minimum, 1920x1080 recommended
- Frame rate: 30fps minimum, 60fps ideal
- Duration: 10 seconds to 2 minutes
- Camera: Stationary, behind baseline, centered on court
- Lighting: Outdoor daylight or well-lit indoor courts

**Court Setup:**
- Single camera mounted on tripod
- Height: 4-6 feet above ground
- Distance from baseline: 10-20 feet
- Field of view: Captures full player stroke and partial court

### 6.4 Output Specifications
**Annotated Images:**
- Format: PNG
- Resolution: Match input video resolution
- Color space: RGB

**Measurements CSV:**
- Format: UTF-8 encoded CSV
- Headers: Descriptive column names
- Units: Both metric and imperial

---

## 7. Constraints & Assumptions

### 7.1 Constraints
**Technical:**
- Colab session timeout: 12 hours maximum (90 min idle on free tier)
- GPU availability not guaranteed on free tier
- No persistent storage between sessions
- Internet connection required
- Cannot process videos >500MB due to upload limitations

**Scope:**
- MVP focuses on behind-baseline perspective only
- Groundstrokes only (no serves, volleys)
- Single player in frame
- No opponent/ball machine visible (reduces detection complexity)

**Accuracy:**
- 3D depth estimation inherently less accurate from monocular video
- Occlusion of body parts may reduce pose estimation quality
- Ball may be occluded at contact (especially backhand)

### 7.2 Assumptions
**User Assumptions:**
- User has basic familiarity with Google Colab (can run cells)
- User can record video with smartphone on tripod
- User positions camera behind baseline as instructed
- User records practice rallies in good lighting conditions

**Technical Assumptions:**
- Pre-trained models (MediaPipe, YOLO, etc.) generalize to user's court/lighting
- Ball is visible in majority of frames
- Player occupies 30-60% of frame height
- Background is relatively static (no moving objects)

**Data Assumptions:**
- Contact point location is consistent enough within good technique to establish baseline
- Players can perceive and act on 2-4 inch differences in contact position
- 3D pose estimation from monocular video provides sufficient accuracy for coaching insights

---

## 8. Development Roadmap

### 8.1 MVP Milestone (Weeks 1-2)
**Phase 1: Core Pipeline (Week 1)**
- [ ] Set up Colab notebook structure
- [ ] Implement video upload and display
- [ ] Integrate ball tracking model (choose MediaPipe or YOLO approach)
- [ ] Implement basic contact detection (proximity-based)
- [ ] Test on 3 sample videos

**Phase 2: Pose & Measurements (Week 2)**
- [ ] Integrate 3D pose estimation (MediaPipe recommended for speed)
- [ ] Implement coordinate extraction and transformation
- [ ] Calculate contact point measurements
- [ ] Generate measurement CSV output
- [ ] Test accuracy on 5 sample videos

**Phase 3: Visualization & Polish (Week 2)**
- [ ] Implement 3D skeleton overlay
- [ ] Add measurement annotations to images
- [ ] Create markdown documentation in notebook
- [ ] Add progress indicators
- [ ] Final testing with 10 diverse rally videos

### 8.2 Post-MVP Enhancements (Future)
**Version 1.1:**
- Automatic shot type detection (forehand vs backhand)
- Batch processing multiple videos
- Summary statistics dashboard

**Version 1.2:**
- Support for additional camera angles (side view)
- Serve analysis
- Racket head speed estimation

**Version 2.0:**
- Web application deployment (Hugging Face Spaces or similar)
- User accounts and historical tracking
- Comparison across sessions

---

## 9. Success Metrics

### 9.1 Technical Metrics
- **Contact Detection Precision:** >80% of detected contacts are true positives
- **Contact Detection Recall:** >70% of actual contacts are detected
- **Processing Speed:** <5 min per 30-second video
- **Pose Estimation Success Rate:** >90% of contact frames yield valid 3D pose

### 9.2 User Validation Metrics (Manual Testing)
- Can user successfully upload and process video in <2 minutes?
- Are measurement outputs interpretable without additional explanation?
- Do visualizations clearly show contact point relative to body?
- Can user identify "late" vs "good" contact from measurements?

### 9.3 MVP Validation Questions
1. Does the tool successfully process at least 8/10 test videos?
2. Do the measurements align with player's subjective experience of contact quality?
3. Can the tool differentiate between intentionally early vs late contact points?
4. Is the visualization clear enough for self-analysis?

---

## 10. GitHub Repository Structure

### 10.1 Recommended Repository Layout
```
tennis-contact-analysis/
├── README.md                          # Overview, setup instructions
├── notebooks/
│   └── tennis_contact_mvp.ipynb      # Main Colab notebook
├── src/
│   ├── __init__.py
│   ├── ball_detection.py             # Ball tracking logic
│   ├── contact_detection.py          # Contact moment identification
│   ├── pose_estimation.py            # 3D pose reconstruction
│   ├── measurements.py               # Contact point calculations
│   └── visualization.py              # Overlay generation
├── utils/
│   ├── __init__.py
│   ├── video_io.py                   # Video loading/saving
│   └── coordinate_transforms.py      # 3D coordinate utilities
├── tests/
│   └── test_sample_video.py          # Unit tests (optional for MVP)
├── data/
│   ├── sample_videos/                # Example rally videos
│   └── sample_outputs/               # Example annotated results
├── docs/
│   ├── PRD.md                        # This document
│   └── USER_GUIDE.md                 # How to record and process videos
├── requirements.txt                   # Python dependencies
└── .gitignore                        # Ignore outputs, large files
```

### 10.2 Development Workflow with Claude Code
1. Clone repository locally
2. Use Claude Code to implement modules in `src/` incrementally
3. Test each module independently in Colab
4. Integrate into main notebook (`tennis_contact_mvp.ipynb`)
5. Commit and push changes to GitHub
6. Share Colab notebook URL (linked to GitHub repo) with testers

---

## 11. Risk Assessment

### 11.1 Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Ball detection fails in poor lighting | High | Medium | Require well-lit videos; add lighting guide in docs |
| Pose estimation inaccurate from back view | High | Medium | Test MediaPipe vs MMPose; consider side-view fallback |
| Colab GPU unavailable or slow | Medium | Low | Provide CPU fallback; document expected processing times |
| Video file too large for upload | Low | Medium | Add file size check; compress video guide in docs |
| Contact detection misses frames due to occlusion | Medium | High | Accept lower recall for MVP; flag low-confidence detections |

### 11.2 User Experience Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| User doesn't position camera correctly | High | High | Provide clear setup guide with example images |
| User expects real-time results | Low | Medium | Set expectation of 3-5 min processing time upfront |
| Measurements not actionable/interpretable | High | Medium | Add interpretation guide; include "good" vs "bad" ranges |
| User abandons due to complexity | Medium | Low | Simplify to 3 cells: Setup, Upload+Process, Results |

---

## 12. Open Questions for Development
1. **Ball detection model choice:** Should we prioritize speed (MediaPipe-based tracking) or accuracy (TrackNet/YOLO)?
2. **Coordinate system definition:** Should ground plane be estimated from pose, or assume fixed camera height?
3. **Contact point location:** Use wrist position as proxy for contact, or attempt to detect racket head?
4. **Measurement units:** Default to metric or imperial? (Suggest: default metric, show both)
5. **Error handling:** How to handle videos with no detected contacts? Show partial results or error message?
6. **Validation dataset:** Do we have access to ground-truth labeled videos for accuracy testing?

---

## 13. Acceptance Criteria

### 13.1 MVP is considered "complete" when:
- [ ] Colab notebook runs end-to-end without errors on 8/10 test videos
- [ ] Contact detection identifies at least 1 contact per rally video
- [ ] 3D pose is successfully estimated for >90% of detected contacts
- [ ] Measurements CSV is generated with all required columns
- [ ] Annotated images clearly show skeleton overlay and contact point
- [ ] Processing completes in <5 minutes on Colab free tier
- [ ] User documentation (markdown cells) explains each step
- [ ] Code is committed to GitHub repository with clear README

### 13.2 User Testing Criteria:
- [ ] 3 target users can successfully process their own videos without assistance
- [ ] Users report that measurements align with their perception of contact quality
- [ ] At least 2/3 users find the tool "useful" or "very useful" for technique analysis

---

## 14. Appendix

### 14.1 Reference Links
- **MediaPipe Pose:** https://google.github.io/mediapipe/solutions/pose.html
- **MMPose:** https://github.com/open-mmlab/mmpose
- **TrackNet (Tennis Ball Tracking):** https://github.com/yastrebksv/TrackNet
- **YOLOv8:** https://github.com/ultralytics/ultralytics
- **Google Colab:** https://colab.research.google.com/

### 14.2 Sample Measurement Interpretation Guide (for future documentation)
**Good Contact Point (Forehand, baseline rally):**
- Lateral offset: 6-12 inches to dominant side of pelvis
- Forward distance: 12-18 inches in front of pelvis
- Height: 30-40 inches above ground (waist to chest height)

**Late Contact Point:**
- Forward distance: <6 inches or behind pelvis
- Often results in pushed/defensive shots

**Crushed/Jammed Contact:**
- Lateral offset: <6 inches (too close to body)
- Reduces racket head speed and control

*(Note: These are rough guidelines and vary by player height, grip, technique)*

---

**Document Status:** Draft for Development  
**Next Steps:** Review with development team (Claude Code), refine technical approach, begin Phase 1 implementation

