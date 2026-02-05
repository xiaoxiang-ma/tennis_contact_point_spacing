"""Tennis Contact Point Detection System.

Two-stage detection pipeline:
- Stage 1: Temporal Detection (WHEN) - Audio + Visual signal fusion
- Stage 2: Spatial Localization (WHERE) - CV + Pose estimation
"""

from .audio_detection import (
    detect_contacts_audio,
    detect_contacts_audio_advanced,
    extract_audio_from_video,
    bandpass_filter,
    highpass_filter,
)
from .contact_detection import (
    detect_contacts,
    get_contact_ball_position,
    compute_ball_velocity,
    detect_trajectory_contacts,
    fuse_contact_signals,
)
from .pose_estimation import PoseEstimator
from .measurements import (
    compute_measurements,
    compute_relative_to_landmarks,
    classify_height,
    classify_forward_back,
    classify_lateral,
)
from .spatial_localization import (
    RacketDetector,
    ContactPointLocalizer,
    localize_contact_point,
)
from .visualization import (
    draw_skeleton,
    draw_contact_point,
    create_diagnostic_video,
)

__all__ = [
    # Audio detection
    'detect_contacts_audio',
    'detect_contacts_audio_advanced',
    'extract_audio_from_video',
    'bandpass_filter',
    'highpass_filter',
    # Contact detection
    'detect_contacts',
    'get_contact_ball_position',
    'compute_ball_velocity',
    'detect_trajectory_contacts',
    'fuse_contact_signals',
    # Pose estimation
    'PoseEstimator',
    # Measurements
    'compute_measurements',
    'compute_relative_to_landmarks',
    'classify_height',
    'classify_forward_back',
    'classify_lateral',
    # Spatial localization
    'RacketDetector',
    'ContactPointLocalizer',
    'localize_contact_point',
    # Visualization
    'draw_skeleton',
    'draw_contact_point',
    'create_diagnostic_video',
]
