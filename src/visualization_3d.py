"""3D interactive contact point visualizer using Plotly.

Build a fully interactive Scatter3d figure showing:
  - Skeleton (bones + joints)
  - Racket (handle, throat, head ellipse)
  - Ball contact point
  - Ground plane
  - Swing velocity arrow
"""

import numpy as np
from typing import Optional

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # graceful — notebook cell 7 will catch this

# ---------------------------------------------------------------------------
# Skeleton connection table (name pairs + Plotly CSS color strings)
# Adapted from src/visualization.py SKELETON_CONNECTIONS (BGR → CSS)
# ---------------------------------------------------------------------------
_SKELETON_CONNECTIONS = [
    # Torso — white
    ("left_shoulder",  "right_shoulder",  "rgb(220,220,220)"),
    ("left_hip",       "right_hip",       "rgb(220,220,220)"),
    ("left_shoulder",  "left_hip",        "rgb(220,220,220)"),
    ("right_shoulder", "right_hip",       "rgb(220,220,220)"),
    # Arms — steel blue
    ("left_shoulder",  "left_elbow",      "rgb(100,160,255)"),
    ("left_elbow",     "left_wrist",      "rgb(100,160,255)"),
    ("right_shoulder", "right_elbow",     "rgb(100,160,255)"),
    ("right_elbow",    "right_wrist",     "rgb(100,160,255)"),
    # Legs — green
    ("left_hip",       "left_knee",       "rgb(60,210,90)"),
    ("left_knee",      "left_ankle",      "rgb(60,210,90)"),
    ("right_hip",      "right_knee",      "rgb(60,210,90)"),
    ("right_knee",     "right_ankle",     "rgb(60,210,90)"),
    # Head
    ("nose",           "left_shoulder",   "rgb(180,180,180)"),
    ("nose",           "right_shoulder",  "rgb(180,180,180)"),
]

_JOINT_NAMES = [
    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "pelvis",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Coordinate transform: MediaPipe → display
# ---------------------------------------------------------------------------

def _to_display(pt: np.ndarray) -> np.ndarray:
    """MediaPipe world → display coordinates.

    display_x =  mp_x   (lateral, unchanged)
    display_y = -mp_y   (flip: up is positive)
    display_z = -mp_z   (flip: in-front-of-player is positive)
    """
    x, y, z = pt
    return np.array([x, -y, -z])


def _d(pt) -> tuple:
    """Shorthand: convert a (3,) array to (dx, dy, dz) display tuple."""
    d = _to_display(np.asarray(pt))
    return float(d[0]), float(d[1]), float(d[2])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_contact_3d_figure(
    analysis: dict,
    contact_info: dict,
    title: str = "Contact Point — 3D View",
) -> "go.Figure":
    """Build the interactive Plotly 3D figure.

    Args:
        analysis:     Output of analyze_contact_window().
        contact_info: Dict with keys 'frame', 'time', 'confidence'.
        title:        Figure title string.

    Returns:
        plotly.graph_objects.Figure ready for fig.show().
    """
    if go is None:
        raise ImportError("plotly is not installed. Run: pip install plotly>=5.0.0")

    landmarks   = analysis.get("landmarks_3d", {})
    racket      = analysis.get("racket_geometry")
    ball_3d     = analysis.get("ball_contact_3d")
    swing_vel   = analysis.get("swing_velocity", np.array([0.0, 0.0, -1.0]))
    quality     = analysis.get("detection_quality", "fallback")
    n_dets      = len(analysis.get("ball_detections", []))

    traces = []

    # ------------------------------------------------------------------
    # 1. Skeleton bones
    # ------------------------------------------------------------------
    for start_name, end_name, color in _SKELETON_CONNECTIONS:
        if start_name not in landmarks or end_name not in landmarks:
            continue
        sx, sy, sz = _d(landmarks[start_name])
        ex, ey, ez = _d(landmarks[end_name])
        traces.append(go.Scatter3d(
            x=[sx, ex], y=[sy, ey], z=[sz, ez],
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ------------------------------------------------------------------
    # 2. Skeleton joints
    # ------------------------------------------------------------------
    jx, jy, jz, jnames = [], [], [], []
    for name in _JOINT_NAMES:
        if name in landmarks:
            dx, dy, dz = _d(landmarks[name])
            jx.append(dx); jy.append(dy); jz.append(dz)
            jnames.append(name.replace("_", " "))

    if jx:
        traces.append(go.Scatter3d(
            x=jx, y=jy, z=jz,
            mode="markers",
            marker=dict(size=5, color="yellow", line=dict(color="black", width=1)),
            text=jnames,
            hovertemplate="%{text}<extra></extra>",
            name="Joints",
            showlegend=False,
        ))

    # ------------------------------------------------------------------
    # 3. Racket
    # ------------------------------------------------------------------
    if racket is not None:
        _add_racket_traces(traces, racket)

    # ------------------------------------------------------------------
    # 4. Ball contact point
    # ------------------------------------------------------------------
    if ball_3d is not None:
        bx, by, bz = _d(ball_3d)
        hover_label = (
            f"Ball contact<br>"
            f"x={bx*100:.1f}cm lateral<br>"
            f"y={by*100:.1f}cm height<br>"
            f"z={bz*100:.1f}cm forward"
        )
        traces.append(go.Scatter3d(
            x=[bx], y=[by], z=[bz],
            mode="markers",
            marker=dict(
                size=18,
                color="yellow",
                line=dict(color="orange", width=2),
                symbol="circle",
            ),
            text=[hover_label],
            hovertemplate="%{text}<extra></extra>",
            name="Ball",
            showlegend=True,
        ))

    # ------------------------------------------------------------------
    # 5. Swing velocity arrow (from wrist, length 0.3m)
    # ------------------------------------------------------------------
    if racket is not None and swing_vel is not None:
        wrist = racket.get("wrist")
        if wrist is not None:
            arrow_end = wrist + 0.3 * swing_vel
            wx, wy, wz = _d(wrist)
            ax, ay, az = _d(arrow_end)
            traces.append(go.Scatter3d(
                x=[wx, ax], y=[wy, ay], z=[wz, az],
                mode="lines",
                line=dict(color="cyan", width=2, dash="dot"),
                name="Swing direction",
                showlegend=True,
                hoverinfo="skip",
            ))

    # ------------------------------------------------------------------
    # 6. Ground plane
    # ------------------------------------------------------------------
    traces += _make_ground_plane(landmarks)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    frame_num  = contact_info.get("frame", "?")
    time_sec   = contact_info.get("time", 0.0)
    confidence = contact_info.get("confidence", 0.0)

    quality_label = {
        "good":     f"Ball detection: good ({n_dets} frames)",
        "partial":  f"Ball detection: partial ({n_dets} frames — one side only)",
        "fallback": "Ball detection: fallback to racket center",
    }.get(quality, quality)

    subtitle = (
        f"Frame {frame_num} | {time_sec:.2f}s | Confidence {confidence:.0%} | {quality_label}"
    )

    # --- Debug ---
    print(f"[3D] traces={len(traces)} | landmarks={len(landmarks)} | racket={'yes' if racket else 'NONE'} | ball={'yes' if ball_3d is not None else 'none'}")
    if landmarks:
        all_d = [_to_display(np.asarray(v)) for v in landmarks.values()]
        xs = [p[0] for p in all_d]; ys = [p[1] for p in all_d]; zs = [p[2] for p in all_d]
        print(f"[3D] Display coord ranges: x=[{min(xs):.3f}, {max(xs):.3f}]  y=[{min(ys):.3f}, {max(ys):.3f}]  z=[{min(zs):.3f}, {max(zs):.3f}]")
    if racket:
        hc = _to_display(racket['head_center'])
        fn = _to_display(racket['face_normal'])
        print(f"[3D] Racket head_center(display)=({hc[0]:.3f}, {hc[1]:.3f}, {hc[2]:.3f})  face_normal=({fn[0]:.3f}, {fn[1]:.3f}, {fn[2]:.3f})")
    if ball_3d is not None:
        bdisp = _to_display(ball_3d)
        print(f"[3D] Ball(display)=({bdisp[0]:.3f}, {bdisp[1]:.3f}, {bdisp[2]:.3f})")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#0d1117",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(
                title="← Right | Left →",
                showbackground=False,
                gridcolor="#333",
                color="#aaa",
            ),
            yaxis=dict(
                title="Height (m)",
                showbackground=False,
                gridcolor="#333",
                color="#aaa",
            ),
            zaxis=dict(
                title="← Behind | Front →",
                showbackground=False,
                gridcolor="#333",
                color="#aaa",
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.4, y=0.8, z=1.2),
                up=dict(x=0, y=1, z=0),
            ),
        ),
        font=dict(color="#cccccc"),
        legend=dict(
            bgcolor="rgba(30,30,40,0.8)",
            bordercolor="#555",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=650,
    )

    # Measurement annotation
    meas_text = _build_measurement_text(ball_3d, landmarks)
    if meas_text:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=meas_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(20,20,30,0.85)",
            bordercolor="#555",
            borderwidth=1,
            font=dict(size=12, color="#eee", family="monospace"),
        )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_racket_traces(traces: list, racket: dict) -> None:
    """Add racket handle, throat, and head ellipse traces."""
    head_center = racket["head_center"]
    right       = racket["right"]
    up_r        = racket["up_r"]
    handle_start = racket["handle_start"]
    handle_end   = racket["handle_end"]  # wrist
    wrist        = racket["wrist"]

    racket_color = "rgb(255,165,50)"  # orange

    # Handle: handle_start → wrist
    hs = _d(handle_start)
    he = _d(handle_end)
    traces.append(go.Scatter3d(
        x=[hs[0], he[0]], y=[hs[1], he[1]], z=[hs[2], he[2]],
        mode="lines",
        line=dict(color=racket_color, width=5),
        name="Racket",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Ellipse: 64-point parametric loop
    t = np.linspace(0, 2 * np.pi, 64)
    ellipse_pts = [
        head_center + np.cos(ti) * 0.125 * right + np.sin(ti) * 0.145 * up_r
        for ti in t
    ]
    ex = [_d(p)[0] for p in ellipse_pts]
    ey = [_d(p)[1] for p in ellipse_pts]
    ez = [_d(p)[2] for p in ellipse_pts]

    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color=racket_color, width=4),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Throat: two short lines from wrist to ellipse (top and bottom of head)
    throat_top    = head_center - 0.145 * up_r  # bottom of ellipse in racket local
    throat_bottom = head_center + 0.145 * up_r

    for tp in [throat_top, throat_bottom]:
        wx, wy, wz = _d(wrist)
        tx, ty, tz = _d(tp)
        traces.append(go.Scatter3d(
            x=[wx, tx], y=[wy, ty], z=[wz, tz],
            mode="lines",
            line=dict(color=racket_color, width=3),
            showlegend=False,
            hoverinfo="skip",
        ))


def _make_ground_plane(landmarks: dict) -> list:
    """Create a ground plane mesh at foot level (y = min ankle y in display coords)."""
    if go is None:
        return []

    # Find ground y from ankles (display y = -mp_y, so ground = max display_y of ankles)
    ankle_ys = []
    for key in ("left_ankle", "right_ankle"):
        if key in landmarks:
            _, dy, _ = _d(landmarks[key])
            ankle_ys.append(dy)

    ground_y = min(ankle_ys) if ankle_ys else -1.0

    # Horizontal extent from pelvis spread
    xs = []
    for key in landmarks:
        dx, _, _ = _d(landmarks[key])
        xs.append(dx)
    zs = []
    for key in landmarks:
        _, _, dz = _d(landmarks[key])
        zs.append(dz)

    x_half = max(0.8, (max(xs) - min(xs)) * 1.2) if xs else 1.0
    z_half = max(0.8, (max(zs) - min(zs)) * 1.5) if zs else 1.2
    cx = (max(xs) + min(xs)) / 2 if xs else 0.0
    cz = (max(zs) + min(zs)) / 2 if zs else 0.0

    x0, x1 = cx - x_half, cx + x_half
    z0, z1 = cz - z_half, cz + z_half

    return [go.Mesh3d(
        x=[x0, x1, x1, x0],
        y=[ground_y, ground_y, ground_y, ground_y],
        z=[z0, z0, z1, z1],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="rgb(50,160,80)",
        opacity=0.12,
        showlegend=False,
        hoverinfo="skip",
        flatshading=True,
    )]


def _build_measurement_text(
    ball_3d: Optional[np.ndarray],
    landmarks: dict,
) -> str:
    """Build the annotation box string with lateral / forward / height measurements."""
    if ball_3d is None:
        return ""

    bx, by, bz = _d(ball_3d)

    # Pelvis as body reference
    pelvis = landmarks.get("pelvis")
    if pelvis is not None:
        px, py, pz = _d(pelvis)
        lat_cm   = (bx - px) * 100
        fwd_cm   = (bz - pz) * 100
        ht_cm    = (by - py) * 100
    else:
        lat_cm = bx * 100
        fwd_cm = bz * 100
        ht_cm  = by * 100

    lines = [
        "<b>Contact Point Measurements</b>",
        f"Lateral:  {lat_cm:+.1f} cm",
        f"Forward:  {fwd_cm:+.1f} cm",
        f"Height:   {ht_cm:+.1f} cm (rel. pelvis)",
    ]

    # vs shoulder height
    left_sh  = landmarks.get("left_shoulder")
    right_sh = landmarks.get("right_shoulder")
    if left_sh is not None and right_sh is not None:
        sh_y = (_d(left_sh)[1] + _d(right_sh)[1]) / 2
        vs_sh = (by - sh_y) * 100
        lines.append(f"vs Shoulder: {vs_sh:+.1f} cm")

    return "<br>".join(lines)
