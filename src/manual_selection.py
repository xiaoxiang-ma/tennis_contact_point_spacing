"""Manual frame selection interface for low-confidence contact detection.

Provides interactive widgets for users to manually select or correct
contact frames when automated detection confidence is below threshold.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets


# Confidence threshold below which manual selection is recommended
LOW_CONFIDENCE_THRESHOLD = 0.7


def needs_manual_selection(contacts: List[Tuple[int, float, str]]) -> bool:
    """Check if any contacts have low confidence requiring manual selection.

    Args:
        contacts: List of (frame_num, confidence, source) from detection.

    Returns:
        True if any contact has confidence < LOW_CONFIDENCE_THRESHOLD.
    """
    if not contacts:
        return True  # No contacts detected, need manual selection

    return any(conf < LOW_CONFIDENCE_THRESHOLD for _, conf, _ in contacts)


def get_low_confidence_contacts(
    contacts: List[Tuple[int, float, str]],
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> List[Tuple[int, float, str]]:
    """Get contacts with confidence below threshold.

    Args:
        contacts: List of (frame_num, confidence, source).
        threshold: Confidence threshold.

    Returns:
        List of low-confidence contacts.
    """
    return [(f, c, s) for f, c, s in contacts if c < threshold]


def create_frame_thumbnail(
    frame: np.ndarray,
    frame_num: int,
    fps: float,
    confidence: Optional[float] = None,
    size: Tuple[int, int] = (320, 180),
) -> np.ndarray:
    """Create a thumbnail image with frame info overlay.

    Args:
        frame: BGR frame.
        frame_num: Frame number.
        fps: Video FPS.
        confidence: Optional detection confidence to display.
        size: Thumbnail size (width, height).

    Returns:
        Resized BGR frame with overlay.
    """
    thumb = cv2.resize(frame, size)

    # Add frame info
    time_sec = frame_num / fps
    text = f"Frame {frame_num} ({time_sec:.2f}s)"
    cv2.putText(thumb, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(thumb, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if confidence is not None:
        conf_text = f"Conf: {confidence:.0%}"
        color = (0, 255, 0) if confidence >= LOW_CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(thumb, conf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(thumb, conf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return thumb


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert BGR frame to base64 for HTML display."""
    import base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


class ManualFrameSelector:
    """Interactive widget for manual frame selection in Jupyter/Colab.

    Provides a slider-based interface to browse frames and select
    the contact frame when automated detection has low confidence.
    """

    def __init__(
        self,
        frames: List[np.ndarray],
        fps: float,
        initial_frame: int = 0,
        suggested_contacts: Optional[List[Tuple[int, float, str]]] = None,
    ):
        """Initialize the frame selector.

        Args:
            frames: List of BGR frames.
            fps: Video FPS.
            initial_frame: Starting frame to display.
            suggested_contacts: Optional list of detected contacts for reference.
        """
        self.frames = frames
        self.fps = fps
        self.num_frames = len(frames)
        self.suggested_contacts = suggested_contacts or []
        self.selected_frame: Optional[int] = None

        # Create widgets
        self.frame_slider = widgets.IntSlider(
            value=initial_frame,
            min=0,
            max=self.num_frames - 1,
            step=1,
            description='Frame:',
            continuous_update=False,
            layout=widgets.Layout(width='80%'),
        )

        self.step_buttons = widgets.HBox([
            widgets.Button(description='<< -10', layout=widgets.Layout(width='80px')),
            widgets.Button(description='< -1', layout=widgets.Layout(width='70px')),
            widgets.Button(description='+1 >', layout=widgets.Layout(width='70px')),
            widgets.Button(description='+10 >>', layout=widgets.Layout(width='80px')),
        ])

        self.select_button = widgets.Button(
            description='Select This Frame as Contact',
            button_style='success',
            layout=widgets.Layout(width='250px'),
        )

        self.output = widgets.Output()

        # Event handlers
        self.frame_slider.observe(self._on_frame_change, names='value')
        self.step_buttons.children[0].on_click(lambda b: self._step(-10))
        self.step_buttons.children[1].on_click(lambda b: self._step(-1))
        self.step_buttons.children[2].on_click(lambda b: self._step(1))
        self.step_buttons.children[3].on_click(lambda b: self._step(10))
        self.select_button.on_click(self._on_select)

    def _step(self, delta: int):
        """Step frame slider by delta."""
        new_val = max(0, min(self.num_frames - 1, self.frame_slider.value + delta))
        self.frame_slider.value = new_val

    def _on_frame_change(self, change):
        """Handle frame slider change."""
        self._update_display()

    def _on_select(self, button):
        """Handle frame selection."""
        self.selected_frame = self.frame_slider.value
        with self.output:
            clear_output(wait=True)
            print(f"✓ Selected frame {self.selected_frame} as contact frame")
            time_sec = self.selected_frame / self.fps
            print(f"  Time: {time_sec:.2f}s")

    def _update_display(self):
        """Update the frame display."""
        frame_num = self.frame_slider.value
        frame = self.frames[frame_num]
        time_sec = frame_num / self.fps

        # Check if this is a suggested contact
        is_suggested = False
        suggested_conf = None
        for f, c, s in self.suggested_contacts:
            if f == frame_num:
                is_suggested = True
                suggested_conf = c
                break

        # Create display frame
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Add frame info
        info_text = f"Frame {frame_num}/{self.num_frames-1} | {time_sec:.2f}s"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if is_suggested:
            marker = f"DETECTED CONTACT (conf: {suggested_conf:.0%})"
            color = (0, 255, 0) if suggested_conf >= LOW_CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(display_frame, marker, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display_frame, marker, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert to HTML for display
        b64 = frame_to_base64(display_frame)

        with self.output:
            clear_output(wait=True)
            display(HTML(f'<img src="data:image/jpeg;base64,{b64}" width="{min(w, 800)}"/>'))

            # Show suggested contacts
            if self.suggested_contacts:
                print("\nSuggested contacts (from automatic detection):")
                for f, c, s in self.suggested_contacts:
                    status = "✓ HIGH" if c >= LOW_CONFIDENCE_THRESHOLD else "⚠ LOW"
                    print(f"  Frame {f} ({f/self.fps:.2f}s): {c:.0%} confidence [{status}]")

    def display(self):
        """Display the selector widget."""
        # Instructions
        instructions = widgets.HTML("""
        <h3>Manual Contact Frame Selection</h3>
        <p>Use the slider or buttons to navigate to the exact frame where ball hits racket.</p>
        <p>Look for: ball touching strings, racket deformation, or direction change.</p>
        """)

        # Layout
        controls = widgets.VBox([
            instructions,
            self.frame_slider,
            self.step_buttons,
            self.select_button,
            self.output,
        ])

        display(controls)
        self._update_display()

    def get_selected_frame(self) -> Optional[int]:
        """Get the manually selected frame number."""
        return self.selected_frame


def manual_contact_selection(
    frames: List[np.ndarray],
    fps: float,
    contacts: List[Tuple[int, float, str]],
    contact_index: int = 0,
) -> Tuple[int, float, str]:
    """Interactive manual selection for a specific contact.

    Args:
        frames: List of BGR frames.
        fps: Video FPS.
        contacts: Detected contacts list.
        contact_index: Which contact to select/adjust.

    Returns:
        Updated contact tuple (frame_num, confidence, source).
    """
    if not contacts:
        initial_frame = len(frames) // 2
    elif contact_index < len(contacts):
        initial_frame = contacts[contact_index][0]
    else:
        initial_frame = contacts[0][0] if contacts else 0

    selector = ManualFrameSelector(
        frames=frames,
        fps=fps,
        initial_frame=initial_frame,
        suggested_contacts=contacts,
    )

    selector.display()

    # Note: In actual use, this would be async/interactive
    # For now, return the original or allow programmatic selection
    if selector.selected_frame is not None:
        return (selector.selected_frame, 1.0, 'manual')

    if contacts and contact_index < len(contacts):
        return contacts[contact_index]

    return (initial_frame, 0.5, 'manual_default')


def create_contact_selection_form(
    frames: List[np.ndarray],
    fps: float,
    suggested_frame: Optional[int] = None,
) -> widgets.VBox:
    """Create a simple form for contact frame input.

    For environments where full interactive widgets may not work,
    this provides a simple numeric input fallback.

    Args:
        frames: List of BGR frames.
        fps: Video FPS.
        suggested_frame: Optional suggested frame number.

    Returns:
        Widget VBox for display.
    """
    num_frames = len(frames)

    frame_input = widgets.BoundedIntText(
        value=suggested_frame or num_frames // 2,
        min=0,
        max=num_frames - 1,
        step=1,
        description='Contact Frame:',
        layout=widgets.Layout(width='250px'),
    )

    time_label = widgets.Label(
        value=f"Time: {(suggested_frame or num_frames // 2) / fps:.2f}s"
    )

    def update_time(change):
        time_label.value = f"Time: {change['new'] / fps:.2f}s"

    frame_input.observe(update_time, names='value')

    preview_output = widgets.Output()

    def show_preview(button):
        frame_num = frame_input.value
        frame = frames[frame_num]
        thumb = create_frame_thumbnail(frame, frame_num, fps, size=(640, 360))
        b64 = frame_to_base64(thumb)
        with preview_output:
            clear_output(wait=True)
            display(HTML(f'<img src="data:image/jpeg;base64,{b64}"/>'))

    preview_button = widgets.Button(description='Preview Frame')
    preview_button.on_click(show_preview)

    form = widgets.VBox([
        widgets.HTML("<h4>Enter Contact Frame Number</h4>"),
        widgets.HBox([frame_input, time_label]),
        preview_button,
        preview_output,
    ])

    return form
