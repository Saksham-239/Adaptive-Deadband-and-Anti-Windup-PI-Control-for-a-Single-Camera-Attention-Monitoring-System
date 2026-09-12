"""
main.py
Adaptive AI Study Attention System — Main Orchestration Loop.

Architecture:
  cv2.VideoCapture (main thread)
        │
        ├──[raw frame]──► face_q  ──► MediaPipe thread → result_q ──► main
        └──[raw frame]──► frame_q ──► YOLO thread → detection_cache  ──► main
                                           ↓
                                    CONTEXT ENGINE
                                    ATTENTION SCORER (EMA)
                                    FSM + INTERVENTION BUCKET
                                    TTS + SQLite logger
                                    OpenCV overlay

Controls:
  q / ESC     — quit
  b           — toggle BREAK state
  c           — re-enter calibration (re-draw book ROI)
  d           — toggle debug overlay
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "CALIBRATING": (255, 200, 0),
    "READING":     (0, 220, 50),
    "WRITING":     (0, 180, 255),
    "THINKING":    (200, 200, 0),
    "DISTRACTED":  (0, 60, 255),
    "PHONE":       (0, 0, 255),
    "BREAK":       (180, 180, 180),
    "UNKNOWN":     (100, 100, 100),
}


def draw_overlay(
    frame: np.ndarray,
    state: str,
    attention: float,
    gaze_s: float,
    head_s: float,
    ctx_s: float,
    blink_s: float,
    bucket: float,
    tier: int,
    fps: float,
    bboxes: list,
    book_roi: Optional[tuple],
    debug_cfg: dict,
    book_q: Optional[float] = None,
) -> np.ndarray:
    """Draw HUD overlay onto frame. Returns modified frame."""
    h, w = frame.shape[:2]
    color = _STATE_COLORS.get(state, (200, 200, 200))

    # Semi-transparent panel
    if debug_cfg.get("show_overlay", True):
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (380, 180), (20, 20, 20), -1)
        alpha = debug_cfg.get("overlay_alpha", 0.7)
        cv2.addWeighted(panel, alpha, frame, 1 - alpha, 0, frame)

        # State badge
        cv2.rectangle(frame, (5, 5), (375, 40), color, -1)
        cv2.putText(frame, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # Score bar
        bar_x, bar_y, bar_w, bar_h = 5, 48, 370, 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * attention)
        bar_color = (0, int(200 * attention), int(200 * (1 - attention)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"A={attention:.2f}", (bar_x + 4, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Component scores
        y0 = 85
        items = [
            ("gaze",    gaze_s),
            ("head",    head_s),
            ("ctx",     ctx_s),
            ("blink",   blink_s),
        ]
        for i, (label, val) in enumerate(items):
            x = 5 + i * 95
            cv2.putText(frame, f"{label}={val:.2f}", (x, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # Bucket / tier
        cv2.putText(frame, f"bucket={bucket:.1f}  tier={tier}  fps={fps:.0f}",
                    (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Controls hint
        cv2.putText(frame, "q=quit  b=break  c=calibrate  d=debug",
                    (5, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    # YOLO bounding boxes
    if debug_cfg.get("show_bboxes", True):
        for bbox in bboxes:
            box_color = (0, 0, 255) if bbox.label == "cell_phone" else (0, 200, 80)
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), box_color, 2)
            cv2.putText(frame, f"{bbox.label} {bbox.confidence:.2f}",
                        (bbox.x1, bbox.y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # Book ROI
    if book_roi is not None:
        cv2.rectangle(frame, (book_roi[0], book_roi[1]),
                      (book_roi[2], book_roi[3]), (0, 255, 0), 2)
        label_y = max(book_roi[1] - 8, 20)
        label_text = f"BOOK ROI Q={book_q:.2f}" if book_q is not None else "BOOK ROI"
        badge_w = 125 if book_q is not None else 90
        cv2.rectangle(frame, (book_roi[0], label_y - 16),
                      (book_roi[0] + badge_w, label_y + 4), (0, 180, 0), -1)
        cv2.putText(frame, label_text, (book_roi[0] + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return frame



# ---------------------------------------------------------------------------
# Calibration phase (mouse-drawn ROI)
# ---------------------------------------------------------------------------

class CalibrationUI:
    """Lets user define the book ROI on the webcam frame.
    
    Supports:
    1. Keyboard corner-marking: Hover cursor over book corners and press 'A'
       - Press 'A' at corner 1, then at corner 2 (or all 4 corners)
       - Automatically calculates the bounding box enclosing all marked points
    2. Click-and-drag: Press down, drag, release
    3. Two-click selection: Click first corner, move, click second corner
    """

    def __init__(self, window_name: str):
        self._window = window_name
        self.roi: Optional[tuple[int, int, int, int]] = None
        self.points: list[tuple[int, int]] = []
        self._drawing = False
        self._start: Optional[tuple[int, int]] = None
        self._end:   Optional[tuple[int, int]] = None
        self._current_mouse: Optional[tuple[int, int]] = None
        self._confirmed = False
        self._registered = False

    def register(self):
        if not self._registered:
            cv2.setMouseCallback(self._window, self._mouse_cb)
            self._registered = True

    def reset(self):
        """Fully reset the calibration state and clear visual guides."""
        self.roi = None
        self.points.clear()
        self._start = None
        self._end = None
        self._drawing = False
        self._confirmed = False

    def add_point(self, x: int, y: int) -> int:
        """Record a corner coordinate marked by pressing 'A'.
        
        When >= 2 points are registered, recalculates the bounding box
        enclosing all marked corners and updates self.roi.
        """
        # Reject duplicate clicks within 8 pixels of previous point
        if self.points and abs(x - self.points[-1][0]) < 8 and abs(y - self.points[-1][1]) < 8:
            return len(self.points)

        self.points.append((int(x), int(y)))
        if len(self.points) == 1:
            self._start = (int(x), int(y))
        elif len(self.points) >= 2:
            x1 = min(p[0] for p in self.points)
            y1 = min(p[1] for p in self.points)
            x2 = max(p[0] for p in self.points)
            y2 = max(p[1] for p in self.points)
            if x2 - x1 >= 15 and y2 - y1 >= 15:
                self.roi = (x1, y1, x2, y2)
                self._drawing = False
        return len(self.points)

    def get_cursor_pos(self, frame_w: int, frame_h: int) -> Optional[tuple[int, int]]:
        """Get current cursor position mapped to frame coordinates."""
        # 1. OpenCV mouse callback (most accurate within window client space)
        if self._current_mouse is not None:
            cx, cy = self._current_mouse
            if 0 <= cx < frame_w and 0 <= cy < frame_h:
                return (cx, cy)

        # 2. Windows API fallback with child window enumeration and scaling
        try:
            import win32gui
            hwnd = win32gui.FindWindow(None, self._window)
            if hwnd:
                children = []
                win32gui.EnumChildWindows(hwnd, lambda h, extra: children.append(h), None)
                target_hwnd = children[0] if children else hwnd
                pt = win32gui.GetCursorPos()
                cx, cy = win32gui.ScreenToClient(target_hwnd, pt)
                rect = win32gui.GetClientRect(target_hwnd)
                win_w = rect[2] - rect[0]
                win_h = rect[3] - rect[1]
                if win_w > 0 and win_h > 0:
                    scaled_x = int(cx * (frame_w / win_w))
                    scaled_y = int(cy * (frame_h / win_h))
                    if 0 <= scaled_x < frame_w and 0 <= scaled_y < frame_h:
                        self._current_mouse = (scaled_x, scaled_y)
                        return (scaled_x, scaled_y)
        except Exception:
            pass

        return self._current_mouse

    def _finish_box(self) -> bool:
        if self._start and self._end:
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            if x2 - x1 >= 15 and y2 - y1 >= 15:  # sanity check
                self.roi = (x1, y1, x2, y2)
                return True
        return False

    def _mouse_cb(self, event, x, y, flags, param):
        self._current_mouse = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self._drawing:
                # Start new selection
                self._drawing = True
                self._start = (x, y)
                self._end   = (x, y)
                self._confirmed = False
            else:
                # Two-click mode: second click finalizes the box
                self._end = (x, y)
                if self._finish_box():
                    self._drawing = False
                else:
                    self._start = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drawing:
                self._end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._drawing:
                self._end = (x, y)
                # If dragged sufficiently, finalize. Otherwise keep drawing active for second click.
                if self._finish_box():
                    self._drawing = False

    def confirm(self) -> None:
        self._confirmed = True

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def draw_guide(self, frame: np.ndarray) -> np.ndarray:
        """Overlay calibration instructions, crosshairs, corner badges, and ROI rectangle."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Top instruction banner
        banner_h = 88
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)

        # Title & instructions
        cv2.putText(overlay, "CALIBRATION: Define your book / study ROI",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(overlay, "[A] = mark corner  |  [Mouse] = drag/click  |  [SPACE] = box selector  |  ENTER = confirm",
                    (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Dynamic step guidance
        n_pts = len(self.points)
        if n_pts == 0:
            step_text = "[Step 1] Move cursor over Corner 1 and press 'A', drag mouse, or press SPACE"
            step_color = (0, 255, 255)
        elif n_pts == 1:
            step_text = f"[Step 2] Corner 1 set at {self.points[0]}. Move to Corner 2 and press 'A'"
            step_color = (0, 220, 255)
        else:
            step_text = f"[Ready!] {n_pts} corners marked. Press ENTER to confirm, or 'A' for next corner"
            step_color = (0, 255, 0)

        cv2.putText(overlay, step_text, (15, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, step_color, 1)

        # Crosshairs at current cursor position
        cur_pt = self.get_cursor_pos(w, h)
        if cur_pt:
            mx, my = cur_pt
            # Thin crosshairs spanning entire window
            cv2.line(overlay, (mx, 0), (mx, h), (0, 180, 255), 1)
            cv2.line(overlay, (0, my), (w, my), (0, 180, 255), 1)
            # Center reticle
            cv2.circle(overlay, (mx, my), 5, (0, 255, 255), -1)
            cv2.circle(overlay, (mx, my), 9, (0, 0, 0), 1)
            # Cursor coordinate label
            cv2.putText(overlay, f"({mx}, {my})", (mx + 8, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        # Live box preview when 1 point is marked and cursor is active
        if n_pts == 1 and cur_pt:
            p1 = self.points[0]
            px1, py1 = min(p1[0], cur_pt[0]), min(p1[1], cur_pt[1])
            px2, py2 = max(p1[0], cur_pt[0]), max(p1[1], cur_pt[1])
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (255, 200, 0), 1)
            cv2.putText(overlay, f"{px2 - px1}x{py2 - py1} (Press 'A' to mark)",
                        (px1 + 4, py1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1)

        # Marked corner pins
        for i, pt in enumerate(self.points):
            cv2.circle(overlay, pt, 7, (0, 255, 255), -1)
            cv2.circle(overlay, pt, 10, (0, 0, 0), 2)
            badge_text = f"C{i+1}"
            bx, by = pt[0] + 12, pt[1] - 4
            cv2.rectangle(overlay, (bx - 2, by - 14), (bx + 30, by + 4), (30, 30, 30), -1)
            cv2.putText(overlay, badge_text, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        # Mouse dragging box
        if self._drawing and self._start and self._end:
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{x2 - x1}x{y2 - y1}", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Finalized / multi-point ROI box
        elif self.roi is not None:
            x1, y1, x2, y2 = self.roi
            # Subtle green fill
            tint = overlay.copy()
            cv2.rectangle(tint, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.addWeighted(tint, 0.12, overlay, 0.88, 0, overlay)

            # Thick solid green outline
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Header badge above the box
            badge_y = max(y1 - 10, banner_h + 20)
            badge_w = 260
            cv2.rectangle(overlay, (x1, badge_y - 20), (x1 + badge_w, badge_y + 4), (0, 180, 0), -1)
            cv2.putText(overlay, f"BOOK ROI: {x2-x1}x{y2-y1} [ENTER to Confirm]",
                        (x1 + 6, badge_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)

        return overlay


# ---------------------------------------------------------------------------
# MediaPipe worker thread
# ---------------------------------------------------------------------------

def mediapipe_worker(
    processor,
    face_q: queue.Queue,
    result_q: queue.Queue,
    target_fps: int,
    stop_event: threading.Event,
) -> None:
    """
    Consumes raw frames from face_q, runs MediaPipe, pushes FaceResult to result_q.
    Runs as a daemon thread.
    """
    from core.detector import put_fresh

    min_interval = 1.0 / max(target_fps, 1)
    last_run     = 0.0

    while not stop_event.is_set():
        now = time.monotonic()
        if now - last_run < min_interval:
            time.sleep(min_interval - (now - last_run))
            continue

        try:
            frame = face_q.get(timeout=0.5)
        except queue.Empty:
            continue

        result = processor.process_frame(frame)
        put_fresh(result_q, (result, now))
        last_run = time.monotonic()

    if hasattr(processor, "close"):
        try:
            processor.close()
        except Exception:
            pass
    logger.info("MediaPipe worker stopped.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    session_id = str(uuid.uuid4())[:8]
    logger.info("Session ID: %s", session_id)

    # --- Camera ---
    cam_cfg = cfg["camera"]
    cap = cv2.VideoCapture(cam_cfg["device_id"])
    if not cap.isOpened():
        logger.error("Cannot open camera %d. Is it in use by another application?", cam_cfg["device_id"])
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS,          cam_cfg["fps"])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera opened: %dx%d", actual_w, actual_h)

    # --- Window & Immediate Loading Splash ---
    WIN = "Face Focus - Attention Monitor"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    calib = CalibrationUI(WIN)
    calib.register()

    # Show live frame immediately so the user sees the camera window without delay
    ret, initial_frame = cap.read()
    if ret:
        splash = initial_frame.copy()
        cv2.rectangle(splash, (0, 0), (actual_w, 64), (20, 20, 20), -1)
        cv2.putText(splash, "Face Focus: Initializing AI perception models...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.imshow(WIN, splash)
        cv2.waitKey(1)

    from core.vision import VisionProcessor
    from core.detector import Detector, put_fresh
    from core.tracker import BookTracker
    from core.attention import AttentionScorer
    from core.fsm import FSMController, FSMState, State
    from core.intervention import TTSEngine, SessionLogger, LogRow

    logger.info("Initializing VisionProcessor (MediaPipe)...")
    processor = VisionProcessor(cfg, actual_w, actual_h)

    logger.info("Initializing Object Detector (YOLO)...")
    detector = Detector(cfg)
    detector.start()

    logger.info("Initializing Dynamic BookTracker...")
    bt_cfg = cfg.get("book_tracker", {})
    book_tracker = BookTracker(
        q_thresh=bt_cfg.get("q_thresh", 0.35),
        color_thresh=bt_cfg.get("color_thresh", 0.25),
        high_conf_thresh=bt_cfg.get("high_conf_thresh", 0.50),
        absence_timeout_sec=bt_cfg.get("absence_timeout_sec", 1.5),
        reacquisition_conf=bt_cfg.get("reacquisition_conf", 0.12),
    )

    logger.info("Initializing Attention Scorer & Controllers...")
    scorer   = AttentionScorer(cfg)
    fsm_ctrl = FSMController(cfg)
    fsm_state = FSMState()
    tts      = TTSEngine(cfg)
    db       = SessionLogger(cfg)

    # --- Queues & Worker Threads ---
    face_q:   queue.Queue = queue.Queue(maxsize=1)
    result_q: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    mp_thread = threading.Thread(
        target=mediapipe_worker,
        args=(processor, face_q, result_q, cfg["mediapipe"]["target_fps"], stop_event),
        name="mediapipe-worker",
        daemon=True,
    )
    mp_thread.start()

    # Automatically bring window to foreground and give input focus
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, WIN)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

    book_roi: Optional[tuple[int, int, int, int]] = None

    # --- State tracking ---
    phone_consecutive  = 0
    phone_absent       = 0
    break_key_held     = False
    show_debug         = cfg["debug"]["show_overlay"]
    debug_cfg          = cfg["debug"]

    # FPS tracking
    fps_counter = 0
    fps_timer   = time.monotonic()
    fps_display = 0.0

    last_tick = time.monotonic()
    last_face_result = None
    last_face_time: float = 0.0
    FACE_STALENESS_TIMEOUT_SEC: float = float(
        cfg.get("mediapipe", {}).get("staleness_timeout_sec", 0.5)
    )
    last_bboxes: list = []
    key_now: int = 0xFF  # safe default before first waitKey

    logger.info("Starting main loop. Draw book ROI then press ENTER.")

    try:
        while True:
            # Detect window closed via X button
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Window closed by user.")
                break

            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed -- camera disconnected?")
                time.sleep(0.1)
                continue

            # Fan out frame to worker threads
            put_fresh(face_q,  frame.copy())
            put_fresh(detector.frame_q, frame.copy())

            # --- Tick timing ---
            now = time.monotonic()
            raw_dt = now - last_tick
            last_tick = now

            # Collect latest results (non-blocking)
            try:
                res, ts = result_q.get_nowait()
                last_face_result = res
                last_face_time = ts
            except queue.Empty:
                pass

            # Expire face result if stale (e.g. worker stalled or frame dropped > 0.5s)
            if last_face_result is not None and (now - last_face_time > FACE_STALENESS_TIMEOUT_SEC):
                last_face_result = None

            last_bboxes, _ = detector.cache.read()
            reacq_bboxes, _ = detector.cache.read_reacq()

            # --- Dynamic Book Tracking ---
            if fsm_state.current != State.CALIBRATING and book_tracker.is_initialized:
                is_book_present, tracked_roi, _ = book_tracker.update(
                    frame, yolo_bboxes=reacq_bboxes, timestamp=now
                )
                if tracked_roi != book_roi:
                    book_roi = tracked_roi
                    processor.set_book_roi(tracked_roi)

            # Decoupled timing: FSM wall-clock tracking vs PI integration
            fsm_dt = max(0.001, min(raw_dt, 60.0))
            if raw_dt > 0.50:
                # Discontinuity detected (e.g. window drag, lag spike) -> hold PI integration
                ctrl_dt = 0.0
            else:
                ctrl_dt = max(0.001, min(raw_dt, 0.10))

            # --- Brightness check ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))

            # --- Extract face data ---
            face_detected = last_face_result is not None
            if face_detected:
                fr = last_face_result
                yaw, pitch, roll = fr.yaw, fr.pitch, fr.roll
                mean_ear  = fr.mean_ear
                gaze_zone = fr.gaze_zone
                iris_cx   = ((fr.left_iris_center[0] + fr.right_iris_center[0]) / 2.0
                             if fr.left_iris_center and fr.right_iris_center else None)
                iris_cy   = ((fr.left_iris_center[1] + fr.right_iris_center[1]) / 2.0
                             if fr.left_iris_center and fr.right_iris_center else None)
            else:
                yaw = pitch = roll = 0.0
                mean_ear  = 0.3
                gaze_zone = "unknown"
                iris_cx = iris_cy = None

            # --- Phone consecutive counter ---
            phone_in_frame = any(b.label == "cell_phone" for b in last_bboxes)
            if phone_in_frame:
                phone_consecutive += 1
                phone_absent = 0
            else:
                phone_absent += 1
                phone_consecutive = 0

            # --- Calibration phase ---
            if fsm_state.current == State.CALIBRATING:
                display = calib.draw_guide(frame.copy())
                cv2.imshow(WIN, display)
                calib.register()  # Safe to register now that window is rendered
                key = cv2.waitKey(30) & 0xFF  # 30ms gives Windows time to process mouse & keyboard events
                if key in (ord('a'), ord('A')):
                    cur_pt = calib.get_cursor_pos(frame.shape[1], frame.shape[0])
                    if cur_pt:
                        n = calib.add_point(cur_pt[0], cur_pt[1])
                        logger.info("Marked corner %d at (%d, %d). Total corners=%d. ROI=%s",
                                    n, cur_pt[0], cur_pt[1], len(calib.points), calib.roi)
                    else:
                        logger.warning("Could not detect cursor position. Move mouse cursor over the window and press 'A'.")
                elif key in (ord('b'), ord('B'), 32):  # 'B' or SPACE: Native OpenCV selectROI
                    logger.info("Opening OpenCV native selectROI...")
                    rect = cv2.selectROI(WIN, frame, fromCenter=False, showCrosshair=True)
                    if rect[2] >= 15 and rect[3] >= 15:
                        rx, ry, rw, rh = rect
                        calib.roi = (rx, ry, rx + rw, ry + rh)
                        calib.points = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
                        calib._drawing = False
                        logger.info("Native selectROI captured: %s", calib.roi)
                    calib._registered = False
                    calib.register()  # Restore callback after selectROI clears it
                elif key == 13:  # ENTER
                    if calib.roi:
                        book_roi = calib.roi
                        book_tracker.set_roi(frame, book_roi, timestamp=time.monotonic())
                        processor.set_book_roi(book_roi)
                        fsm_ctrl.calibration_complete(fsm_state)
                        logger.info("Calibration confirmed: ROI=%s, BookTracker initialized", book_roi)
                    else:
                        logger.info("ENTER pressed with no ROI selected. Hover over corners and press 'A', press SPACE for box selector, or press 's' to skip.")
                elif key in (ord('r'), ord('R')):
                    calib.reset()
                    book_tracker.reset()
                    processor.set_book_roi(None)
                    logger.info("Calibration reset.")
                elif key in (ord('s'), ord('S')):  # 's' to skip calibration entirely
                    book_roi = None
                    book_tracker.reset()
                    processor.set_book_roi(None)
                    fsm_ctrl.calibration_complete(fsm_state)
                    logger.info("Calibration skipped by user (no book ROI).")
                elif key in (ord('q'), 27):
                    break
                continue

            # --- Attention score ---
            attention = scorer.update(
                gaze_zone=gaze_zone,
                yaw=yaw,
                pitch=pitch,
                mean_ear=mean_ear,
                bboxes=last_bboxes,
                book_roi=book_roi,
                iris_cx=iris_cx,
                iris_cy=iris_cy,
            )

            # --- FSM tick ---
            from core.fsm import FSMInput
            # key_now is read AFTER imshow below; use last value here for break_key
            fsm_input = FSMInput(
                attention_filtered=attention,
                gaze_zone=gaze_zone,
                head_pitch_deg=pitch,
                phone_consecutive=phone_consecutive,
                phone_absent_frames=phone_absent,
                mean_brightness=brightness,
                face_detected=face_detected,
                break_key_pressed=(key_now == ord('b')),
                dt=fsm_dt,
                ctrl_dt=ctrl_dt,
            )
            current_state, tier = fsm_ctrl.tick(fsm_state, fsm_input)

            # --- Intervention ---
            intervention_fired = False
            if tier > 0:
                intervention_fired = tts.speak(tier)

            # --- Logging ---
            row = LogRow(
                session_id=session_id,
                timestamp=time.time(),
                state=current_state.value,
                attention_raw=scorer.last_raw,
                attention_filtered=attention,
                gaze_score=scorer.last_gaze_score,
                head_score=scorer.last_head_score,
                context_score=scorer.last_context_score,
                blink_score=scorer.last_blink_score,
                yaw=yaw, pitch=pitch, roll=roll,
                gaze_zone=gaze_zone,
                intervention_tier=tier,
                intervention_fired=intervention_fired,
            )
            db.maybe_log(row)

            # --- FPS tracking ---
            fps_counter += 1
            if time.monotonic() - fps_timer >= 1.0:
                fps_display = fps_counter / (time.monotonic() - fps_timer)
                fps_counter = 0
                fps_timer   = time.monotonic()

            # --- Overlay & display ---
            display_frame = draw_overlay(
                frame=frame.copy(),
                state=current_state.value,
                attention=attention,
                gaze_s=scorer.last_gaze_score,
                head_s=scorer.last_head_score,
                ctx_s=scorer.last_context_score,
                blink_s=scorer.last_blink_score,
                bucket=fsm_state.intervention_bucket,
                tier=tier,
                fps=fps_display,
                bboxes=last_bboxes,
                book_roi=book_roi,
                debug_cfg=debug_cfg,
                book_q=book_tracker.last_q if book_tracker.is_initialized and book_roi else None,
            )
            cv2.imshow(WIN, display_frame)

            # --- Key handling ---
            key_now = cv2.waitKey(1) & 0xFF
            if key_now == ord('q') or key_now == 27:
                logger.info("Quit requested.")
                break
            elif key_now == ord('c'):
                # Re-enter calibration
                fsm_state.current = State.CALIBRATING
                fsm_state.state_duration = 0.0
                calib.reset()
                book_tracker.reset()
                book_roi  = None
                processor.set_book_roi(None)
                logger.info("Re-entering calibration.")
            elif key_now == ord('d'):
                debug_cfg["show_overlay"] = not debug_cfg.get("show_overlay", True)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stop_event.set()
        detector.stop()
        tts.stop()
        db.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Pump event loop one last time to force window closure on Windows
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
