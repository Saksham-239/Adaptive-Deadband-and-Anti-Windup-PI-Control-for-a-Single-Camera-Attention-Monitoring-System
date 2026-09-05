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
    fps: float,
    bboxes: list,
    book_roi: Optional[tuple],
    debug_cfg: dict,
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
        tier = min(int(bucket), 3)
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
                      (book_roi[2], book_roi[3]), (0, 255, 200), 2)
        cv2.putText(frame, "BOOK ROI", (book_roi[0], book_roi[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

    return frame


# ---------------------------------------------------------------------------
# Calibration phase (mouse-drawn ROI)
# ---------------------------------------------------------------------------

class CalibrationUI:
    """Lets user draw a rectangle on the frame to define the book ROI."""

    def __init__(self, window_name: str):
        self._window = window_name
        self.roi: Optional[tuple[int, int, int, int]] = None
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

    def _finish_box(self):
        if self._start and self._end:
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            if x2 - x1 > 20 and y2 - y1 > 20:  # sanity check
                self.roi = (x1, y1, x2, y2)

    def _mouse_cb(self, event, x, y, flags, param):
        self._current_mouse = (x, y)
        is_lbutton = (flags & cv2.EVENT_FLAG_LBUTTON) != 0

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._end   = (x, y)
            self._confirmed = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if is_lbutton:
                if not self._drawing:
                    # Windows touchpad dropped the LBUTTONDOWN event
                    self._drawing = True
                    self._start = (x, y)
                    self._confirmed = False
                self._end = (x, y)
            else:
                # Button released during move
                if self._drawing:
                    self._drawing = False
                    self._finish_box()

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._end = (x, y)
            self._finish_box()

    def confirm(self) -> None:
        self._confirmed = True

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def draw_guide(self, frame: np.ndarray) -> np.ndarray:
        """Overlay calibration instructions and current rectangle."""
        overlay = frame.copy()
        cv2.putText(overlay, "CALIBRATION: Click and drag to draw a rectangle around your book/study material",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay, "ENTER = confirm (or skip ROI)   R = reset   S = skip calibration",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)

        # Draw mouse crosshairs to confirm tracking is active
        if self._current_mouse:
            mx, my = self._current_mouse
            h, w = overlay.shape[:2]
            cv2.line(overlay, (mx, 0), (mx, h), (0, 150, 255), 1)
            cv2.line(overlay, (0, my), (w, my), (0, 150, 255), 1)
            cv2.circle(overlay, (mx, my), 4, (0, 255, 255), -1)

        if self._start and self._end:
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 100), 2)
            if self.roi:
                cv2.putText(overlay, "Press ENTER to confirm",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
        return overlay


# ---------------------------------------------------------------------------
# MediaPipe worker thread
# ---------------------------------------------------------------------------

def mediapipe_worker(
    face_q: queue.Queue,
    result_q: queue.Queue,
    cfg: dict,
    frame_w: int,
    frame_h: int,
    stop_event: threading.Event,
) -> None:
    """
    Consumes raw frames from face_q, runs MediaPipe, pushes FaceResult to result_q.
    Runs as a daemon thread.
    """
    from core.vision import VisionProcessor
    from core.detector import put_fresh

    try:
        processor = VisionProcessor(cfg, frame_w, frame_h)
    except Exception as exc:
        logger.error("VisionProcessor init failed: %s", exc)
        return

    target_fps   = cfg["mediapipe"]["target_fps"]
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
        if result is not None:
            put_fresh(result_q, result)
        last_run = time.monotonic()

    processor.close()
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
        logger.error("Cannot open camera %d. Is it in use?", cam_cfg["device_id"])
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS,          cam_cfg["fps"])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera opened: %dx%d", actual_w, actual_h)

    # --- Queues ---
    face_q:   queue.Queue = queue.Queue(maxsize=1)
    result_q: queue.Queue = queue.Queue(maxsize=1)
    frame_q:  queue.Queue = queue.Queue(maxsize=1)

    # --- Threads ---
    stop_event = threading.Event()

    mp_thread = threading.Thread(
        target=mediapipe_worker,
        args=(face_q, result_q, cfg, actual_w, actual_h, stop_event),
        name="mediapipe-worker",
        daemon=True,
    )
    mp_thread.start()

    from core.detector import Detector, put_fresh
    detector = Detector(cfg)
    detector.start()

    # --- Core modules ---
    from core.attention import AttentionScorer
    from core.fsm import FSMController, FSMState, State
    from core.intervention import TTSEngine, SessionLogger, LogRow

    scorer  = AttentionScorer(cfg)
    fsm_ctrl = FSMController(cfg)
    fsm_state = FSMState()
    tts     = TTSEngine(cfg)
    db      = SessionLogger(cfg)

    # --- Window & Calibration ---
    WIN = "Face Focus — Study Attention Monitor"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    calib = CalibrationUI(WIN)
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

            # Fan out frame to both worker threads
            put_fresh(face_q,  frame.copy())
            put_fresh(frame_q, frame.copy())
            put_fresh(detector.frame_q, frame.copy())

            # Collect latest results (non-blocking)
            try:
                last_face_result = result_q.get_nowait()
            except queue.Empty:
                pass

            last_bboxes, _ = detector.cache.read()

            # --- Tick timing ---
            now = time.monotonic()
            dt  = now - last_tick
            last_tick = now

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
                key = cv2.waitKey(30) & 0xFF  # 30ms gives Windows time to process mouse events
                if key == 13:  # ENTER
                    if calib.roi:
                        book_roi = calib.roi
                    else:
                        # No ROI drawn — use full frame as book area
                        book_roi = (0, 0, actual_w, actual_h)
                        logger.info("No ROI drawn; using full frame as book ROI.")
                    fsm_ctrl.calibration_complete(fsm_state)
                    logger.info("Calibration confirmed: ROI=%s", book_roi)
                elif key == ord('r'):
                    calib.roi = None
                elif key == ord('s'):  # 's' to skip calibration entirely
                    book_roi = None
                    fsm_ctrl.calibration_complete(fsm_state)
                    logger.info("Calibration skipped.")
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
                dt=dt,
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
                fps=fps_display,
                bboxes=last_bboxes,
                book_roi=book_roi,
                debug_cfg=debug_cfg,
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
                calib.roi = None
                book_roi  = None
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
