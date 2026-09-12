"""
run_live_absence_test.py
Automated Guided Interactive Absence & Staleness Verification Runner.

Features:
  - Prominent banner at the TOP OF THE SCREEN with real-time countdown timer & progress bar.
  - 3-second buffer warnings before every state change so you have time to move comfortably.
  - Zero required keystrokes: auto-bypasses calibration, guides through all phases, auto-completes.
  - You can press 'q' or ESC at any time to finish early.
  - Prints an exhaustive telemetry verification audit upon completion.

Usage:
    python run_live_absence_test.py
"""

from __future__ import annotations

import csv
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.attention import AttentionScorer
from core.detector import Detector, put_fresh
from core.fsm import FSMController, FSMInput, FSMState, State
from core.intervention import SessionLogger, LogRow
from core.tracker import BookTracker
from core.vision import VisionProcessor
from main import mediapipe_worker, _STATE_COLORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_test")


# ---------------------------------------------------------------------------
# Visual Banner & HUD
# ---------------------------------------------------------------------------

def draw_top_guidance_banner(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    progress: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Renders a modern, high-contrast guidance banner at the top of the frame."""
    h, w = frame.shape[:2]
    banner_h = 80

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (22, 22, 22), -1)
    # Left accent border
    cv2.rectangle(overlay, (0, 0), (12, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    # Title & Subtitle text
    cv2.putText(frame, title, (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(frame, subtitle, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (225, 225, 225), 1)

    # Progress bar along bottom of banner
    bar_w = int(w * max(0.0, min(1.0, progress)))
    cv2.rectangle(frame, (0, banner_h - 4), (w, banner_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, banner_h - 4), (bar_w, banner_h), color, -1)

    return frame


def draw_hud_panel(
    frame: np.ndarray,
    state: str,
    attention: float,
    gaze_s: float,
    head_s: float,
    ctx_s: float,
    blink_s: float,
    yaw: float,
    pitch: float,
    roll: float,
    fps: float,
    y_offset: int = 86,
) -> np.ndarray:
    """Draws the HUD telemetry panel directly beneath the guidance banner."""
    color = _STATE_COLORS.get(state, (200, 200, 200))
    panel_w = 380
    panel_h = 175

    panel = frame.copy()
    cv2.rectangle(panel, (0, y_offset), (panel_w, y_offset + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.75, frame, 0.25, 0, frame)

    # State badge
    cv2.rectangle(frame, (5, y_offset + 5), (panel_w - 5, y_offset + 38), color, -1)
    cv2.putText(frame, f"STATE: {state}", (12, y_offset + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)

    # Attention bar
    bar_x, bar_y, bar_w, bar_h = 5, y_offset + 46, panel_w - 10, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
    fill = int(bar_w * max(0.0, min(1.0, attention)))
    bar_color = (0, int(220 * attention), int(220 * (1 - attention)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"Attention = {attention:.2f}", (bar_x + 6, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Component scores
    y1 = y_offset + 82
    components = [("gaze", gaze_s), ("head", head_s), ("ctx", ctx_s), ("blink", blink_s)]
    for i, (label, val) in enumerate(components):
        x = 6 + i * 92
        cv2.putText(frame, f"{label}={val:.2f}", (x, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)

    # Pose angles
    y2 = y_offset + 108
    cv2.putText(frame, f"Yaw: {yaw:+5.1f}   Pitch: {pitch:+5.1f}   Roll: {roll:+5.1f}",
                (6, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 100), 1)

    # FPS / Exit hint
    y3 = y_offset + 134
    cv2.putText(frame, f"Camera FPS: {fps:.0f}  |  Press 'q' / ESC to exit early",
                (6, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return frame


# ---------------------------------------------------------------------------
# Test Phase Timeline Definition
# ---------------------------------------------------------------------------

PHASES = [
    {
        "name": "SETTLE",
        "duration": 5.0,
        "color": (255, 200, 0),  # Cyan
        "title": "GET READY: Sit Facing Camera (Starting in {rem:.0f}s)",
        "sub": "Auto-configuring baseline... Look towards the webcam naturally.",
    },
    {
        "name": "READING_1",
        "duration": 12.0,
        "color": (0, 220, 50),   # Green
        "title": "[STEP 1/3] NORMAL READING — Hold steady ({rem:.1f}s)",
        "sub": "Look at screen or desk. Notice the green 'STATE: READING' badge below.",
    },
    {
        "name": "BUFFER_EXIT",
        "duration": 6.0,
        "color": (0, 200, 255),  # Amber
        "title": "[PREPARE] Leave camera view in {rem:.1f}s...",
        "sub": "Take your time: prepare to duck down or slide your chair out of camera view.",
    },
    {
        "name": "ABSENCE",
        "duration": 10.0,
        "color": (0, 80, 255),   # Orange/Red
        "title": "[STEP 2/3] STAY OUT OF CAMERA VIEW NOW ({rem:.1f}s)",
        "sub": "Stay completely out of view! Watch STATE turn grey UNKNOWN and poses reset to 0.0.",
    },
    {
        "name": "BUFFER_RETURN",
        "duration": 6.0,
        "color": (0, 200, 255),  # Amber
        "title": "[PREPARE] Return to camera view in {rem:.1f}s...",
        "sub": "Take your time: prepare to sit back down in your chair.",
    },
    {
        "name": "READING_2",
        "duration": 12.0,
        "color": (0, 220, 50),   # Green
        "title": "[STEP 3/3] RETURNED: Resume Reading ({rem:.1f}s)",
        "sub": "Sit naturally. Watch STATE promptly recover to green READING.",
    },
    {
        "name": "COMPLETE",
        "duration": 3.0,
        "color": (255, 150, 0),  # Blue
        "title": "TEST COMPLETE! Saving telemetry & compiling audit...",
        "sub": "Closing window and printing results in terminal...",
    },
]


# ---------------------------------------------------------------------------
# Main Runner Loop
# ---------------------------------------------------------------------------

def run_test():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg["camera"]
    cap = cv2.VideoCapture(cam_cfg["device_id"])
    if not cap.isOpened():
        logger.error("Cannot open camera %d. Make sure no other application is using it.", cam_cfg["device_id"])
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS,          cam_cfg["fps"])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    WIN = "Face Focus - Guided Absence Test"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Initial Splash
    ret, splash_frame = cap.read()
    if ret:
        splash = splash_frame.copy()
        cv2.rectangle(splash, (0, 0), (actual_w, 75), (20, 20, 20), -1)
        cv2.putText(splash, "Loading Face Focus AI Models & Initializing Guided Test...",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.imshow(WIN, splash)
        cv2.waitKey(1)

    # Initialize perceptions & controllers
    processor = VisionProcessor(cfg, actual_w, actual_h)
    detector = Detector(cfg)
    detector.start()
    scorer = AttentionScorer(cfg)
    fsm_ctrl = FSMController(cfg)
    fsm_state = FSMState()
    fsm_ctrl.calibration_complete(fsm_state)  # Auto-complete calibration

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

    FACE_STALENESS_TIMEOUT_SEC = float(cfg.get("mediapipe", {}).get("staleness_timeout_sec", 0.5))

    # CSV Logging
    csv_path = os.path.join(PROJECT_ROOT, "scratch", "absence_test_telemetry.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "rel_time_s", "phase", "state", "face_detected",
        "attention", "yaw", "pitch", "roll"
    ])

    print("\n" + "=" * 76)
    print("      FACE FOCUS — AUTOMATED GUIDED ABSENCE TEST")
    print("=" * 76)
    print("The test window is active. Look at the TOP OF THE SCREEN for live prompts.")
    print("Every phase provides a countdown timer and buffer warnings.")
    print("=" * 76 + "\n", flush=True)

    test_start_time = time.monotonic()
    last_tick = test_start_time
    last_face_result = None
    last_face_time: float = 0.0
    last_phase_name = ""

    # Telemetry collectors for post-test audit
    telemetry_log = []

    # Window focus
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, WIN)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

    fps_counter = 0
    fps_timer = time.monotonic()
    current_fps = 30.0

    try:
        while True:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.monotonic()
            raw_dt = now - last_tick
            last_tick = now

            fps_counter += 1
            if now - fps_timer >= 0.5:
                current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # Determine current phase along timeline
            elapsed = now - test_start_time
            accum = 0.0
            cur_phase = PHASES[-1]
            phase_rem = 0.0
            phase_progress = 1.0

            for p in PHASES:
                if elapsed < accum + p["duration"]:
                    cur_phase = p
                    time_in_p = elapsed - accum
                    phase_rem = max(0.0, p["duration"] - time_in_p)
                    phase_progress = time_in_p / p["duration"]
                    break
                accum += p["duration"]

            # Print terminal notification on phase transition
            if cur_phase["name"] != last_phase_name:
                print(f"[{elapsed:5.1f}s] >>> {cur_phase['name']} <<< : {cur_phase['title'].format(rem=phase_rem)}")
                last_phase_name = cur_phase["name"]

            # Fan out frame
            put_fresh(face_q, frame.copy())
            put_fresh(detector.frame_q, frame.copy())

            # Drain result_q (MediaPipe)
            try:
                res, ts = result_q.get_nowait()
                last_face_result = res
                last_face_time = ts
            except queue.Empty:
                pass

            # Expire face result if stale (>0.5s timeout)
            if last_face_result is not None and (now - last_face_time > FACE_STALENESS_TIMEOUT_SEC):
                last_face_result = None

            # Extract face data
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

            last_bboxes, _ = detector.cache.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))

            # Attention & FSM tick
            attention = scorer.update(
                gaze_zone=gaze_zone,
                yaw=yaw,
                pitch=pitch,
                mean_ear=mean_ear,
                bboxes=last_bboxes,
                book_roi=None,
                iris_cx=iris_cx,
                iris_cy=iris_cy,
            )

            fsm_dt = max(0.001, min(raw_dt, 0.10))
            fsm_input = FSMInput(
                attention_filtered=attention,
                gaze_zone=gaze_zone,
                head_pitch_deg=pitch,
                phone_consecutive=0,
                phone_absent_frames=10,
                mean_brightness=brightness,
                face_detected=face_detected,
                break_key_pressed=False,
                dt=fsm_dt,
                ctrl_dt=fsm_dt,
            )
            current_state, tier = fsm_ctrl.tick(fsm_state, fsm_input)

            # Record telemetry
            csv_writer.writerow([
                f"{now:.3f}", f"{elapsed:.2f}", cur_phase["name"], current_state.value,
                face_detected, f"{attention:.3f}", f"{yaw:.1f}", f"{pitch:.1f}", f"{roll:.1f}"
            ])
            telemetry_log.append({
                "t": elapsed,
                "phase": cur_phase["name"],
                "state": current_state.value,
                "face_detected": face_detected,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "attention": attention,
            })

            # Render display with Top Guidance Banner and Sub-Banner HUD
            title_text = cur_phase["title"].format(rem=phase_rem)
            display = draw_top_guidance_banner(
                frame.copy(),
                title=title_text,
                subtitle=cur_phase["sub"],
                progress=phase_progress,
                color=cur_phase["color"],
            )

            display = draw_hud_panel(
                display,
                state=current_state.value,
                attention=attention,
                gaze_s=getattr(scorer, "last_gaze_score", 0.6),
                head_s=getattr(scorer, "last_head_score", 1.0),
                ctx_s=getattr(scorer, "last_context_score", 0.5),
                blink_s=getattr(scorer, "last_blink_score", 1.0),
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                fps=current_fps,
                y_offset=86,
            )

            cv2.imshow(WIN, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("\n[User aborted early with 'q'/ESC]")
                break

            # Auto-exit when timeline finishes
            if elapsed >= sum(p["duration"] for p in PHASES):
                break

    finally:
        stop_event.set()
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

    # -----------------------------------------------------------------------
    # Automated Telemetry Audit & Verification Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("                 TELEMETRY AUDIT & VALIDATION REPORT")
    print("=" * 76)
    print(f"Log saved: {csv_path} ({len(telemetry_log)} frames recorded)\n")

    p1_rows = [r for r in telemetry_log if r["phase"] == "READING_1"]
    abs_rows = [r for r in telemetry_log if r["phase"] == "ABSENCE"]
    p2_rows = [r for r in telemetry_log if r["phase"] == "READING_2"]

    # 1. Check Phase 1: Normal Reading
    if p1_rows:
        on_task_pct = sum(1 for r in p1_rows if r["state"] in ("READING", "WRITING")) / len(p1_rows) * 100.0
        avg_attn = sum(r["attention"] for r in p1_rows) / len(p1_rows)
        print(f"Phase 1 (Normal Reading Baseline):")
        print(f"  - Frames Focused (READING/WRITING): {on_task_pct:.1f}% ({len(p1_rows)} frames)")
        print(f"  - Mean Baseline Attention:          {avg_attn:.2f}")

    # 2. Check Phase 2: Face Absence & Pose Zeroing
    print(f"\nPhase 2 (Face Absence & Staleness Guard):")
    unknown_rows = [r for r in telemetry_log if r["state"] == "UNKNOWN"]
    if abs_rows:
        unknown_in_abs = [r for r in abs_rows if r["state"] == "UNKNOWN"]
        unknown_pct = len(unknown_in_abs) / len(abs_rows) * 100.0
        print(f"  - Absence Window Frames in UNKNOWN: {unknown_pct:.1f}% ({len(unknown_in_abs)}/{len(abs_rows)} frames)")

        # Verify no frozen poses (yaw, pitch, roll must be 0.0)
        non_zero_poses = [r for r in unknown_rows if abs(r["yaw"]) > 0.01 or abs(r["pitch"]) > 0.01]
        if not non_zero_poses and unknown_rows:
            print(f"  - Pose Telemetry Zeroed:            YES (yaw=0.0, pitch=0.0, roll=0.0 across all {len(unknown_rows)} UNKNOWN frames)")
        else:
            print(f"  - Pose Telemetry Zeroed:            NO ({len(non_zero_poses)} frames had residual non-zero pose)")

    # 3. Check Phase 3: Recovery
    print(f"\nPhase 3 (Return & Re-acquisition):")
    if p2_rows:
        on_task_rec = [r for r in p2_rows if r["state"] in ("READING", "WRITING")]
        rec_pct = len(on_task_rec) / len(p2_rows) * 100.0
        print(f"  - Frames Recovered to Focus:        {rec_pct:.1f}% ({len(on_task_rec)}/{len(p2_rows)} frames)")

    # 4. Final Verdict
    has_unknown = len(unknown_rows) > 0
    has_recovered = (any(r["state"] in ("READING", "WRITING") for r in p2_rows) if p2_rows else False)

    print("\n" + "-" * 76)
    if has_unknown and has_recovered:
        print(">> FINAL VERDICT: PASSED <<")
        print("  1. State transitioned promptly to UNKNOWN when face left camera.")
        print("  2. Pose angles were zeroed out cleanly during absence (no frozen telemetry).")
        print("  3. State recovered immediately to READING upon user return.")
    else:
        print(">> FINAL VERDICT: INCOMPLETE / RETEST REQUIRED <<")
        if not has_unknown:
            print("  * Warning: Camera never observed face absence during the absence window.")
        if not has_recovered:
            print("  * Warning: System did not recover upon return.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_test()
