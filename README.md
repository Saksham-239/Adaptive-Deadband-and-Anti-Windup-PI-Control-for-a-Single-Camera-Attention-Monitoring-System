# Face Focus — Real-Time Study Attention Monitor

A real-time computer-vision system that estimates study attention from a webcam and responds to sustained distraction with state-aware interventions.

It combines **MediaPipe face/pose landmarks**, **YOLOv8 object detection**, a **dynamic visual book tracker (CSRT + NCC + Color Histogram)**, an **attention score with EMA smoothing**, an **8-state finite-state machine with Adaptive Deadband & Anti-Windup PI Control**, **text-to-speech interventions**, and **SQLite session logging**.

> The goal is not to classify every frame as “focused” or “distracted”. The system models attention as a **temporal state** and avoids nagging the user during brief, legitimate gaze shifts such as thinking or writing.

## What It Does

```text
Webcam
  │
  ├──► MediaPipe (15Hz) ──► gaze / head pose / blink features (staleness timeout: 0.5s)
  │
  └──► YOLOv8 (5Hz) ─────► contextual objects & reacquisition proposals
                               │
                               ├──► Dynamic BookTracker (CSRT + NCC + HSV Color Gating)
                               │           ↓
                               ├──► Context Engine → Attention Scorer → EMA
                               │           ↓
                               └──► 8-State FSM (Adaptive Deadband)
                                           │
                                 ┌─────────┴─────────┐
                                 ▼                   ▼
                          Anti-Windup PI       SQLite logger
                                 │                   │
                                 ▼                   ▼
                            Voice TTS           Session data
```

### States

| State | Meaning |
|---|---|
| `CALIBRATING` | Initial study-material / book ROI setup |
| `READING` | Attention is consistent with reading (screen or calibrated desk ROI) |
| `WRITING` | Looking down at the desk / notes with sustained head pitch |
| `THINKING` | Brief gaze-away / cognitive pause (≤8s); intervention suppressed |
| `DISTRACTED` | Sustained inattention exceeding adaptive threshold |
| `PHONE` | Phone detected in active view |
| `BREAK` | User-declared break (interventions suspended) |
| `UNKNOWN` | Face absent / dark frame — scoring paused, interventions suspended |

## Key Engineering Decisions

### Temporal attention instead of frame-by-frame classification
A single gaze-away frame should not trigger a voice warning. The system uses an EMA-smoothed attention score and an FSM with explicit timing thresholds so transient behaviour can be distinguished from sustained distraction.

### Concurrent vision pipelines & absence decoupling
MediaPipe and YOLO run asynchronously so the main camera/UI loop is not blocked by inference. Fresh frames are pushed through bounded queues, and a 0.5s wall-clock staleness guard expires face results immediately upon camera exit, preventing frozen telemetry.

### Dynamic visual book tracking
Anchored by initial user calibration, `core/tracker.py` maintains tracking independently of repeated YOLO classifications. It combines sub-millisecond OpenCV CSRT visual tracking with a dual appearance gate ($Q_{\text{ncc}} \ge 0.50 \lor (Q_{\text{ncc}} \ge 0.35 \land Q_{\text{color}} \ge 0.25)$) to accommodate desk sliding, perspective foreshortening (30–40° tilt), and page flips while cleanly rejecting background drift. Low-confidence YOLO proposals ($conf \ge 0.12$) enable automatic reacquisition upon return to desk.

### Adaptive deadband and anti-windup PI control
The intervention bucket uses an anti-windup PI controller governed by an exponentially weighted moving variance (EMV) adaptive deadband. Inattention accumulates proportional-integral intervention debt, quantizing into escalating voice intervention tiers while preventing integrator windup and rapid threshold chattering.

## Project Structure

```text
.
├── main.py                 # Camera loop, HUD overlay, and system orchestration
├── dashboard.py            # Streamlit session analytics dashboard
├── setup.py                # One-time setup / model / GPU validation
├── config.yaml             # Runtime thresholds and tunable parameters
├── requirements.txt        # Pinned Python dependencies
├── core/
│   ├── vision.py           # MediaPipe processing, head pose (solvePnP), gaze
│   ├── detector.py         # YOLOv8 worker, detection cache, and put_fresh utility
│   ├── tracker.py          # Dynamic BookTracker (CSRT + NCC + Color Histogram + Reacquisition)
│   ├── attention.py        # Attention scoring and EMA filter
│   ├── fsm.py              # 8-state FSM, adaptive deadband, and anti-windup PI controller
│   └── intervention.py     # TTS engine and SQLite session logging
├── models/
│   ├── face_landmarker.task # Downloaded during setup
│   └── yolov8n.pt          # YOLOv8 nano weights
├── data/
│   └── sessions.db         # Created at runtime
└── tests/
    ├── test_attention.py   # Attention scoring & gaze weighting unit tests
    ├── test_fsm.py         # FSM transitions, deadband, & PI controller tests
    ├── test_intervention.py# Intervention bucket, quantizer, & logging tests
    ├── test_tracker.py     # CSRT tracking, NCC, color correlation, & absence tests
    └── test_vision.py      # Queue absence semantics & staleness timeout tests
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch

For the CUDA setup used by this project, install the matching PyTorch wheel **before** the rest of the dependencies.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run one-time setup

```bash
python setup.py
```

### 5. Start the monitor

```bash
python main.py
```

### 6. Start the dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

### 7. Run tests

```bash
pytest tests/ -v
```

## Controls

### Runtime Controls
| Key | Action |
|---|---|
| `q` / `ESC` | Quit application |
| `b` | Toggle `BREAK` state (pause interventions) |
| `c` | Re-enter calibration mode (re-define study ROI) |
| `d` | Toggle HUD debug overlay |

### Calibration Controls (Startup & 'c' Mode)
| Action | Description |
|---|---|
| `A` / `a` | Mark corner pin (C1, C2, C3, C4) at cursor position (precision touchpad friendly) |
| `SPACE` / `b` / `B` | Open native OpenCV interactive box selector (`cv2.selectROI`) |
| Left Mouse Drag | Drag rectangle or click corner-to-corner directly on frame |
| `r` / `R` | Reset marked calibration points and tracker |
| `s` / `S` | Skip calibration (screen study mode / no physical book) |
| `ENTER` | Confirm calibrated study ROI and begin session |

## Main Tuning Parameters

All runtime thresholds are centralized in `config.yaml`.

- `mediapipe.staleness_timeout_sec` — max duration without fresh face result before marking absent (`0.5s`)
- `book_tracker.q_thresh` / `color_thresh` / `high_conf_thresh` — appearance verification gates for study material
- `book_tracker.absence_timeout_sec` — sustained low-quality duration before book declared absent (`1.5s`)
- `fsm.thinking_entry_sec` — delay before entering `THINKING`
- `fsm.thinking_max_sec` — maximum thinking duration before `DISTRACTED` (`8.0s`)
- `fsm.base_focused_threshold` / `fsm.base_distracted_threshold` — baseline attention hysteresis band
- `intervention.Kp` / `Ki_up` / `Ki_down` — PI controller gains for the distraction intervention bucket
- `intervention.min_interval_sec` — minimum time between voice interventions
- `attention.weights` — relative contribution of gaze, head pose, context, and blink signals

## Technology

**Python · OpenCV · MediaPipe · YOLOv8 · PyTorch · Streamlit · Plotly · SQLite · pytest**

## Notes

- The current setup is optimized around a Windows desktop/laptop workflow and SAPI5-based TTS.
- CUDA availability depends on the installed NVIDIA driver and compatible PyTorch build.
- Model and threshold behaviour should be validated on the user's actual camera/environment before treating attention scores as meaningful measurements.
