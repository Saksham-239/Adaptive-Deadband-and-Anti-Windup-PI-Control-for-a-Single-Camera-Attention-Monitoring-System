# Face Focus — Real-Time Study Attention Monitor

A real-time computer-vision system that estimates study attention from a webcam and responds to sustained distraction with state-aware interventions.

It combines **MediaPipe face/pose landmarks**, **YOLOv8 object detection**, an **attention score with EMA smoothing**, an **8-state finite-state machine**, **text-to-speech interventions**, and **SQLite session logging**.

> The goal is not to classify every frame as “focused” or “distracted”. The system models attention as a **temporal state** and avoids nagging the user during brief, legitimate gaze shifts such as thinking or writing.

## What It Does

```text
Webcam
  │
  ├──► MediaPipe ──► gaze / head pose / blink features
  │
  └──► YOLOv8 ────► contextual objects (e.g. phone)
                       │
                       ▼
                Context Engine
                       │
                       ▼
                Attention Score
                       │
                       ▼
                EMA smoothing
                       │
                       ▼
                 8-State FSM
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Intervention logic      SQLite logger
             │                   │
             ▼                   ▼
            TTS              Session data
```

### States

| State | Meaning |
|---|---|
| `CALIBRATING` | Initial study-material / book ROI setup |
| `READING` | Attention is consistent with reading |
| `WRITING` | Looking down at the desk / notes |
| `THINKING` | Brief gaze-away; intervention suppressed |
| `DISTRACTED` | Sustained inattention |
| `PHONE` | Phone detected with attention directed toward it |
| `BREAK` | User-declared break |
| `UNKNOWN` | No reliable face/frame information |

## Key Engineering Decisions

### Temporal attention instead of frame-by-frame classification

A single gaze-away frame should not trigger a voice warning. The system uses an EMA-smoothed attention score and an FSM with explicit timing thresholds so transient behaviour can be distinguished from sustained distraction.

### Concurrent vision pipelines

MediaPipe and YOLO run asynchronously so the main camera/UI loop is not blocked by inference. Fresh frames are pushed through bounded queues and stale results are discarded.

### Context-aware interventions

The system combines gaze, head pose, blink behaviour, study-material context, and object detection rather than treating “looking away” as automatic distraction.

### Configurable control behaviour

Thresholds and weights live in `config.yaml`, including the attention hysteresis band, thinking-state timing, intervention cooldown, and component weights.

## Project Structure

```text
.
├── main.py                 # Camera loop and system orchestration
├── dashboard.py            # Streamlit session dashboard
├── setup.py                # One-time setup / model / GPU validation
├── config.yaml             # Runtime thresholds and tunable parameters
├── requirements.txt        # Pinned Python dependencies
├── core/
│   ├── vision.py           # MediaPipe processing, head pose, gaze
│   ├── detector.py         # YOLOv8 worker and detection cache
│   ├── attention.py        # Attention scoring and EMA
│   ├── fsm.py              # State machine and intervention logic
│   └── intervention.py     # TTS and SQLite session logging
├── models/
│   └── face_landmarker.task # Downloaded during setup
├── data/
│   └── sessions.db         # Created at runtime
└── tests/
    ├── test_attention.py
    ├── test_fsm.py
    └── test_intervention.py
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
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
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

| Key | Action |
|---|---|
| `q` / `ESC` | Quit |
| `b` | Toggle `BREAK` state |
| `c` | Re-enter calibration |
| `d` | Toggle debug overlay |
| `r` | Reset ROI during calibration |
| `ENTER` | Confirm the book ROI |

## Main Tuning Parameters

All runtime thresholds are centralized in `config.yaml`.

- `fsm.thinking_entry_sec` — delay before entering `THINKING`
- `fsm.thinking_max_sec` — maximum thinking duration before `DISTRACTED`
- `fsm.focused_threshold` / `fsm.distracted_threshold` — attention hysteresis band
- `intervention.min_interval_sec` — minimum time between voice interventions
- `attention.weights` — relative contribution of gaze, head pose, context, and blink signals

## Technology

**Python · OpenCV · MediaPipe · YOLOv8 · PyTorch · Streamlit · Plotly · SQLite · pytest**

## Notes

- The current setup is optimized around a Windows desktop/laptop workflow and SAPI5-based TTS.
- CUDA availability depends on the installed NVIDIA driver and compatible PyTorch build.
- Model and threshold behaviour should be validated on the user's actual camera/environment before treating attention scores as meaningful measurements.
