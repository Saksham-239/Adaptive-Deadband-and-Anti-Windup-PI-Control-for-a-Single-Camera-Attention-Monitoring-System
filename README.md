# Face Focus — Adaptive AI Study Attention System

Real-time study attention monitor using webcam, MediaPipe, and YOLOv8.

## Quick Start

### 1. Install PyTorch CUDA first (critical — must be before requirements.txt)
```bash
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 2. Install remaining dependencies
```bash
.venv\Scripts\pip install -r requirements.txt
```

### 3. One-time setup (downloads MediaPipe model, validates GPU)
```bash
.venv\Scripts\python setup.py
```

### 4. Run the system
```bash
.venv\Scripts\python main.py
```

### 5. Run the dashboard (in a separate terminal)
```bash
.venv\Scripts\streamlit run dashboard.py
```

### 6. Run tests
```bash
.venv\Scripts\pytest tests/ -v
```

---

## Controls
| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `b` | Toggle BREAK state (pause interventions) |
| `c` | Re-enter calibration (re-draw book ROI) |
| `d` | Toggle debug overlay |
| `r` | Reset ROI during calibration |
| `ENTER` | Confirm book ROI during calibration |

## Architecture
```
cv2.VideoCapture (main thread)
      │
      ├──► face_q  ──► MediaPipe thread (15Hz) ──► result_q ──► main
      └──► frame_q ──► YOLO thread (3Hz) ──► detection_cache ──► main
                             ↓
                      Context Engine → Attention Scorer → EMA
                             ↓
                      FSM (8 states) → Intervention Bucket
                             ↓
                      TTS (daemon thread) + SQLite Logger
```

## Project Structure
```
face focus/
├── main.py           # Orchestration loop
├── dashboard.py      # Streamlit analytics dashboard
├── setup.py          # One-time setup & GPU validation
├── config.yaml       # All tunables (single source of truth)
├── requirements.txt  # Pinned dependencies
├── core/
│   ├── vision.py     # MediaPipe + head pose + gaze zone
│   ├── detector.py   # YOLOv8n thread + detection cache
│   ├── attention.py  # Attention scorer + EMA (pure functions)
│   ├── fsm.py        # 8-state FSM + intervention bucket (pure functions)
│   └── intervention.py  # TTS thread + SQLite logger
├── models/
│   └── face_landmarker.task  # Downloaded by setup.py
├── data/
│   └── sessions.db   # Created at runtime
└── tests/
    ├── test_attention.py
    ├── test_fsm.py
    └── test_intervention.py
```

## States
| State | Description |
|-------|-------------|
| `CALIBRATING` | Draw book ROI (startup) |
| `READING` | Focused on study material |
| `WRITING` | Looking down at desk (notes) |
| `THINKING` | Brief gaze-away (≤8s) — no intervention |
| `DISTRACTED` | Sustained inattention |
| `PHONE` | Phone detected + gaze toward it (highest priority) |
| `BREAK` | User-declared pause |
| `UNKNOWN` | No face / dark frame — interventions suspended |

## Tuning
All thresholds are in `config.yaml`. Key values to tune after first run:
- `fsm.thinking_entry_sec` — how long to tolerate gaze-away before THINKING
- `fsm.thinking_max_sec` — how long THINKING before DISTRACTED
- `fsm.focused_threshold` / `fsm.distracted_threshold` — hysteresis band
- `intervention.min_interval_sec` — minimum gap between voice warnings
- `attention.weights` — relative importance of gaze / head / context / blink
