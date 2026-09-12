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
Anchored by initial user calibration, `core/tracker.py` maintains tracking independently of repeated YOLO classifications. It combines sub-millisecond OpenCV CSRT visual tracking with a dual appearance gate (`Q_ncc >= 0.50` or `Q_ncc >= 0.35` and `Q_color >= 0.25`) to accommodate desk sliding, perspective foreshortening (30–40° tilt), and page flips while cleanly rejecting background drift. Low-confidence YOLO proposals (`conf >= 0.12`) enable automatic reacquisition upon return to desk.

### Adaptive deadband and anti-windup PI control
The intervention bucket uses an anti-windup PI controller governed by an exponentially weighted moving variance (EMV) adaptive deadband. Inattention accumulates proportional-integral intervention debt, quantizing into escalating voice intervention tiers while preventing integrator windup and rapid threshold chattering.

## Control System Loop & Dynamics

The core control-theoretic architecture regulates human-in-the-loop attention using closed-loop feedback. Rather than relying on static heuristic rules, the system models attention regulation as a continuous feedback loop featuring an adaptive deadband, an anti-windup PI controller, and a stateful Schmitt quantizer.

```text
               Target Focus Reference (T_c)
                             │
                             ▼
Human Plant ──► [ Perception & Estimator ] ──( A_f )──► [ Adaptive Deadband (EMV) ]
(Gaze, Head,     (MediaPipe, solvePnP,                   (Rolling Variance σ_attn)
 Desk, Blink)     YOLO, BookTracker)                                 │
                                                                     ▼  e_db
                                                         [ Anti-Windup PI Controller ]
                                                         (Kp, Ki_up, Ki_down, Cool-down)
                                                                     │
                                                                     ▼  u(t)
                                                      [ Stateful Schmitt Quantizer ]
                                                      (4 Tiers with Hysteresis δ)
                                                                     │
                                                                     ▼  Tier {0, 1, 2, 3}
Auditory Feedback ◄─────────────────────────────────── [ Voice TTS Actuator ]
```

### 1. Multi-Modal Signal Estimation
The raw attention score `A(t)` in `[0, 1]` fuses four sensory signals:

```text
A(t) = w_gaze * S_gaze + w_head * S_head + w_ctx * S_ctx + w_blink * S_blink
```

`A(t)` is filtered using a 1-Euro observer filter (suppressing micro-jitter without adding latency) and an exponential moving average (EMA) to produce the continuous state estimate `A_f(t)`.

### 2. Adaptive Deadband (EMV Variance Scaling)
To eliminate control chattering when the user's attention signal hovers near threshold boundaries, an Exponential Moving Variance (EMV) filter dynamically tracks signal noise:

```text
Rolling Mean:     μ(t)  = α_emv * A_f(t) + (1 - α_emv) * μ(t-1)
Rolling Variance: σ²(t) = (1 - α_emv) * [σ²(t-1) + α_emv * (A_f(t) - μ(t-1))²]
```

The hysteresis bandwidth `B(t)` scales dynamically with standard deviation `σ(t)`:

```text
B(t) = clamp(B_min + k_band * σ(t), B_min, B_max)
T_focused    = T_c + B(t) / 2
T_distracted = T_c - B(t) / 2
```

The deadband error `e_db(t)` feeds directly into the controller:

```text
e_db = T_distracted - A_f    (if A_f < T_distracted: positive distraction debt)
e_db = T_focused - A_f       (if A_f > T_focused: negative recovery credit)
e_db = 0.0                   (inside neutral deadband gap)
```

### 3. Anti-Windup PI Controller
Intervention demand `u(t)` is governed by a proportional-integral control law with asymmetric accumulation and recovery gains:

```text
u(t) = Kp * e_db(t) + I(t)
```

* **Distraction Accumulation (`e_db > 0`):** Integrates at rate `Ki_up`. If demand reaches the maximum actuator saturation boundary (`u >= u_act_max = 2.25`), positive integration is immediately clamped (conditional anti-windup).
* **Focus Recovery (`e_db < 0`):** Discharges intervention debt at an accelerated rate `Ki_down` (`Ki_down > Ki_up`), rewarding rapid refocusing without residual lag.
* **Neutral Cool-Down (`e_db == 0`):** Decays accumulated debt exponentially: `I(t) = max(0, I(t) - λ_cool * I(t) * Δt)`.
* **Discontinuity Protection:** Integrator updates are frozen (`Δt_ctrl = 0`) if an OS scheduling jitter or frame lag spike exceeds 0.5s, preventing artificial numerical accumulation.

### 4. Stateful Schmitt Quantizer (Actuator Interface)
Continuous intervention demand `u(t)` is mapped into discrete voice intervention tiers `{0, 1, 2, 3}` using a stateful 4-tier Schmitt trigger with downward hysteresis `δ = 0.05`:
* **Tier 0 (Silent):** `u < 0.75` (nominal focused state)
* **Tier 1 (Gentle Reminder):** `u >= 0.75` (downward exit at `0.70`)
* **Tier 2 (Firm Prompt):** `u >= 1.50` (downward exit at `1.45`)
* **Tier 3 (Urgent Reset):** `u >= 2.25` (downward exit at `2.20`, aligned with `u_act_max`)

This hysteresis prevents auditory oscillation and prompt spamming when attention fluctuates near a tier boundary.

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

### 6. Start the Web Analytics Dashboard

Face Focus includes a browser-based analytics application ([`dashboard.py`](dashboard.py)) built with **Streamlit** and **Plotly Dark**. It connects directly to the local SQLite database (`data/sessions.db`) to visualize study telemetry both live and post-hoc.

#### Launching the Dashboard

Run this command in a separate terminal (or anytime to review past study sessions):

```bash
# If using the project's virtual environment (recommended):
.venv\Scripts\python -m streamlit run dashboard.py

# Or if your virtual environment is already activated:
streamlit run dashboard.py
```

The dashboard will automatically open in your web browser at:
> **`http://localhost:8501`**

#### Dashboard Features & Usage

* **Live & Historical Operation Modes**:
  - **Live Monitoring**: Run concurrently alongside `python main.py`. With the sidebar **Auto-refresh** toggle enabled (polled every `dashboard.refresh_interval_sec` in `config.yaml`, default `2.0s`), graphs update in real-time as you study.
  - **Historical Review**: Run standalone without activating the camera to analyze, compare, and audit past recorded sessions using the sidebar **Select session** dropdown.
* **Executive Summary Cards**: Displays total session duration (minutes), mean attention score (`0.0–1.0`), focus ratio (`% time in READING / WRITING`), distraction ratio (`% time in DISTRACTED / PHONE`), and count of TTS voice interventions dispatched.
* **Attention Score Over Time**: Interactive Plotly graph plotting both Raw fused attention and Filtered (1-Euro + EMA) attention, with shaded red vertical regions highlighting contiguous distraction episodes.
* **Time in State (Donut Chart)**: Percentage breakdown of study time spent across FSM states (`READING`, `WRITING`, `THINKING`, `DISTRACTED`, `PHONE`, `BREAK`, `UNKNOWN`, `CALIBRATING`).
* **Component Score Trajectories**: Multi-line tracking of the four underlying sensory components: Gaze (`S_gaze`), Head Pose (`S_head`), Context/YOLO (`S_ctx`), and Blink/EAR (`S_blink`).
* **Gaze Zone Distribution**: Bar chart comparing fixation counts across detected zones (`screen`, `desk`, `book`, and `away`).
* **Intervention Event Scatter**: Timestamped timeline of auditory intervention alerts showing Tier 1 (Soft), Tier 2 (Firm), and Tier 3 (Urgent) triggers.
* **Raw Data Inspector**: Collapsible table viewer allowing you to inspect the latest raw tabular rows stored in SQLite.

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
