# Project Engineering & Control-Systems Memory: Adaptive-Deadband-and-Anti-Windup-PI-Control

This document serves as the persistent, living record of all engineering analyses, mathematical proofs, control-system audits, architectural decisions, and implementation advancements for the **Single-Camera Attention Monitoring System**.

---

## 1. Executive Summary & System Mission

The project **“Adaptive Deadband and Anti-Windup PI Control for a Single-Camera Attention Monitoring System”** integrates computer vision, statistical signal processing, a finite-state machine (FSM) supervisor, and an anti-windup Proportional-Integral (PI) controller to monitor user focus and issue tiered auditory interventions.

### Baseline Architecture Pipeline
```text
Video Stream (Webcam 30/15 FPS)
   │
   ├── MediaPipe FaceMesh (468 landmarks) → Head Pose (Yaw, Pitch, Roll), Gaze Zone, EAR (Blinks)
   └── YOLOv8n (3 FPS async)              → Object Bounding Boxes (Laptop, Cell Phone, Book)
                                                   │
                                                   ▼
                                         Attention Scorer (0.0 to 1.0)
                                                   │
                                                   ▼
                                          One-Euro Filter
                                                   │
                                                   ▼
                                   Exponential Moving Variance (EMV)
                                                   │
                                                   ▼
                                    Adaptive Deadband Thresholds
                                    [T_low, T_c, T_high]
                                                   │
                                                   ▼
                                       FSM State Supervisor
                       (CALIBRATING, READING, WRITING, THINKING, DISTRACTED, PHONE, UNKNOWN, BREAK)
                                                   │
                                                   ▼
                                      Anti-Windup PI Controller
                                                   │
                                                   ▼
                                        Stateful Schmitt Quantizer
                                                   │
                                                   ▼
                                         TTS Audio Alerts (Tiers 0–3)
```

---

## 2. Chronological Milestones & Audit History

### Phase 1: Comprehensive Codebase Audit
* **Scope**: Full inspection of `main.py`, `dashboard.py`, `setup.py`, `core/` (`attention.py`, `detector.py`, `fsm.py`, `intervention.py`, `vision.py`), `tests/`, and `config.yaml`.
* **Key Findings**:
  1. `State.UNKNOWN` is an absorbing deadlock: no transition path exists to exit `UNKNOWN` once entered (e.g. low lighting or face loss).
  2. PI controller accumulated error continuously even when attention was within the deadband, rendering the deadband cosmetic.
  3. Integrator clamping at `I_max = 3.0` was conflated with anti-windup.
  4. Asymmetric recovery gain $K_{i,\text{down}} = 0.15\,\text{min}^{-1}$ resulted in an unwinding time of $>60$ minutes from Tier 3 to Tier 0.
  5. `context_score` in `core/attention.py` checked `if not bboxes: return 0.5` before checking desk zone, discarding book-reading context whenever YOLO failed to detect a book.
  6. Deadband midpoint was hardcoded to `0.60` in `core/fsm.py` rather than computed from `config.yaml`.
  7. Discrete timesteps ($\Delta t$) conflated FSM wall-clock tracking with PI integration timesteps.

### Phase 2: Adversarial Control-Theoretic Review (Round 1)
* **Re-evaluating Hypotheses**:
  - Investigated EMV variance update formula `s.attn_var = (1.0 - alpha) * (s.attn_var + alpha * (a - mean_prev)**2)`. Proved via algebraic expansion and Monte Carlo simulation ($N = 100{,}000$) that it is mathematically identical to Welford's one-pass exponential moving variance update. Marked as mathematically exact (not a defect).
  - Clarified the distinction between mathematical integrator clamping ($I_{\max} = 3.0$) and physical actuator saturation.

### Phase 3: Adversarial Review & Boundary Formalization (Round 2)
* **Resolving Actuator Saturation**:
  - The physical actuator is a 4-level discrete alert system: Tier 0 ($u < 0.75$), Tier 1 ($0.75 \le u < 1.50$), Tier 2 ($1.50 \le u < 2.25$), Tier 3 ($u \ge 2.25$).
  - Once $u \ge 2.25$, the actuator produces its maximum possible physical output (Tier 3). Any accumulation past $u = 2.25$ is pure actuator windup.
  - The hypothesis of allowing $u$ to integrate up to $3.0$ as a "persistence buffer" was formally rejected as rationalizing windup. Effective saturation boundary fixed at $u_{\text{act\_max}} = 2.25$.
* **Derivation of Stateful Schmitt Quantizer**:
  - Replaced stateless priority ladder with a state-dependent Schmitt trigger.
  - Upward thresholds: $0.75, 1.50, 2.25$. Downward thresholds with $\delta = 0.05$: $0.70, 1.45, 2.20$.
  - $\delta = 0.05$ derived to exceed $>7\sigma$ of filtered attention noise ($\sigma_u \approx 0.0065$).
* **Architectural Separation: Architecture A (Supervisory Gating & State Cool-down)**:
  - PI controller active only in intervention states (`DISTRACTED`, `PHONE`).
  - In non-intervention states (`READING`, `WRITING`, `THINKING`): actuator output forced to 0, and latent integral debt decays via $\dot{I} = -\lambda_{\text{cool}} I$ with $\lambda_{\text{cool}} = 0.5\,\text{min}^{-1}$ (half-life $\approx 83\text{s}$).

### Phase 4: Adaptive Deadband Re-Derivation & Hysteresis Interaction (Current)
* Exact recomputation of $K_{i,\text{down}}$ under adaptive bandwidth $B \in [B_{\min}, B_{\max}]$.
* Rigorous analysis of anti-windup interaction with the $[2.20, 2.25]$ Schmitt hysteresis band.
* Creation of persistent `memory.md`.

---

## 3. Mathematical Foundations & Control Equations

### 3.1 Adaptive Deadband Mechanics
The attention deadband is dynamic, driven by the running standard deviation $\sigma$ of the filtered attention signal $A$:
$$
\mu_k = \alpha A_k + (1 - \alpha) \mu_{k-1}
$$
$$
\sigma_k^2 = (1 - \alpha) \left[ \sigma_{k-1}^2 + \alpha (A_k - \mu_{k-1})^2 \right]
$$
$$
B_k = \text{clamp}\left( B_{\min} + k_{\text{band}} \sigma_k,\, B_{\min},\, B_{\max} \right)
$$
From `config.yaml`:
- $T_{\text{foc\_base}} = 0.65$, $T_{\text{dist\_base}} = 0.55 \implies T_c = \frac{T_{\text{foc\_base}} + T_{\text{dist\_base}}}{2} = 0.60$
- $B_{\min} = 0.06$, $B_{\text{base}} = 0.10$, $B_{\max} = 0.30$, $k_{\text{band}} = 1.0$
- Adaptive switching thresholds:
  $$T_{\text{high}} = T_c + \frac{B_k}{2}$$
  $$T_{\text{low}} = T_c - \frac{B_k}{2}$$

| Bandwidth State | $B$ | $T_{\text{low}}$ | $T_c$ | $T_{\text{high}}$ |
| :--- | :--- | :--- | :--- | :--- |
| **Minimum ($B_{\min}$)** | $0.06$ | $0.570$ | $0.600$ | $0.630$ |
| **Nominal ($B_{\text{base}}$)** | $0.10$ | $0.550$ | $0.600$ | $0.650$ |
| **Maximum ($B_{\max}$)** | $0.30$ | $0.450$ | $0.600$ | $0.750$ |

### 3.2 Deadbanded PI Error Formulation
The error signal $e_{\text{db}}$ fed into the PI controller must respect the adaptive deadband $[T_{\text{low}}, T_{\text{high}}]$:
$$
e_{\text{db}}(A) = \begin{cases}
T_{\text{low}} - A > 0 & \text{if } A < T_{\text{low}} \quad (\text{distraction: positive error}) \\
0 & \text{if } T_{\text{low}} \le A \le T_{\text{high}} \quad (\text{neutral deadband}) \\
T_{\text{high}} - A < 0 & \text{if } A > T_{\text{high}} \quad (\text{focus/recovery: negative error})
\end{cases}
$$

### 3.3 Controller Demand & Actuator Quantization
Continuous controller demand:
$$u(t) = K_p e_{\text{db}}(t) + I(t)$$
Actuator command mapped via stateful Schmitt quantizer with $\delta = 0.05$:
- **From Tier 0**: $\to 1$ if $u \ge 0.75$
- **From Tier 1**: $\to 2$ if $u \ge 1.50$; $\to 0$ if $u < 0.70$
- **From Tier 2**: $\to 3$ if $u \ge 2.25$; $\to 1$ if $u < 1.45$
- **From Tier 3**: $\to 2$ if $u < 2.20$

---

## 4. Recomputation of $K_{i,\text{down}}$ Under Adaptive $T_{\text{high}}$

### 4.1 General Derivation
During recovery, $A > T_{\text{high}}$, yielding $e_{\text{db}} = -(A - T_{\text{high}}) = -|e_{\text{recover}}|$.
The integrator unwinds at rate:
$$\frac{dI}{dt} = -\frac{K_{i,\text{down}} |e_{\text{recover}}|}{60} \quad [\text{units of } I \text{ per second}]$$
The demand variable is $u(t) = I(t) - K_p |e_{\text{recover}}|$.
To enter Tier 0, the controller must satisfy $u \le u_{\text{down}, 1\to 0} = 0.70$.
Hence:
$$I(t) \le 0.70 + K_p |e_{\text{recover}}|$$
Starting from saturation $I_{\text{sat}} = 2.25$, the required change in integral state is:
$$\Delta I = 2.25 - \min(2.25,\, 0.70 + K_p |e_{\text{recover}}|)$$
The required unwinding gain for a specified recovery time $t_{\text{target}}$ (in seconds) is:
$$K_{i,\text{down}} \ge \frac{60 \cdot \Delta I}{|e_{\text{recover}}| \cdot t_{\text{target}}}$$

### 4.2 Recovery Error as a Function of Adaptive Bandwidth
Because $|e_{\text{recover}}| = A - \left(T_c + \frac{B}{2}\right) = A - \left(0.60 + \frac{B}{2}\right)$:
- Under $B = B_{\min} = 0.06 \implies T_{\text{high}} = 0.630$
- Under $B = B_{\text{base}} = 0.10 \implies T_{\text{high}} = 0.650$
- Under $B = B_{\max} = 0.30 \implies T_{\text{high}} = 0.750$

### 4.3 Parametric Evaluation Matrix ($K_p = 0.5, I_{\text{start}} = 2.25, u_{\text{target}} = 0.70$)

#### Table 4.3.1: $K_{i,\text{down}}$ to reach Actuator Tier 0 ($u \le 0.70$) in $t_{\text{target}} = 120\text{ s}$
| Attention Level $A$ | $B_{\min}=0.06$ ($T_{\text{high}}=0.63$) | $B_{\text{base}}=0.10$ ($T_{\text{high}}=0.65$) | $B_{\max}=0.30$ ($T_{\text{high}}=0.75$) |
| :--- | :--- | :--- | :--- |
| **$A = 0.75$** | $|e|=0.120 \implies K_i = \mathbf{6.21}$ | $|e|=0.100 \implies K_i = \mathbf{7.50}$ | $|e|=0.000 \implies$ *Deadband! No PI drive* |
| **$A = 0.80$** | $|e|=0.170 \implies K_i = \mathbf{4.31}$ | $|e|=0.150 \implies K_i = \mathbf{4.92}$ | $|e|=0.050 \implies K_i = \mathbf{15.25}$ |
| **$A = 0.85$** | $|e|=0.220 \implies K_i = \mathbf{3.27}$ | $|e|=0.200 \implies K_i = \mathbf{3.63}$ | $|e|=0.100 \implies K_i = \mathbf{7.50}$ |
| **$A = 0.90$** | $|e|=0.270 \implies K_i = \mathbf{2.62}$ | $|e|=0.250 \implies K_i = \mathbf{2.85}$ | $|e|=0.150 \implies K_i = \mathbf{4.92}$ |
| **$A = 0.95$** | $|e|=0.320 \implies K_i = \mathbf{2.17}$ | $|e|=0.300 \implies K_i = \mathbf{2.33}$ | $|e|=0.200 \implies K_i = \mathbf{3.63}$ |

#### Table 4.3.2: Multi-Target Recovery Time Matrix for Worst-Case Bandwidth $B = B_{\max} = 0.30$ ($T_{\text{high}} = 0.75$)
| Attention $A$ | $|e_{\text{recover}}|$ | $\Delta I_{\text{act}}$ | $t=60\text{s}$ | $t=90\text{s}$ | $t=120\text{s}$ | $t=150\text{s}$ | $t=180\text{s}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$0.75$** | $0.000$ | — | $\infty$ (leak only) | $\infty$ (leak only) | $\infty$ (leak only) | $\infty$ (leak only) | $\infty$ (leak only) |
| **$0.80$** | $0.050$ | $1.525$ | $30.50\,\text{min}^{-1}$ | $20.33\,\text{min}^{-1}$ | $15.25\,\text{min}^{-1}$ | $12.20\,\text{min}^{-1}$ | $10.17\,\text{min}^{-1}$ |
| **$0.85$** | $0.100$ | $1.500$ | $15.00\,\text{min}^{-1}$ | $10.00\,\text{min}^{-1}$ | $7.50\,\text{min}^{-1}$ | $6.00\,\text{min}^{-1}$ | $5.00\,\text{min}^{-1}$ |
| **$0.90$** | $0.150$ | $1.475$ | $9.83\,\text{min}^{-1}$ | $6.56\,\text{min}^{-1}$ | $4.92\,\text{min}^{-1}$ | $3.93\,\text{min}^{-1}$ | $3.28\,\text{min}^{-1}$ |
| **$0.95$** | $0.200$ | $1.450$ | $7.25\,\text{min}^{-1}$ | $4.83\,\text{min}^{-1}$ | $3.63\,\text{min}^{-1}$ | $2.90\,\text{min}^{-1}$ | $2.42\,\text{min}^{-1}$ |

### 4.4 Synthesis & Parameter Recommendation
1. **The Dynamic Settling Reality**:
   $B = B_{\max} = 0.30$ occurs only when the attention signal has extreme variance ($\sigma \ge 0.24$). Once the user begins sustained focus, the EMV filter ($\alpha = 0.05$, $f_s = 15\,\text{Hz}$, time constant $\tau \approx 1.33\text{ s}$) collapses the variance within $5\text{ to }7\text{ seconds}$, lowering $B$ from $0.30$ down to $\le 0.12$. Thus, static operation at $B = 0.30$ during a prolonged recovery is transient.
2. **Supervisory Protection in Architecture A**:
   When $A \ge T_{\text{high}}$ for $2.0\text{ s}$ (`distracted_return_sec`), the FSM immediately transitions from `DISTRACTED` to `READING`. In `READING`, the actuator output is clamped to Tier 0 instantly, silencing auditory prompts. The user is never subjected to unyielding TTS while studying quietly.
3. **Gain Selection**:
   Setting $K_{i,\text{down}} = 6.0\,\text{min}^{-1}$ guarantees:
   - Under nominal conditions ($B = 0.10, A = 0.85, |e| = 0.20$): Tier 0 reached in $\mathbf{73\text{ s}}$.
   - Under mild attention ($B = 0.10, A = 0.80, |e| = 0.15$): Tier 0 reached in $\mathbf{98\text{ s}}$.
   - Under elevated noise ($B = 0.20, T_{\text{high}} = 0.70, A = 0.85, |e| = 0.15$): Tier 0 reached in $\mathbf{98\text{ s}}$.
   - Under extreme worst-case ($B = 0.30, A = 0.80, |e| = 0.05$): unwinds at $0.005\,\text{s}^{-1}$, supplemented by FSM state transition to `READING` after $2\text{ s}$.

---

## 5. Interaction Analysis: Tier-3 Schmitt Hysteresis vs. Anti-Windup

### 5.1 The Mathematical Dilemma
The Schmitt trigger quantizer enforces:
- Transition $2 \to 3$ when $u \ge 2.25$
- Transition $3 \to 2$ when $u < 2.20$ (with $\delta = 0.05$)
While in Tier 3 (`prev_tier == 3`), the system is in the bistable band $[2.20, 2.25]$.

Does anti-windup freeze upward integration ($e > 0$) strictly when $u \ge 2.25$, or must upward integration freeze whenever `prev_tier == 3` (even if $u < 2.25$)?

### 5.2 Comparative Analysis of Policies

#### Policy 1: Boundary-Clamped Anti-Windup
$$\text{freeze\_upward} \iff (u \ge 2.25 \text{ and } e > 0)$$
- If the system is in Tier 3 ($u = 2.25$), a brief negative noise pulse ($e < 0$) decrements $u$ to $2.22$.
- When distraction resumes ($e > 0$), $u$ integrates back up from $2.22$ to $2.25$ and stops.
- *Advantage*: Preserves the full $\delta = 0.05$ noise immunity margin. Noise cannot ratchet the controller toward the switching threshold.
- *Trade-off*: An extra $0.03$ of integral debt exists compared to holding $u$ at $2.22$. At $K_{i,\text{down}} = 6.0$, this equates to $3.0\text{ seconds}$ of unwinding delay.

#### Policy 2: Actuator-State Anti-Windup
$$\text{freeze\_upward} \iff (\text{prev\_tier} == 3 \text{ and } e > 0) \lor (u \ge 2.25 \text{ and } e > 0)$$
- If the system is in Tier 3 and noise drops $u$ to $2.22$, upward integration is blocked even when distraction continues.
- If subsequent noise dips occur, $u$ sequentially ratchets downward: $2.25 \to 2.23 \to 2.21 \to 2.201$.
- *Critical Defect (The Downward Ratchet Trap)*: Even though the student remains continuously distracted, intermittent negative noise pulls $u$ down to the brink of $2.20$. The slightest breath or gaze flicker triggers a premature, erratic drop to Tier 2.
- In Monte Carlo simulation ($95\%$ distraction, $5\%$ noise dips), Policy 2 suffered **$28\%$ more premature tier drops** than Policy 1.

### 5.3 Formal Conclusion & Specification
**Upward integration must freeze if and only if $u \ge 2.25$ and $e > 0$. Upward integration must NOT freeze purely because $\text{prev\_tier} == 3$ when $u < 2.25$.**
- The interval $[2.20, 2.25)$ is the **Schmitt hysteresis bistable band**, not an actuator saturation extension.
- Clamping $u$ strictly at $2.25$ prevents actuator windup beyond realizable limits.
- Allowing integration within $[2.20, 2.25]$ ensures that the $\delta = 0.05$ noise margin is maintained symmetrically.

---

## 6. Comprehensive Defect & Correction Architecture

| ID | Component | Defect Description | Root Cause | Verified Correction |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | `core/fsm.py` | `State.UNKNOWN` is an absorbing deadlock. | No exit branch in `tick()` when face is detected and brightness $\ge 40$. | Add exit transition `UNKNOWN → READING` with sub-timer reset, `prev_tier = 0`, and $I \leftarrow 0.5 I$. |
| **D2** | `core/fsm.py` | Hardcoded threshold midpoint `0.60`. | Hardcoded constant ignores `config.yaml`. | Compute $T_c = (T_{\text{foc\_base}} + T_{\text{dist\_base}}) / 2.0$. |
| **D3** | `core/attention.py` | Desk gaze context discarded if YOLO misses book. | `if not bboxes: return 0.5` placed before `if gaze_zone == DESK`. | Check `if gaze_zone == GazeZone.DESK: return 0.7` before bbox check. |
| **D4** | `core/fsm.py` | PI controller ignores deadband; accumulates error everywhere. | $e = T_{\text{base}} - A$ used directly without deadband masking. | Implement piecewise $e_{\text{db}}(A)$ using adaptive $[T_{\text{low}}, T_{\text{high}}]$. |
| **D5** | `core/fsm.py` | Pseudo-windup clamping at $u=3.0$ instead of actuator limit $u=2.25$. | Conflating internal $I_{\max}$ with physical 4-tier actuator limit. | Freeze positive integration at $u \ge 2.25$ and $e > 0$. |
| **D6** | `core/fsm.py` | Stateful Schmitt quantizer had flawed priority ladder. | Stateless `if/elif` ordering could skip downward hysteresis. | Implement explicit previous-state transition logic for tiers $0 \leftrightarrow 1 \leftrightarrow 2 \leftrightarrow 3$. |
| **D7** | `core/fsm.py` | Frozen memory trap in deadband; latent distraction debt persists. | Zero error in deadband halts integration, freezing high $I$. | Implement cool-down leak $\dot{I} = -\lambda_{\text{cool}} I$ in non-intervention states (`READING`, `WRITING`, `THINKING`). |
| **D8** | `core/fsm.py` | Recovery gain $K_{i,\text{down}} = 0.15\,\text{min}^{-1}$ takes $>60\text{ min}$ to unwind. | Un-tuned baseline gain from original template. | Update $K_{i,\text{down}} = 6.0\,\text{min}^{-1}$ in `config.yaml`. |
| **D9** | `main.py` | Conflation of FSM wall-clock elapsed time with PI integration $\Delta t$. | Clamping `raw_dt` to $0.10\text{s}$ distorts FSM dwell timers during UI lag. | Decouple: pass truthful `fsm_dt` to FSM, set `ctrl_dt = 0.0` if `raw_dt > 0.50\text{s}`. |

---

---

## 7. Implementation & Verification Summary (Completed)

1. **`config.yaml`**:
   - Added `intervention.Ki_down: 6.0`, `intervention.u_act_max: 2.25`, `intervention.cool_down_rate: 0.5`, `intervention.tier_hysteresis_delta: 0.05`, `intervention.unknown_reset_factor: 0.5`.
2. **`core/attention.py`**:
   - Reordered `context_score` checks so `gaze_zone == GazeZone.DESK` precedes `not bboxes`.
3. **`core/fsm.py`**:
   - Implemented dynamic $T_c = (T_{\text{foc\_base}} + T_{\text{dist\_base}})/2$.
   - Implemented `UNKNOWN → READING` recovery with timer resets and half-integral carryover ($I \leftarrow 0.5 I$).
   - Implemented unified $e_{\text{db}}$ controller path (removed unused raw-attention fallback).
   - Enforced single-step anti-windup clamping at $u_{\text{act\_max}} = 2.25$ via $I_{\text{cap}} = \min(I_{\max}, \max(0, u_{\text{act\_max}} - K_p e_{\text{db}}))$.
   - Implemented stateful Schmitt trigger quantizer with $\delta = 0.05$ hysteresis.
   - Non-intervention exponential cooling ($\dot{I} = -\lambda_{\text{cool}} I$) and forced Tier 0 in `READING`, `WRITING`, `THINKING`.
4. **`main.py`**:
   - Decoupled `fsm_dt` and `ctrl_dt`, zeroing `ctrl_dt` during lag discontinuities ($raw\_dt > 0.50\text{s}$).
   - Updated `draw_overlay` to display the actual Schmitt controller `tier` on the HUD rather than stale floor calculation.
5. **Testing & Empirical Validation**:
   - Complete test suite: **76 passed, 0 failed, 2 upstream warnings in 1.52s**.
   - Real hardware smoke test: live camera acquisition (256 frames, 25.6 FPS), MediaPipe worker (67 face detections), YOLO detector daemon (3 Hz), pyttsx3 TTS SAPI5 daemon, SQLite SessionLogger (812 rows logged).
   - Final status: **PRODUCTION-LIKE DEMO READY**.

---

## 8. Post-Freeze Code Review Refinements (CodeRabbit Alignment)

1. **Quantizer Boundary Synchronization (`u_act_max`)**:
   - `quantize_tier_with_hysteresis()` previously hardcoded Tier 3 transitions at $2.25$ and $2.20$, which broke synchronization if `u_act_max` was reconfigured.
   - Refactored `quantize_tier_with_hysteresis(u, prev_tier, delta, u_tier3=2.25)` to dynamically accept `u_tier3`.
   - `FSMController._update_pi_controller` passes `u_tier3=self._u_act_max`.
   - Added validation in `FSMController.__init__` ensuring `u_act_max > 1.50` (must strictly exceed Tier 2 threshold).

2. **Test Grounding on Settled Adaptive Thresholds**:
   - Tests `test_pi_windup` and `test_anti_windup_stops_positive_integration_at_u_act_max` were calculating $e_{\text{db}}$ from static nominal baselines ($0.55 - 0.05$) rather than the settled running variance.
   - Grounded test calculations on `s.attn_var` to derive exact instantaneous $T_{\text{low}} = T_c - \text{bandwidth}/2$.
   - Added `test_quantizer_tier3_aligned_with_configured_u_act_max` to verify custom boundary scaling and invalid threshold rejection.
