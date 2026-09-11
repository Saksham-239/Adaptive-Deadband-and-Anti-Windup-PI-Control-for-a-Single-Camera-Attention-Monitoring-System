"""
core/fsm.py
Finite State Machine for attention monitoring — PURE FUNCTIONS.

States: CALIBRATING, READING, WRITING, THINKING, DISTRACTED, PHONE, BREAK, UNKNOWN
All transition logic is in the FSMController.tick() method.
tick() is a pure function given the current state + inputs → returns new state.
This makes the FSM unit-testable with synthetic inputs.

Hysteresis:
  Downward transitions (READING → THINKING): A_f < distracted_threshold
  Upward transitions (→ READING):            A_f >= focused_threshold
  Gap = focused_threshold - distracted_threshold = 0.10 (enforced by config construction)

Supervisory PI Controller & Intervention:
  Active regulation occurs in DISTRACTED and PHONE states using adaptive deadband error e_db.
  Conditional anti-windup halts integration at physical actuator saturation u_act_max = 2.25.
  Stateful Schmitt quantizer with hysteresis maps u = Kp*e_db + I to tiers {0, 1, 2, 3}.
  Non-intervention states (READING, WRITING, THINKING) force tier 0 and cool down the integrator.
  BREAK and CALIBRATING reset the integrator; UNKNOWN freezes it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(str, Enum):
    CALIBRATING = "CALIBRATING"
    READING     = "READING"
    WRITING     = "WRITING"
    THINKING    = "THINKING"
    DISTRACTED  = "DISTRACTED"
    PHONE       = "PHONE"
    BREAK       = "BREAK"
    UNKNOWN     = "UNKNOWN"


# ---------------------------------------------------------------------------
# Input snapshot passed to tick()
# ---------------------------------------------------------------------------

@dataclass
class FSMInput:
    """Snapshot of all sensor-derived values for one FSM tick."""
    attention_filtered: float       # EMA-smoothed attention score [0, 1]
    gaze_zone: str                  # from GazeZone constants
    head_pitch_deg: float           # positive = looking down
    phone_consecutive: int          # consecutive frames with phone detected
    phone_absent_frames: int        # consecutive frames with phone absent
    mean_brightness: float          # for UNKNOWN detection
    face_detected: bool             # True if MediaPipe found a face this tick
    break_key_pressed: bool         # True if user pressed 'b'
    dt: float                       # seconds since last tick (FSM elapsed-time dt)
    ctrl_dt: Optional[float] = None # Controller integration dt; if None, derived from dt


# ---------------------------------------------------------------------------
# FSM state
# ---------------------------------------------------------------------------

@dataclass
class FSMState:
    """
    Mutable FSM state — all timing accumulators and counters.
    Pass this as a reference into tick(); do NOT create a new one each frame.
    """
    current: State = State.CALIBRATING

    # Time spent in current state continuously
    state_duration: float = 0.0

    # Sub-state timers (seconds of consecutive condition being true)
    time_below_distracted_thresh: float = 0.0  # for READING → THINKING trigger
    time_above_focused_thresh: float    = 0.0  # for → READING trigger

    # Integrator state I(t) [0.0, u_act_max] (I_max is numerical safety ceiling)
    intervention_bucket: float = 0.0

    # For EMV (Exponential Moving Variance)
    attn_mean: float = 0.5
    attn_var: float = 0.0

    # Writing sub-state timer
    writing_timer: float = 0.0

    # Break toggle (True = currently in BREAK)
    in_break: bool = False

    # Previous break_key state to detect edges
    prev_break_key: bool = False

    # Previous intervention tier for stateful Schmitt quantizer
    prev_tier: int = 0


# ---------------------------------------------------------------------------
# Stateful Schmitt Quantizer
# ---------------------------------------------------------------------------

def quantize_tier_with_hysteresis(u: float, prev_tier: int, delta: float = 0.05) -> int:
    """
    Stateful 4-tier Schmitt quantizer with hysteresis.

    Nominal upward thresholds:
        Tier 0 -> 1: 0.75
        Tier 1 -> 2: 1.50
        Tier 2 -> 3: 2.25

    Downward thresholds (nominal - delta):
        Tier 1 -> 0: 0.75 - delta (0.70)
        Tier 2 -> 1: 1.50 - delta (1.45)
        Tier 3 -> 2: 2.25 - delta (2.20)
    """
    if prev_tier == 0:
        if u >= 2.25:
            return 3
        elif u >= 1.50:
            return 2
        elif u >= 0.75:
            return 1
        else:
            return 0
    elif prev_tier == 1:
        if u >= 2.25:
            return 3
        elif u >= 1.50:
            return 2
        elif u < (0.75 - delta):
            return 0
        else:
            return 1
    elif prev_tier == 2:
        if u >= 2.25:
            return 3
        elif u < (1.50 - delta):
            if u < (0.75 - delta):
                return 0
            return 1
        else:
            return 2
    elif prev_tier == 3:
        if u < (2.25 - delta):
            if u < (0.75 - delta):
                return 0
            elif u < (1.50 - delta):
                return 1
            else:
                return 2
        else:
            return 3
    else:
        if u >= 2.25:
            return 3
        elif u >= 1.50:
            return 2
        elif u >= 0.75:
            return 1
        else:
            return 0


# ---------------------------------------------------------------------------
# FSMController
# ---------------------------------------------------------------------------

class FSMController:
    """
    Manages FSM state transitions and the intervention bucket.

    Usage:
        fsm = FSMController(cfg)
        fsm_state = FSMState()
        new_state, bucket_tier = fsm.tick(fsm_state, inputs)

    tick() mutates fsm_state in place and returns (new_state, tier).
    tier = int in {0, 1, 2, 3} indicating intervention urgency.
    """

    def __init__(self, cfg: dict):
        f = cfg.get("fsm", {})
        self._thinking_entry    = f["thinking_entry_sec"]
        self._thinking_max      = f["thinking_max_sec"]
        self._writing_entry     = f["writing_entry_sec"]
        self._writing_pitch     = f["writing_pitch_deg"]
        self._return_reading    = f["return_to_reading_sec"]
        self._distracted_return = f["distracted_return_sec"]
        self._base_focused_thresh = f["base_focused_threshold"]
        self._base_distracted_thresh = f["base_distracted_threshold"]
        self._brightness_floor  = f["brightness_floor"]
        
        self._emv_alpha         = f["emv_alpha"]
        self._min_bandwidth     = f["min_bandwidth"]
        self._max_bandwidth     = f["max_bandwidth"]
        self._k_band            = f["k_band"]

        iv = cfg.get("intervention", {})
        self._Kp                = iv.get("Kp", 0.5)
        self._Ki_up             = iv.get("Ki_up", 1.2)
        self._Ki_down           = iv.get("Ki_down", 6.0)
        self._cool_down_rate    = iv.get("cool_down_rate", 0.5)
        self._u_act_max         = iv.get("u_act_max", 2.25)
        self._tier_hysteresis_delta = iv.get("tier_hysteresis_delta", 0.05)
        self._unknown_reset_factor = iv.get("unknown_reset_factor", 0.5)
        self._I_max             = iv.get("I_max", 3.0)

        yolo = cfg.get("yolo", {})
        self._phone_n           = yolo.get("phone_consecutive_frames", 5)
        self._phone_exit_n      = yolo.get("phone_exit_frames", 10)

    def tick(self, s: FSMState, inp: FSMInput) -> tuple[State, int]:
        """
        Advance the FSM by one tick.
        Mutates s in place.
        Returns (new_state, intervention_tier).
        """
        fsm_dt = inp.dt
        if inp.ctrl_dt is not None:
            ctrl_dt = inp.ctrl_dt
        else:
            ctrl_dt = 0.0 if fsm_dt > 0.50 else max(0.001, min(fsm_dt, 0.10))

        # --- BREAK toggle (edge-triggered) ---
        break_edge = inp.break_key_pressed and not s.prev_break_key
        s.prev_break_key = inp.break_key_pressed

        if break_edge:
            if s.current == State.BREAK:
                # Exit BREAK → READING
                s.current        = State.READING
                s.state_duration = 0.0
                s.intervention_bucket = 0.0
                s.prev_tier      = 0
                logger.info("FSM: BREAK → READING (user resumed)")
            else:
                # Enter BREAK
                s.current        = State.BREAK
                s.state_duration = 0.0
                s.intervention_bucket = 0.0
                s.prev_tier      = 0
                logger.info("FSM: %s → BREAK (user declared)", s.current.value)
            s.prev_break_key = inp.break_key_pressed
            return s.current, 0

        if s.current == State.BREAK:
            s.state_duration += fsm_dt
            s.intervention_bucket = 0.0
            s.prev_tier = 0
            return State.BREAK, 0

        # --- CALIBRATING ---
        if s.current == State.CALIBRATING:
            s.state_duration += fsm_dt
            s.intervention_bucket = 0.0
            s.prev_tier = 0
            return State.CALIBRATING, 0

        # --- Exit UNKNOWN → READING if face restored and brightness adequate ---
        if s.current == State.UNKNOWN:
            if inp.face_detected and inp.mean_brightness >= self._brightness_floor:
                logger.info("FSM: UNKNOWN → READING (recovered face=%s, brightness=%.0f)",
                            inp.face_detected, inp.mean_brightness)
                s.current = State.READING
                s.state_duration = 0.0
                s.time_below_distracted_thresh = 0.0
                s.time_above_focused_thresh = 0.0
                s.writing_timer = 0.0
                s.prev_tier = 0
                s.intervention_bucket *= self._unknown_reset_factor
                return State.READING, 0

        # --- UNKNOWN: low brightness or no face ---
        if not inp.face_detected or inp.mean_brightness < self._brightness_floor:
            if s.current != State.UNKNOWN:
                logger.debug("FSM: → UNKNOWN (face=%s, brightness=%.0f)",
                             inp.face_detected, inp.mean_brightness)
            s.current        = State.UNKNOWN
            s.state_duration += fsm_dt
            s.prev_tier      = 0
            # Bucket does NOT accumulate in UNKNOWN; freeze integrator
            return State.UNKNOWN, 0

        a = inp.attention_filtered

        # --- Update EMV and calculate adaptive thresholds ---
        # EMV tracks rolling standard deviation to scale hysteresis bandwidth.
        # This prevents thrashing under changing noise floors (e.g. user fidgeting).
        mean_prev = s.attn_mean
        s.attn_mean = self._emv_alpha * a + (1.0 - self._emv_alpha) * mean_prev
        s.attn_var = (1.0 - self._emv_alpha) * (s.attn_var + self._emv_alpha * (a - mean_prev)**2)
        std_dev = s.attn_var ** 0.5

        # Adaptive Deadband (Schmitt trigger logic)
        # Bandwidth scales with noise. We clamp to [min_bandwidth, max_bandwidth].
        bandwidth = self._min_bandwidth + self._k_band * std_dev
        bandwidth = max(self._min_bandwidth, min(bandwidth, self._max_bandwidth))

        # Shift thresholds symmetrically around config-derived Tc
        Tc = (self._base_focused_thresh + self._base_distracted_thresh) / 2.0
        focused_thresh = Tc + (bandwidth / 2.0)
        distracted_thresh = Tc - (bandwidth / 2.0)

        # Deadbanded error calculation:
        # e_db = T_low - A,   if A < T_low  (distraction: positive error)
        # e_db = T_high - A,  if A > T_high (recovery: negative error)
        # e_db = 0,           otherwise     (neutral deadband)
        if a < distracted_thresh:
            e_db = distracted_thresh - a
            s.time_below_distracted_thresh += fsm_dt
            s.time_above_focused_thresh     = 0.0
        elif a > focused_thresh:
            e_db = focused_thresh - a
            s.time_above_focused_thresh     += fsm_dt
            s.time_below_distracted_thresh   = 0.0
        else:
            e_db = 0.0
            # In the hysteresis gap — don't accumulate either dwell timer

        # --- PHONE: preempts all other states ---
        if inp.phone_consecutive >= self._phone_n:
            if s.current != State.PHONE:
                logger.info("FSM: %s → PHONE", s.current.value)
                s.state_duration = 0.0
            s.current = State.PHONE
            s.state_duration += fsm_dt
            return State.PHONE, self._update_pi_controller(s, e_db, ctrl_dt)

        # --- Exit PHONE if phone absent long enough ---
        if s.current == State.PHONE:
            if inp.phone_absent_frames >= self._phone_exit_n:
                logger.info("FSM: PHONE → DISTRACTED (phone removed)")
                s.current        = State.DISTRACTED
                s.state_duration = 0.0

        # --- State transitions ---
        if s.current == State.READING:
            # READING → WRITING
            if (inp.gaze_zone == "desk_zone" and
                    inp.head_pitch_deg > self._writing_pitch):
                s.time_below_distracted_thresh = 0.0  # reset distraction timer
                s.writing_timer += fsm_dt
                if s.writing_timer >= self._writing_entry:
                    s.current = State.WRITING
                    s.state_duration = 0.0
                    s.writing_timer = 0.0
                    logger.info("FSM: READING → WRITING")
            else:
                s.writing_timer = 0.0

            # READING → THINKING
            if (s.time_below_distracted_thresh >= self._thinking_entry and
                    s.current == State.READING):
                s.current = State.THINKING
                s.state_duration = 0.0
                s.time_below_distracted_thresh = 0.0
                logger.info("FSM: READING → THINKING")

        elif s.current == State.WRITING:
            # WRITING → READING if gaze/pitch return to normal
            if not (inp.gaze_zone == "desk_zone" and
                    inp.head_pitch_deg > self._writing_pitch):
                if s.time_above_focused_thresh >= self._return_reading:
                    s.current = State.READING
                    s.state_duration = 0.0
                    logger.info("FSM: WRITING → READING")
            # WRITING → THINKING if attention drops
            if s.time_below_distracted_thresh >= self._thinking_entry:
                s.current = State.THINKING
                s.state_duration = 0.0
                s.time_below_distracted_thresh = 0.0
                logger.info("FSM: WRITING → THINKING")

        elif s.current == State.THINKING:
            # THINKING → READING
            if s.time_above_focused_thresh >= self._return_reading:
                s.current = State.READING
                s.state_duration = 0.0
                s.time_above_focused_thresh = 0.0
                logger.info("FSM: THINKING → READING")
            # THINKING → DISTRACTED (timeout)
            elif s.state_duration >= self._thinking_max:
                s.current = State.DISTRACTED
                s.state_duration = 0.0
                logger.info("FSM: THINKING → DISTRACTED (timeout %.1fs)", self._thinking_max)

        elif s.current == State.DISTRACTED:
            # DISTRACTED → READING
            if s.time_above_focused_thresh >= self._distracted_return:
                s.current = State.READING
                s.state_duration = 0.0
                s.time_above_focused_thresh = 0.0
                logger.info("FSM: DISTRACTED → READING")

        s.state_duration += fsm_dt
        return s.current, self._update_pi_controller(s, e_db, ctrl_dt)

    def _update_pi_controller(self, s: FSMState, e_db: float = 0.0, dt: float = 0.0) -> int:
        """
        Human-in-the-loop supervisory PI controller mapping deadbanded error to intervention tier.
        
        Active PI regulation occurs strictly in DISTRACTED and PHONE states.
        In READING, WRITING, and THINKING: actuator tier is forced to 0 and the integrator cools down.
        In BREAK and CALIBRATING: integrator is reset to 0.
        In UNKNOWN: integrator is frozen and tier 0 is returned.
        """
        if s.current in (State.UNKNOWN, State.BREAK, State.CALIBRATING):
            s.prev_tier = 0
            if s.current in (State.BREAK, State.CALIBRATING):
                s.intervention_bucket = 0.0
            return 0

        dt_min = dt / 60.0

        if s.current in (State.READING, State.WRITING, State.THINKING):
            # Non-intervention states: force actuator tier = 0 and exponentially cool the integrator
            if s.intervention_bucket > 0.0 and dt_min > 0.0:
                s.intervention_bucket = max(0.0, s.intervention_bucket - self._cool_down_rate * s.intervention_bucket * dt_min)
            s.prev_tier = 0
            return 0

        # Active PI control in DISTRACTED or PHONE
        u_raw = self._Kp * e_db + s.intervention_bucket

        # Conditional anti-windup (effective saturation boundary u_act_max = 2.25)
        if e_db > 0:
            if u_raw >= self._u_act_max:
                # Actuator saturated at Tier 3: stop positive integration
                pass
            else:
                I_new = s.intervention_bucket + self._Ki_up * e_db * dt_min
                # Cap integration so positive integration cannot drive next controller demand beyond u_act_max
                I_cap = min(self._I_max, max(0.0, self._u_act_max - self._Kp * e_db))
                s.intervention_bucket = min(I_new, I_cap)
        elif e_db < 0:
            if u_raw <= 0.0:
                # Actuator saturated at Tier 0: stop negative integration
                s.intervention_bucket = max(0.0, s.intervention_bucket)
            else:
                I_new = s.intervention_bucket + self._Ki_down * e_db * dt_min
                s.intervention_bucket = max(0.0, I_new)
        else:
            # e_db == 0 (neutral deadband inside DISTRACTED/PHONE):
            # Apply cool-down decay rather than freezing accumulated integral indefinitely
            if s.intervention_bucket > 0.0 and dt_min > 0.0:
                s.intervention_bucket = max(0.0, s.intervention_bucket - self._cool_down_rate * s.intervention_bucket * dt_min)

        u = self._Kp * e_db + s.intervention_bucket
        tier = quantize_tier_with_hysteresis(u, s.prev_tier, self._tier_hysteresis_delta)
        s.prev_tier = tier
        return tier

    def calibration_complete(self, s: FSMState) -> None:
        """Call when the user confirms the book ROI to leave CALIBRATING."""
        if s.current == State.CALIBRATING:
            s.current = State.READING
            s.state_duration = 0.0
            s.intervention_bucket = 0.0
            s.prev_tier = 0
            logger.info("FSM: CALIBRATING → READING")
