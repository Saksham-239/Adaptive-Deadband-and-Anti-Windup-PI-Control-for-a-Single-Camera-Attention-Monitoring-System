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

Intervention bucket:
  Accumulates while DISTRACTED, drains while FOCUSED.
  floor(bucket) → intervention tier (0=silent, 1=soft, 2=firm, 3=urgent).
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
    dt: float                       # seconds since last tick


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

    # Intervention bucket [0.0, I_max] (now the Integrator state)
    intervention_bucket: float = 0.0

    # For EMV (Exponential Moving Variance)
    attn_mean: float = 0.5
    attn_var: float = 0.0

    # Break toggle (True = currently in BREAK)
    in_break: bool = False

    # Previous break_key state to detect edges
    prev_break_key: bool = False


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
        f = cfg["fsm"]
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

        iv = cfg["intervention"]
        self._Kp                = iv["Kp"]
        self._Ki_up             = iv["Ki_up"]
        self._Ki_down           = iv["Ki_down"]
        self._I_max             = iv["I_max"]

        yolo = cfg["yolo"]
        self._phone_n           = yolo["phone_consecutive_frames"]
        self._phone_exit_n      = yolo["phone_exit_frames"]

    def tick(self, s: FSMState, inp: FSMInput) -> tuple[State, int]:
        """
        Advance the FSM by one tick.
        Mutates s in place.
        Returns (new_state, intervention_tier).
        """
        dt = inp.dt

        # --- BREAK toggle (edge-triggered) ---
        break_edge = inp.break_key_pressed and not s.prev_break_key
        s.prev_break_key = inp.break_key_pressed

        if break_edge:
            if s.current == State.BREAK:
                # Exit BREAK → READING
                s.current        = State.READING
                s.state_duration = 0.0
                s.intervention_bucket = 0.0
                logger.info("FSM: BREAK → READING (user resumed)")
            else:
                # Enter BREAK
                s.current        = State.BREAK
                s.state_duration = 0.0
                logger.info("FSM: %s → BREAK (user declared)", s.current.value)
            s.prev_break_key = inp.break_key_pressed
            return s.current, 0

        if s.current == State.BREAK:
            s.state_duration += dt
            return State.BREAK, 0

        # --- UNKNOWN: low brightness or no face ---
        if not inp.face_detected or inp.mean_brightness < self._brightness_floor:
            if s.current != State.UNKNOWN:
                logger.debug("FSM: → UNKNOWN (face=%s, brightness=%.0f)",
                             inp.face_detected, inp.mean_brightness)
            s.current        = State.UNKNOWN
            s.state_duration += dt
            # Bucket does NOT accumulate in UNKNOWN
            return State.UNKNOWN, 0

        # --- PHONE: preempts all other states ---
        if inp.phone_consecutive >= self._phone_n:
            if s.current != State.PHONE:
                logger.info("FSM: %s → PHONE", s.current.value)
                s.state_duration = 0.0
            s.current = State.PHONE
            s.state_duration += dt
            return State.PHONE, self._update_pi_controller(s, inp.attention_filtered, dt)

        # --- Exit PHONE if phone absent long enough ---
        if s.current == State.PHONE:
            if inp.phone_absent_frames >= self._phone_exit_n:
                logger.info("FSM: PHONE → DISTRACTED (phone removed)")
                s.current        = State.DISTRACTED
                s.state_duration = 0.0

        # --- CALIBRATING ---
        if s.current == State.CALIBRATING:
            s.state_duration += dt
            return State.CALIBRATING, 0

        # --- Update sub-timers ---
        a = inp.attention_filtered
        
        # --- Update EMV and calculate adaptive thresholds ---
        # EMV tracks rolling standard deviation to scale hysteresis bandwidth.
        # This prevents thrashing under changing noise floors (e.g. user fidgeting).
        mean_prev = s.attn_mean
        s.attn_mean = self._emv_alpha * a + (1.0 - self._emv_alpha) * mean_prev
        s.attn_var = (1.0 - self._emv_alpha) * (s.attn_var + self._emv_alpha * (a - mean_prev)**2)
        std_dev = s.attn_var ** 0.5
        
        # Adaptive Deadband (Schmitt trigger logic)
        # Bandwidth scales with noise. We clamp to max_bandwidth.
        bandwidth = self._min_bandwidth + self._k_band * std_dev
        bandwidth = min(bandwidth, self._max_bandwidth)
        
        # Shift thresholds symmetrically around the 0.60 midpoint
        focused_thresh = 0.60 + (bandwidth / 2.0)
        distracted_thresh = 0.60 - (bandwidth / 2.0)

        if a < distracted_thresh:
            s.time_below_distracted_thresh += dt
            s.time_above_focused_thresh     = 0.0
        elif a >= focused_thresh:
            s.time_above_focused_thresh     += dt
            s.time_below_distracted_thresh   = 0.0
        else:
            # In the hysteresis gap — don't accumulate either timer
            pass

        # --- State transitions ---
        prev = s.current

        if s.current == State.READING:
            # READING → WRITING
            if (inp.gaze_zone == "desk_zone" and
                    inp.head_pitch_deg > self._writing_pitch):
                s.time_below_distracted_thresh = 0.0  # reset distraction timer
                if not hasattr(s, '_writing_timer'):
                    s._writing_timer = 0.0
                s._writing_timer = getattr(s, '_writing_timer', 0.0) + dt
                if s._writing_timer >= self._writing_entry:
                    s.current = State.WRITING
                    s.state_duration = 0.0
                    s._writing_timer = 0.0
                    logger.info("FSM: READING → WRITING")
            else:
                s._writing_timer = 0.0

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

        s.state_duration += dt
        return s.current, self._update_pi_controller(s, a, dt)

    def _update_pi_controller(self, s: FSMState, a: float, dt: float) -> int:
        """
        PI Controller mapping error to intervention tier severity.
        No derivative term because attention-signal derivative is noise even 
        post-filtering, and the P term already captures sudden severity via context-override.
        """
        if s.current in (State.UNKNOWN, State.BREAK, State.CALIBRATING):
            return 0
            
        # Error signal decoupled from adaptive deadband. e > 0 means distracted.
        e = self._base_focused_thresh - a
        
        # Convert dt (seconds) to minutes to match the per-minute Ki units from config
        dt_min = dt / 60.0
        
        # Asymmetric leaky integrator with anti-windup (conditional integration)
        Ki = self._Ki_up if e > 0 else self._Ki_down
        I_new = s.intervention_bucket + Ki * e * dt_min
        
        if I_new > self._I_max and e > 0:
            pass # Freeze at saturation
        elif I_new < 0 and e < 0:
            s.intervention_bucket = 0.0
        else:
            s.intervention_bucket = max(0.0, min(I_new, self._I_max))
            
        u = self._Kp * e + s.intervention_bucket
        
        # Tier quantization mapping [0, I_max] into 4 discrete bins [0, 1, 2, 3]
        # Explicit division by I_max/4 and min(3, ...) clamping.
        tier = min(3, int(u // (self._I_max / 4.0)))
        return max(0, tier)

    def calibration_complete(self, s: FSMState) -> None:
        """Call when the user confirms the book ROI to leave CALIBRATING."""
        if s.current == State.CALIBRATING:
            s.current = State.READING
            s.state_duration = 0.0
            logger.info("FSM: CALIBRATING → READING")
