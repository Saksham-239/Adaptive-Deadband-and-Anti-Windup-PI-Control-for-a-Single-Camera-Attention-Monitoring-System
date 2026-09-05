"""
tests/test_fsm.py
Unit tests for core/fsm.py — pure FSM transitions.
Drives the FSM with synthetic FSMInput values, no webcam required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.fsm import FSMController, FSMState, FSMInput, State


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

BASE_CFG = {
    "fsm": {
        "thinking_entry_sec": 3.0,
        "thinking_max_sec":   8.0,
        "writing_entry_sec":  2.0,
        "writing_pitch_deg":  15.0,
        "return_to_reading_sec":  2.0,
        "distracted_return_sec":  2.0,
        "base_focused_threshold": 0.60,
        "base_distracted_threshold": 0.60,
        "emv_alpha": 0.1,
        "min_bandwidth": 0.1,
        "max_bandwidth": 0.3,
        "k_band": 2.0,
        "brightness_floor":       40.0,
    },
    "intervention": {
        "Kp": 0.5,
        "Ki_up": 1.2,
        "Ki_down": 0.15,
        "I_max": 3.0,
    },
    "yolo": {
        "phone_consecutive_frames": 5,
        "phone_exit_frames":        10,
    },
}


def make_ctrl() -> FSMController:
    return FSMController(BASE_CFG)


def make_input(**kwargs) -> FSMInput:
    defaults = dict(
        attention_filtered=0.80,
        gaze_zone="book_zone",
        head_pitch_deg=5.0,
        phone_consecutive=0,
        phone_absent_frames=0,
        mean_brightness=120.0,
        face_detected=True,
        break_key_pressed=False,
        dt=0.1,
    )
    defaults.update(kwargs)
    return FSMInput(**defaults)


def tick_n(ctrl, state, inp, n):
    """Tick FSM n times with the same input."""
    result = (state.current, 0)
    for _ in range(n):
        result = ctrl.tick(state, inp)
    return result


# ---------------------------------------------------------------------------
# Calibration → Reading
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_starts_in_calibrating(self):
        s = FSMState()
        assert s.current == State.CALIBRATING

    def test_calibration_complete_transitions_to_reading(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        assert s.current == State.READING


# ---------------------------------------------------------------------------
# READING → THINKING → DISTRACTED
# ---------------------------------------------------------------------------

class TestReadingToDistracted:
    def _reading_state(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        return ctrl, s

    def test_reading_stays_reading_when_focused(self):
        ctrl, s = self._reading_state()
        for _ in range(100):
            state, _ = ctrl.tick(s, make_input(attention_filtered=0.80))
        assert state == State.READING

    def test_reading_to_thinking_after_3s_below_threshold(self):
        ctrl, s = self._reading_state()
        # 3.0s / 0.1dt = 30 ticks
        inp = make_input(attention_filtered=0.40)  # below 0.55
        tick_n(ctrl, s, inp, 29)
        assert s.current == State.READING  # not yet
        tick_n(ctrl, s, inp, 2)
        assert s.current == State.THINKING

    def test_thinking_to_distracted_after_8s(self):
        ctrl, s = self._reading_state()
        # Drive to THINKING first
        inp_low = make_input(attention_filtered=0.40)
        tick_n(ctrl, s, inp_low, 31)
        assert s.current == State.THINKING
        # Stay in THINKING for 8s → DISTRACTED
        tick_n(ctrl, s, inp_low, 80)
        assert s.current == State.DISTRACTED

    def test_thinking_to_reading_when_score_recovers(self):
        ctrl, s = self._reading_state()
        tick_n(ctrl, s, make_input(attention_filtered=0.40), 31)
        assert s.current == State.THINKING
        # High score for 2s → back to READING
        tick_n(ctrl, s, make_input(attention_filtered=0.80), 21)
        assert s.current == State.READING

    def test_distracted_to_reading_when_score_recovers(self):
        ctrl, s = self._reading_state()
        # Drive to DISTRACTED
        tick_n(ctrl, s, make_input(attention_filtered=0.40), 31 + 81)
        assert s.current == State.DISTRACTED
        # Recover
        tick_n(ctrl, s, make_input(attention_filtered=0.80), 21)
        assert s.current == State.READING


# ---------------------------------------------------------------------------
# Hysteresis — no flicker around threshold
# ---------------------------------------------------------------------------

class TestHysteresis:
    def test_no_flicker_at_gap_midpoint(self):
        """Score oscillating in the hysteresis gap should not cause state flips."""
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        states_seen = set()
        # Score oscillates between 0.57 and 0.63 — inside the 0.55–0.65 gap
        for i in range(200):
            score = 0.57 if i % 2 == 0 else 0.63
            state, _ = ctrl.tick(s, make_input(attention_filtered=score))
            states_seen.add(state)
        # Should stay in READING (started there, never went low enough to leave)
        assert State.DISTRACTED not in states_seen
        assert State.THINKING not in states_seen


# ---------------------------------------------------------------------------
# PHONE state
# ---------------------------------------------------------------------------

class TestPhoneState:
    def _reading_state(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        return ctrl, s

    def test_phone_preempts_reading(self):
        ctrl, s = self._reading_state()
        inp = make_input(phone_consecutive=5)
        state, _ = ctrl.tick(s, inp)
        assert state == State.PHONE

    def test_phone_requires_n_consecutive(self):
        ctrl, s = self._reading_state()
        # Only 4 consecutive frames — should NOT trigger
        state, _ = ctrl.tick(s, make_input(phone_consecutive=4))
        assert state == State.READING

    def test_phone_exits_to_distracted_after_absence(self):
        ctrl, s = self._reading_state()
        ctrl.tick(s, make_input(phone_consecutive=5))
        assert s.current == State.PHONE
        # Phone absent for 10 frames
        state, _ = ctrl.tick(s, make_input(phone_consecutive=0, phone_absent_frames=10))
        assert state == State.DISTRACTED

    def test_phone_does_not_exit_early(self):
        ctrl, s = self._reading_state()
        ctrl.tick(s, make_input(phone_consecutive=5))
        state, _ = ctrl.tick(s, make_input(phone_consecutive=0, phone_absent_frames=5))
        assert state == State.PHONE  # needs 10, not 5


# ---------------------------------------------------------------------------
# BREAK state
# ---------------------------------------------------------------------------

class TestBreakState:
    def _reading_state(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        return ctrl, s

    def test_break_enters_on_key_edge(self):
        ctrl, s = self._reading_state()
        # First tick with break_key_pressed=True → enter BREAK
        state, _ = ctrl.tick(s, make_input(break_key_pressed=True))
        assert state == State.BREAK

    def test_break_held_stays_in_break(self):
        ctrl, s = self._reading_state()
        ctrl.tick(s, make_input(break_key_pressed=True))  # enter
        state, _ = ctrl.tick(s, make_input(break_key_pressed=True))  # held
        assert state == State.BREAK  # no exit while held

    def test_break_exits_on_second_edge(self):
        ctrl, s = self._reading_state()
        ctrl.tick(s, make_input(break_key_pressed=True))   # enter
        ctrl.tick(s, make_input(break_key_pressed=False))  # release
        state, _ = ctrl.tick(s, make_input(break_key_pressed=True))   # second press
        assert state == State.READING

    def test_break_resets_intervention_bucket(self):
        ctrl, s = self._reading_state()
        # Build up bucket
        tick_n(ctrl, s, make_input(attention_filtered=0.40), 31 + 81)
        assert s.intervention_bucket > 0
        # Enter and exit BREAK
        ctrl.tick(s, make_input(break_key_pressed=True))
        ctrl.tick(s, make_input(break_key_pressed=False))
        ctrl.tick(s, make_input(break_key_pressed=True))
        assert s.intervention_bucket == pytest.approx(0.0)

    def test_no_intervention_during_break(self):
        ctrl, s = self._reading_state()
        ctrl.tick(s, make_input(break_key_pressed=True))
        _, tier = ctrl.tick(s, make_input(break_key_pressed=False))
        assert tier == 0


# ---------------------------------------------------------------------------
# UNKNOWN state
# ---------------------------------------------------------------------------

class TestUnknownState:
    def test_unknown_on_no_face(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        state, _ = ctrl.tick(s, make_input(face_detected=False))
        assert state == State.UNKNOWN

    def test_unknown_on_dark_frame(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        state, _ = ctrl.tick(s, make_input(mean_brightness=20.0))
        assert state == State.UNKNOWN

    def test_unknown_no_bucket_accumulation(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        bucket_before = s.intervention_bucket
        for _ in range(100):
            ctrl.tick(s, make_input(face_detected=False))
        assert s.intervention_bucket == pytest.approx(bucket_before)


# ---------------------------------------------------------------------------
# Intervention bucket & tier
# ---------------------------------------------------------------------------

class TestInterventionBucket:
    def _distracted_state(self):
        ctrl = make_ctrl()
        s = FSMState()
        ctrl.calibration_complete(s)
        tick_n(ctrl, s, make_input(attention_filtered=0.40), 31 + 81)
        assert s.current == State.DISTRACTED
        return ctrl, s

    def test_pi_windup(self):
        ctrl, s = self._distracted_state()
        
        # Drive the bucket to saturation (e = 0.60 - 0.20 = 0.40 > 0)
        # Ki_up is 1.2/min = 0.02/sec. e * Ki_up = 0.008/sec
        # So we need to stay distracted for a while to reach I_max=3.0
        # 3.0 / 0.008 = 375 seconds. At 0.1s dt, we need 3750 ticks.
        tick_n(ctrl, s, make_input(attention_filtered=0.20), 4000)
        
        assert s.intervention_bucket == pytest.approx(3.0)  # Capped at I_max
        
        # When distraction ends, it should drain immediately at Ki_down rate, no windup overshoot.
        # e = 0.60 - 0.80 = -0.20. Ki_down is 0.15/min = 0.0025/sec
        # Rate: e * Ki_down = -0.0005/sec.
        # For 1 tick (0.1s), drain is 0.00005.
        _, tier = ctrl.tick(s, make_input(attention_filtered=0.80))
        assert s.intervention_bucket == pytest.approx(3.0 - 0.00005)

    def test_tier_boundaries(self):
        ctrl, s = self._distracted_state()
        # Mock u directly to test the explicit tier binning logic
        # Bin boundaries for I_max=3.0 are multiples of 0.75:
        # Tier 0: [0, 0.75)
        # Tier 1: [0.75, 1.5)
        # Tier 2: [1.5, 2.25)
        # Tier 3: [2.25, 3.0+]
        
        def mock_pi_and_get_tier(u_val):
            # Hack the state and PI calculation to force a specific `u`
            # For e = 0, u = s.intervention_bucket = u_val
            s.intervention_bucket = u_val
            # Calling _update_pi_controller with e=0 (a=0.6) and dt=0
            # will result in u = u_val and return the tier.
            return ctrl._update_pi_controller(s, a=0.60, dt=0.0)
            
        assert mock_pi_and_get_tier(0.0) == 0
        assert mock_pi_and_get_tier(0.74) == 0
        
        assert mock_pi_and_get_tier(0.75) == 1
        assert mock_pi_and_get_tier(1.49) == 1
        
        assert mock_pi_and_get_tier(1.50) == 2
        assert mock_pi_and_get_tier(2.24) == 2
        
        assert mock_pi_and_get_tier(2.25) == 3
        assert mock_pi_and_get_tier(3.00) == 3
        
        # Even if P-term pushes u > I_max, tier must clamp to 3
        assert mock_pi_and_get_tier(3.50) == 3
        assert mock_pi_and_get_tier(10.0) == 3
