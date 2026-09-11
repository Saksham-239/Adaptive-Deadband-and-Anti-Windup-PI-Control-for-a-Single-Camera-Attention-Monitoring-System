"""
tests/test_fsm.py
Unit tests for core/fsm.py — pure FSM transitions.
Drives the FSM with synthetic FSMInput values, no webcam required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.fsm import FSMController, FSMState, FSMInput, State, quantize_tier_with_hysteresis


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
        "Ki_down": 6.0,
        "cool_down_rate": 0.5,
        "u_act_max": 2.25,
        "tier_hysteresis_delta": 0.05,
        "unknown_reset_factor": 0.5,
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
        
        # Drive distraction (a = 0.20).
        # Positive integration must stop when u_raw reaches u_act_max = 2.25
        # It must NOT continue accumulating up to I_max = 3.0.
        tick_n(ctrl, s, make_input(attention_filtered=0.20), 4000)
        
        # u_raw = Kp * e_db + I >= 2.25. Since e_db > 0, I < 2.25 and I << 3.0
        assert s.intervention_bucket < 2.25
        # u must saturate at u_act_max (2.25), not 3.0
        std_dev = s.attn_var ** 0.5
        bw = max(ctrl._min_bandwidth, min(ctrl._min_bandwidth + ctrl._k_band * std_dev, ctrl._max_bandwidth))
        T_c = (ctrl._base_focused_thresh + ctrl._base_distracted_thresh) / 2.0
        T_low = T_c - bw / 2.0
        e_db = T_low - 0.20 # positive error
        u_sat = ctrl._Kp * e_db + s.intervention_bucket
        assert u_sat >= 2.25
        assert s.intervention_bucket < 3.0
        
        # When distraction ends, negative integration should drain immediately
        # with no windup overshoot delay.
        bucket_before = s.intervention_bucket
        _, tier = ctrl.tick(s, make_input(attention_filtered=0.80))
        assert s.intervention_bucket < bucket_before

    def test_tier_boundaries(self):
        ctrl, s = self._distracted_state()
        s.prev_tier = 0  # start from Tier 0
        
        def mock_pi_and_get_tier(u_val):
            s.intervention_bucket = u_val
            # Calling _update_pi_controller with e_db=0 and dt=0 returns tier from u_val
            tier = ctrl._update_pi_controller(s, e_db=0.0, dt=0.0)
            return tier
            
        s.prev_tier = 0
        assert mock_pi_and_get_tier(0.0) == 0
        s.prev_tier = 0
        assert mock_pi_and_get_tier(0.74) == 0
        
        s.prev_tier = 0
        assert mock_pi_and_get_tier(0.75) == 1
        s.prev_tier = 1
        assert mock_pi_and_get_tier(1.49) == 1
        
        s.prev_tier = 1
        assert mock_pi_and_get_tier(1.50) == 2
        s.prev_tier = 2
        assert mock_pi_and_get_tier(2.24) == 2
        
        s.prev_tier = 2
        assert mock_pi_and_get_tier(2.25) == 3
        s.prev_tier = 3
        assert mock_pi_and_get_tier(3.00) == 3
        
        # Even if P-term pushes u > u_act_max, tier must clamp to 3
        assert mock_pi_and_get_tier(3.50) == 3
        assert mock_pi_and_get_tier(10.0) == 3


# ---------------------------------------------------------------------------
# Comprehensive Architecture Verification Tests
# ---------------------------------------------------------------------------

class TestFinalizedArchitecture:
    def test_unknown_to_reading_recovery(self):
        """1. UNKNOWN -> READING recovery when face is detected and brightness >= floor."""
        ctrl = make_ctrl()
        s = FSMState(current=State.UNKNOWN, state_duration=10.0,
                     time_below_distracted_thresh=5.0, time_above_focused_thresh=3.0,
                     writing_timer=2.0, intervention_bucket=2.0, prev_tier=3)
        
        # Restore face and brightness
        inp = make_input(face_detected=True, mean_brightness=100.0, attention_filtered=0.80)
        state, tier = ctrl.tick(s, inp)
        
        assert state == State.READING
        assert tier == 0
        assert s.state_duration == 0.0
        assert s.time_below_distracted_thresh == 0.0
        assert s.time_above_focused_thresh == 0.0
        assert s.writing_timer == 0.0
        assert s.prev_tier == 0
        # Bucket scaled by unknown_reset_factor (0.5 * 2.0 = 1.0)
        assert s.intervention_bucket == pytest.approx(1.0)

    def test_integrator_cooling_in_reading(self):
        """2. Integrator cooling in READING state."""
        ctrl = make_ctrl()
        s = FSMState(current=State.READING, intervention_bucket=1.0)
        
        # Tick in READING for 60 seconds (cool_down_rate = 0.5/min)
        # 1 tick with dt=60.0 (simulated) or 600 ticks of dt=0.1
        inp = make_input(attention_filtered=0.80, dt=0.1, ctrl_dt=0.1)
        for _ in range(600):
            state, tier = ctrl.tick(s, inp)
            assert tier == 0  # Must always force tier = 0 in READING
            
        # After 60 seconds at cool_down_rate = 0.5/min, I(t) should decay significantly
        assert s.intervention_bucket < 0.70
        assert s.intervention_bucket > 0.0

    def test_integrator_reset_in_break_and_calibrating(self):
        """3. Integrator reset in BREAK and CALIBRATING states."""
        ctrl = make_ctrl()
        s = FSMState(current=State.CALIBRATING, intervention_bucket=1.5, prev_tier=2)
        state, tier = ctrl.tick(s, make_input(attention_filtered=0.50))
        assert tier == 0
        assert s.intervention_bucket == 0.0
        assert s.prev_tier == 0

        # Now test BREAK
        s.current = State.READING
        s.intervention_bucket = 1.8
        s.prev_tier = 2
        # Press 'b' to enter BREAK
        state, tier = ctrl.tick(s, make_input(break_key_pressed=True))
        assert state == State.BREAK
        assert tier == 0
        assert s.intervention_bucket == 0.0
        assert s.prev_tier == 0

    def test_anti_windup_stops_positive_integration_at_u_act_max(self):
        """4. Anti-windup stops positive integration at u_raw >= 2.25."""
        ctrl = make_ctrl()
        s = FSMState(current=State.DISTRACTED, intervention_bucket=2.0)
        
        # Deep distraction: e_db = 0.55 - 0.10 = 0.45.
        # u_raw = 0.5 * 0.45 + 2.0 = 2.225 initially.
        # Within a few ticks, u_raw >= 2.25 and positive integration freezes.
        for _ in range(500):
            ctrl.tick(s, make_input(attention_filtered=0.10, dt=0.1, ctrl_dt=0.1))
            
        # Integrator should NOT reach I_max (3.0); it must stop once u_raw >= u_act_max (2.25)
        assert s.intervention_bucket < ctrl._u_act_max
        std_dev = s.attn_var ** 0.5
        bw = max(ctrl._min_bandwidth, min(ctrl._min_bandwidth + ctrl._k_band * std_dev, ctrl._max_bandwidth))
        T_c = (ctrl._base_focused_thresh + ctrl._base_distracted_thresh) / 2.0
        T_low = T_c - bw / 2.0
        e_db = T_low - 0.10
        u_raw = ctrl._Kp * e_db + s.intervention_bucket
        assert u_raw >= ctrl._u_act_max

    def test_one_step_anti_windup_boundary_crossing(self):
        """Positive integration cannot drive controller demand beyond u_act_max in a single step."""
        ctrl = make_ctrl()
        s = FSMState(current=State.DISTRACTED, intervention_bucket=2.04)
        e_db = 0.40
        # u_raw = Kp * e_db + I = 0.5 * 0.40 + 2.04 = 2.24 < 2.25
        u_raw_initial = ctrl._Kp * e_db + s.intervention_bucket
        assert u_raw_initial < ctrl._u_act_max

        # With dt=60.0s (1 min), Ki_up=1.2:
        # Uncapped I_new would be 2.04 + 1.2 * 0.40 * 1.0 = 2.52, pushing u to 2.72.
        # With boundary capping, I is capped at u_act_max - Kp*e_db = 2.25 - 0.20 = 2.05.
        tier = ctrl._update_pi_controller(s, e_db=e_db, dt=60.0)

        u_final = ctrl._Kp * e_db + s.intervention_bucket
        assert u_final == pytest.approx(ctrl._u_act_max)
        assert s.intervention_bucket == pytest.approx(ctrl._u_act_max - ctrl._Kp * e_db)
        assert tier == 3

    def test_quantizer_tier3_aligned_with_configured_u_act_max(self):
        """Tier 3 threshold scales with configured u_act_max, and invalid values <= 1.50 are rejected."""
        # 1. Custom valid u_act_max (e.g. 2.00)
        custom_cfg = {**BASE_CFG, "intervention": {**BASE_CFG["intervention"], "u_act_max": 2.00}}
        ctrl_custom = FSMController(custom_cfg)
        s = FSMState(current=State.DISTRACTED, prev_tier=2)
        # At u = 1.99, tier should be 2
        tier_below = ctrl_custom._update_pi_controller(s, e_db=0.0, dt=0.0)
        s.intervention_bucket = 1.99
        tier_below = ctrl_custom._update_pi_controller(s, e_db=0.0, dt=0.0)
        assert tier_below == 2
        # At u = 2.00, tier should be 3
        s.intervention_bucket = 2.00
        tier_at = ctrl_custom._update_pi_controller(s, e_db=0.0, dt=0.0)
        assert tier_at == 3

        # 2. Reject u_act_max <= 1.50 (must exceed Tier 2 entry)
        invalid_cfg = {**BASE_CFG, "intervention": {**BASE_CFG["intervention"], "u_act_max": 1.40}}
        with pytest.raises(ValueError, match="must be greater than Tier 2 threshold"):
            FSMController(invalid_cfg)

    def test_negative_integration_below_saturation_boundary(self):
        """5. Negative integration remains possible below saturation boundary."""
        ctrl = make_ctrl()
        s = FSMState(current=State.DISTRACTED, intervention_bucket=2.0, prev_tier=3)
        
        # User refocuses with high attention (e_db < 0)
        inp = make_input(attention_filtered=0.90, dt=0.1, ctrl_dt=0.1)
        ctrl.tick(s, inp)
        
        # Negative integration must engage immediately
        assert s.intervention_bucket < 2.0

    def test_schmitt_upward_transitions(self):
        """6. Schmitt upward transitions at 0.75 / 1.50 / 2.25."""
        assert quantize_tier_with_hysteresis(0.74, prev_tier=0, delta=0.05) == 0
        assert quantize_tier_with_hysteresis(0.75, prev_tier=0, delta=0.05) == 1
        
        assert quantize_tier_with_hysteresis(1.49, prev_tier=1, delta=0.05) == 1
        assert quantize_tier_with_hysteresis(1.50, prev_tier=1, delta=0.05) == 2
        
        assert quantize_tier_with_hysteresis(2.24, prev_tier=2, delta=0.05) == 2
        assert quantize_tier_with_hysteresis(2.25, prev_tier=2, delta=0.05) == 3

    def test_schmitt_downward_transitions(self):
        """7. Schmitt downward transitions at 0.70 / 1.45 / 2.20."""
        # From Tier 3 downward:
        assert quantize_tier_with_hysteresis(2.20, prev_tier=3, delta=0.05) == 3
        assert quantize_tier_with_hysteresis(2.19, prev_tier=3, delta=0.05) == 2
        
        # From Tier 2 downward:
        assert quantize_tier_with_hysteresis(1.45, prev_tier=2, delta=0.05) == 2
        assert quantize_tier_with_hysteresis(1.44, prev_tier=2, delta=0.05) == 1
        
        # From Tier 1 downward:
        assert quantize_tier_with_hysteresis(0.70, prev_tier=1, delta=0.05) == 1
        assert quantize_tier_with_hysteresis(0.69, prev_tier=1, delta=0.05) == 0

    def test_tier_hysteresis_prevents_chatter(self):
        """8. Tier hysteresis prevents chatter around each boundary."""
        # Around Tier 3 boundary (2.25):
        # Enter Tier 3:
        tier = quantize_tier_with_hysteresis(2.25, prev_tier=2, delta=0.05)
        assert tier == 3
        # Small drop into hysteresis gap [2.20, 2.25) stays Tier 3:
        assert quantize_tier_with_hysteresis(2.24, prev_tier=3, delta=0.05) == 3
        assert quantize_tier_with_hysteresis(2.21, prev_tier=3, delta=0.05) == 3
        
        # Around Tier 2 boundary (1.50):
        assert quantize_tier_with_hysteresis(1.50, prev_tier=1, delta=0.05) == 2
        assert quantize_tier_with_hysteresis(1.49, prev_tier=2, delta=0.05) == 2
        assert quantize_tier_with_hysteresis(1.46, prev_tier=2, delta=0.05) == 2
        
        # Around Tier 1 boundary (0.75):
        assert quantize_tier_with_hysteresis(0.75, prev_tier=0, delta=0.05) == 1
        assert quantize_tier_with_hysteresis(0.74, prev_tier=1, delta=0.05) == 1
        assert quantize_tier_with_hysteresis(0.71, prev_tier=1, delta=0.05) == 1

    def test_no_actuator_output_in_non_intervention_states(self):
        """9. No actuator output (tier = 0) in READING, WRITING, THINKING."""
        ctrl = make_ctrl()
        
        for st in (State.READING, State.WRITING, State.THINKING):
            s = FSMState(current=st, intervention_bucket=2.20, prev_tier=3)
            # Even with high integral demand, tier must be forced to 0
            tier = ctrl._update_pi_controller(s, e_db=0.30, dt=0.1)
            assert tier == 0
            assert s.prev_tier == 0

    def test_timing_discontinuity_zeros_ctrl_dt(self):
        """11. Timing discontinuity: ctrl_dt = 0 while FSM elapsed time still advances."""
        ctrl = make_ctrl()
        s = FSMState(current=State.DISTRACTED, intervention_bucket=1.0)
        
        # Simulate a 5.0 second lag spike / UI freeze (dt=5.0, ctrl_dt=0.0)
        inp = make_input(attention_filtered=0.20, dt=5.0, ctrl_dt=0.0)
        ctrl.tick(s, inp)
        
        # FSM state duration advances by 5.0s
        assert s.state_duration == pytest.approx(5.0)
        # Integrator bucket must NOT accumulate 5 seconds of error (dt_ctrl was 0)
        assert s.intervention_bucket == pytest.approx(1.0)

    def test_neutral_deadband_controlled_decay(self):
        """12. Controller does not integrate indefinitely inside neutral deadband."""
        ctrl = make_ctrl()
        s = FSMState(current=State.DISTRACTED, intervention_bucket=1.5)
        
        # Inside deadband: a = 0.60 -> e_db = 0.0
        inp = make_input(attention_filtered=0.60, dt=0.1, ctrl_dt=0.1)
        for _ in range(100):
            ctrl.tick(s, inp)
            
        # Integral must decay via cool_down_rate, not freeze forever
        assert s.intervention_bucket < 1.5
        assert s.intervention_bucket > 0.0

    def test_progressive_tier_deescalation(self):
        """13. Recovery from high intervention demand produces progressive tier de-escalation."""
        ctrl = make_ctrl()
        s = FSMState(current=State.DISTRACTED, intervention_bucket=2.25, prev_tier=3)
        
        # In DISTRACTED, user refocuses with recovery error e_db = -0.10.
        # As the integrator unwinds from 2.25, the quantizer must progressively
        # de-escalate through each discrete tier: 3 -> 2 -> 1 -> 0 without skipping.
        observed_tiers = [s.prev_tier]
        
        for _ in range(2500):
            tier = ctrl._update_pi_controller(s, e_db=-0.10, dt=0.1)
            if tier != observed_tiers[-1]:
                observed_tiers.append(tier)
            if tier == 0:
                break
                
        # Must observe progressive de-escalation: 3 -> 2 -> 1 -> 0
        assert observed_tiers == [3, 2, 1, 0]
