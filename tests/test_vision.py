"""
tests/test_vision.py
Unit tests for face detection staleness timeout and queue absence semantics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import queue
import time
import pytest
from unittest.mock import MagicMock

from core.detector import put_fresh
from core.fsm import FSMController, FSMState, FSMInput, State
from core.vision import FaceResult


def test_put_fresh_pushes_none_on_absence():
    """Verify put_fresh reliably pushes (None, ts) when detection produces no face."""
    q = queue.Queue(maxsize=1)
    
    # 1. First face detected
    dummy_face = MagicMock(spec=FaceResult)
    t0 = 100.0
    put_fresh(q, (dummy_face, t0))
    
    res, ts = q.get_nowait()
    assert res is dummy_face
    assert ts == 100.0
    
    # 2. Face leaves camera -> worker pushes None
    t1 = 100.1
    put_fresh(q, (None, t1))
    
    res2, ts2 = q.get_nowait()
    assert res2 is None
    assert ts2 == 100.1


def test_face_staleness_timeout_expires_result():
    """Simulate main loop queue consumption and wall-clock staleness expiration."""
    FACE_STALENESS_TIMEOUT_SEC = 0.5
    result_q = queue.Queue(maxsize=1)
    
    last_face_result = None
    last_face_time = 0.0
    
    # Tick 1 at t=10.0: Face detected
    t_tick1 = 10.0
    dummy_face = MagicMock(spec=FaceResult)
    put_fresh(result_q, (dummy_face, t_tick1))
    
    try:
        res, ts = result_q.get_nowait()
        last_face_result = res
        last_face_time = ts
    except queue.Empty:
        pass
    
    if last_face_result is not None and (t_tick1 - last_face_time > FACE_STALENESS_TIMEOUT_SEC):
        last_face_result = None
        
    assert last_face_result is not None
    assert last_face_result is dummy_face
    
    # Tick 2 at t=10.3 (300ms later, result_q empty): Within 0.5s timeout, face retained
    t_tick2 = 10.3
    try:
        res, ts = result_q.get_nowait()
        last_face_result = res
        last_face_time = ts
    except queue.Empty:
        pass
        
    if last_face_result is not None and (t_tick2 - last_face_time > FACE_STALENESS_TIMEOUT_SEC):
        last_face_result = None
        
    assert last_face_result is not None
    
    # Tick 3 at t=10.6 (600ms after last detection, result_q empty): Timeout expires -> None
    t_tick3 = 10.6
    try:
        res, ts = result_q.get_nowait()
        last_face_result = res
        last_face_time = ts
    except queue.Empty:
        pass
        
    if last_face_result is not None and (t_tick3 - last_face_time > FACE_STALENESS_TIMEOUT_SEC):
        last_face_result = None
        
    assert last_face_result is None


def test_face_absence_drives_fsm_unknown():
    """Verify that when last_face_result is None, FSM immediately enters State.UNKNOWN."""
    cfg = {
        "fsm": {
            "brightness_floor": 40.0,
            "thinking_entry_sec": 3.0,
            "thinking_max_sec": 8.0,
            "writing_entry_sec": 2.0,
            "writing_pitch_deg": 15.0,
            "return_to_reading_sec": 2.0,
            "distracted_return_sec": 2.0,
            "base_focused_threshold": 0.65,
            "base_distracted_threshold": 0.55,
            "emv_alpha": 0.05,
            "min_bandwidth": 0.06,
            "max_bandwidth": 0.30,
            "k_band": 1.0,
        }
    }
    fsm = FSMController(cfg)
    state = FSMState()
    state.current = State.READING
    
    # Simulated frame where face is absent (last_face_result is None)
    last_face_result = None
    face_detected = last_face_result is not None
    
    inp = FSMInput(
        attention_filtered=0.8,
        gaze_zone="unknown",
        head_pitch_deg=0.0,
        phone_consecutive=0,
        phone_absent_frames=10,
        mean_brightness=100.0,
        face_detected=face_detected,
        break_key_pressed=False,
        dt=0.033,
    )
    
    new_state, _ = fsm.tick(state, inp)
    assert new_state == State.UNKNOWN
    assert state.current == State.UNKNOWN
