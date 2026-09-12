"""
tests/test_attention.py
Unit tests for core/attention.py — pure functions only.
No webcam, no MediaPipe, no YOLO required.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.attention import (
    gaze_score,
    head_score,
    context_score,
    blink_score,
    compute_raw_score,
    AttentionScorer,
)
from core.detector import BBox
from core.vision import GazeZone, classify_gaze_zone


# ---------------------------------------------------------------------------
# gaze_score
# ---------------------------------------------------------------------------

class TestGazeScore:
    def test_book_zone_returns_1(self):
        assert gaze_score(GazeZone.BOOK) == pytest.approx(1.0)

    def test_away_zone_returns_0(self):
        assert gaze_score(GazeZone.AWAY) == pytest.approx(0.0)

    def test_screen_zone_returns_neutral(self):
        score = gaze_score(GazeZone.SCREEN)
        assert 0.0 < score < 1.0

    def test_desk_zone_returns_point_8(self):
        assert gaze_score(GazeZone.DESK) == pytest.approx(0.8)

    def test_unknown_zone_returns_point_3(self):
        assert gaze_score(GazeZone.UNKNOWN) == pytest.approx(0.3)

    def test_book_zone_pitch_down_disjoint_iris_roi(self):
        """
        Verify that when pitched down with a calibrated ROI on the desk (pitch > pitch_down_threshold),
        classify_gaze_zone() prioritizes GazeZone.BOOK over GazeZone.DESK, and gaze_score(zone) evaluates to 1.0.
        When uncalibrated (book_roi=None) with identical posture, it yields GazeZone.DESK and gaze_score(zone) evaluates to 0.8.
        """
        desk_book_roi = (100, 300, 500, 480)
        pitch_down = 20.0     # above pitch_down_threshold (15.0)

        # 1. Calibrated book ROI present -> BOOK zone (score = 1.0)
        zone_calibrated = classify_gaze_zone(
            yaw=0.0,
            pitch=pitch_down,
            iris_dev_x=0.0,
            iris_dev_y=0.0,
            book_roi=desk_book_roi,
            frame_w=640,
            frame_h=480,
            pitch_down_threshold=15.0,
        )
        assert zone_calibrated == GazeZone.BOOK
        assert gaze_score(zone_calibrated) == pytest.approx(1.0)

        # 2. Uncalibrated (no book ROI) -> DESK zone (score = 0.8)
        zone_uncalibrated = classify_gaze_zone(
            yaw=0.0,
            pitch=pitch_down,
            iris_dev_x=0.0,
            iris_dev_y=0.0,
            book_roi=None,
            frame_w=640,
            frame_h=480,
            pitch_down_threshold=15.0,
        )
        assert zone_uncalibrated == GazeZone.DESK
        assert gaze_score(zone_uncalibrated) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# head_score
# ---------------------------------------------------------------------------

class TestHeadScore:
    def test_zero_angles_is_1(self):
        assert head_score(0.0, 0.0, sigma_deg=20.0) == pytest.approx(1.0)

    def test_at_sigma_approx_point_6(self):
        score = head_score(20.0, 0.0, sigma_deg=20.0)
        assert abs(score - math.exp(-0.5)) < 0.01

    def test_large_angle_approaches_0(self):
        assert head_score(90.0, 90.0, sigma_deg=20.0) < 0.01

    def test_asymmetric_yaw_pitch(self):
        # head_score explicitly does not penalize looking down (positive pitch).
        yaw_score = head_score(10, 0)
        pitch_down_score = head_score(0, 10)
        assert yaw_score != pytest.approx(pitch_down_score)
        assert pitch_down_score > yaw_score

    def test_output_range(self):
        for yaw in [-45, 0, 45]:
            for pitch in [-30, 0, 30]:
                s = head_score(yaw, pitch)
                assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# context_score
# ---------------------------------------------------------------------------

class TestContextScore:
    def _make_bbox(self, label: str, cls_id: int, x1=0, y1=0, x2=300, y2=300):
        return BBox(label=label, class_id=cls_id, confidence=0.9,
                    x1=x1, y1=y1, x2=x2, y2=y2)

    def test_phone_under_gaze_returns_0(self):
        bboxes = [self._make_bbox("cell_phone", 67)]
        score = context_score(GazeZone.BOOK, bboxes, iris_cx=150, iris_cy=150)
        assert score == pytest.approx(0.0)

    def test_book_under_gaze_returns_1(self):
        bboxes = [self._make_bbox("book", 73)]
        score = context_score(GazeZone.BOOK, bboxes, iris_cx=150, iris_cy=150)
        assert score == pytest.approx(1.0)

    def test_laptop_under_gaze_returns_half(self):
        bboxes = [self._make_bbox("laptop", 63)]
        score = context_score(GazeZone.SCREEN, bboxes, iris_cx=150, iris_cy=150)
        assert score == pytest.approx(0.5)

    def test_no_bboxes_returns_neutral(self):
        score = context_score(GazeZone.BOOK, [], iris_cx=150, iris_cy=150)
        assert score == pytest.approx(0.5)

    def test_desk_zone_no_match_returns_07(self):
        # No YOLO object detected, but gaze is on desk zone -> pen/paper implied (0.7)
        score = context_score(GazeZone.DESK, [], iris_cx=500, iris_cy=500)
        assert score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# blink_score
# ---------------------------------------------------------------------------

class TestBlinkScore:
    def test_open_eye_returns_1(self):
        assert blink_score(0.30, ear_threshold=0.20) == pytest.approx(1.0)

    def test_closed_eye_returns_0(self):
        assert blink_score(0.10, ear_threshold=0.20) == pytest.approx(0.0)

    def test_threshold_midpoint_returns_half(self):
        score = blink_score(0.20, ear_threshold=0.20)  # midpoint = threshold
        assert 0.4 <= score <= 0.6

    def test_range(self):
        for ear in [0.0, 0.1, 0.2, 0.3, 0.4]:
            s = blink_score(ear)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# compute_raw_score
# ---------------------------------------------------------------------------

class TestComputeRawScore:
    def test_all_ones_returns_1(self):
        weights = [0.35, 0.30, 0.25, 0.10]
        assert compute_raw_score(1, 1, 1, 1, weights) == pytest.approx(1.0)

    def test_all_zeros_returns_0(self):
        weights = [0.35, 0.30, 0.25, 0.10]
        assert compute_raw_score(0, 0, 0, 0, weights) == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        weights = [0.35, 0.30, 0.25, 0.10]
        assert sum(weights) == pytest.approx(1.0)

    def test_output_clamped_to_01(self):
        weights = [1.0, 0.0, 0.0, 0.0]
        # Even with wrong weights, output stays in [0,1]
        assert 0.0 <= compute_raw_score(2.0, 0, 0, 0, weights) <= 1.0


# ---------------------------------------------------------------------------
# OneEuroFilter
# ---------------------------------------------------------------------------

from core.vision import OneEuroFilter

class TestOneEuroFilter:
    def test_first_call_guard(self):
        f = OneEuroFilter()
        # On first call, dt is implicitly None. Should return raw value.
        assert f(0.1, 10.0) == pytest.approx(10.0)
        assert f.x_prev == 10.0
        assert f.dx_prev == 0.0

    def test_dt_zero_or_negative_guard(self):
        f = OneEuroFilter()
        f(0.1, 10.0)  # Init
        
        # Non-monotonic / identical timestamp
        res = f(0.1, 20.0)
        assert res == pytest.approx(10.0)  # Should return previous filtered value, skipping update
        
        res = f(0.05, 30.0)
        assert res == pytest.approx(10.0)  # Same for negative dt


# ---------------------------------------------------------------------------
# AttentionScorer integration
# ---------------------------------------------------------------------------

class TestAttentionScorer:
    def _make_cfg(self):
        return {
            "attention": {
                "weights": [0.35, 0.30, 0.25, 0.10],
                "ema_alpha": 0.25,
                "head_sigma_deg": 20.0,
                "gaze_zone_margin_px": 30,
                "blink_ear_threshold": 0.20,
            }
        }

    def test_full_attention_scenario(self):
        scorer = AttentionScorer(self._make_cfg())
        result = scorer.update(
            gaze_zone=GazeZone.BOOK,
            yaw=0.0, pitch=0.0,
            mean_ear=0.30,
            bboxes=[],
            book_roi=None,
            iris_cx=None, iris_cy=None,
        )
        assert result > 0.55

    def test_full_distraction_scenario(self):
        scorer = AttentionScorer(self._make_cfg())
        for _ in range(20):
            result = scorer.update(
                gaze_zone=GazeZone.AWAY,
                yaw=45.0, pitch=0.0,
                mean_ear=0.30,
                bboxes=[],
                book_roi=None,
                iris_cx=None, iris_cy=None,
            )
        assert result < 0.4

    def test_reset_returns_to_neutral(self):
        scorer = AttentionScorer(self._make_cfg())
        for _ in range(20):
            scorer.update(GazeZone.AWAY, 45.0, 0.0, 0.30, [], None, None, None)
        scorer.reset()
        assert scorer.filtered == pytest.approx(0.5)
