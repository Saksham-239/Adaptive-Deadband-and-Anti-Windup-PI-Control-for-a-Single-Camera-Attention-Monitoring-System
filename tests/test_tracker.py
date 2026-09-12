"""
tests/test_tracker.py
Unit tests for BookTracker (OpenCV CSRT + NCC quality scoring + YOLO reacquisition).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import cv2

from core.detector import BBox
from core.tracker import BookTracker


@pytest.fixture
def test_scene():
    """Create a realistic canvas using the real calibrated notebook crop."""
    crop_path = os.path.join(os.path.dirname(__file__), "..", "scratch", "crop_box2.png")
    book_patch = cv2.imread(crop_path)
    if book_patch is None:
        # Fallback if image not on disk
        book_patch = np.full((80, 280, 3), 200, dtype=np.uint8)
        cv2.putText(book_patch, "STUDY NOTEBOOK", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    h, w = book_patch.shape[:2]
    canvas = np.full((480, 640, 3), 120, dtype=np.uint8)
    x1, y1 = 100, 150
    x2, y2 = x1 + w, y1 + h
    canvas[y1:y2, x1:x2] = book_patch
    return canvas, (x1, y1, x2, y2)


class TestBookTracker:
    def test_initialization(self, test_scene):
        canvas, roi = test_scene
        tracker = BookTracker(q_thresh=0.35, absence_timeout_sec=1.5)
        ok = tracker.set_roi(canvas, roi, timestamp=100.0)

        assert ok is True
        assert tracker.is_initialized is True
        assert tracker.is_present is True
        assert tracker.current_roi == roi
        assert tracker.last_q == pytest.approx(1.0)

    def test_exact_frame_high_quality(self, test_scene):
        canvas, roi = test_scene
        tracker = BookTracker(q_thresh=0.35, absence_timeout_sec=1.5)
        tracker.set_roi(canvas, roi, timestamp=100.0)

        present, box, q = tracker.update(canvas, timestamp=100.1)
        assert present is True
        assert box is not None
        assert q > 0.95
        assert abs(box[0] - roi[0]) <= 2
        assert abs(box[1] - roi[1]) <= 2

    def test_shifted_motion_tracking(self, test_scene):
        canvas, roi = test_scene
        tracker = BookTracker(q_thresh=0.35, absence_timeout_sec=1.5)
        tracker.set_roi(canvas, roi, timestamp=100.0)

        x1, y1, x2, y2 = roi
        book_patch = canvas[y1:y2, x1:x2].copy()

        # Shift target by +15px in X and +10px in Y
        shift_x, shift_y = 15, 10
        shifted_canvas = np.full((480, 640, 3), 120, dtype=np.uint8)
        new_x1, new_y1 = x1 + shift_x, y1 + shift_y
        new_x2, new_y2 = x2 + shift_x, y2 + shift_y
        shifted_canvas[new_y1:new_y2, new_x1:new_x2] = book_patch

        present, box, q = tracker.update(shifted_canvas, timestamp=100.1)

        assert present is True
        assert box is not None
        assert q > 0.85
        assert abs(box[0] - new_x1) <= 3
        assert abs(box[1] - new_y1) <= 3

    def test_no_aspect_ratio_gate(self):
        """Verify wide or narrow books are tracked without arbitrary aspect ratio rejections."""
        canvas = np.full((480, 640, 3), 100, dtype=np.uint8)
        # Narrow book: 240x60 (aspect ratio 4:1)
        roi = (100, 100, 340, 160)
        x1, y1, x2, y2 = roi
        w, h = x2 - x1, y2 - y1
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                patch[y, x] = [int(200 * (x / w)), int(150 * (y / h)), 180]
        cv2.putText(patch, "STUDY NOTEBOOK", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        canvas[y1:y2, x1:x2] = patch

        tracker = BookTracker(q_thresh=0.35, absence_timeout_sec=1.5)
        ok = tracker.set_roi(canvas, roi, timestamp=100.0)
        assert ok is True

        present, box, q = tracker.update(canvas, timestamp=100.1)
        assert present is True
        assert box is not None
        assert q > 0.90

    def test_absence_timeout_on_target_removal(self, test_scene):
        canvas, roi = test_scene
        tracker = BookTracker(q_thresh=0.35, absence_timeout_sec=1.5)
        tracker.set_roi(canvas, roi, timestamp=100.0)

        # Book is removed: canvas replaced with uniform desk / noise
        empty_desk = np.full((480, 640, 3), 120, dtype=np.uint8)

        # Frame 1 at t=100.5s (within 1.5s timeout):
        # Q drops below threshold, but timeout hasn't expired yet
        present, box, q = tracker.update(empty_desk, timestamp=100.5)
        assert q < 0.35
        assert present is True  # still holding within timeout

        # Frame 2 at t=101.6s (> 1.5s elapsed with low Q):
        present, box, q = tracker.update(empty_desk, timestamp=101.6)
        assert present is False
        assert box is None
        assert q < 0.35

    def test_yolo_reacquisition_after_absence(self, test_scene):
        canvas, roi = test_scene
        tracker = BookTracker(q_thresh=0.35, absence_timeout_sec=1.5)
        tracker.set_roi(canvas, roi, timestamp=100.0)

        empty_desk = np.full((480, 640, 3), 120, dtype=np.uint8)

        # Force absence
        tracker.update(empty_desk, timestamp=102.0)
        assert tracker.is_present is False

        # Now book reappears at a new position (e.g. shifted +50px)
        new_canvas = np.full((480, 640, 3), 120, dtype=np.uint8)
        nx1, ny1, nx2, ny2 = roi[0] + 50, roi[1] + 50, roi[2] + 50, roi[3] + 50
        new_canvas[ny1:ny2, nx1:nx2] = canvas[roi[1]:roi[3], roi[0]:roi[2]]  # copy book patch

        yolo_proposal = BBox(
            label="book",
            class_id=73,
            confidence=0.45,
            x1=float(nx1),
            y1=float(ny1),
            x2=float(nx2),
            y2=float(ny2),
        )

        present, box, q = tracker.update(new_canvas, yolo_bboxes=[yolo_proposal], timestamp=103.0)
        assert present is True
        assert box is not None
        assert q > 0.85
        assert abs(box[0] - nx1) <= 3
        assert abs(box[1] - ny1) <= 3

    def test_yolo_reacquisition_ignores_unrelated_boxes(self, test_scene):
        canvas, roi = test_scene
        tracker = BookTracker(q_thresh=0.35, absence_timeout_sec=1.5)
        tracker.set_roi(canvas, roi, timestamp=100.0)

        empty_desk = np.full((480, 640, 3), 120, dtype=np.uint8)
        tracker.update(empty_desk, timestamp=102.0)
        assert tracker.is_present is False

        # Proposal with low Q or wrong label
        unrelated_proposal = BBox(
            label="cell_phone",
            class_id=67,
            confidence=0.85,
            x1=10.0,
            y1=10.0,
            x2=50.0,
            y2=100.0,
        )
        present, box, q = tracker.update(empty_desk, yolo_bboxes=[unrelated_proposal], timestamp=103.0)
        assert present is False
        assert box is None

    def test_tiered_presence_formula_direct(self):
        """Verify the exact formula: (Q_ncc >= high_conf_thresh) OR (Q_ncc >= q_thresh AND Q_color >= color_thresh)."""
        tracker = BookTracker(q_thresh=0.35, color_thresh=0.25, high_conf_thresh=0.50)

        # High NCC alone passes unconditionally (even with zero or negative color)
        assert tracker.check_presence(0.50, 0.0) is True
        assert tracker.check_presence(0.85, -0.1) is True

        # Marginal NCC requires color >= 0.25
        assert tracker.check_presence(0.40, 0.30) is True
        assert tracker.check_presence(0.35, 0.25) is True

        # Marginal NCC with low color is vetoed (shirt collar case)
        assert tracker.check_presence(0.48, 0.15) is False
        assert tracker.check_presence(0.38, 0.20) is False

        # Low NCC fails even if color is perfect (cover / desk case)
        assert tracker.check_presence(0.30, 0.95) is False
        assert tracker.check_presence(0.15, 0.80) is False

    def test_yolo_reacquisition_rejects_marginal_ncc_with_bad_color(self, test_scene):
        """Verify reacquisition rejects candidate that fails the tiered appearance gate."""
        canvas, roi = test_scene
        tracker = BookTracker(q_thresh=0.35, color_thresh=0.25, high_conf_thresh=0.50)
        tracker.set_roi(canvas, roi, timestamp=100.0)

        empty_desk = np.full((480, 640, 3), 120, dtype=np.uint8)
        tracker.update(empty_desk, timestamp=102.0)
        assert tracker.is_present is False

        # Mock compute_quality to return marginal NCC (0.45) and mock color to return 0.10 (collar drift)
        original_ncc = tracker.compute_quality
        original_col = tracker.compute_color_quality
        try:
            tracker.compute_quality = lambda frame, x1, y1, x2, y2: 0.45
            tracker.compute_color_quality = lambda frame, x1, y1, x2, y2: 0.10

            collar_proposal = BBox(
                label="book",
                class_id=73,
                confidence=0.40,
                x1=100.0,
                y1=100.0,
                x2=200.0,
                y2=200.0,
            )
            present, box, q = tracker.update(empty_desk, yolo_bboxes=[collar_proposal], timestamp=103.0)
            assert present is False
            assert box is None

            # Now mock good color (0.50) -> should be accepted
            tracker.compute_color_quality = lambda frame, x1, y1, x2, y2: 0.50
            present, box, q = tracker.update(empty_desk, yolo_bboxes=[collar_proposal], timestamp=103.1)
            assert present is True
            assert box is not None
        finally:
            tracker.compute_quality = original_ncc
            tracker.compute_color_quality = original_col
