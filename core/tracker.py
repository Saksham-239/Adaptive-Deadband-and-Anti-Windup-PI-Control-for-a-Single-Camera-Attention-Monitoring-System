"""
core/tracker.py
Dynamic Book Tracker using OpenCV CSRT with NCC appearance quality scoring.

Architecture:
  - Initialized on user-calibrated book ROI (x1, y1, x2, y2).
  - CSRT tracks frame-to-frame position dynamically without aspect-ratio gating.
  - Normalized Cross-Correlation (NCC) against the calibrated template provides
    a continuous quality metric Q in [-1.0, 1.0].
  - When Q < q_thresh for >= absence_timeout_sec, the book is declared absent.
  - YOLO proposals are strictly used for cold-start / reacquisition when the
    tracker has lost the book, never to continuously gate or override valid CSRT tracks.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from core.detector import BBox

logger = logging.getLogger("tracker")


class BookTracker:
    def __init__(
        self,
        q_thresh: float = 0.35,
        color_thresh: float = 0.25,
        high_conf_thresh: float = 0.50,
        absence_timeout_sec: float = 1.5,
        reacquisition_conf: float = 0.12,
    ) -> None:
        self.q_thresh = q_thresh
        self.color_thresh = color_thresh
        self.high_conf_thresh = high_conf_thresh
        self.absence_timeout_sec = absence_timeout_sec
        self.reacquisition_conf = reacquisition_conf

        self._tracker: Optional[cv2.Tracker] = None
        self._template: Optional[np.ndarray] = None  # Grayscale template image
        self._template_size: Optional[Tuple[int, int]] = None  # (w, h)
        self._template_hist: Optional[np.ndarray] = None  # Normalized 2D H-S histogram

        self.is_initialized: bool = False
        self.is_present: bool = False
        self.current_roi: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
        self.last_q: float = 0.0
        self.last_q_color: float = 0.0
        self._last_good_time: float = 0.0

    @staticmethod
    def compute_hsv_hist(crop_bgr: np.ndarray, h_bins: int = 30, s_bins: int = 32) -> Optional[np.ndarray]:
        """Compute normalized 2D Hue-Saturation histogram from a BGR crop."""
        if crop_bgr is None or crop_bgr.size == 0 or len(crop_bgr.shape) < 3 or crop_bgr.shape[2] != 3:
            return None
        try:
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        except Exception:
            return None

    def set_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Initialize the CSRT tracker and store reference appearance templates
        (grayscale NCC template + HSV color histogram).
        ROI format: (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = roi
        h_f, w_f = frame.shape[:2]

        x1 = max(0, min(x1, w_f - 1))
        y1 = max(0, min(y1, h_f - 1))
        x2 = max(0, min(x2, w_f))
        y2 = max(0, min(y2, h_f))

        w = x2 - x1
        h = y2 - y1
        if w < 10 or h < 10:
            logger.warning("Rejected set_roi: bbox too small (%dx%d)", w, h)
            return False

        bgr_crop = None
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bgr_crop = frame[y1:y2, x1:x2]
        else:
            gray = frame

        crop = gray[y1:y2, x1:x2]
        if crop.size == 0 or np.std(crop) < 1.0:
            logger.warning("Rejected set_roi: crop is uniform / zero-std")
            return False

        self._template = crop.copy()
        self._template_size = (w, h)
        if bgr_crop is not None:
            self._template_hist = self.compute_hsv_hist(bgr_crop)
        else:
            self._template_hist = None

        # Initialize CSRT tracker (OpenCV expects (x, y, w, h) as integers)
        csrt_bbox = (int(x1), int(y1), int(w), int(h))
        self._tracker = cv2.TrackerCSRT_create()
        self._tracker.init(frame, csrt_bbox)

        now = timestamp if timestamp is not None else time.monotonic()
        self.is_initialized = True
        self.is_present = True
        self.current_roi = (x1, y1, x2, y2)
        self.last_q = 1.0
        self.last_q_color = 1.0
        self._last_good_time = now

        logger.info("BookTracker initialized on ROI %s (template %dx%d)", self.current_roi, w, h)
        return True

    def compute_quality(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Compute Normalized Cross-Correlation (NCC) between the candidate crop and
        the calibrated reference template. Returns Q in [-1.0, 1.0].
        """
        if self._template is None or self._template_size is None:
            return 0.0

        h_f, w_f = frame.shape[:2]
        x1_c = max(0, min(x1, w_f - 1))
        y1_c = max(0, min(y1, h_f - 1))
        x2_c = max(0, min(x2, w_f))
        y2_c = max(0, min(y2, h_f))

        w = x2_c - x1_c
        h = y2_c - y1_c
        if w < 5 or h < 5:
            return 0.0

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        crop = gray[y1_c:y2_c, x1_c:x2_c]
        if crop.size == 0 or np.std(crop) < 1e-4:
            return 0.0

        target_w, target_h = self._template_size
        # Scale tolerance check: reject collapsed (<0.25x) or runaway (>3.0x) boxes
        if w < 0.25 * target_w or h < 0.25 * target_h or w > 3.0 * target_w or h > 3.0 * target_h:
            return 0.0

        # Resize candidate crop to match template dimensions for direct correlation
        if (w, h) != (target_w, target_h):
            crop_resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            crop_resized = crop

        try:
            res = cv2.matchTemplate(crop_resized, self._template, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            return float(np.clip(score, -1.0, 1.0))
        except Exception:
            return 0.0

    def compute_color_quality(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Compute HSV Hue-Saturation histogram correlation between the candidate crop
        and the calibrated reference template. Returns Q_color in [-1.0, 1.0].
        """
        if self._template_hist is None or len(frame.shape) < 3:
            return 0.0

        h_f, w_f = frame.shape[:2]
        x1_c = max(0, min(x1, w_f - 1))
        y1_c = max(0, min(y1, h_f - 1))
        x2_c = max(0, min(x2, w_f))
        y2_c = max(0, min(y2, h_f))

        w = x2_c - x1_c
        h = y2_c - y1_c
        if w < 5 or h < 5:
            return 0.0

        crop = frame[y1_c:y2_c, x1_c:x2_c]
        cand_hist = self.compute_hsv_hist(crop)
        if cand_hist is None:
            return 0.0

        try:
            score = float(cv2.compareHist(self._template_hist, cand_hist, cv2.HISTCMP_CORREL))
            return float(np.clip(score, -1.0, 1.0))
        except Exception:
            return 0.0

    def check_presence(self, q_ncc: float, q_color: float) -> bool:
        """
        Tiered presence formula:
        is_present_this_frame = (Q_ncc >= high_conf_thresh) OR (Q_ncc >= q_thresh AND Q_color >= color_thresh)
        """
        return (q_ncc >= self.high_conf_thresh) or (q_ncc >= self.q_thresh and q_color >= self.color_thresh)

    def update(
        self,
        frame: np.ndarray,
        yolo_bboxes: Optional[list[BBox]] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]], float]:
        """
        Update the tracker with the incoming frame.
        Returns:
            (is_present, current_roi, Q)
        where current_roi is (x1, y1, x2, y2) or None if absent.
        """
        now = timestamp if timestamp is not None else time.monotonic()

        if not self.is_initialized or self._tracker is None:
            return False, None, 0.0

        # Step 1: Query CSRT tracker (only while target is actively present)
        if self.is_present:
            ok, csrt_box = self._tracker.update(frame)
            q_ncc = 0.0
            q_color = 0.0

            if ok:
                bx, by, bw, bh = [int(v) for v in csrt_box]
                h_f, w_f = frame.shape[:2]
                x1 = max(0, min(bx, w_f - 1))
                y1 = max(0, min(by, h_f - 1))
                x2 = max(0, min(bx + bw, w_f))
                y2 = max(0, min(by + bh, h_f))

                q_ncc = self.compute_quality(frame, x1, y1, x2, y2)
                q_color = self.compute_color_quality(frame, x1, y1, x2, y2)
                self.last_q = q_ncc
                self.last_q_color = q_color

                is_present_this_frame = self.check_presence(q_ncc, q_color)

                if is_present_this_frame:
                    # Strong track: target present and verified
                    self.current_roi = (x1, y1, x2, y2)
                    self._last_good_time = now
                else:
                    # Degraded / drifted / occluded: trust CSRT position during grace period
                    self.current_roi = (x1, y1, x2, y2)
                    elapsed = now - self._last_good_time
                    if elapsed >= self.absence_timeout_sec:
                        self.is_present = False
                        self.current_roi = None
            else:
                # CSRT internal failure
                self.last_q = 0.0
                self.last_q_color = 0.0
                elapsed = now - self._last_good_time
                if elapsed >= self.absence_timeout_sec:
                    self.is_present = False
                    self.current_roi = None
        else:
            # Target absent: CSRT is dormant/unanchored; do not self-recover on background clutter
            self.last_q = 0.0
            self.last_q_color = 0.0
            self.current_roi = None

        # Step 2: YOLO Reacquisition (strictly when absent or lost)
        if not self.is_present and yolo_bboxes:
            best_cand: Optional[Tuple[int, int, int, int]] = None
            best_cand_q = -1.0
            best_cand_color = -1.0

            for b in yolo_bboxes:
                if b.label == "book" and b.confidence >= self.reacquisition_conf:
                    cand_roi = (int(b.x1), int(b.y1), int(b.x2), int(b.y2))
                    cand_q_ncc = self.compute_quality(frame, *cand_roi)
                    cand_q_color = self.compute_color_quality(frame, *cand_roi)
                    if self.check_presence(cand_q_ncc, cand_q_color) and cand_q_ncc > best_cand_q:
                        best_cand_q = cand_q_ncc
                        best_cand_color = cand_q_color
                        best_cand = cand_roi

            if best_cand is not None:
                # Re-initialize CSRT tracker on the verified YOLO candidate
                cx1, cy1, cx2, cy2 = best_cand
                cw = cx2 - cx1
                ch = cy2 - cy1
                self._tracker = cv2.TrackerCSRT_create()
                self._tracker.init(frame, (int(cx1), int(cy1), int(cw), int(ch)))
                self.is_present = True
                self.current_roi = best_cand
                self.last_q = best_cand_q
                self.last_q_color = best_cand_color
                self._last_good_time = now
                logger.info("Book reacquired via YOLO proposal at %s (Q_ncc=%.2f, Q_color=%.2f)", best_cand, best_cand_q, best_cand_color)

        return self.is_present, self.current_roi, self.last_q

    def reset(self) -> None:
        """Reset the tracker to uncalibrated state."""
        self._tracker = None
        self._template = None
        self._template_size = None
        self._template_hist = None
        self.is_initialized = False
        self.is_present = False
        self.current_roi = None
        self.last_q = 0.0
        self.last_q_color = 0.0
        self._last_good_time = 0.0
