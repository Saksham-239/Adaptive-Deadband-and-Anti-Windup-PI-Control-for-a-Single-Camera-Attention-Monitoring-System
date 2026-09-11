"""
core/attention.py
Attention scorer and EMA temporal filter — PURE FUNCTIONS.

No side effects, no global state. All inputs are explicit arguments.
This makes every function unit-testable with synthetic data.

Attention score formula:
  A_raw = w1*gaze_score + w2*head_score + w3*context_score + w4*blink_score
  All components ∈ [0.0, 1.0].  Weights sum to 1.0.
  A_filtered[k] = α * A_raw[k] + (1-α) * A_filtered[k-1]

Context score rules (context_score):
  - YOLO detects 'book' under gaze → 1.0
  - YOLO detects 'laptop' under gaze → 0.5
  - YOLO detects 'cell_phone' under gaze → 0.0
  - Gaze in calibrated desk_zone, no YOLO match → 0.7  (pen/paper implied)
  - Gaze in away_zone → 0.2
  - No YOLO data yet (cache empty) → 0.5  (neutral)
"""

from __future__ import annotations

import math
import time
from typing import Optional

from core.detector import BBox
from core.vision import GazeZone, OneEuroFilter


# ---------------------------------------------------------------------------
# Component scorers (pure functions)
# ---------------------------------------------------------------------------

def gaze_score(
    gaze_zone: str,
    book_roi: Optional[tuple[int, int, int, int]],
    iris_cx: Optional[float],
    iris_cy: Optional[float],
    margin_px: int = 30,
) -> float:
    """
    Score ∈ [0, 1] based on whether gaze is directed at study material.

    If a book ROI is calibrated and iris coords are available, uses
    spatial overlap (1.0 inside ROI+margin, 0 if far outside, linear falloff).
    Otherwise falls back to zone-based classification.
    """
    if book_roi is not None and iris_cx is not None and iris_cy is not None:
        x1, y1, x2, y2 = book_roi
        # Expand ROI by margin
        rx1, ry1 = x1 - margin_px, y1 - margin_px
        rx2, ry2 = x2 + margin_px, y2 + margin_px
        # Check if inside
        if rx1 <= iris_cx <= rx2 and ry1 <= iris_cy <= ry2:
            return 1.0
        # Linear falloff from edge to 2× margin outside
        dist_x = max(rx1 - iris_cx, 0, iris_cx - rx2)
        dist_y = max(ry1 - iris_cy, 0, iris_cy - ry2)
        dist   = math.sqrt(dist_x ** 2 + dist_y ** 2)
        roi_diag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 1e-6
        return float(max(0.0, 1.0 - dist / (roi_diag * 0.5)))

    # Zone-based fallback
    zone_scores = {
        GazeZone.BOOK:    1.0,
        GazeZone.SCREEN:  0.6,   # head straight — neutral study posture
        GazeZone.DESK:    0.8,   # looking down at desk — actively studying
        GazeZone.AWAY:    0.0,
        GazeZone.UNKNOWN: 0.3,
    }
    return zone_scores.get(gaze_zone, 0.3)


def head_score(yaw: float, pitch: float, sigma_deg: float = 20.0) -> float:
    """
    Score ∈ [0, 1] based on head orientation (after axis remapping).
    
    With remapped axes:
      yaw:   left/right turn (-48 left, +41 right, 0 straight)
      pitch: up/down (+320 down at desk, -26 up at ceiling, 0 straight)
    
    Does NOT penalize looking down (positive pitch) since that's studying.
    Only penalizes yaw (looking sideways) and negative pitch (looking up).
    """
    effective_pitch = min(pitch, 0.0)  # don't penalize looking down
    exponent = (yaw ** 2 + effective_pitch ** 2) / (2.0 * sigma_deg ** 2)
    return float(math.exp(-exponent))


def context_score(
    gaze_zone: str,
    bboxes: list[BBox],
    iris_cx: Optional[float],
    iris_cy: Optional[float],
    margin_px: int = 20,
) -> float:
    """
    Score ∈ [0, 1] based on what YOLO detected under the gaze point.

    Priority order: cell_phone (override 0.0) > book (1.0) > laptop (0.5).
    If gaze is in desk_zone but no YOLO object matches → 0.7 (pen/paper implied).
    COCO has no pen/paper class — this is zone-based, not class-based.
    """
    if gaze_zone == GazeZone.DESK and not bboxes:
        return 0.7  # pen/paper implied when looking at desk with no detections

    if not bboxes:
        # No detections yet (YOLO cache empty at startup)
        return 0.5  # neutral — don't penalise startup

    # If we have iris position, find which YOLO box contains it
    matched_label: Optional[str] = None
    if iris_cx is not None and iris_cy is not None:
        for bbox in bboxes:
            if bbox.contains_point(iris_cx, iris_cy, margin=margin_px):
                # phone is max priority — immediately return 0.0
                if bbox.label == "cell_phone":
                    return 0.0
                matched_label = bbox.label
                break  # take first match (sorted by YOLO confidence)

    if matched_label == "book":
        return 1.0
    if matched_label == "laptop":
        return 0.5

    # No YOLO object under gaze — use zone
    if gaze_zone == GazeZone.DESK:
        return 0.7  # pen/paper implied
    if gaze_zone == GazeZone.AWAY:
        return 0.2

    return 0.4  # default when looking at nothing detected


def blink_score(mean_ear: float, ear_threshold: float = 0.20) -> float:
    """
    Score ∈ [0, 1]. Returns 1.0 if eyes are open, 0.0 if closed.
    Smooth version: linear between (threshold-0.05) and (threshold+0.05).
    This is deliberately simple — v2 adds a rolling-window PERCLOS metric.
    """
    open_val  = ear_threshold + 0.05
    close_val = ear_threshold - 0.05
    if mean_ear >= open_val:
        return 1.0
    if mean_ear <= close_val:
        return 0.0
    return float((mean_ear - close_val) / (open_val - close_val))


def compute_raw_score(
    g_score: float,
    h_score: float,
    c_score: float,
    b_score: float,
    weights: list[float],
) -> float:
    """
    Weighted sum of the four components.
    Weights should sum to 1.0; function normalises defensively.
    Returns A_raw ∈ [0.0, 1.0].
    """
    w_total = sum(weights) or 1.0
    raw = (
        weights[0] * g_score +
        weights[1] * h_score +
        weights[2] * c_score +
        weights[3] * b_score
    ) / w_total
    return float(min(max(raw, 0.0), 1.0))


# ---------------------------------------------------------------------------
# AttentionScorer — stateful wrapper (holds EMA state + component history)
# ---------------------------------------------------------------------------

class AttentionScorer:
    """
    Stateful scorer that computes and tracks the attention score over time.
    Holds only EMA state; all computation delegates to pure functions above.
    """

    def __init__(self, cfg: dict):
        a_cfg = cfg["attention"]
        self._weights        = a_cfg["weights"]
        self._sigma_deg      = a_cfg["head_sigma_deg"]
        self._margin_px      = a_cfg["gaze_zone_margin_px"]
        self._ear_threshold  = a_cfg["blink_ear_threshold"]
        
        f_cfg = cfg.get("filters", {}).get("attn_score", {})
        mincutoff = f_cfg.get("mincutoff", 0.5)
        beta = f_cfg.get("beta", 0.05)
        self._filter = OneEuroFilter(mincutoff=mincutoff, beta=beta)
        self._filtered       = 0.5   # initial neutral score

        # Last computed components (for debug overlay)
        self.last_gaze_score:    float = 0.5
        self.last_head_score:    float = 0.5
        self.last_context_score: float = 0.5
        self.last_blink_score:   float = 0.5
        self.last_raw:           float = 0.5
        self.last_filtered:      float = 0.5

    def update(
        self,
        gaze_zone: str,
        yaw: float,
        pitch: float,
        mean_ear: float,
        bboxes: list[BBox],
        book_roi: Optional[tuple[int, int, int, int]],
        iris_cx: Optional[float],
        iris_cy: Optional[float],
    ) -> float:
        """
        Compute one step and return the filtered attention score.
        Call this once per main-loop iteration.
        """
        g = gaze_score(gaze_zone, book_roi, iris_cx, iris_cy, self._margin_px)
        h = head_score(yaw, pitch, self._sigma_deg)
        c = context_score(gaze_zone, bboxes, iris_cx, iris_cy, self._margin_px)
        b = blink_score(mean_ear, self._ear_threshold)

        raw      = compute_raw_score(g, h, c, b, self._weights)
        
        # 1€ Filter as observer: superior to simple EMA because it dynamically 
        # tunes cutoff frequency, crushing jitter during stability but responding 
        # instantly (without lag) to sudden drops in attention.
        filtered = self._filter(time.monotonic(), raw)

        self._filtered = filtered

        # Cache for debug overlay
        self.last_gaze_score    = g
        self.last_head_score    = h
        self.last_context_score = c
        self.last_blink_score   = b
        self.last_raw           = raw
        self.last_filtered      = filtered

        return filtered

    def reset(self) -> None:
        """Reset state (e.g. after BREAK state)."""
        self._filtered = 0.5
        self._filter.x_prev = 0.5
        self._filter.dx_prev = 0.0
        self._filter.t_prev = None

    @property
    def filtered(self) -> float:
        return self._filtered
