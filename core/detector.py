"""
core/detector.py
YOLOv8n object detector running on a background thread.

Design:
  - Main thread creates Detector and calls start().
  - Internally runs a loop consuming frames from frame_q (Queue, maxsize=1).
  - Writes results into detection_cache under detection_lock.
  - Main thread reads detection_cache at any time (lock-protected, non-blocking).
  - put_fresh() utility ensures stale frames are evicted before inserting new ones.

Thread safety:
  - frame_q: thread-safe by construction (Queue).
  - detection_cache: protected by detection_lock (Lock).
  - No shared mutable state outside these two primitives.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO class names for the IDs we care about
_COCO_NAMES = {
    63: "laptop",
    67: "cell_phone",
    73: "book",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Single detected object bounding box."""
    label: str
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def contains_point(self, px: float, py: float, margin: int = 0) -> bool:
        """Return True if (px, py) is inside this box (with optional margin)."""
        return (
            self.x1 - margin <= px <= self.x2 + margin
            and self.y1 - margin <= py <= self.y2 + margin
        )


@dataclass
class DetectionCache:
    """Lock-protected latest YOLO detection results."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    bboxes: list[BBox] = field(default_factory=list)
    reacq_proposals: list[BBox] = field(default_factory=list)
    timestamp: float = 0.0

    def update(self, bboxes: list[BBox], reacq_proposals: Optional[list[BBox]] = None) -> None:
        with self.lock:
            self.bboxes = bboxes
            self.reacq_proposals = reacq_proposals if reacq_proposals is not None else []
            self.timestamp = time.monotonic()

    def read(self) -> tuple[list[BBox], float]:
        """Returns (bboxes, timestamp). Safe to call from any thread."""
        with self.lock:
            return list(self.bboxes), self.timestamp

    def read_reacq(self) -> tuple[list[BBox], float]:
        """Returns (reacq_proposals, timestamp). For BookTracker reacquisition only."""
        with self.lock:
            return list(self.reacq_proposals), self.timestamp


# ---------------------------------------------------------------------------
# put_fresh utility
# ---------------------------------------------------------------------------

def put_fresh(q: queue.Queue, item) -> None:
    """
    Evict any stale item from q, then insert item.
    Ensures the queue always holds the freshest frame, never drops the new one.
    """
    try:
        q.get_nowait()   # evict stale
    except queue.Empty:
        pass
    q.put_nowait(item)   # insert fresh (queue guaranteed empty now)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class Detector:
    """
    YOLOv8n detector running on a daemon thread.
    Usage:
        detector = Detector(cfg)
        detector.start()
        # in main loop:
        put_fresh(detector.frame_q, frame)
        bboxes, ts = detector.cache.read()
        detector.stop()
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self.frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.cache = DetectionCache()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model = None

        # Config values
        self._model_path    = cfg["yolo"]["model_path"]
        self._conf_threshs  = cfg["yolo"]["confidence_thresholds"]
        self._iou_thresh    = cfg["yolo"]["iou_threshold"]
        self._classes       = cfg["yolo"]["classes_of_interest"]
        self._target_fps    = cfg["yolo"]["target_fps"]
        self._min_interval  = 1.0 / max(self._target_fps, 1)
        self._reacq_conf    = cfg.get("book_tracker", {}).get("reacquisition_conf", 0.12)
        # Compute model inference floor dynamically from all active thresholds in config
        all_thresholds      = list(self._conf_threshs.values()) + [self._reacq_conf]
        self._model_floor   = float(min(all_thresholds))

    def start(self) -> None:
        """Load YOLO model and start background inference thread."""
        self._load_model()
        self._thread = threading.Thread(
            target=self._run, name="yolo-detector", daemon=True
        )
        self._thread.start()
        logger.info("Detector thread started (target %.1f Hz).", self._target_fps)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("Detector thread stopped.")

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            # Warm up on a dummy frame to avoid first-frame latency spike
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            logger.info("YOLO model loaded: %s  device=%s", self._model_path, self._model.device)
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise

    def _run(self) -> None:
        """Background inference loop. Runs at target_fps."""
        last_run = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            # Rate-limit to target_fps
            elapsed = now - last_run
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
                continue

            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            normal_bboxes, reacq_proposals = self._infer(frame)
            self.cache.update(normal_bboxes, reacq_proposals)
            last_run = time.monotonic()

    def _infer(self, frame: np.ndarray) -> tuple[list[BBox], list[BBox]]:
        """
        Run YOLO inference on frame.
        Returns:
            (normal_bboxes, reacq_proposals)
            - normal_bboxes: strictly gated by cfg['confidence_thresholds'] (phone, book >= 0.35)
            - reacq_proposals: loose candidate proposals (book only, >= reacq_conf) for BookTracker
        """
        try:
            results = self._model(
                frame,
                conf=self._model_floor,  # Dynamically computed from config
                iou=self._iou_thresh,
                classes=self._classes,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
            return [], []

        normal_bboxes: list[BBox] = []
        reacq_proposals: list[BBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = _COCO_NAMES.get(cls_id, f"cls_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = BBox(
                    label=label, class_id=cls_id, confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
                
                # Standard authority gate (e.g. phone >= 0.75, book >= 0.35)
                req_conf = self._conf_threshs.get(label, 0.50)
                if conf >= req_conf:
                    normal_bboxes.append(bbox)

                # Reacquisition proposal gate (book only, >= reacq_conf e.g. 0.12)
                if label == "book" and conf >= self._reacq_conf:
                    reacq_proposals.append(bbox)

        # Sort by confidence descending
        normal_bboxes.sort(key=lambda b: b.confidence, reverse=True)
        reacq_proposals.sort(key=lambda b: b.confidence, reverse=True)
        return normal_bboxes, reacq_proposals
