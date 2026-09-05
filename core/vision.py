"""
core/vision.py
MediaPipe Face Landmarker integration + head pose (solvePnP) + iris/gaze extraction.

Responsibilities:
  - Load and run MediaPipe Tasks Face Landmarker on a frame.
  - Compute head yaw/pitch/roll via solvePnP with approximated camera intrinsics.
  - Extract iris center position and eye-aspect-ratio (EAR) for blink detection.
  - Classify coarse gaze zone (book_zone / screen_zone / desk_zone / away_zone)
    based on head pose quadrant + iris deviation within the eye socket.
  - Returns a FaceResult dataclass. Returns None if no face is detected.

NOTE on GPU delegate: MediaPipe Python Tasks API on Windows may only expose the
CPU delegate. Test with BaseOptions(model_asset_path=..., delegate=GPU) at startup.
If it raises, fall back to CPU silently.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1€ Filter (One Euro Filter) for State Estimation
# ---------------------------------------------------------------------------
def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    """
    1€ Filter as an observer. Balances lag and jitter by dynamically
    adjusting the cutoff frequency based on the derivative of the signal.
    """
    def __init__(self, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None

    def __call__(self, t: float, x: float) -> float:
        # First call guard: initialize filter state directly with raw value (skip filtering).
        # This prevents NaN/inf on first run since dt is undefined.
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        # dt <= 0 guard (non-monotonic timestamps, e.g., from threaded capture):
        # We skip the update and hold the previous filtered value. 
        # We do this instead of clamping dt to epsilon because a tiny dt would 
        # create an artificial, massive spike in the derivative term (dx/dt -> inf),
        # permanently poisoning the recursive filter state.
        if t_e <= 0.0:
            return self.x_prev

        # The filtered derivative of the signal.
        a_d = _smoothing_factor(t_e, self.dcutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.components.containers import landmark as mp_landmark
except ImportError as e:
    raise ImportError(
        "mediapipe not installed. Run: pip install mediapipe==0.10.14"
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — 3D model face landmarks for solvePnP (in cm, origin at nose tip)
# Indices into MediaPipe's 478-landmark model.
# ---------------------------------------------------------------------------
_MP_6PT_INDICES = [1, 33, 263, 61, 291, 199]  # nose, l-eye, r-eye, l-mouth, r-mouth, chin

_FACE_3D_MODEL = np.array([
    [0.0,      0.0,      0.0   ],  # Nose tip (1)
    [-225.0,  170.0,   -135.0 ],  # Left eye left corner (33)
    [ 225.0,  170.0,   -135.0 ],  # Right eye right corner (263)
    [-150.0, -150.0,   -125.0 ],  # Left mouth corner (61)
    [ 150.0, -150.0,   -125.0 ],  # Right mouth corner (291)
    [0.0,    -330.0,   -65.0  ],  # Chin (199)
], dtype=np.float64)

# MediaPipe iris landmark indices
_LEFT_IRIS  = [474, 475, 476, 477]
_RIGHT_IRIS = [469, 470, 471, 472]

# EAR landmark indices (6 points per eye)
_LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GazeZone:
    BOOK   = "book_zone"
    SCREEN = "screen_zone"
    DESK   = "desk_zone"
    AWAY   = "away_zone"
    UNKNOWN = "unknown"


@dataclass
class FaceResult:
    """All per-frame outputs from the vision pipeline."""
    timestamp: float = 0.0

    # Head pose (degrees; positive yaw = turned right, positive pitch = looking down)
    yaw:   float = 0.0
    pitch: float = 0.0
    roll:  float = 0.0
    pose_valid: bool = False

    # Iris position in frame (pixel coords), left and right
    left_iris_center:  Optional[tuple[float, float]] = None
    right_iris_center: Optional[tuple[float, float]] = None

    # Eye-aspect ratio per eye (0 = closed, ~0.25–0.35 = open)
    left_ear:  float = 0.0
    right_ear: float = 0.0
    mean_ear:  float = 0.0

    # Gaze zone classification
    gaze_zone: str = GazeZone.UNKNOWN

    # Iris deviation within eye socket: normalized [-1, +1] (left/right, up/down)
    iris_deviation_x: float = 0.0
    iris_deviation_y: float = 0.0

    # Raw landmark array (shape [478, 3]) for downstream use
    landmarks_px: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Camera intrinsics approximation
# ---------------------------------------------------------------------------

def approx_camera_matrix(frame_w: int, frame_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate webcam intrinsics. No physical calibration needed.
    focal_length ≈ frame_width (standard approximation at typical webcam FOV ~60°).
    Error is ~3–5° for head pose at 0.5–1.5m — acceptable for zone classification.
    """
    f = float(frame_w)
    cx, cy = frame_w / 2.0, frame_h / 2.0
    cam_matrix = np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume no distortion
    return cam_matrix, dist_coeffs


# ---------------------------------------------------------------------------
# Helper functions (pure, testable)
# ---------------------------------------------------------------------------

def _ear(landmarks_px: np.ndarray, indices: list[int]) -> float:
    """Eye Aspect Ratio from 6 landmark points."""
    pts = landmarks_px[indices]
    # Vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def _iris_center(landmarks_px: np.ndarray, indices: list[int]) -> tuple[float, float]:
    """Mean position of iris landmarks."""
    pts = landmarks_px[indices, :2]
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])


def _iris_deviation(
    iris_cx: float, iris_cy: float,
    eye_indices: list[int],
    landmarks_px: np.ndarray
) -> tuple[float, float]:
    """
    Normalized iris deviation within the eye socket.
    Returns (dev_x, dev_y) each in [-1, +1].
    dev_x > 0 means iris is to the right of eye center.
    """
    pts = landmarks_px[eye_indices, :2]
    eye_min = pts.min(axis=0)
    eye_max = pts.max(axis=0)
    eye_center = (eye_min + eye_max) / 2.0
    eye_span   = (eye_max - eye_min) / 2.0 + 1e-6  # half-width, half-height
    dev_x = (iris_cx - eye_center[0]) / eye_span[0]
    dev_y = (iris_cy - eye_center[1]) / eye_span[1]
    return float(np.clip(dev_x, -1, 1)), float(np.clip(dev_y, -1, 1))


def classify_gaze_zone(
    yaw: float,
    pitch: float,
    iris_dev_x: float,
    iris_dev_y: float,
    book_roi: Optional[tuple[int, int, int, int]],  # (x1, y1, x2, y2) or None
    frame_w: int,
    frame_h: int,
    yaw_threshold: float = 25.0,
    pitch_down_threshold: float = 15.0,
) -> str:
    """
    Coarse gaze zone classification from head pose + iris deviation.

    After axis remapping (see _compute_head_pose):
      yaw:   left/right turn.  ~0 = straight. -48 = left, +41 = right.
      pitch: up/down tilt.     ~0 = straight. +320 = looking down. -26 = looking up.

    Zones:
      desk_zone   — head pitched down (studying on desk)
      away_zone   — head turned sideways or tilted up
      screen_zone — head roughly centered (default neutral posture)
      book_zone   — head centered + calibrated book ROI present
    """
    # Strong sideways head turn → away
    if abs(yaw) > yaw_threshold:
        return GazeZone.AWAY

    # Looking up at ceiling (negative pitch) → away
    if pitch < -15.0:
        return GazeZone.AWAY

    # Looking down at desk (positive pitch) → desk
    if pitch > pitch_down_threshold:
        return GazeZone.DESK

    # Significant eye deviation without head turn → away (side-eye)
    if abs(iris_dev_x) > 0.55:
        return GazeZone.AWAY

    # If a book ROI has been calibrated, centered gaze = book zone
    if book_roi is not None:
        return GazeZone.BOOK

    # Default: head centered, not looking down or up → screen
    return GazeZone.SCREEN


# ---------------------------------------------------------------------------
# VisionProcessor class
# ---------------------------------------------------------------------------

class VisionProcessor:
    """
    Wraps MediaPipe Face Landmarker.
    Call process_frame(frame) → FaceResult | None.
    """

    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self._cfg = cfg
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._cam_matrix, self._dist_coeffs = approx_camera_matrix(frame_w, frame_h)
        self._book_roi: Optional[tuple[int, int, int, int]] = None

        f_cfg = cfg.get("filters", {}).get("head_pose", {})
        mincutoff = f_cfg.get("mincutoff", 1.0)
        beta = f_cfg.get("beta", 0.007)
        self._yaw_filter = OneEuroFilter(mincutoff=mincutoff, beta=beta)
        self._pitch_filter = OneEuroFilter(mincutoff=mincutoff, beta=beta)
        self._roll_filter = OneEuroFilter(mincutoff=mincutoff, beta=beta)

        model_path = cfg["mediapipe"]["model_asset_path"]
        det_conf   = cfg["mediapipe"]["min_detection_confidence"]
        pres_conf  = cfg["mediapipe"]["min_presence_confidence"]
        track_conf = cfg["mediapipe"]["min_tracking_confidence"]

        def _make_landmarker(use_gpu: bool) -> mp_vision.FaceLandmarker:
            delegate = (mp_python.BaseOptions.Delegate.GPU if use_gpu
                        else mp_python.BaseOptions.Delegate.CPU)
            base_opts = mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=delegate,
            )
            face_opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=cfg["mediapipe"]["num_faces"],
                min_face_detection_confidence=det_conf,
                min_face_presence_confidence=pres_conf,
                min_tracking_confidence=track_conf,
            )
            return mp_vision.FaceLandmarker.create_from_options(face_opts)

        # Try GPU first; the error only surfaces at create_from_options, not at
        # BaseOptions construction, so the try/except must wrap the whole call.
        try:
            self._landmarker = _make_landmarker(use_gpu=True)
            logger.info("MediaPipe: running on GPU delegate.")
        except Exception as exc:
            logger.warning("MediaPipe GPU delegate failed (%s). Falling back to CPU.", exc)
            self._landmarker = _make_landmarker(use_gpu=False)
            logger.info("MediaPipe: running on CPU delegate.")

        logger.info("VisionProcessor initialised (frame %dx%d).", frame_w, frame_h)

    @staticmethod
    def _make_base_options(model_path: str) -> mp_python.BaseOptions:
        """Try GPU delegate first; fall back to CPU if unavailable on Windows."""
        try:
            opts = mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.GPU,
            )
            logger.info("MediaPipe: GPU delegate requested.")
            return opts
        except Exception as exc:
            logger.warning("MediaPipe GPU delegate unavailable (%s). Using CPU.", exc)
            return mp_python.BaseOptions(model_asset_path=model_path)

    def set_book_roi(self, roi: tuple[int, int, int, int]) -> None:
        """Set the calibrated book region of interest (x1, y1, x2, y2)."""
        self._book_roi = roi
        logger.info("Book ROI set: %s", roi)

    def process_frame(self, frame: np.ndarray) -> Optional[FaceResult]:
        """
        Run Face Landmarker on frame (BGR, HxWxC).
        Returns FaceResult or None if no face detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection = self._landmarker.detect(mp_image)
        if not detection.face_landmarks:
            return None

        # Use first face only
        lm_norm = detection.face_landmarks[0]
        h, w = frame.shape[:2]

        # Convert normalised landmarks → pixel coords (Nx3)
        lm_px = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in lm_norm],
            dtype=np.float32,
        )

        result = FaceResult(
            timestamp=time.monotonic(),
            landmarks_px=lm_px,
        )

        # --- Head pose via solvePnP ---
        self._compute_head_pose(lm_px, result)

        # --- Iris centers ---
        lx, ly = _iris_center(lm_px, _LEFT_IRIS)
        rx, ry = _iris_center(lm_px, _RIGHT_IRIS)
        result.left_iris_center  = (lx, ly)
        result.right_iris_center = (rx, ry)

        # Average iris center
        iris_cx = (lx + rx) / 2.0
        iris_cy = (ly + ry) / 2.0

        # --- Iris deviation ---
        dev_x_l, dev_y_l = _iris_deviation(lx, ly, _LEFT_EYE_EAR, lm_px)
        dev_x_r, dev_y_r = _iris_deviation(rx, ry, _RIGHT_EYE_EAR, lm_px)
        result.iris_deviation_x = (dev_x_l + dev_x_r) / 2.0
        result.iris_deviation_y = (dev_y_l + dev_y_r) / 2.0

        # --- EAR / blink ---
        result.left_ear  = _ear(lm_px, _LEFT_EYE_EAR)
        result.right_ear = _ear(lm_px, _RIGHT_EYE_EAR)
        result.mean_ear  = (result.left_ear + result.right_ear) / 2.0

        # --- Gaze zone classification ---
        result.gaze_zone = classify_gaze_zone(
            yaw=result.yaw,
            pitch=result.pitch,
            iris_dev_x=result.iris_deviation_x,
            iris_dev_y=result.iris_deviation_y,
            book_roi=self._book_roi,
            frame_w=w,
            frame_h=h,
        )

        return result

    def _compute_head_pose(self, lm_px: np.ndarray, result: FaceResult) -> None:
        """Fill yaw/pitch/roll in result via solvePnP."""
        img_pts = lm_px[_MP_6PT_INDICES, :2].astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            _FACE_3D_MODEL,
            img_pts,
            self._cam_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            result.pose_valid = False
            return

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix → raw Euler angles
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            raw_roll  = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            raw_pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            raw_yaw   = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            raw_roll  = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            raw_pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            raw_yaw   = 0.0

        # REMAPPING based on real diagnostic data:
        #   raw_pitch = real-world yaw (left/right turn: -48 left, +41 right)
        #   raw_roll  = real-world pitch (up/down: +170 straight, +144 up, -150 down)
        
        def normalize_angle(angle, offset=0.0):
            # Normalizes angle relative to offset to [-180, 180]
            return ((angle - offset + 180.0) % 360.0) - 180.0

        # 1€ Filter as observer to smooth jitter while preserving dynamic responsiveness
        t = result.timestamp
        result.yaw   = self._yaw_filter(t, normalize_angle(raw_pitch, offset=-4.0))
        result.pitch = self._pitch_filter(t, normalize_angle(raw_roll, offset=170.0))
        result.roll  = self._roll_filter(t, normalize_angle(raw_yaw, offset=0.0))
        
        result.pose_valid = True

    def close(self) -> None:
        self._landmarker.close()
