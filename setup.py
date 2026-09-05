"""
setup.py
One-time setup helper:
  1. Downloads the MediaPipe Face Landmarker model to models/
  2. Validates PyTorch CUDA is active
  3. Validates YOLOv8n auto-download (ultralytics handles this automatically)

Run once before first use:
  python setup.py
"""

import os
import sys
import urllib.request

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def download_mediapipe_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"[OK] MediaPipe model already present: {MODEL_PATH}")
        return
    print(f"Downloading MediaPipe Face Landmarker to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MODEL_PATH)
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"[OK] Downloaded {size_mb:.1f} MB")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("Download manually from:")
        print(f"  {MEDIAPIPE_MODEL_URL}")
        sys.exit(1)


def validate_torch_cuda():
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
            print(f"[OK] PyTorch CUDA available: {name}")
        else:
            print("[WARNING] PyTorch CUDA not available — running on CPU.")
            print("  Reinstall torch from: https://download.pytorch.org/whl/cu126")
    except ImportError:
        print("[ERROR] torch not installed. See requirements.txt.")
        sys.exit(1)


def validate_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # auto-downloads on first call
        device = str(model.device)
        print(f"[OK] YOLOv8n ready on device: {device}")
    except Exception as e:
        print(f"[ERROR] YOLO validation failed: {e}")


def validate_mediapipe_delegate():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        try:
            mp_python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=mp_python.BaseOptions.Delegate.GPU,
            )
            print("[OK] MediaPipe GPU delegate available.")
        except Exception:
            print("[WARNING] MediaPipe GPU delegate NOT available on this platform.")
            print("  MediaPipe will run on CPU. Reduce resolution to 640x480 in config.yaml")
            print("  if performance is inadequate.")
    except ImportError:
        print("[ERROR] mediapipe not installed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Face Focus — Setup & Validation")
    print("=" * 60)
    download_mediapipe_model()
    validate_torch_cuda()
    validate_yolo()
    validate_mediapipe_delegate()
    print("=" * 60)
    print("Setup complete. Run: python main.py")
