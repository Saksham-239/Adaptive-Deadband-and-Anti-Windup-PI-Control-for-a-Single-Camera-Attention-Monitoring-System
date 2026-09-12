"""Quick verification of remapped axes — captures 3 seconds of straight-ahead data."""
import os, sys, time
import cv2
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from core.vision import VisionProcessor
from core.attention import gaze_score, head_score, blink_score

cfg = yaml.safe_load(open("config.yaml"))
cam = cv2.VideoCapture(cfg["camera"]["device_id"])
cam.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
proc = VisionProcessor(cfg, w, h)
print(f"Camera: {w}x{h}. Capturing 3s — just look at camera.", flush=True)

start = time.time()
last = 0
while time.time() - start < 3:
    ret, frame = cam.read()
    if not ret: continue
    r = proc.process_frame(frame)
    t = time.time() - start
    if r and r.pose_valid and t - last >= 0.3:
        gs = gaze_score(gz)
        hs = head_score(r.yaw, r.pitch, cfg["attention"]["head_sigma_deg"])
        bs = blink_score(r.mean_ear, cfg["attention"]["blink_ear_threshold"])
        print(f"[{t:4.1f}s] yaw={r.yaw:+6.1f} pitch={r.pitch:+6.1f} roll={r.roll:+6.1f} "
              f"zone={gz:<15s} gaze_s={gs:.2f} head_s={hs:.2f} blink={bs:.2f}", flush=True)
        last = t
    time.sleep(0.03)

cam.release()
proc.close()
print("DONE", flush=True)
