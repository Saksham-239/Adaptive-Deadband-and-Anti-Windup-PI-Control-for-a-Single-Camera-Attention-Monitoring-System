"""
diagnose_poses.py - Interactive pose diagnostic, one pose at a time.
Each ENTER press captures 3 seconds of data, then waits for next ENTER.
"""
import csv, os, sys, time
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from core.vision import VisionProcessor
from core.attention import gaze_score, head_score, blink_score

CAPTURE_SEC = 3

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    cam = cv2.VideoCapture(cfg["camera"]["device_id"])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = VisionProcessor(cfg, w, h)
    print(f"Camera ready: {w}x{h}", flush=True)

    poses = [
        "STRAIGHT", "DOWN_DESK", "UP_CEILING",
        "TURN_LEFT", "TURN_RIGHT", "SIDE_EYE_RIGHT",
    ]
    all_rows = []
    header = ["pose","yaw","pitch","roll","iris_dx","iris_dy","ear","zone","gaze_s","head_s","blink_s"]

    for pose in poses:
        print(f"\nREADY_FOR: {pose}", flush=True)
        input()  # wait for ENTER
        print(f"CAPTURING: {pose} for {CAPTURE_SEC}s...", flush=True)

        rows = []
        start = time.time()
        while time.time() - start < CAPTURE_SEC:
            ret, frame = cam.read()
            if not ret:
                continue
            result = proc.process_frame(frame)
            if result and result.pose_valid:
                g_zone = result.gaze_zone
                g_s = gaze_score(g_zone, None, None, None, cfg["attention"]["gaze_zone_margin_px"])
                h_s = head_score(result.yaw, result.pitch, cfg["attention"]["head_sigma_deg"])
                b_s = blink_score(result.mean_ear, cfg["attention"]["blink_ear_threshold"])
                rows.append({
                    "pose": pose, "yaw": round(result.yaw,1), "pitch": round(result.pitch,1),
                    "roll": round(result.roll,1), "iris_dx": round(result.iris_deviation_x,3),
                    "iris_dy": round(result.iris_deviation_y,3), "ear": round(result.mean_ear,3),
                    "zone": g_zone, "gaze_s": round(g_s,3), "head_s": round(h_s,3), "blink_s": round(b_s,3),
                })
            time.sleep(0.03)

        if rows:
            n = len(rows)
            avg = lambda k: round(sum(r[k] for r in rows)/n, 1)
            from collections import Counter
            top_zone = Counter(r["zone"] for r in rows).most_common(1)[0][0]
            print(f"RESULT: {pose}  n={n}  yaw={avg('yaw'):+.1f}  pitch={avg('pitch'):+.1f}  "
                  f"roll={avg('roll'):+.1f}  iris_dx={avg('iris_dx'):+.2f}  zone={top_zone}  "
                  f"gaze_s={avg('gaze_s'):.2f}  head_s={avg('head_s'):.2f}", flush=True)
            all_rows.extend(rows)
        else:
            print(f"RESULT: {pose}  NO FACE DETECTED", flush=True)

    out = os.path.join(os.path.dirname(__file__), "diagnose_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nALL_DONE. Saved {len(all_rows)} samples to {out}", flush=True)

    cam.release()
    proc.close()

if __name__ == "__main__":
    main()
