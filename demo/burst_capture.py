"""
Hands-free burst capture: point the webcam at a fixed area, hold one card
up and slowly move it through poses (near, far, tilted near, tilted far),
and this saves a steady stream of clean, auto-labelled training frames
without needing to press a key each time.

This is the live-camera counterpart to extract_webcam_video.py -- same
per-frame logic (reject blur, auto-box with the classical detector, skip
anything ambiguous rather than guess), just sourced frame-by-frame from a
live feed instead of a pre-recorded file. One run = one card/class, same
convention as extract_webcam_video.py, and both write into the same
yolo_dataset/images/train + labels/train using the same `webcam_<label>_`
naming, so their outputs are interchangeable.

Never shows a detection box on screen and never saves one into a frame --
this is a capture tool, not the demo. Every frame that gets kept is exactly
what the camera saw, nothing drawn on top of it.

Usage
-----
    python burst_capture.py --label G
    python burst_capture.py --label G --roi          # drag a capture-area
                                                       # box first, ESC to
                                                       # skip and use the
                                                       # full frame instead
    python burst_capture.py --label G --interval 0.2 --max-frames 150

Keys while running
-------------------
    q / Esc   quit
    space     pause / resume capturing (repositioning without saving junk)
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_CNN_DIR = os.path.join(_REPO, "cnn")
sys.path.insert(0, _CNN_DIR)
sys.path.insert(0, os.path.join(_CNN_DIR, "scripts"))
sys.path.insert(0, os.path.join(_REPO, "yolo", "scripts"))

import cv2
import numpy as np
from PIL import Image

from detect_cards import detect_cards

REPO = os.path.join(_REPO, "yolo")
IMG_DIR = os.path.join(REPO, "yolo_dataset", "images", "train")
LBL_DIR = os.path.join(REPO, "yolo_dataset", "labels", "train")

with open(os.path.join(REPO, "yolo_dataset", "data.yaml")) as f:
    CLASSES = [line.split(": ")[1].strip() for line in f if line.strip().startswith(tuple("0123456789"))]


def blur_score(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def open_camera(index, cap_width, cap_height):
    """Same backend-verification approach as webcam_demo.py's open_source():
    isOpened() alone is not trustworthy on Windows, so every candidate
    backend is confirmed with a real frame read."""
    backends = [(None, "default")]
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append((cv2.CAP_DSHOW, "DirectShow"))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append((cv2.CAP_MSMF, "MSMF"))
    for api, label in backends:
        cap = cv2.VideoCapture(index) if api is None else cv2.VideoCapture(index, api)
        if cap.isOpened():
            if cap_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
            if cap_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"camera {index} opened via {label} backend at {w}x{h}")
                return cap
        cap.release()
    raise SystemExit(f"Could not read from camera {index}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, choices=CLASSES,
                     help="the single class shown throughout this session")
    ap.add_argument("--source", type=int, default=0, help="camera index")
    ap.add_argument("--cap-width", type=int, default=1280)
    ap.add_argument("--cap-height", type=int, default=720)
    ap.add_argument("--interval", type=float, default=0.3,
                     help="seconds between capture attempts (default 0.3, "
                          "~3/sec -- slow enough to avoid near-duplicate "
                          "frames, fast enough to cover a lot of poses "
                          "quickly)")
    ap.add_argument("--blur-thresh", type=float, default=60.0,
                     help="minimum Laplacian variance to keep a frame")
    ap.add_argument("--roi", action="store_true",
                     help="drag out a fixed capture-area rectangle before "
                          "starting (ESC during selection to skip and use "
                          "the full frame instead). Only this region of "
                          "each frame is processed and saved -- useful for "
                          "excluding background clutter, other people, etc. "
                          "from a busy room.")
    ap.add_argument("--max-frames", type=int, default=0,
                     help="stop after accepting this many frames (0 = run "
                          "until q/Esc)")
    args = ap.parse_args()

    cap = open_camera(args.source, args.cap_width, args.cap_height)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    roi = None  # (x0, y0, x1, y1) in the raw frame
    if args.roi:
        ok, frame = cap.read()
        if ok:
            print("Drag a box around the capture area, then press ENTER/SPACE. "
                  "Esc to skip and use the full frame.")
            x, y, w, h = cv2.selectROI("select capture area", frame,
                                        showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("select capture area")
            if w > 0 and h > 0:
                roi = (x, y, x + w, y + h)
                print(f"capture area: {roi}")

    cls_id = CLASSES.index(args.label)
    existing = [f for f in os.listdir(IMG_DIR) if f.startswith(f"webcam_{args.label}_")]
    start_idx = len(existing)

    accepted = 0
    reject_blur = reject_zero = reject_multi = 0
    paused = False
    last_capture = 0.0
    last_status = "starting..."

    print(f"Ready. Label='{args.label}', saving to {IMG_DIR}")
    print("space = pause/resume   q/Esc = quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            display = frame.copy()
            if roi:
                x0, y0, x1, y1 = roi
                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

            now = time.time()
            if not paused and (now - last_capture) >= args.interval:
                last_capture = now
                work = frame
                if roi:
                    x0, y0, x1, y1 = roi
                    work = frame[y0:y1, x0:x1]

                score = blur_score(work)
                if score < args.blur_thresh:
                    reject_blur += 1
                    last_status = f"skip: blurry ({score:.0f})"
                else:
                    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    boxes = detect_cards(
                        pil, min_area_frac=0.02, max_area_frac=0.97,
                        min_aspect=0.5, max_aspect=2.0, border_margin_px=0,
                    )
                    if len(boxes) == 0:
                        reject_zero += 1
                        last_status = "skip: no card found"
                    elif len(boxes) > 1:
                        reject_multi += 1
                        last_status = "skip: ambiguous (multiple regions)"
                    else:
                        w, h = pil.size
                        x0b, y0b, x1b, y1b = boxes[0]
                        cx, cy = (x0b + x1b) / 2 / w, (y0b + y1b) / 2 / h
                        bw, bh = (x1b - x0b) / w, (y1b - y0b) / h
                        stem = f"webcam_{args.label}_{start_idx + accepted:04d}"
                        pil.save(os.path.join(IMG_DIR, stem + ".jpg"), quality=95)
                        with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as f:
                            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                        accepted += 1
                        last_status = f"saved #{accepted}"

                        if args.max_frames and accepted >= args.max_frames:
                            print(f"reached --max-frames={args.max_frames}, stopping")
                            break

            hud = f"{args.label} | saved:{accepted} blur:{reject_blur} " \
                  f"none:{reject_zero} ambig:{reject_multi} | {last_status}"
            if paused:
                hud = "[PAUSED] " + hud
            cv2.putText(display, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("burst_capture (raw feed, nothing drawn is saved)", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\ndone: saved {accepted} frames for '{args.label}' "
          f"(rejected {reject_blur} blurry, {reject_zero} no-card, "
          f"{reject_multi} ambiguous)")


if __name__ == "__main__":
    main()
