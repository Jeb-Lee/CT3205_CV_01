"""
Extract clean, auto-labelled YOLO training frames from a plain webcam video
of one card being moved through the camera's field of view -- straight-on,
tilted, pulled near and far. This is the real-world counterpart to
augment_scale_tilt.py's synthetic scale/rotation/perspective crops: instead
of manufacturing scale and angle variety from 18 static photos, it comes
from actually recording a card at hand-held scale and angle, on the actual
deployment camera, so it carries real lighting, real motion, real webcam
colour/compression characteristics that no synthetic transform can invent.

IMPORTANT -- record with a plain video tool (Windows Camera app, OBS,
anything), NOT through webcam_demo.py's own 'r' recorder. That recorder
burns the detected box, label and HUD into the saved frames (same issue the
'c' vs 's' screenshot keys exist to avoid) -- feeding those back into
training would teach the model to key off the drawn box, not the card.

One video = one card/class. Recording several cards in one clip isn't
supported here (there'd be no automatic way to know which class a given
frame belongs to); record one short clip per card instead.

Method, per sampled frame:
  1. Reject if too blurry (variance of Laplacian below --blur-thresh) --
     expected during the "pull in/out" and fast-tilt parts of a slow pan,
     and a blurry frame is a bad training example regardless of whether a
     box could be found for it.
  2. Run detect_cards() with the same loosened area/aspect bounds used for
     the solo-crop dataset (a hand-held card fills far more of a frame than
     the multi-card photos these bounds were originally tuned for).
  3. Keep the frame only if detect_cards() finds exactly one region -- a
     frame with zero (card out of frame, or a hand-shadow merge rejected by
     the aspect/area filters) or multiple (transition, or an ambiguous
     merge) is skipped rather than guessed at, so every label this script
     produces is exactly as deterministic and reproducible as the rest of
     this project's dataset-building scripts. Expect a real, non-trivial
     rejection rate -- that's the filter working, not a bug.

Frames are saved at full scale, uncropped -- unlike build_solo_crops.py,
which re-crops already-isolated card photos, these frames are the whole
realistic hand-held scene (hand, background, real scale-in-frame) and
should stay that way; that context is the entire point of collecting them.

Output goes into yolo_dataset/images/train + labels/train as
`webcam_<label>_NNNN.jpg`, gitignored like the other build scripts'
regeneratable outputs (the source video is expected to live outside the
repo).

Usage (from yolo/):
    python scripts/extract_webcam_video.py path/to/video.mp4 --label G
    python scripts/extract_webcam_video.py path/to/video.mp4 --label G --preview
"""
import argparse
import os
import sys

_CNN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cnn")
sys.path.insert(0, _CNN_DIR)
sys.path.insert(0, os.path.join(_CNN_DIR, "scripts"))

import cv2
import numpy as np
from PIL import Image

from detect_cards import detect_cards, save_contact_sheet

REPO = os.path.join(os.path.dirname(__file__), "..")
IMG_DIR = os.path.join(REPO, "yolo_dataset", "images", "train")
LBL_DIR = os.path.join(REPO, "yolo_dataset", "labels", "train")

with open(os.path.join(REPO, "yolo_dataset", "data.yaml")) as f:
    CLASSES = [line.split(": ")[1].strip() for line in f if line.strip().startswith(tuple("0123456789"))]

MAX_LONG_SIDE = 1280  # downscale larger frames to keep dataset size reasonable


def blur_score(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to a raw (no overlay) webcam video")
    ap.add_argument("--label", required=True, choices=CLASSES,
                     help="the single class shown throughout this video")
    ap.add_argument("--stride", type=int, default=5,
                     help="sample every Nth frame (default 5)")
    ap.add_argument("--blur-thresh", type=float, default=60.0,
                     help="minimum Laplacian variance to keep a frame "
                          "(lower = more permissive). Tune after checking "
                          "the printed score distribution on a first run.")
    ap.add_argument("--preview", action="store_true",
                     help="also write a contact sheet of accepted frames "
                          "(with boxes drawn) for visual spot-checking")
    ap.add_argument("--max-frames", type=int, default=0,
                     help="stop after accepting this many frames (0 = no limit)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    cls_id = CLASSES.index(args.label)
    existing = [f for f in os.listdir(IMG_DIR) if f.startswith(f"webcam_{args.label}_")]
    start_idx = len(existing)

    frame_idx = -1
    accepted = 0
    blur_scores = []
    reject_blur = reject_zero = reject_multi = 0
    preview_crops = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride != 0:
            continue

        score = blur_score(frame)
        blur_scores.append(score)
        if score < args.blur_thresh:
            reject_blur += 1
            continue

        h, w = frame.shape[:2]
        if max(h, w) > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        boxes = detect_cards(
            pil, min_area_frac=0.02, max_area_frac=0.97,
            min_aspect=0.5, max_aspect=2.0, border_margin_px=0,
        )
        if len(boxes) == 0:
            reject_zero += 1
            continue
        if len(boxes) > 1:
            reject_multi += 1
            continue

        x0, y0, x1, y1 = boxes[0]
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h

        stem = f"webcam_{args.label}_{start_idx + accepted:04d}"
        pil.save(os.path.join(IMG_DIR, stem + ".jpg"), quality=95)
        with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as f:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        accepted += 1

        if args.preview and len(preview_crops) < 64:
            marked = pil.copy()
            from PIL import ImageDraw
            d = ImageDraw.Draw(marked)
            d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=max(2, w // 300))
            preview_crops.append(marked)

        if args.max_frames and accepted >= args.max_frames:
            break

    cap.release()

    scanned = len(blur_scores)
    print(f"scanned {scanned} sampled frames (stride={args.stride})")
    if blur_scores:
        arr = np.array(blur_scores)
        print(f"blur score: min={arr.min():.0f} median={np.median(arr):.0f} "
              f"max={arr.max():.0f} (threshold={args.blur_thresh:.0f})")
    print(f"rejected: {reject_blur} too blurry, {reject_zero} no card found, "
          f"{reject_multi} ambiguous (multiple regions)")
    print(f"accepted: {accepted} new '{args.label}' training frames "
          f"-> {IMG_DIR}")

    if args.preview and preview_crops:
        out = os.path.join(REPO, f"_preview_webcam_{args.label}.jpg")
        save_contact_sheet(preview_crops, out, cell=200)
        print(f"wrote preview contact sheet: {out}")


if __name__ == "__main__":
    main()
