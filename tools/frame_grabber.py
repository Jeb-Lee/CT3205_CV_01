"""
frame_grabber.py - pull training frames out of a video, cropped to a fixed region.

Requires: pip install opencv-python

Examples
--------
    # interactive: scrub through the video, press S to save
    python frame_grabber.py clip.mp4 --roi

    # automatic: one frame every 0.5 seconds
    python frame_grabber.py clip.mp4 --roi --every 0.5

    # start 30s in, save as jpg, into a custom folder
    python frame_grabber.py clip.mp4 --start 30 --ext jpg -o dataset/raw

Interactive keys
----------------
    s / SPACE   save current frame (cropped)
    d           forward 1 frame          a   back 1 frame
    e           forward 10 frames        w   back 10 frames
    f           forward 100 frames       r   back 100 frames
    c           re-draw the crop box
    x           clear the crop box (save full frames)
    q / ESC     quit
"""

import argparse
import os
import sys

import cv2

WIN = "frame_grabber"


def parse_args():
    p = argparse.ArgumentParser(description="Extract cropped frames from a video.")
    p.add_argument("video", help="Path to the video file.")
    p.add_argument("-o", "--out", default="frames", help="Output folder.")
    p.add_argument("--every", type=float, default=None,
                   help="Automatic mode: seconds between saved frames.")
    p.add_argument("--roi", action="store_true",
                   help="Drag a crop box before extracting.")
    p.add_argument("--prefix", default="frame", help="Filename prefix.")
    p.add_argument("--ext", default="png", choices=["png", "jpg"],
                   help="png keeps the pixels exact; jpg is smaller.")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Shrink the preview window only (does not affect saved files).")
    return p.parse_args()


def pick_roi(frame, scale):
    """Let the user drag a box. Returns (x, y, w, h) in ORIGINAL pixel coords."""
    shown = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale)
    box = cv2.selectROI("Drag crop area, then ENTER (C to cancel)", shown,
                        showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Drag crop area, then ENTER (C to cancel)")
    x, y, w, h = (int(v / scale) for v in box)
    return (x, y, w, h) if w > 0 and h > 0 else None


def crop(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def save(frame, roi, out_dir, prefix, ext, index):
    path = os.path.join(out_dir, f"{prefix}_{index:05d}.{ext}")
    params = [cv2.IMWRITE_JPEG_QUALITY, 98] if ext == "jpg" else []
    cv2.imwrite(path, crop(frame, roi), params)
    return path


def draw_hud(frame, roi, pos, total, fps, saved, scale):
    view = frame.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if scale != 1.0:
        view = cv2.resize(view, None, fx=scale, fy=scale)
    secs = pos / fps if fps else 0
    label = f"frame {pos}/{total}  {secs:6.2f}s  saved:{saved}"
    cv2.putText(view, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(view, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return view


def run_auto(cap, args, roi, fps, out_dir):
    """Read straight through and save every Nth frame. Faster and more reliable
    than seeking, because seeking on compressed video snaps to keyframes."""
    step = max(1, int(round(fps * args.every)))
    start_frame = int(args.start * fps)
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start_frame and (idx - start_frame) % step == 0:
            save(frame, roi, out_dir, args.prefix, args.ext, saved)
            saved += 1
        idx += 1
    print(f"Saved {saved} frames to {out_dir}")


def run_interactive(cap, args, roi, fps, total, out_dir):
    pos = int(args.start * fps)
    saved = 0
    jumps = {ord('d'): 1, ord('a'): -1,
             ord('e'): 10, ord('w'): -10,
             ord('f'): 100, ord('r'): -100}

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            pos = max(0, pos - 1)
            if pos == 0:
                break
            continue

        cv2.imshow(WIN, draw_hud(frame, roi, pos, total, fps, saved, args.scale))
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('s'), 32):
            path = save(frame, roi, out_dir, args.prefix, args.ext, saved)
            saved += 1
            print(f"  -> {path}   (source frame {pos})")
        elif key == ord('c'):
            roi = pick_roi(frame, args.scale)
        elif key == ord('x'):
            roi = None
        elif key in jumps:
            pos = max(0, min(total - 1, pos + jumps[key]))

    cv2.destroyAllWindows()
    print(f"Saved {saved} frames to {out_dir}")


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(args.out, exist_ok=True)
    print(f"{args.video}: {total} frames @ {fps:.2f} fps")

    roi = None
    if args.roi:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
        ok, first = cap.read()
        if not ok:
            sys.exit("Could not read a frame to select the crop area from.")
        roi = pick_roi(first, args.scale)
        print(f"Crop region: {roi}" if roi else "No crop - saving full frames.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.every:
        run_auto(cap, args, roi, fps, args.out)
    else:
        run_interactive(cap, args, roi, fps, total, args.out)

    cap.release()


if __name__ == "__main__":
    main()
