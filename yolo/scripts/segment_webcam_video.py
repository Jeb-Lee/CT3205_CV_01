"""
Split a single "card-by-card through the whole deck" webcam video into
per-card segments automatically, so the only thing a human needs to supply
afterward is the simple part: the order the cards were filmed in.

Recorded pattern this expects (per card): brought near, pulled slowly far,
rotated a little clockwise and counter-clockwise, then swapped for the next
card. During a swap the frame briefly has no plausible card-shaped region
(hand moving off-frame, card out of view, or a blurred transition) -- this
script finds those gaps and treats the runs between them as segments.

Method: sample frames at a stride, run the classical detector (loosened
bounds, same reasoning as build_solo_crops.py / extract_webcam_video.py --
a hand-held card fills far more of frame than the multi-card photos these
defaults were tuned for) to get a per-sample "card visible" signal, then
find contiguous True runs at least --min-seconds long. Short gaps (a single
noisy sample) are bridged rather than treated as a real transition, since a
momentary misdetection mid-segment is expected and shouldn't fragment one
card's footage into two.

Output: a JSON manifest of segments (index, start/end seconds, frame
range) plus a contact sheet with one representative thumbnail per segment,
numbered, for a human to check the count and order against the actual
filming order before any labels are assigned. Nothing is extracted or
written into the training set at this stage -- see extract_webcam_video.py
(to be pointed at each confirmed segment) for that.

Usage (from yolo/):
    python scripts/segment_webcam_video.py path/to/video.mp4
"""
import argparse
import json
import os
import sys

_CNN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cnn")
sys.path.insert(0, _CNN_DIR)
sys.path.insert(0, os.path.join(_CNN_DIR, "scripts"))

import cv2
from PIL import Image

from detect_cards import detect_cards, save_contact_sheet

REPO = os.path.join(os.path.dirname(__file__), "..")


def has_card(pil):
    boxes = detect_cards(
        pil, min_area_frac=0.015, max_area_frac=0.97,
        min_aspect=0.4, max_aspect=2.5, border_margin_px=0,
    )
    return len(boxes) >= 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--sample-stride", type=int, default=3,
                     help="check every Nth frame for a card (default 3, "
                          "i.e. ~10 checks/sec on a 30fps video)")
    ap.add_argument("--min-seconds", type=float, default=1.5,
                     help="shorter runs are discarded as noise, not a segment")
    ap.add_argument("--bridge-seconds", type=float, default=0.5,
                     help="gaps in detection shorter than this are bridged "
                          "(treated as still the same segment) rather than "
                          "splitting it")
    ap.add_argument("--out", default=None,
                     help="output JSON path (default: <video>.segments.json)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_run = max(1, int(args.min_seconds * fps / args.sample_stride))
    bridge = max(0, int(args.bridge_seconds * fps / args.sample_stride))

    signal = []  # (frame_idx, bool)
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.sample_stride != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        signal.append((frame_idx, has_card(pil)))
    cap.release()

    # bridge short False gaps
    bridged = [v for _, v in signal]
    i = 0
    while i < len(bridged):
        if not bridged[i]:
            j = i
            while j < len(bridged) and not bridged[j]:
                j += 1
            if (j - i) <= bridge and i > 0 and j < len(bridged):
                for k in range(i, j):
                    bridged[k] = True
            i = j
        else:
            i += 1

    # find True runs
    runs = []
    i = 0
    while i < len(bridged):
        if bridged[i]:
            j = i
            while j < len(bridged) and bridged[j]:
                j += 1
            runs.append((signal[i][0], signal[j - 1][0]))
            i = j
        else:
            i += 1

    runs = [(s, e) for s, e in runs if (e - s) >= min_run * args.sample_stride]

    segments = []
    for idx, (start_f, end_f) in enumerate(runs):
        segments.append({
            "index": idx,
            "start_frame": start_f,
            "end_frame": end_f,
            "start_s": round(start_f / fps, 2),
            "end_s": round(end_f / fps, 2),
            "duration_s": round((end_f - start_f) / fps, 2),
        })

    out_path = args.out or (args.video + ".segments.json")
    with open(out_path, "w") as f:
        json.dump({"video": args.video, "fps": fps, "segments": segments}, f, indent=2)
    print(f"found {len(segments)} segments -> {out_path}")

    # contact sheet: middle frame of each segment
    cap = cv2.VideoCapture(args.video)
    thumbs = []
    for seg in segments:
        mid = (seg["start_frame"] + seg["end_frame"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thumbs.append(Image.fromarray(rgb))
    cap.release()

    sheet_path = os.path.splitext(out_path)[0] + "_contact_sheet.jpg"
    save_contact_sheet(thumbs, sheet_path, cell=160)
    print(f"contact sheet ({len(thumbs)} segments, numbered 0..{len(thumbs)-1}) -> {sheet_path}")
    for seg in segments:
        print(f"  [{seg['index']:2d}] {seg['start_s']:6.1f}s - {seg['end_s']:6.1f}s "
              f"({seg['duration_s']:.1f}s)")


if __name__ == "__main__":
    main()
