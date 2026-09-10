"""
Convert a folder of already-extracted, per-class-named webcam frames (e.g.
cnn/images/extra3, named "<label>_Img_NNN.jpg") into auto-labelled YOLO
training images. The live-camera counterpart is burst_capture.py; the
video counterpart is extract_webcam_video.py -- all three write into the
same yolo_dataset/images/train + labels/train under the `webcam_<label>_`
prefix.

Per frame:
  1. Reject if too blurry (Laplacian variance) -- same as the other two
     tools.
  2. Run detect_cards() with loosened bounds (a hand-held card fills far
     more of frame than the multi-card photos these defaults were tuned
     for) and take the LARGEST detected region as the card, even when a
     smaller region is also found nearby -- verified by hand on two full
     classes (S, "4") that this is reliable for ordinary background
     clutter and stray hand fragments.
  3. Reject if that largest region's area, as a fraction of the frame, is
     an outlier relative to the *same class's own* other frames (more than
     `--outlier-ratio` away from that class's median). Aimed at a dark
     object *behind* the card (a tinted window, a soft toy) fusing with the card
     under the same brightness threshold and inflating the box.
  4. Skip any frame named in `--exclude` -- the frames rejected by visual
     review. Step 3 only partly catches the background merge: where enough
     of a class's frames share it, the class median itself shifts, and
     neither aspect ratio nor mask solidity separated the merged boxes
     either. For cnn/images/extra3 the review decisions are recorded in
     yolo_dataset/extra3_rejected.txt, so they are auditable and rerunnable
     rather than living only in which files happen to have been deleted.

Output names are `webcam_<label>_<k>` where k is the frame's position among
the frames that passed steps 1-3, counted from 0. A frame excluded in step
4 leaves a gap in that numbering instead of shifting every later name, so
rerunning with the same inputs and the same --exclude list reproduces the
same files, byte for byte, overwriting rather than appending.

Usage (from yolo/):
    python scripts/extract_frame_folder.py ../cnn/images/extra3 \
        --classes yolo_dataset/data_v2.yaml \
        --exclude yolo_dataset/extra3_rejected.txt
    add --preview to write a contact sheet of accepted frames per class
"""
import argparse
import os
import re
import sys
from collections import defaultdict

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

NAME_RE = re.compile(r"^(?P<label>.+)_Img_\d+\.jpg$", re.IGNORECASE)


def blur_score(im_pil):
    arr = np.array(im_pil.convert("L"))
    return cv2.Laplacian(arr, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of <label>_Img_NNN.jpg frames")
    ap.add_argument("--classes", default=None,
                     help="path to a data.yaml to validate labels against "
                          "(default: yolo_dataset/data.yaml). Any label in "
                          "the folder not in this class list is an error, "
                          "not silently skipped.")
    ap.add_argument("--blur-thresh", type=float, default=60.0)
    ap.add_argument("--outlier-ratio", type=float, default=0.35,
                     help="reject a frame if its largest-region area "
                          "fraction differs from its class's own median by "
                          "more than this fraction (default 0.35)")
    ap.add_argument("--exclude", default=None,
                     help="text file of source filenames (one per line) "
                          "rejected by visual review")
    ap.add_argument("--preview", action="store_true",
                     help="write a per-class contact sheet of accepted "
                          "frames (boxes drawn) for a final visual check")
    args = ap.parse_args()

    classes_path = args.classes or os.path.join(REPO, "yolo_dataset", "data.yaml")
    with open(classes_path) as f:
        classes = [line.split(": ")[1].strip() for line in f
                   if line.strip().startswith(tuple("0123456789"))]

    excluded = set()
    if args.exclude:
        with open(args.exclude) as f:
            excluded = {line.strip() for line in f
                        if line.strip() and not line.startswith("#")}

    by_label = defaultdict(list)
    for fname in sorted(os.listdir(args.folder)):
        m = NAME_RE.match(fname)
        if not m:
            continue
        by_label[m.group("label")].append(fname)

    unknown = set(by_label) - set(classes)
    if unknown:
        raise SystemExit(f"Labels in folder not found in {classes_path}: "
                          f"{sorted(unknown)}\nFix the class list or the "
                          f"filenames before continuing.")
    all_names = {f for names in by_label.values() for f in names}
    missing = excluded - all_names
    if missing:
        raise SystemExit(f"--exclude names {len(missing)} file(s) not in "
                          f"{args.folder}, e.g. {sorted(missing)[:3]}")

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    totals = defaultdict(int)

    for label in sorted(by_label):
        fnames = by_label[label]
        cls_id = classes.index(label)

        # steps 1-2: blur filter + detect, keep area fractions of survivors
        candidates = []  # (fname, pil, box, area_frac)
        n_blur = n_zero = 0
        for fname in fnames:
            path = os.path.join(args.folder, fname)
            pil = Image.open(path).convert("RGB")
            if blur_score(pil) < args.blur_thresh:
                n_blur += 1
                continue
            boxes = detect_cards(
                pil, min_area_frac=0.02, max_area_frac=0.97,
                min_aspect=0.4, max_aspect=2.5, border_margin_px=0,
            )
            if not boxes:
                n_zero += 1
                continue
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            w, h = pil.size
            area_frac = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            candidates.append((fname, pil, box, area_frac))

        # step 3: per-class outlier rejection on area fraction
        median = float(np.median([c[3] for c in candidates])) if candidates else 0.0
        accepted = [c for c in candidates
                    if not (median > 0 and abs(c[3] - median) / median > args.outlier_ratio)]
        n_outlier = len(candidates) - len(accepted)

        # step 4: visual-review exclusions, numbered by position in `accepted`
        kept = n_excluded = 0
        accepted_previews = []
        for k, (fname, pil, box, area_frac) in enumerate(accepted):
            if fname in excluded:
                n_excluded += 1
                continue
            w, h = pil.size
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            stem = f"webcam_{label}_{k:04d}"
            pil.save(os.path.join(IMG_DIR, stem + ".jpg"), quality=95)
            with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            kept += 1
            if args.preview:
                from PIL import ImageDraw
                marked = pil.copy()
                ImageDraw.Draw(marked).rectangle([x0, y0, x1, y1],
                                                  outline=(255, 0, 0), width=4)
                accepted_previews.append(marked)

        print(f"{label:12s}: {len(fnames):3d} frames -> kept {kept:3d}  "
              f"(blur {n_blur}, no-card {n_zero}, area-outlier {n_outlier}, "
              f"visual-review {n_excluded})")
        totals["kept"] += kept
        totals["blur"] += n_blur
        totals["zero"] += n_zero
        totals["outlier"] += n_outlier
        totals["excluded"] += n_excluded

        if args.preview and accepted_previews:
            out = os.path.join(args.folder, f"_accepted_{label}.jpg")
            save_contact_sheet(accepted_previews, out, cell=180)

    print(f"\nTOTAL kept: {totals['kept']}  (rejected: blur {totals['blur']}, "
          f"no-card {totals['zero']}, area-outlier {totals['outlier']}, "
          f"visual-review {totals['excluded']})")


if __name__ == "__main__":
    main()
