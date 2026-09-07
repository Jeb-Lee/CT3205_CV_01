"""
Feed YOLO the same real photographic material the CNN classifier already
gets: cnn/images/manifest.json's 211 individually-cropped, single-card
real photos (from images/extra/, images/extra2/, and the original
img1..img25 templates), turned into single-box YOLO training images.

Until now YOLO only ever trained on the 18 *whole* multi-card photographs
(see build_yolo_dataset.py) -- it never saw any of these solo crops, even
though the CNN's much larger effective training set (211 examples, the
"more real photos" experiment that moved classifier accuracy from ~69% to
~81%) is built entirely from them. This is the other half of that same
real photographic material, restructured for detection instead of
classification -- for a fair test of whether YOLO benefits from it the
same way, it should see it too.

Method: for each manifest entry, re-run detect_cards() directly on the
already-cropped image (loosened area/aspect bounds, since a solo crop's
card fills far more of the frame than in a busy multi-card photo -- the
original defaults would reject it as "too large"). Take the largest
detected region as the card -- reliable even when a second, smaller region
is also flagged, which usually turns out to be a glyph's own enclosed loop
(D, G, 6, 9, ...) registering as a separate dark island, not a second
card. Re-crop tightly around just that region with a small margin, so any
genuine neighbouring card sliver visible in the original generously-padded
crop (built with crop_with_padding's 35% margin, for the classical mask
extractor's benefit) is excluded rather than left in-frame and unlabelled.

Output goes into the same images/train + labels/train directories as
build_yolo_dataset.py and augment_scale_tilt.py, using a `solo_NNNN` stem
so none of the three ever collide or overwrite each other. Regenerated
deterministically from files already in the repo, so (like synth_*.jpg)
these are gitignored rather than committed.

Usage (from yolo/):
    python scripts/build_solo_crops.py
"""
import json
import os
import sys

_CNN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cnn")
sys.path.insert(0, _CNN_DIR)
sys.path.insert(0, os.path.join(_CNN_DIR, "scripts"))

from PIL import Image
from detect_cards import detect_cards

REPO = os.path.join(os.path.dirname(__file__), "..")
MANIFEST = os.path.join(_CNN_DIR, "images", "manifest.json")
IMG_DIR = os.path.join(REPO, "yolo_dataset", "images", "train")
LBL_DIR = os.path.join(REPO, "yolo_dataset", "labels", "train")

with open(os.path.join(REPO, "yolo_dataset", "data.yaml")) as f:
    CLASSES = [line.split(": ")[1].strip() for line in f if line.strip().startswith(tuple("0123456789"))]

PAD_FRAC = 0.08  # margin around the detected card, as a fraction of its own size


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    written = 0
    skipped = []
    for rel_path, label in manifest.items():
        src_path = os.path.join(_CNN_DIR, "images", rel_path)
        im = Image.open(src_path).convert("RGB")
        W, H = im.size

        boxes = detect_cards(
            im, min_area_frac=0.03, max_area_frac=0.97,
            min_aspect=0.5, max_aspect=2.0, border_margin_px=0,
        )
        if not boxes:
            skipped.append((rel_path, "no box found"))
            continue

        # the card itself -- always the largest region, even when a smaller
        # one is also flagged (usually a glyph's own enclosed loop).
        x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        bw, bh = x1 - x0, y1 - y0
        pad_x, pad_y = bw * PAD_FRAC, bh * PAD_FRAC

        cx0 = max(0, int(x0 - pad_x))
        cy0 = max(0, int(y0 - pad_y))
        cx1 = min(W, int(x1 + pad_x))
        cy1 = min(H, int(y1 + pad_y))

        crop = im.crop((cx0, cy0, cx1, cy1))
        cw, ch = crop.size
        if cw <= 0 or ch <= 0:
            skipped.append((rel_path, "degenerate crop"))
            continue

        # the box's position within the new, tighter crop
        nx0, ny0 = x0 - cx0, y0 - cy0
        nx1, ny1 = x1 - cx0, y1 - cy0
        ncx, ncy = (nx0 + nx1) / 2 / cw, (ny0 + ny1) / 2 / ch
        nw, nh = (nx1 - nx0) / cw, (ny1 - ny0) / ch

        if label not in CLASSES:
            skipped.append((rel_path, f"label {label!r} not a YOLO class"))
            continue
        cls_id = CLASSES.index(label)

        stem = f"solo_{written:04d}"
        crop.save(os.path.join(IMG_DIR, stem + ".jpg"), quality=95)
        with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as f:
            f.write(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
        written += 1

    print(f"wrote {written} solo-crop training images from {len(manifest)} "
          f"manifest entries")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for rel_path, reason in skipped:
            print(f"  {rel_path}: {reason}")


if __name__ == "__main__":
    main()
