"""
Synthetic scale + tilt augmentation for the YOLO training set.

Root cause under test: every training photo shows several cards spread flat
on a table, each occupying roughly 5-11% of the frame's linear dimension per
side (well under 15% of frame area), photographed close to head-on. A
hand-held card in the real-time demo fills 35-90%+ of the frame instead, and
is naturally rotated in-plane (the hand twists it) and tilted out-of-plane
(viewed at an angle, not head-on) -- scales and viewpoints the fine-tuned
detector never saw, which is why it returns literally zero boxes on
hand-held photos at any confidence threshold (see demo/README.md, "Hold the
card flat, not up"). This is a training-distribution gap, not a threshold
problem, and transfer learning from COCO weights doesn't fix it either -- the
network still needs *some* gradient signal from examples at that scale and
viewpoint.

This script manufactures that signal without new photography. For each
original training photo it picks a handful of that photo's own boxes as
"anchors" and, per anchor:
  1. sizes a square sampling window so the anchor card fills 35-80% of the
     eventual output frame (the zoom / scale half of the fix);
  2. rotates that window in-plane by a random angle (the "twisted in hand"
     half of the fix);
  3. independently jitters each of its 4 corners (a projective/perspective
     warp -- the "viewed at an angle, not head-on" half of the fix),
then samples the source image through the resulting quad with a single
perspective transform (`PIL.Image.transform(..., Image.PERSPECTIVE, ...)`),
and re-derives every box's label by mapping its 4 corners through the same
homography and taking their axis-aligned bounding box in the output frame
(the standard approximation for carrying axis-aligned boxes through a
rotation/homography, since YOLO's label format has no rotated-box notion).
Any box, not just the anchor, that still has enough of its own area inside
the sampled quad is kept -- so one synthetic frame can still show more than
one card, the same way a real zoomed-in shot of a cluttered desk would.

If a candidate quad would sample outside the original photo (i.e. rotation
+ jitter pushed a corner past the image edge), the rotation/jitter magnitude
is backed off and retried a few times before falling back to a plain
axis-aligned (zoom-only, no tilt) crop for that anchor, so every anchor still
contributes at least the scale half of the augmentation.

Output goes into the same images/train + labels/train directories, using a
`synth_NNN_k` stem so build_yolo_dataset.py's own `train_NNN` outputs are
never touched or overwritten by a rerun of this script (or vice versa) --
delete synth_*.jpg/.txt by hand to clear a previous run before regenerating.

Usage (from yolo/):
    python scripts/augment_scale_tilt.py
"""
import math
import os
import random

import numpy as np
from PIL import Image

REPO = os.path.join(os.path.dirname(__file__), "..")
IMG_DIR = os.path.join(REPO, "yolo_dataset", "images", "train")
LBL_DIR = os.path.join(REPO, "yolo_dataset", "labels", "train")

CROPS_PER_IMAGE = 4                # distinct anchor boxes per source photo
TARGET_FRAC_RANGE = (0.35, 0.80)   # anchor card's fraction of the new frame
ROTATION_RANGE_DEG = (-30.0, 30.0)  # in-plane twist
PERSPECTIVE_JITTER_FRAC = 0.14      # per-corner jitter, as a fraction of
                                     # crop_size -- the out-of-plane "tilt"
MIN_VISIBLE_FRAC = 0.4              # keep a non-anchor box only if this much
                                     # of its original area survives the warp
MAX_RETRIES = 5                     # back off rotation/jitter this many
                                     # times before falling back to zoom-only
SEED = 42


def load_boxes(label_path):
    """Returns list of (cls, cx, cy, w, h), all normalized 0-1."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def find_coeffs(pa, pb):
    """The 8-parameter homography H with H(pa_i) = pb_i for i=0..3."""
    matrix = []
    for (x, y), (X, Y) in zip(pa, pb):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(pb, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B)


def apply_homography(coeffs, points):
    a, b, c, d, e, f, g, h = coeffs
    out = []
    for x, y in points:
        denom = g * x + h * y + 1.0
        out.append(((a * x + b * y + c) / denom, (d * x + e * y + f) / denom))
    return out


def rotated_square(cx, cy, size, angle_deg):
    """4 corners of a `size`x`size` square centered on (cx, cy), rotated
    in-plane by angle_deg. Order: TL, TR, BR, BL."""
    half = size / 2
    base = [(-half, -half), (half, -half), (half, half), (-half, half)]
    t = math.radians(angle_deg)
    ct, st = math.cos(t), math.sin(t)
    return [(cx + x * ct - y * st, cy + x * st + y * ct) for x, y in base]


def jitter_quad(quad, size, frac, rng):
    j = size * frac
    return [(x + rng.uniform(-j, j), y + rng.uniform(-j, j)) for x, y in quad]


def quad_in_bounds(quad, W, H, margin=1.0):
    return all(margin <= x <= W - margin and margin <= y <= H - margin for x, y in quad)


def build_source_quad(a_cx_px, a_cy_px, crop_size, W, H, rng):
    """Rotated + perspective-jittered quad in source-image coordinates,
    backing off toward a plain axis-aligned square if it would sample
    outside the image. Returns (quad, rotated_bool)."""
    base = rotated_square(a_cx_px, a_cy_px, crop_size, 0.0)
    if quad_in_bounds(base, W, H):
        pass  # even the unrotated square just barely fits; still try tilt below
    for attempt in range(MAX_RETRIES):
        backoff = 1.0 - attempt / MAX_RETRIES  # 1.0, 0.8, 0.6, 0.4, 0.2
        angle = rng.uniform(*ROTATION_RANGE_DEG) * backoff
        quad = rotated_square(a_cx_px, a_cy_px, crop_size, angle)
        quad = jitter_quad(quad, crop_size, PERSPECTIVE_JITTER_FRAC * backoff, rng)
        if quad_in_bounds(quad, W, H):
            return quad, True
    # fall back to the plain axis-aligned square (zoom only, no tilt)
    return base, False


def make_crop(im, boxes, anchor_idx, target_frac, min_visible_frac, rng):
    W, H = im.size
    a_cls, a_cx, a_cy, a_w, a_h = boxes[anchor_idx]
    a_cx_px, a_cy_px = a_cx * W, a_cy * H
    a_w_px, a_h_px = a_w * W, a_h * H

    anchor_dim = max(a_w_px, a_h_px)
    crop_size = anchor_dim / target_frac
    crop_size = min(crop_size, min(W, H))
    if crop_size < anchor_dim:
        return None  # degenerate: box already bigger than any valid crop

    in_quad, tilted = build_source_quad(a_cx_px, a_cy_px, crop_size, W, H, rng)
    out_size = int(round(crop_size))
    out_quad = [(0, 0), (out_size, 0), (out_size, out_size), (0, out_size)]

    # PIL samples output->input, i.e. wants coeffs H with H(out) = in.
    pil_coeffs = find_coeffs(out_quad, in_quad)
    # for carrying box corners forward we want the opposite direction,
    # H' with H'(in) = out -- same solver, roles swapped.
    fwd_coeffs = find_coeffs(in_quad, out_quad)

    crop = im.transform((out_size, out_size), Image.PERSPECTIVE, pil_coeffs,
                         resample=Image.BICUBIC)

    new_boxes = []
    for cls, cx, cy, w, h in boxes:
        bx0, by0 = (cx - w / 2) * W, (cy - h / 2) * H
        bx1, by1 = (cx + w / 2) * W, (cy + h / 2) * H
        orig_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        if orig_area <= 0:
            continue
        corners = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
        warped = apply_homography(fwd_coeffs, corners)
        xs = [p[0] for p in warped]
        ys = [p[1] for p in warped]
        wx0, wx1 = min(xs), max(xs)
        wy0, wy1 = min(ys), max(ys)
        warped_area = max(0.0, wx1 - wx0) * max(0.0, wy1 - wy0)
        if warped_area <= 0:
            continue
        # clip the warped bbox to the output canvas
        cx0, cy0 = max(0.0, wx0), max(0.0, wy0)
        cx1, cy1 = min(out_size, wx1), min(out_size, wy1)
        vis_w, vis_h = max(0.0, cx1 - cx0), max(0.0, cy1 - cy0)
        vis_area = vis_w * vis_h
        # approximate visibility against the warped (post-transform) box
        # area, which already reflects any scale change from the homography
        if vis_area / warped_area < min_visible_frac:
            continue
        ncx = (cx0 + cx1) / 2 / out_size
        ncy = (cy0 + cy1) / 2 / out_size
        nw = vis_w / out_size
        nh = vis_h / out_size
        new_boxes.append((cls, ncx, ncy, nw, nh))

    if not new_boxes:
        return None
    return crop, new_boxes, tilted


def main():
    rng = random.Random(SEED)
    src_stems = sorted(
        f[:-4] for f in os.listdir(IMG_DIR)
        if f.startswith("train_") and f.endswith(".jpg")
    )
    print(f"{len(src_stems)} source training photos found")

    written = 0
    tilted_count = 0
    for stem in src_stems:
        img_path = os.path.join(IMG_DIR, stem + ".jpg")
        lbl_path = os.path.join(LBL_DIR, stem + ".txt")
        im = Image.open(img_path).convert("RGB")
        boxes = load_boxes(lbl_path)
        if not boxes:
            continue

        n_pick = min(CROPS_PER_IMAGE, len(boxes))
        anchors = rng.sample(range(len(boxes)), n_pick)
        for k, anchor_idx in enumerate(anchors):
            target_frac = rng.uniform(*TARGET_FRAC_RANGE)
            result = make_crop(im, boxes, anchor_idx, target_frac, MIN_VISIBLE_FRAC, rng)
            if result is None:
                continue
            crop, new_boxes, tilted = result
            tilted_count += int(tilted)

            out_stem = f"synth_{stem[6:]}_{k}"
            crop.convert("RGB").save(os.path.join(IMG_DIR, out_stem + ".jpg"), quality=95)
            with open(os.path.join(LBL_DIR, out_stem + ".txt"), "w") as f:
                for cls, cx, cy, w, h in new_boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            written += 1

    print(f"wrote {written} synthetic scale+tilt training images "
          f"({tilted_count} with rotation/perspective applied, "
          f"{written - tilted_count} fell back to zoom-only) "
          f"targeting {TARGET_FRAC_RANGE[0]:.0%}-{TARGET_FRAC_RANGE[1]:.0%} "
          f"of frame, {ROTATION_RANGE_DEG[0]:.0f} to {ROTATION_RANGE_DEG[1]:.0f} "
          f"degree rotation")


if __name__ == "__main__":
    main()
