"""
Find every card-like blob in a photo of several flashcards, crop each with
padding, and (optionally) save a contact sheet for labeling by eye.

Reuses the exact black-card / convex-hull segmentation primitives from
cnn_char_reader.py — this is the same idea as extract_card_mask(), just
applied to find *every* card in a photo instead of assuming there's one.

Usage (exploratory — to figure out what's in a new photo and label it):
    python scripts/detect_cards.py path/to/photo.jpg out_dir/

That writes out_dir/cand_000.jpg, cand_001.jpg, ... and out_dir/_contact_sheet.jpg
(a numbered grid of thumbnails) so each index's actual label can be read off
by eye and fed into build_dataset.py's PHOTOS maps.
"""
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from cnn_char_reader import _dark_neutral_mask


def detect_cards(image: Image.Image, min_area_frac=0.0015, max_area_frac=0.25,
                  min_aspect=0.72, max_aspect=1.4, border_margin_px=1):
    """Return a list of (x0, y0, x1, y1) boxes for card-like blobs.

    Filters by area (as a fraction of the whole photo) and aspect ratio
    (cards are close to square regardless of tilt — see the write-up's
    convex-hull section for why), and drops anything within
    `border_margin_px` of the frame edge since that's usually a cut-off
    object, not a full card. The default (1px) matches the dataset-building
    scripts' original behaviour exactly; pass 0 to disable this filter
    entirely for a hand-held live feed, where a card near the edge of a
    narrow field of view is common and usually still whole.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    H, W = rgb.shape[:2]
    total_area = H * W

    mask = _dark_neutral_mask(rgb)
    # mild closing to bridge small JPEG-blur gaps in a card's outline
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))

    boxes = []
    for i in range(1, n + 1):
        component = labeled == i
        area = component.sum()
        if not (min_area_frac * total_area <= area <= max_area_frac * total_area):
            continue
        ys, xs = np.where(component)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = x1 - x0, y1 - y0
        if h == 0 or w == 0:
            continue
        aspect = w / h
        if not (min_aspect <= aspect <= max_aspect):
            continue
        m = border_margin_px
        if x0 <= m or y0 <= m or x1 >= (W - 1) - m or y1 >= (H - 1) - m:
            continue
        boxes.append((x0, y0, x1, y1))

    return boxes


def crop_with_padding(image: Image.Image, box, pad_frac=0.35):
    """Crop a generous padded region around a detected box — extract_card_mask
    needs some background margin around the card to find its footprint."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad_frac), int(h * pad_frac)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(image.width, x1 + px), min(image.height, y1 + py)
    return image.crop((cx0, cy0, cx1, cy1))


def save_contact_sheet(crops, out_path, cell=130):
    """A numbered grid of thumbnails — view this once to read off every
    crop's actual label instead of guessing at reading order."""
    if not crops:
        return
    cols = math.ceil(math.sqrt(len(crops)))
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, crop in enumerate(crops):
        thumb = crop.copy()
        thumb.thumbnail((cell - 10, cell - 24))
        r, c = divmod(i, cols)
        sheet.paste(thumb, (c * cell + 5, r * cell + 20))
        draw.text((c * cell + 4, r * cell + 2), str(i), fill=(255, 255, 0))
    sheet.save(out_path, quality=90)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/detect_cards.py <photo.jpg> <out_dir>")
        sys.exit(1)

    photo_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    im = Image.open(photo_path).convert("RGB")
    boxes = detect_cards(im)
    print(f"{photo_path}: found {len(boxes)} candidate boxes")

    crops = [crop_with_padding(im, box) for box in boxes]
    for i, crop in enumerate(crops):
        crop.save(os.path.join(out_dir, f"cand_{i:03d}.jpg"), quality=92)

    save_contact_sheet(crops, os.path.join(out_dir, "_contact_sheet.jpg"))
    print(f"saved {len(boxes)} crops + contact sheet to {out_dir}")
