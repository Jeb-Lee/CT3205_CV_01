"""
Rebuilds images/manifest.json from scratch: the original 25 one-per-class
photos, plus every labeled card crop pulled out of the two later batches of
multi-card photos, minus the 9 templates a visual audit found were silently
corrupted (see the write-up, "Testing the prediction: real photos, at
scale"). This is what actually produced the images/extra/, images/extra2/,
and images/manifest.json shipped in this repo.

NOT meant to be run as-is here: it reads from images/raw/, the original
WhatsApp photos, which aren't included in this repo (bulky, and every card
they contain is already extracted into images/extra and images/extra2).
It's included for reproducibility and so the labeling decisions are visible
and auditable, not just their output. To actually rerun it, supply your own
images/raw/ with the same file layout, or adapt the PHOTOS maps below to a
new set of photos (photo -> {detected index: character label}), built by
first running detect_cards.py on each photo and reading labels off its
contact sheet.
"""
import os
import json
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from cnn_char_reader import extract_card_mask, LABELS
from detect_cards import detect_cards, crop_with_padding

REPO = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(REPO, "images", "raw")
IMAGES_DIR = os.path.join(REPO, "images")

SKIP = {"circle", "left_arrow", "right_arrow", "up_arrow", "BAD"}

# Batch 1: five re-shoots of the full 25-card deck (different backgrounds/
# lighting/angles), each mapped as {detected-candidate-index: label}.
BATCH_1 = {
    "WhatsApp Image 2026-09-06 at 11.41.02.jpeg": {
        0: "5", 1: "6", 2: "7", 3: "4", 4: "2", 5: "3", 6: "1", 7: "C", 8: "B",
        9: "D", 10: "A", 11: "9", 12: "E", 13: "8", 14: "V", 15: "S", 16: "U",
        17: "T", 18: "H", 19: "F", 20: "G", 21: "Z", 22: "circle", 23: "Y",
        24: "right_arrow", 25: "left_arrow", 26: "X", 27: "W",
    },
    "WhatsApp Image 2026-09-06 at 11.41.03.jpeg": {
        0: "5", 1: "4", 2: "3", 3: "2", 4: "E", 5: "1", 6: "C", 7: "B", 8: "A",
        9: "9", 10: "BAD", 11: "V", 12: "U", 13: "T", 14: "S", 15: "H", 16: "G",
        17: "F", 18: "Z", 19: "Y", 20: "circle", 21: "X", 22: "right_arrow",
        23: "left_arrow",
    },
    "WhatsApp Image 2026-09-06 at 11.41.04.jpeg": {
        0: "1", 1: "A", 2: "9", 3: "T", 4: "S", 5: "8", 6: "H", 7: "right_arrow",
        8: "G", 9: "circle", 10: "Z", 11: "F", 12: "Y", 13: "X", 14: "W",
    },
    "WhatsApp Image 2026-09-06 at 11.41.06.jpeg": {
        0: "BAD", 1: "BAD", 2: "7", 3: "4", 4: "V", 5: "U", 6: "T",
        7: "left_arrow", 8: "right_arrow", 9: "circle", 10: "Z", 11: "Y",
        12: "X", 13: "W",
    },
    "WhatsApp Image 2026-09-06 at 11.41.07.jpeg": {
        0: "B", 1: "6", 2: "C", 3: "7", 4: "S", 5: "D", 6: "X", 7: "T", 8: "E",
        9: "Y", 10: "U", 11: "Z", 12: "circle", 13: "left_arrow",
    },
}

# Batch 2: ten photos from the second, better-spaced session (subfolder
# "New folder" in the raw dump).
BATCH_2 = {
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (25).jpeg": {
        0: "3", 1: "4", 2: "2", 3: "1", 4: "6", 5: "7", 6: "5", 7: "8", 8: "A",
        9: "B", 10: "9", 11: "C", 12: "F", 13: "E", 14: "D", 15: "G", 16: "G",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (33).jpeg": {
        0: "S", 1: "H", 2: "T", 3: "U", 4: "V", 5: "W", 6: "Y", 7: "X",
        8: "up_arrow", 9: "right_arrow", 10: "circle", 11: "Z",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (26).jpeg": {
        0: "BAD", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
        9: "A", 10: "9", 11: "B", 12: "C", 13: "F", 14: "E", 15: "G", 16: "D",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (28).jpeg": {
        0: "3", 1: "1", 2: "2", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "A",
        9: "B", 10: "9", 11: "C", 12: "F", 13: "E", 14: "D", 15: "G",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (24).jpeg": {
        0: "3", 1: "1", 2: "2", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "A",
        9: "9", 10: "B", 11: "C", 12: "F", 13: "E", 14: "D", 15: "G",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (32).jpeg": {
        0: "U", 1: "T", 2: "S", 3: "H", 4: "Y", 5: "W", 6: "V", 7: "X",
        8: "up_arrow", 9: "circle", 10: "Z", 11: "right_arrow",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (6).jpeg": {
        0: "B", 1: "C", 2: "A", 3: "BAD", 4: "D", 5: "E", 6: "F", 7: "D",
        8: "G", 9: "H", 10: "S",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (31).jpeg": {
        0: "BAD", 1: "H", 2: "S", 3: "T", 4: "V", 5: "W", 6: "X", 7: "Y",
        8: "up_arrow", 9: "circle", 10: "Z",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (12).jpeg": {
        0: "V", 1: "U", 2: "T", 3: "Y", 4: "X", 5: "W", 6: "W",
        7: "left_arrow", 8: "circle", 9: "Z", 10: "right_arrow",
    },
    "New folder/WhatsApp Image 2026-09-06 at 12.48.23 (16).jpeg": {
        0: "2", 1: "1", 2: "3", 3: "5", 4: "6", 5: "4", 6: "circle", 7: "8",
        8: "9", 9: "7", 10: "circle",
    },
}

# Found by a visual audit of every extracted template's segmented mask (not
# just its pixel count) — see the write-up. Four of these are from the
# original day-one 25, which had never been individually inspected before.
KNOWN_CORRUPTED = {
    "img2.jpg", "img3.jpg", "img8.jpg", "img12.jpg", "img19.jpg",
    "extra/extra_7_11.41.07.jpg", "extra/extra_Z_11.41.07.jpg",
    "extra2/e2_D_6at1248246_4.jpg", "extra2/e2_D_6at1248246_7.jpg",
    "extra2/e2_S_6at1248246_10.jpg",
}


def process_batch(batch, extra_dir, name_fn):
    """For each (photo, {index: label}) pair: detect cards in the photo,
    crop the ones with a known label, segment-check them, and save."""
    os.makedirs(os.path.join(IMAGES_DIR, extra_dir), exist_ok=True)
    entries = {}
    for rel_photo, mapping in batch.items():
        photo_path = os.path.join(RAW_DIR, rel_photo)
        if not os.path.isfile(photo_path):
            print(f"SKIP (not found): {photo_path}")
            continue
        im = Image.open(photo_path).convert("RGB")
        boxes = detect_cards(im)
        for idx, label in mapping.items():
            if label in SKIP or idx >= len(boxes):
                continue
            crop = crop_with_padding(im, boxes[idx])
            mask = extract_card_mask(crop)
            if mask is None:
                print(f"FAILED to segment: {rel_photo} idx {idx} (label {label})")
                continue
            out_name = name_fn(rel_photo, idx, label)
            out_path = os.path.join(IMAGES_DIR, extra_dir, out_name)
            crop.save(out_path, quality=92)
            entries[f"{extra_dir}/{out_name}"] = label
    return entries


if __name__ == "__main__":
    manifest = {f"img{n}.jpg": label for n, label in LABELS.items()}

    manifest.update(process_batch(
        BATCH_1, "extra",
        lambda photo, idx, label: f"extra_{label}_{os.path.splitext(os.path.basename(photo))[0][-8:]}.jpg",
    ))
    manifest.update(process_batch(
        BATCH_2, "extra2",
        lambda photo, idx, label: f"e2_{label}_{os.path.splitext(os.path.basename(photo))[0][-10:]}_{idx}.jpg",
    ))

    removed = [k for k in KNOWN_CORRUPTED if manifest.pop(k, None) is not None]

    with open(os.path.join(IMAGES_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nRemoved {len(removed)} known-corrupted templates: {removed}")
    print(f"Final manifest: {len(manifest)} entries across {len(set(manifest.values()))} classes.")
