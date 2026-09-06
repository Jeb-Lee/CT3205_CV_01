"""
Build a YOLO-format detection dataset from data this project already has:
- Image1/2/3.jpeg: the very first three multi-card photos, with the exact
  manual boxes used to build the original img1-25 CNN templates.
- Five "batch 1" grid re-shoots + ten "batch 2" photos: no saved box
  coordinates, but detect_cards() is deterministic, so re-running it on the
  same photo regenerates the same boxes in the same order used when their
  crops were labeled for the CNN dataset — so we can pair boxes with the
  already-known index -> label maps directly.
- Nine more "batch 2" photos held out from CNN training, used the same way,
  as a genuinely unseen YOLO validation set.

Writes standard ultralytics format:
  yolo_dataset/images/{train,val}/*.jpg
  yolo_dataset/labels/{train,val}/*.txt   (class cx cy w h, normalized)
  yolo_dataset/data.yaml
"""
import os
import sys
import shutil

# cnn/ holds the segmentation code this reuses (detect_cards, extract_card_mask
# via cnn_char_reader) — CT3205_CV_01/{cnn,yolo} are siblings.
_CNN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cnn")
sys.path.insert(0, _CNN_DIR)
sys.path.insert(0, os.path.join(_CNN_DIR, "scripts"))
from PIL import Image
from detect_cards import detect_cards

REPO = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = r"C:\Users\jeble\PycharmProjects\CT3079_Capstone\images\Raw"
OUT = os.path.join(REPO, "yolo_dataset")

# ---------------------------------------------------------------------------
# Image1/2/3.jpeg: original manual boxes (generous padding, from the very
# first card-isolation pass in this project).
# ---------------------------------------------------------------------------
MANUAL = {
    "Image1.jpeg": [
        ((80, 195, 490, 575), "1"), ((445, 185, 815, 550), "2"),
        ((790, 185, 1135, 550), "3"), ((105, 540, 490, 920), "4"),
        ((420, 475, 785, 820), "5"), ((720, 560, 1085, 920), "6"),
        ((105, 810, 490, 1290), "7"), ((420, 810, 800, 1290), "8"),
        ((720, 895, 1170, 1350), "9"),
    ],
    "Image2.jpeg": [
        ((48, 128, 425, 441), "A"), ((423, 140, 741, 443), "B"),
        ((703, 75, 1186, 510), "C"), ((32, 405, 393, 752), "D"),
        ((418, 439, 740, 759), "E"), ((692, 460, 1055, 828), "F"),
        ((19, 725, 411, 1119), "G"), ((389, 748, 718, 1079), "H"),
        ((762, 775, 1124, 1145), "S"), ((52, 1047, 473, 1414), "T"),
        ((432, 1070, 771, 1416), "U"), ((739, 1093, 1143, 1487), "V"),
    ],
    "Image3.jpeg": [
        ((60, 160, 500, 500), "W"), ((410, 100, 850, 500), "X"),
        ((740, 130, 1130, 530), "Y"), ((30, 510, 480, 950), "Z"),
        ((420, 510, 840, 890), "circle"), ((110, 980, 520, 1320), "left_arrow"),
        ((640, 980, 1100, 1340), "right_arrow"),
    ],
}

# ---------------------------------------------------------------------------
# Batch 1 (used to build images/extra) and batch 2 train split (images/extra2)
# — the same PHOTOS maps from scripts/build_dataset.py.
# ---------------------------------------------------------------------------
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

BATCH_2_TRAIN = {
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

# Batch 2 photos NEVER used for CNN training — the exact held-out set from
# multi_seed_eval_big.py, minus the tarp photo (systematically bad).
BATCH_2_VAL = {
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (29).jpeg": {0: "S", 1: "H", 2: "T", 3: "U", 4: "V", 5: "W", 6: "Y", 7: "X", 11: "Z"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (27).jpeg": {0: "1", 1: "5", 2: "6", 3: "A", 4: "B", 5: "9", 6: "C", 7: "F", 8: "E", 9: "G", 10: "D"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (9).jpeg": {0: "T", 1: "U", 2: "V", 3: "X", 4: "Y", 5: "W", 6: "Z"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (4).jpeg": {0: "B", 1: "C", 2: "A", 3: "D", 4: "E", 5: "F", 7: "G", 8: "H", 9: "S"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (30).jpeg": {1: "U", 2: "W", 3: "V", 4: "Y", 5: "X", 9: "Z"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (11).jpeg": {0: "T", 1: "U", 2: "V", 3: "W", 4: "X", 5: "Y", 6: "Z"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.24 (10).jpeg": {0: "T", 1: "U", 2: "V", 3: "X", 4: "Y", 5: "W", 6: "Z"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.23 (5).jpeg": {0: "B", 1: "A", 2: "C", 3: "D", 4: "E", 6: "F", 7: "G", 8: "H", 9: "S"},
    "New folder/WhatsApp Image 2026-09-06 at 12.48.23 (3).jpeg": {0: "1", 1: "5", 2: "6", 3: "4", 5: "8", 6: "9", 7: "7"},
}

SKIP = {"BAD"}


def yolo_line(class_id, box, w, h):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def write_split(entries, split, classes):
    img_dir = os.path.join(OUT, "images", split)
    lbl_dir = os.path.join(OUT, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    n_boxes = 0
    for i, (photo_path, boxes_labels) in enumerate(entries):
        im = Image.open(photo_path)
        w, h = im.size
        stem = f"{split}_{i:03d}"
        im.convert("RGB").save(os.path.join(img_dir, stem + ".jpg"), quality=95)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            for box, label in boxes_labels:
                f.write(yolo_line(classes.index(label), box, w, h))
                n_boxes += 1
    print(f"{split}: {len(entries)} images, {n_boxes} boxes")


if __name__ == "__main__":
    all_labels = set()
    for v in MANUAL.values():
        all_labels |= {lbl for _, lbl in v}
    for d in (BATCH_1, BATCH_2_TRAIN, BATCH_2_VAL):
        for mapping in d.values():
            all_labels |= set(mapping.values())
    all_labels -= SKIP
    classes = sorted(all_labels)
    print(f"{len(classes)} classes: {classes}")

    train_entries, val_entries = [], []

    for photo, boxes_labels in MANUAL.items():
        train_entries.append((os.path.join(_CNN_DIR, "images", photo), boxes_labels))

    for source, dest in ((BATCH_1, train_entries), (BATCH_2_TRAIN, train_entries), (BATCH_2_VAL, val_entries)):
        for rel_photo, mapping in source.items():
            photo_path = os.path.join(RAW_DIR, rel_photo)
            if not os.path.isfile(photo_path):
                print(f"MISSING SOURCE PHOTO: {photo_path}")
                continue
            im = Image.open(photo_path).convert("RGB")
            boxes = detect_cards(im)
            pairs = []
            for idx, label in mapping.items():
                if label in SKIP or idx >= len(boxes):
                    continue
                pairs.append((boxes[idx], label))
            dest.append((photo_path, pairs))

    write_split(train_entries, "train", classes)
    write_split(val_entries, "val", classes)

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

    print(f"\nWrote dataset to {OUT}")
