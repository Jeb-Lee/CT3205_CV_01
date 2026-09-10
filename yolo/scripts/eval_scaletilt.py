"""
Two checks across the YOLO fine-tunes -- the untouched baseline
(yolo_runs/cards), the scale+tilt-augmented run (yolo_runs/cards_scaletilt),
the all-real run trained on the CNN's own 211 solo-crop photos plus the 18
whole-photo images (yolo_runs/cards_allreal, see build_solo_crops.py), and
the final 30-class run that adds 350 hand-held webcam frames
(yolo_runs/cards_v2, see extract_frame_folder.py):

1. Standard held-out detection/classification metrics on yolo_dataset's own
   val split (the flat-on-table photos) -- confirms each variant didn't
   regress on the original distribution.
2. Direct test against real hand-held/novel-background photos captured live
   during demo testing (demo/captures/), at a spread of confidence
   thresholds including very low ones -- this is the actual question: did
   each round of training data make the detector work under the conditions
   it is really used in, where the baseline produced literally zero boxes
   down to conf 0.01.
   Note: the two novel-background (whiteboard) photos already have a
   detection box drawn into their pixels from a prior live demo run -- fine
   for evaluation (no training happens here), but worth knowing if a result
   looks odd. demo/captures/ is not committed; part 2 reports MISSING for
   any photo that isn't present locally.

Class names for predictions come from each model's own `names`, not from
data.yaml: cards_v2 has a 30th class (down_arrow) that the 29-class
data.yaml doesn't list. The first 29 are identical across all four, so
ground truth and predictions still line up.
"""
import os
import sys

_CNN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cnn")
sys.path.insert(0, _CNN_DIR)
sys.path.insert(0, os.path.join(_CNN_DIR, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO

from eval_detection import load_ground_truth, evaluate, print_report

REPO = os.path.join(os.path.dirname(__file__), "..")
DEMO_CAPTURES = os.path.join(REPO, "..", "demo", "captures")
LABELS_DIR = os.path.join(REPO, "yolo_dataset", "labels", "val")
IMAGES_DIR = os.path.join(REPO, "yolo_dataset", "images", "val")

WEIGHTS = {
    "baseline (cards)": os.path.join(REPO, "yolo_runs", "cards", "weights", "best.pt"),
    "scale+tilt (cards_scaletilt)": os.path.join(REPO, "yolo_runs", "cards_scaletilt", "weights", "best.pt"),
    "all-real (cards_allreal)": os.path.join(REPO, "yolo_runs", "cards_allreal", "weights", "best.pt"),
    "deployment (cards_v2)": os.path.join(REPO, "yolo_runs", "cards_v2", "weights", "best.pt"),
}

HAND_HELD_PHOTOS = [
    ("shot_20260906_220843.jpg", "G"),
    ("shot_20260906_220904.jpg", "G"),
    ("shot_20260906_221417.jpg", "V"),
    ("shot_20260906_221422.jpg", "V"),
    ("shot_20260906_222158.jpg", "G"),
    ("shot_20260906_222202.jpg", "G"),
    ("shot_20260907_201820.jpg", "A"),   # card stuck to a whiteboard --
    ("shot_20260907_202847.jpg", "3"),   # novel background, not hand-held
]

with open(os.path.join(REPO, "yolo_dataset", "data.yaml")) as f:
    yolo_classes = [line.split(": ")[1].strip() for line in f if line.strip().startswith(tuple("0123456789"))]

gt_by_image = {img: boxes for img, boxes in load_ground_truth(LABELS_DIR, IMAGES_DIR, yolo_classes)}

print("=" * 70)
print("PART 1: held-out val set (flat-on-table photos, same distribution as training)")
print("=" * 70)
for name, weights_path in WEIGHTS.items():
    if not os.path.isfile(weights_path):
        print(f"{name}: weights not found at {weights_path}, skipping")
        continue
    model = YOLO(weights_path)
    preds = {}
    for img_path in gt_by_image:
        result = model.predict(img_path, verbose=False, conf=0.25)[0]
        boxes = []
        for box_t, cls_t in zip(result.boxes.xyxy, result.boxes.cls):
            boxes.append((tuple(box_t.tolist()), model.names[int(cls_t.item())]))
        preds[img_path] = boxes
    print_report(name, evaluate(gt_by_image, preds))

print()
print("=" * 70)
print("PART 2: real hand-held photos (demo/captures/) -- the actual test case")
print("=" * 70)
conf_levels = [0.25, 0.10, 0.05, 0.01]
for name, weights_path in WEIGHTS.items():
    if not os.path.isfile(weights_path):
        continue
    model = YOLO(weights_path)
    print(f"\n--- {name} ---")
    for fname, true_label in HAND_HELD_PHOTOS:
        photo_path = os.path.join(DEMO_CAPTURES, fname)
        if not os.path.isfile(photo_path):
            print(f"  {fname}: MISSING")
            continue
        line = f"  {fname} (true={true_label}):"
        any_hit = False
        for conf in conf_levels:
            result = model.predict(photo_path, verbose=False, conf=conf)[0]
            n = len(result.boxes)
            if n:
                any_hit = True
                top_i = result.boxes.conf.argmax().item()
                top_conf = result.boxes.conf[top_i].item()
                top_cls = model.names[int(result.boxes.cls[top_i].item())]
                line += f" [conf>={conf}: {n} box(es), top={top_cls}@{top_conf:.2f}]"
            else:
                line += f" [conf>={conf}: 0]"
        print(line + ("" if any_hit else "  <-- ZERO detections at every threshold"))
