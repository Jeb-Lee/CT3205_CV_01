"""
Runs all three pipelines against the same val set and prints a side-by-side
report:
  1. Current stack   — classical (threshold + convex hull) detection, then
                        the existing CNN classifies each crop.
  2. YOLO alone      — YOLO's own detection AND its own classification.
  3. Hybrid          — YOLO detection, then the existing CNN classifies
                        each crop (ignoring YOLO's own class prediction).
"""
import os
import sys

# cnn/ holds the CNN + segmentation code this reuses — CT3205_CV_01/{cnn,yolo}
# are siblings.
_CNN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cnn")
sys.path.insert(0, _CNN_DIR)
sys.path.insert(0, os.path.join(_CNN_DIR, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

import cnn_char_reader as M
from detect_cards import detect_cards, crop_with_padding
from eval_detection import load_ground_truth, evaluate, print_report

REPO = os.path.join(os.path.dirname(__file__), "..")
LABELS_DIR = os.path.join(REPO, "yolo_dataset", "labels", "val")
IMAGES_DIR = os.path.join(REPO, "yolo_dataset", "images", "val")
YOLO_WEIGHTS = os.path.join(REPO, "yolo_runs", "cards", "weights", "best.pt")

with open(os.path.join(REPO, "yolo_dataset", "data.yaml")) as f:
    yolo_classes = [line.split(": ")[1].strip() for line in f if line.strip().startswith(tuple("0123456789"))]

gt_by_image = {img: boxes for img, boxes in load_ground_truth(LABELS_DIR, IMAGES_DIR, yolo_classes)}
cnn_model, cnn_classes = M._load_model_for_inference()


def classify_crop_with_cnn(crop):
    mask = M.extract_card_mask(crop)
    if mask is None:
        return None
    arr = np.asarray(mask).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        pred_idx = cnn_model(tensor).argmax(1).item()
    return cnn_classes[pred_idx]


# ---- Pipeline 1: current stack (classical detect + CNN classify) ----------
pred_current = {}
for img_path in gt_by_image:
    im = Image.open(img_path).convert("RGB")
    boxes = detect_cards(im)
    preds = []
    for box in boxes:
        crop = crop_with_padding(im, box)
        label = classify_crop_with_cnn(crop)
        if label is not None:
            preds.append((box, label))
    pred_current[img_path] = preds

# ---- Pipeline 2 & 3: YOLO alone, and YOLO-detect + CNN-classify hybrid ----
yolo = YOLO(YOLO_WEIGHTS)
pred_yolo_alone = {}
pred_hybrid = {}
for img_path in gt_by_image:
    im = Image.open(img_path).convert("RGB")
    result = yolo.predict(img_path, verbose=False, conf=0.25)[0]
    yolo_preds, hybrid_preds = [], []
    for box_t, cls_t in zip(result.boxes.xyxy, result.boxes.cls):
        box = tuple(box_t.tolist())
        yolo_label = yolo_classes[int(cls_t.item())]
        yolo_preds.append((box, yolo_label))

        crop = crop_with_padding(im, box)
        cnn_label = classify_crop_with_cnn(crop)
        if cnn_label is not None:
            hybrid_preds.append((box, cnn_label))
    pred_yolo_alone[img_path] = yolo_preds
    pred_hybrid[img_path] = hybrid_preds

print_report("1. Current stack (classical detect + CNN classify)", evaluate(gt_by_image, pred_current))
print_report("2. YOLO alone (YOLO detect + YOLO classify)", evaluate(gt_by_image, pred_yolo_alone))
print_report("3. Hybrid (YOLO detect + CNN classify)", evaluate(gt_by_image, pred_hybrid))
