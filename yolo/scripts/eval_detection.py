"""
Shared evaluation: given ground-truth boxes+labels for the val set and a
list of predicted boxes+labels(+optional confidence), report:
  - detection recall: fraction of real cards a method actually found (IoU >= 0.5)
  - detection precision: fraction of its predictions that were real cards
  - classification accuracy GIVEN a correct detection: of the cards it
    found, how many did it also read correctly

Splitting these apart matters here: a method can fail by not finding a card
at all (detection) or by finding it and misreading it (classification) —
conflating them hides which part of the pipeline is actually the bottleneck.
"""
import os


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(labels_dir, images_dir, classes):
    """Yields (image_path, [(box_px, label), ...]) for every val image."""
    from PIL import Image
    for fname in sorted(os.listdir(labels_dir)):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(images_dir, stem + ".jpg")
        w, h = Image.open(img_path).size
        boxes = []
        with open(os.path.join(labels_dir, fname)) as f:
            for line in f:
                cid, cx, cy, bw, bh = line.split()
                cid, cx, cy, bw, bh = int(cid), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                boxes.append(((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), classes[cid]))
        yield img_path, boxes


def evaluate(gt_by_image: dict, pred_by_image: dict, iou_thresh=0.5):
    """
    gt_by_image, pred_by_image: {image_path: [(box, label), ...]}
    Greedy IoU matching per image (each GT box matched to at most one
    prediction and vice versa).
    """
    total_gt, total_pred, matched, correct_class = 0, 0, 0, 0
    per_class_errors = []

    for img_path, gt_boxes in gt_by_image.items():
        preds = pred_by_image.get(img_path, [])
        total_gt += len(gt_boxes)
        total_pred += len(preds)

        used_pred = set()
        for gt_box, gt_label in gt_boxes:
            best_iou, best_j = 0, None
            for j, (p_box, p_label) in enumerate(preds):
                if j in used_pred:
                    continue
                v = iou(gt_box, p_box)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j is not None and best_iou >= iou_thresh:
                used_pred.add(best_j)
                matched += 1
                p_label = preds[best_j][1]
                if p_label == gt_label:
                    correct_class += 1
                else:
                    per_class_errors.append((gt_label, p_label))

    recall = matched / total_gt if total_gt else 0.0
    precision = matched / total_pred if total_pred else 0.0
    class_acc_given_detected = correct_class / matched if matched else 0.0
    overall_acc = correct_class / total_gt if total_gt else 0.0  # detect AND classify correctly

    return {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "matched": matched,
        "detection_recall": recall,
        "detection_precision": precision,
        "classification_acc_given_detected": class_acc_given_detected,
        "overall_acc": overall_acc,
        "errors": per_class_errors,
    }


def print_report(name, result):
    r = result
    print(f"\n=== {name} ===")
    print(f"  ground-truth cards: {r['total_gt']}   predictions made: {r['total_pred']}")
    print(f"  detection recall:    {r['detection_recall']:.1%}  (found {r['matched']}/{r['total_gt']} real cards)")
    print(f"  detection precision: {r['detection_precision']:.1%}")
    print(f"  classification acc. given correct detection: {r['classification_acc_given_detected']:.1%}")
    print(f"  overall (detect AND classify correctly):      {r['overall_acc']:.1%}")
    if r["errors"]:
        print("  misclassifications (expected -> predicted):", ", ".join(f"{e}->{p}" for e, p in r["errors"]))
