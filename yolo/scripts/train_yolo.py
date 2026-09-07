import argparse
import os

from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(REPO, "yolo_dataset", "data.yaml")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cards",
                     help="run name under yolo_runs/ (default 'cards', the "
                          "original baseline run -- pass a different name "
                          "to keep that run's weights/curves intact when "
                          "training a variant, e.g. on an augmented dataset)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--data", default=DATA_YAML,
                     help="path to a data.yaml (default: the original "
                          "18-photo dataset). Pass a variant, e.g. "
                          "yolo_dataset/data_allreal.yaml, to train on a "
                          "different image set without touching the default.")
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")  # COCO-pretrained nano checkpoint
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        patience=40,
        project=os.path.join(REPO, "yolo_runs"),
        name=args.name,
        seed=42,
        verbose=True,
    )
