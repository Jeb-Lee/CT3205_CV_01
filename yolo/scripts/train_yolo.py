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
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")  # COCO-pretrained nano checkpoint
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        patience=40,
        project=os.path.join(REPO, "yolo_runs"),
        name=args.name,
        seed=42,
        verbose=True,
    )
