import os
from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(REPO, "yolo_dataset", "data.yaml")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # COCO-pretrained nano checkpoint
    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=8,
        patience=40,
        project=os.path.join(REPO, "yolo_runs"),
        name="cards",
        seed=42,
        verbose=True,
    )
