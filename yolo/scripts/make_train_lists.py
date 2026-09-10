"""
Write the explicit training-image list each detector variant trains on.

Every image lives in the one folder, yolo_dataset/images/train, so pointing
a data.yaml at the folder would train on whatever happens to be in it --
the baseline would silently pick up every image added by later rounds. So
each variant's data*.yaml points at a list instead, and the list pins down
exactly which images that run saw:

    list                   images                                  run
    baseline_train.txt     train_*  (18 whole photos)              yolo_runs/cards
    scaletilt_train.txt    + synth_*  (72, augment_scale_tilt.py)   yolo_runs/cards_scaletilt
    allreal_train.txt      train_* + solo_*  (211, build_solo_crops) yolo_runs/cards_allreal
    v2_train.txt           + webcam_*  (350, extract_frame_folder)  yolo_runs/cards_v2

Entries are written as "./images/train/<file>". Ultralytics resolves a
"./" entry against the list file's own folder, so the lists (and the
data*.yaml files, which have no `path:` key for the same reason) work from
any clone, on any OS, without editing.

Usage (from yolo/):
    python scripts/make_train_lists.py
"""
import os

DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolo_dataset")
IMG_DIR = os.path.join(DS, "images", "train")

LISTS = {
    "baseline_train.txt": (("train_",), 18),
    "scaletilt_train.txt": (("train_", "synth_"), 90),
    "allreal_train.txt": (("train_", "solo_"), 229),
    "v2_train.txt": (("train_", "solo_", "webcam_"), 579),
}


def main():
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    for name, (prefixes, expected) in LISTS.items():
        chosen = [f for f in files if f.startswith(prefixes)]
        with open(os.path.join(DS, name), "w", newline="\n") as out:
            for f in chosen:
                out.write(f"./images/train/{f}\n")
        note = "" if len(chosen) == expected else (
            f"   <-- expected {expected}; generate the missing images first "
            f"(see the README)")
        print(f"{name:20s} {len(chosen):4d} images{note}")


if __name__ == "__main__":
    main()
