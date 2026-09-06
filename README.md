# CT3205_CV_01 — Flashcard Character Recognition

Two approaches to the same problem, kept in separate subfolders on purpose:
a from-scratch CNN classifier ([`cnn/`](cnn/)) and a fine-tuned YOLO detector
compared against it ([`yolo/`](yolo/)). Built for a computer vision course
assignment, starting from a genuinely hard constraint: **one real photograph
per class**.

The full write-up — architecture, two segmentation bugs found and fixed, why
a bigger model didn't help without more data, going out and collecting that
data, the corrupted-training-data discovery, the single-run measurement
trap, a tested-and-rejected architecture change, a real-world flash-glare
failure case, and the YOLO comparison — is in
[`docs/one-photo-per-card.html`](docs/one-photo-per-card.html). Open it in a
browser; it's a standalone page.

## Results

| | |
| --- | --- |
| CNN, held-out accuracy | **~81% (± 6.5% across 5 seeds)**, 211 real photos / 25 classes, grown from 1 photo/class |
| YOLOv8n, held-out detection recall | 84.7% (vs. the classical detector's 100%) |
| Real-world test (3 fresh photos, novel backgrounds) | CNN pipeline: 2/3 correct, 1 diagnosed segmentation failure (flash glare). YOLO: detected 0/3. |

## Project layout

```
cnn/            the CNN classifier — the main, working pipeline
  cnn_char_reader.py        segmentation, augmentation, CharCNN, training, predict()
  char_cnn_model.pt, char_cnn_labels.json    trained weights + class list
  images/                   every real training photo used, plus the manifest
  scripts/                  detect_cards.py (multi-card detection), build_dataset.py

yolo/           the YOLO detection comparison — reuses cnn/ code, never modifies it
  scripts/                  build_yolo_dataset.py, train_yolo.py, run_comparison.py, eval_detection.py
  yolo_dataset/             YOLO-format images/labels, train + held-out val split
  yolo_runs/cards/          the fine-tuned weights and training curves

demo/           real-time webcam demonstration
  webcam_demo.py            live boxes, class labels, confidence scores; runs
                            either pipeline, switchable at runtime

docs/one-photo-per-card.html   the write-up
RngTest/                        three real-world test photos (not training/eval data)
requirements.txt, requirements-yolo.txt
```

**`cnn/images/`** holds `img1.jpg`–`img25.jpg` (the original one-photo-per-class
templates), `extra/` and `extra2/` (real photos added across two later
shoots), `Image1–3.jpeg` (the source photos the write-up's independent test
crops come from), and `manifest.json` (`{relative path: character label}`
for every training photo actually used).

**On `cnn/scripts/build_dataset.py`:** it reads from `images/raw/`, the
original photo dump, which isn't included here (bulky, and every card in it
is already extracted into `images/extra`/`images/extra2`). It's included so
every labeling decision is visible and auditable, not just its output —
rerunning it requires supplying your own `images/raw/` with a matching
layout, or adapting its `BATCH_1`/`BATCH_2` maps to a new set of photos.
`yolo/scripts/build_yolo_dataset.py` works the same way, and reconstructs
bounding boxes for the CNN's own training/held-out photos by re-running the
deterministic detector rather than re-annotating anything by hand.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt          # CNN only
pip install -r requirements-yolo.txt      # adds ultralytics, if you want the YOLO side too
```

`requirements.txt` installs the CPU build of PyTorch by default. For an
NVIDIA GPU, install a CUDA build instead (match the index to your driver's
CUDA version, e.g. this project was built and measured against CUDA 12.4):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Note:** installing `ultralytics` (in `requirements-yolo.txt`) will pull in
its own CPU-only `torch`/`torchvision` and silently downgrade a CUDA install
in the same environment — reinstall the CUDA build afterward if you need it
back. A separate venv per subfolder avoids this entirely if you want the CNN
side to stay untouched.

## Usage

```bash
# CNN — from cnn/
python cnn_char_reader.py train                    # retrain from images/manifest.json
python cnn_char_reader.py predict path/to/photo.jpg # classify one card photo

# YOLO comparison — from yolo/ (needs cnn/ present as a sibling folder)
python scripts/build_yolo_dataset.py    # regenerate yolo_dataset/ from cnn/images
python scripts/train_yolo.py            # fine-tune YOLOv8n from COCO weights
python scripts/run_comparison.py        # current stack vs. YOLO vs. hybrid, side by side

# Real-time demo — from demo/ (needs opencv-python)
python webcam_demo.py                   # live webcam with boxes, labels, confidences
python webcam_demo.py --pipeline hybrid # same, using classical detect + CNN
```

Measured live at 640x480: **87.6 FPS** on the RTX 3050, 15.8 FPS on CPU, and
24.2 FPS for the hybrid pipeline — all comfortably real-time. See
[`demo/README.md`](demo/README.md) for controls and troubleshooting.

## Design notes (short version — see the write-up for the full story)

- **Input is a binary glyph mask, not the raw photo.** Each card is
  segmented down to "glyph vs. not", independent of its color, so the
  network can't shortcut on the coincidence that (in the original 25
  photos) every character happened to be printed in a different color.
- **Segmentation uses the convex hull of the card's black pixels**, not its
  bounding box or a hole-fill — the only approach that survives both a
  tilted card (whose bbox includes background in its corners) and a glyph
  stroke that touches the card's physical edge (which breaks hole-filling).
  It still assumes the background stays reasonably dark — camera flash on a
  glossy card near a reflective surface defeats it (see the write-up's
  real-world test section).
- **More real photos, not a bigger model, is what actually improved
  accuracy.** A parameter-count argument for a smaller, global-average-pooling
  head was tested and rejected — measured 26 points worse, not better. Kept
  in the code as a documented negative result rather than deleted.
- **A fine-tuned YOLO detector was compared head-to-head against the
  classical detector**, split into detection and classification metrics
  rather than one blended number. YOLO's own classifier is slightly weaker
  than the CNN's; its detector's recall gap is what actually decides the
  ranking; and on brand-new real-world photos it detected nothing at all,
  while the classical rule still worked on most of them — evidence that 18
  training photos of one desk isn't enough diversity for a learned detector
  to generalize, not that the architecture choice was wrong.
- **Every CNN accuracy number here comes from a 5-seed average on a
  held-out test set**, not a single run — single runs on a test set this
  size swing by double-digit percentage points from random seed alone.
