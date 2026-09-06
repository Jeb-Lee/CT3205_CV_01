# CT3205_CV_01 — Flashcard Character Recognition CNN

A from-scratch PyTorch CNN that reads a photo of one flashcard (a digit 1–9
or a letter A–H/S–Z, printed in color on a black square) and predicts which
character it is. Built for a computer vision course assignment, starting
from a genuinely hard constraint: **one real photograph per class**.

The full write-up — architecture, two segmentation bugs found and fixed,
why a bigger model didn't help without more data, going out and collecting
that data, the corrupted-training-data discovery, the single-run
measurement trap, and a tested-and-rejected architecture change — is in
[`docs/one-photo-per-card.html`](docs/one-photo-per-card.html). Open it in
a browser; it's a standalone page.

## Result

A ~551,000-parameter CNN, trained on 211 real photos across 25 classes
(grown from 1 photo/class), scores **~81% (± 6.5% across 5 random seeds)**
on a 72-example held-out test set of photos never used in training.

## Project layout

| Path | What it is |
| --- | --- |
| `cnn_char_reader.py` | The model: segmentation (photo → binary glyph mask), augmentation, `CharCNN`, training loop, `predict()` |
| `images/img1.jpg`–`img25.jpg` | The original one-photo-per-class templates |
| `images/extra/`, `images/extra2/` | Additional real photos added across two later shoots |
| `images/Image1.jpeg`–`Image3.jpeg` | The three source photos the write-up's independent test crops come from (never used in training) |
| `images/manifest.json` | `{relative path: character label}` for every training photo actually used |
| `char_cnn_model.pt`, `char_cnn_labels.json` | The trained model weights and its class list |
| `scripts/detect_cards.py` | Finds every card in a multi-card photo (reuses the same segmentation as `cnn_char_reader.py`), for labeling new photos |
| `scripts/build_dataset.py` | Reproduces `images/manifest.json` from the raw photos — see note below |
| `docs/one-photo-per-card.html` | The write-up |

**Note on `scripts/build_dataset.py`:** it reads from `images/raw/`, the
original photo dump, which isn't included here (bulky, and every card in it
is already extracted into `images/extra`/`images/extra2`). It's included so
every labeling decision is visible and auditable, not just its output —
rerunning it requires supplying your own `images/raw/` with a matching
layout, or adapting its `BATCH_1`/`BATCH_2` maps to a new set of photos.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch by default. For an
NVIDIA GPU, install a CUDA build instead (match the index to your driver's
CUDA version, e.g. this project was built and measured against CUDA 12.4):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

```bash
# Retrain from images/manifest.json (~2-3 min on a GPU, longer on CPU)
python cnn_char_reader.py train

# Classify one card photo
python cnn_char_reader.py predict path/to/photo.jpg
```

## Design notes (short version — see the write-up for the full story)

- **Input is a binary glyph mask, not the raw photo.** Each card is
  segmented down to "glyph vs. not", independent of its color, so the
  network can't shortcut on the coincidence that (in the original 25
  photos) every character happened to be printed in a different color.
- **Segmentation uses the convex hull of the card's black pixels**, not its
  bounding box or a hole-fill — the only approach that survives both a
  tilted card (whose bbox includes background in its corners) and a glyph
  stroke that touches the card's physical edge (which breaks hole-filling).
- **More real photos, not a bigger model, is what actually improved
  accuracy.** A parameter-count argument for a smaller, global-average-pooling
  head was tested and rejected — measured 26 points worse, not better. Kept
  in the code as a documented negative result rather than deleted.
- **Every accuracy number here comes from a 5-seed average on a held-out
  test set**, not a single run — single runs on a test set this size swing
  by double-digit percentage points from random seed alone.
