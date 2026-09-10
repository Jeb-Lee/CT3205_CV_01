# CT3205 Assignment 1 — Custom Object Detection for Text Symbols

A YOLOv8n detector, adapted by transfer learning from COCO-pretrained weights,
that finds and reads 30 kinds of symbol card — the digits 1–9, the letters A–H
and S–Z, a circle and four arrows — live from a webcam, reporting a class,
a confidence score and a bounding box for each. Alongside the required YOLO
baseline, an independent comparison system (classical segmentation feeding a
small CNN trained from scratch) is built and measured on identical data.

The detector was improved over three rounds, and **the architecture never
changed**: every gain came from the training data. The final round —
training on cards actually held in a hand, filmed on the demo's own webcam —
took the detector from reading 1 of 8 real hand-held photos correctly to
8 of 8.

**Demo video:** https://youtu.be/7xptfqeyB-0
**Report:** [`report_build/CT3205_Assignment1_Report_rev03.docx`](report_build/CT3205_Assignment1_Report_rev03.docx) (IEEE format, 12 pages)

## Assignment deliverables

| Requirement | Where |
| --- | --- |
| Source code | this repository |
| Trained model weights | `yolo/yolo_runs/<run>/weights/best.pt` (four detectors, below); `cnn/char_cnn_model.pt` (the CNN) |
| README and reproduction instructions | this file — see [Reproducing the work](#reproducing-the-work) |
| Report (≤ 15 pages) | `report_build/CT3205_Assignment1_Report_rev03.docx` — 12 pages |
| AI Usage Declaration | report, Section IX |
| Real-time demonstration | [`demo/webcam_demo.py`](demo/); recording at https://youtu.be/7xptfqeyB-0 |

## Quick start — run the demo

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt -r requirements-yolo.txt

cd demo
python webcam_demo.py --weights ../yolo/yolo_runs/cards_v2/weights/best.pt
```

Hold a card up to the camera. `q` quits, `+`/`-` change the confidence
threshold, `p` switches to the classical + CNN pipeline. With no `--weights`
the demo runs the original baseline detector, which cannot see hand-held
cards — see [`demo/README.md`](demo/README.md) for all options, controls and
troubleshooting.

## Results

Four detectors, all YOLOv8n fine-tuned from the same COCO checkpoint with the
same settings — only the training data differs. Held-out metrics are on the
same 9 photographs / 72 instances throughout; the real-world column is 8
photographs taken during live testing (6 hand-held, 2 on a whiteboard, a
background never seen in training), counted correct at the demo's 0.25
confidence threshold.

| Run (`yolo/yolo_runs/…`) | Training images | Classes | P | R | mAP@0.5 | mAP@0.5:0.95 | Real-world |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cards` — baseline | 18 whole photos | 29 | 0.729 | 0.794 | 0.914 | 0.894 | 1 / 8 |
| `cards_scaletilt` | + 72 synthetic zoom/rotate/warp crops | 29 | 0.892 | 0.981 | 0.959 | 0.942 | 2 / 8 |
| `cards_allreal` | + 211 real single-card crops | 29 | 0.895 | 0.973 | 0.995 | 0.988 | 7 / 8 |
| **`cards_v2` — final** | + 350 hand-held webcam frames (579 total) | 30 | **0.931** | **0.990** | 0.988 | 0.983 | **8 / 8** |

What the held-out aggregates hide, the per-class and real-world columns show:
in `cards_allreal` class 4 was still at recall 0.54; in `cards_v2` it is 1.00.
Synthetic variation helped least, new real photographs helped more, and
photographs taken under the conditions of use helped most.

Comparison system (classical detection + from-scratch CNN) on the same
held-out split: 100% detection recall and 94.4% overall accuracy, against the
baseline detector's 84.7% and 76.4%. The CNN alone reads 81.4% ± 6.5% of
held-out crops, averaged over five random seeds. It has no classes for the
circle or arrows; only the YOLO detector can recognise those.

**Read these numbers with the caveats the report gives them:** every detector
figure is a single training run; the held-out set is small; and seven of the
final thirty classes (Y, Z, circle and all four arrows) have *no* instances in
the held-out split, so their accuracy is unmeasured rather than confirmed —
the evidence for them is live observation (see the demo video). The
real-world photos are kept locally (`demo/captures/`, not committed).

## Repository layout

```
yolo/                    the required YOLO detector
  scripts/
    train_yolo.py                 fine-tune YOLOv8n from COCO weights (--name, --data, --epochs)
    make_train_lists.py           write the explicit image list each variant trains on
    augment_scale_tilt.py         round 1: synthetic zoom/rotate/perspective crops  -> synth_*
    build_solo_crops.py           round 2: the CNN's real crops as detection data   -> solo_*
    extract_frame_folder.py       round 3: auto-label hand-held webcam frames       -> webcam_*
    eval_scaletilt.py             the four-detector comparison (held-out + real-world)
    run_comparison.py, eval_detection.py   YOLO vs. classical+CNN vs. hybrid
    build_yolo_dataset.py         builds the baseline dataset from raw photos (not included)
    segment_webcam_video.py, extract_webcam_video.py   video tooling (see report III-F)
  yolo_dataset/
    images/, labels/              train (all rounds) + the fixed 9-photo val split
    data.yaml, data_scaletilt.yaml, data_allreal.yaml, data_v2.yaml   one per run
    *_train.txt                   the image list each run trains on
    extra3_rejected.txt           the 51 frames rejected on visual review (round 3)
  yolo_runs/{cards, cards_scaletilt, cards_allreal, cards_v2}/   weights + training curves

cnn/                     the comparison system: classical segmentation + from-scratch CNN
  cnn_char_reader.py              segmentation, augmentation, CharCNN, train, predict
  char_cnn_model.pt, char_cnn_labels.json
  scripts/detect_cards.py         the classical card detector (also used to auto-label YOLO data)
  scripts/build_dataset.py
  images/                         img1–25 (originals), extra/, extra2/ (later shoots), manifest.json,
                                  extra3/ (446 hand-held webcam frames, round 3),
                                  extra3_vid/New_Vid.mp4 (the recording they came from)

demo/                    real-time demonstration
  webcam_demo.py                  live boxes, labels, confidences; either pipeline
  burst_capture.py                hands-free capture of auto-labelled training frames
tools/
  frame_grabber.py                pull cropped frames out of a video (used for round 3)
  identify_cards.py               GPT-4o via Azure OpenAI — a third approach, reference only
  add_subtitles.py, captions.txt  helpers for the demo video
report_build/            the IEEE report: build_report.py -> CT3205_Assignment1_Report_rev03.docx
docs/one-photo-per-card.html     earlier technical write-up of the CNN side
RngTest/                 three real-world test photos used by the report's flash-glare case
```

## Setup

Built and measured with Python 3.13 on Windows 11, an Intel Core
i5-12450HX and an NVIDIA RTX 3050 (6 GB).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # numpy, scipy, pillow, torch (CPU build)
pip install -r requirements-yolo.txt     # ultralytics 8.4.142 (brings in OpenCV)
```

For an NVIDIA GPU, install a CUDA build of PyTorch **after** the two lines
above — installing `ultralytics` silently replaces a CUDA build with a
CPU-only one (it happened twice during this project, and one training run
ended up on the CPU without anyone noticing until the log was read):

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## Reproducing the work

Everything needed to evaluate and retrain all four detectors is committed:
the images, the labels, the exact image list for each run, and the weights.
The only generated set not committed is round 1's 72 synthetic crops, which
one command regenerates.

### 1. Evaluate the committed weights (no training)

```bash
cd yolo
yolo val model=yolo_runs/cards_v2/weights/best.pt data=yolo_dataset/data_v2.yaml    # final detector, per-class table
yolo val model=yolo_runs/cards/weights/best.pt    data=yolo_dataset/data.yaml       # baseline
python scripts/eval_scaletilt.py      # all four side by side: held-out, then the real-world photos
```

`yolo val` reproduces the P / R / mAP columns in the table above exactly.
`eval_scaletilt.py`'s real-world part needs the 8 photos in `demo/captures/`,
which are not committed; it reports them as MISSING rather than failing.

### 2. Retrain each detector

```bash
cd yolo
python scripts/augment_scale_tilt.py       # regenerate round 1's synth_* crops (only uncommitted input)
python scripts/make_train_lists.py         # rewrite the four image lists; prints 18 / 90 / 229 / 579

python scripts/train_yolo.py --name repro_cards                                              # baseline
python scripts/train_yolo.py --name repro_scaletilt --data yolo_dataset/data_scaletilt.yaml
python scripts/train_yolo.py --name repro_allreal   --data yolo_dataset/data_allreal.yaml
python scripts/train_yolo.py --name repro_v2        --data yolo_dataset/data_v2.yaml         # final
```

All four use 150 epochs (early stopping after 40 without improvement), batch
8, 640×640 input, seed 42, and Ultralytics' default optimiser selection
(AdamW, lr ≈ 0.0003). On the RTX 3050 the final run takes about 90 minutes.
Expect numbers close to, not identical with, the table: GPU training is not
bit-for-bit deterministic, and each reported figure is one run.

Why each run has its own image list: every training image sits in one folder
(`yolo_dataset/images/train/`), so a `data.yaml` pointing at the folder would
make the baseline silently train on every later round's images too. Each
`data*.yaml` points at an explicit list instead. The lists use `./`-relative
paths and the yaml files have no `path:` key, so they work from any clone
without editing.

### 3. Rebuild the final round's training data from the recording

The 350 hand-held frames in `cards_v2`'s training set are committed, but the
whole chain back to the raw recording is here too:

```bash
# (a) pull cropped frames out of the recording, e.g. one every 0.5 s
python tools/frame_grabber.py cnn/images/extra3_vid/New_Vid.mp4 --roi --every 0.5 --ext jpg -o my_frames

# (b) by hand: sort the frames into cnn/images/extra3/ as <label>_Img_NNN.jpg,
#     dropping frames caught mid-swap   (the one step that is not scripted)

# (c) auto-label, applying the recorded visual-review rejections
cd yolo
python scripts/extract_frame_folder.py ../cnn/images/extra3 --classes yolo_dataset/data_v2.yaml --exclude yolo_dataset/extra3_rejected.txt
```

Step (c) on the committed `extra3/` reproduces the committed 350 training
frames **byte for byte** — verified by regenerating them. It auto-labels with
the classical detector, rejects blurry frames and per-class box-size
outliers, then skips the 51 frames listed in `extra3_rejected.txt`. Those
were rejected by eye: a dark shelf and a soft toy behind the card fused with
it under the detector's brightness threshold, producing a box that included
background. No automatic test caught them (area, aspect ratio and mask
solidity were all tried — report Section V-G). Add `--preview` to get a
contact sheet per class for exactly that kind of review.

Round 2's crops regenerate from the CNN's committed photos with
`python scripts/build_solo_crops.py`.

### 4. The comparison system

```bash
cd cnn
python cnn_char_reader.py train                        # one training run, from images/manifest.json
python cnn_char_reader.py predict ../RngTest/Test1.jpeg
cd ../yolo
python scripts/run_comparison.py                       # classical+CNN vs. YOLO vs. hybrid, held-out split
```

The report's 81.4% ± 6.5% is a mean over five seeds; the script that ran the
five seeds is not part of this repository, so a single `train` run is
expected to land somewhere inside that spread rather than on it.

### 5. The report

```bash
cd report_build
python build_report.py        # -> CT3205_Assignment1_Report_rev03.docx, from template.docx
python render.py              # docx -> PDF -> page PNGs, for checking (Windows + Word + pywin32 + pymupdf)
```

### What the repository cannot reproduce on its own

- **The raw multi-card photo dump** behind `cnn/scripts/build_dataset.py` and
  `yolo/scripts/build_yolo_dataset.py`. Their outputs (every crop, box and
  label) are committed; the scripts are included so every labelling decision
  is auditable.
- **The 8 real-world test photos** (`demo/captures/`, kept local).
- **The five-seed CNN evaluation** (see step 4).
- **Frame-for-frame round 3 sorting** — step 3(b) is manual; everything after
  it is deterministic.

## Other tools

- **`demo/burst_capture.py`** — hold a card up and move it around; it saves
  clean, auto-labelled frames about three times a second. `--roi` fences out
  background clutter. See [`demo/README.md`](demo/README.md).
- **`tools/identify_cards.py`** — a third, very different approach carried
  over from an earlier project: it sends the photo to GPT-4o through Azure
  OpenAI and asks for a plain-English description of each card. Not
  evaluated in the report and contributes to none of its numbers. Needs
  `pip install openai python-dotenv` and a `.env` (git-ignored) defining
  `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` and optionally
  `AZURE_OPENAI_DEPLOYMENT`.
- **`tools/add_subtitles.py`** — burns captions (`tools/captions.txt`) into
  a video with ffmpeg, keeping the audio track.

## Key findings (short version — the report has the full story)

- **Data, not architecture.** The one architectural experiment (a smaller,
  global-average-pooling head for the CNN) cost 26 points. Every detector
  improvement came from changing what it was shown.
- **Held-out scores can hide the failure that matters.** The baseline
  reached mAP@0.5 of 0.914 on the held-out split and returned *zero* boxes
  on hand-held cards, at any confidence threshold. The held-out split shows
  cards lying flat on a table; the demo shows them held in a hand.
- **Measure with seeds, not single runs.** Identical CNN configurations
  ranged from 73.6% to 93.1% across random seeds, so every CNN figure is a
  five-seed mean.
- **Check the data by eye.** Nine corrupted CNN training crops, and 51
  background-merged detector frames, were only caught by looking at them.
- **Symbols are not natural images.** Ultralytics' default horizontal flip
  turns a left arrow into a right arrow that is still labelled "left". This
  is diagnosed but not yet tested by an ablation (report Section VII-A).
- **The classical detector has a structural limit** that more data cannot
  fix: it thresholds on dark pixels, so a hand's shadow — or a dark object
  behind the card — can fuse with the card. The YOLO detector learns past it;
  the rule cannot.
