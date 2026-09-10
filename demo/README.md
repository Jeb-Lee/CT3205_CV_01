# Real-Time Detection Demo

Live camera demonstration of the trained symbol detector. Shows the camera
feed with bounding boxes, class labels and confidence scores drawn on every
detected symbol, plus a HUD reporting device, frame rate, current confidence
threshold and detection count.

A recording of this demo running with the final detector:
**https://youtu.be/7xptfqeyB-0**

## Run it

```bash
cd demo
python webcam_demo.py --weights ../yolo/yolo_runs/cards_v2/weights/best.pt   # final detector (recommended)
```

That's the detector used in the demo video and the one the report's final
results describe (`cards_v2`, 30 classes). Other options:

```bash
python webcam_demo.py                      # original baseline detector (the default; matches the report's baseline numbers)
python webcam_demo.py --pipeline hybrid    # classical detect + from-scratch CNN classify
python webcam_demo.py --source clip.mp4    # from a video file instead of a webcam
python webcam_demo.py --conf 0.1           # lower the confidence threshold (default 0.15)
python webcam_demo.py --cap-width 1920 --cap-height 1080   # request a capture resolution
python webcam_demo.py --scale-tilt         # the synthetic-augmentation checkpoint (superseded, see below)
python webcam_demo.py --pipeline hybrid --loose             # widen the classical detector's bounds (usually not what you want)
```

`--weights` accepts any of the four checkpoints in `yolo/yolo_runs/`. The
default was deliberately left at the original baseline so that running the
demo with no arguments reproduces the report's baseline behaviour.

Requires the packages in the repository's `requirements.txt` and
`requirements-yolo.txt` (OpenCV comes in with `ultralytics`).

`--cap-width`/`--cap-height` (default 1280×720) request a capture resolution
from the camera driver; the demo reports what it actually got, since drivers
don't always grant the exact size asked for.

## Controls

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `s` | save a screenshot to `captures/`, boxes/labels/HUD drawn on |
| `c` | save a **clean** screenshot (no overlay) -- use this, not `s`, for anything you might later add to a training set |
| `r` | start / stop recording video to `captures/` |
| `+` / `-` | raise / lower the confidence threshold |
| `p` | switch pipeline (YOLO ↔ hybrid) live |

## Two pipelines

`--pipeline yolo` (default) runs a fine-tuned YOLOv8n detector, which does
localisation and classification in a single pass. This is the assignment's
required baseline, and — with the `cards_v2` weights — the system shown in
the demo video.

`--pipeline hybrid` runs the classical detector (dark-card thresholding plus
convex-hull reconstruction) for localisation and the from-scratch CNN for
classification. Pressing `p` switches between them live.

The CNN has no classes for the pictographic symbols (circle and arrows), so
in hybrid mode it will confidently mislabel those as letters. That is
expected — it was only ever trained on the 25 letters and digits — and is a
documented limitation rather than a bug. Only the YOLO detector can
recognise the pictographs.

## Hand-held cards: what fails, and what fixed it

Holding a card up to the camera is the natural way to use this demo, and
for most of the project it was exactly what broke it. The history is worth
knowing, because it explains why there are four checkpoints:

- **Baseline (`cards`, the default)** was trained only on photos of cards
  lying flat on a table, each filling ~5–11% of the frame's width. A
  hand-held card fills 35–90%+ — a scale it never saw. It returns *zero*
  boxes on hand-held cards, even at `--conf 0.01`: no signal at all, not a
  low-confidence one. Threshold tuning cannot fix it.
- **Scale+tilt (`--scale-tilt`)** added synthetic zoomed, rotated and
  perspective-warped crops of the same photos. It detects most hand-held
  cards, but often misclassifies them (2 of 8 correct on our real-world
  sample). Superseded.
- **All-real (`cards_allreal`)** added the 211 real, individually-cropped
  card photos the CNN was already trained on. 7 of 8 correct.
- **Deployment (`cards_v2`, recommended)** added 350 frames of cards
  actually held in a hand, filmed on an iPhone rather than this webcam. **8 of 8
  correct**, and the only checkpoint with the 30th class (`down_arrow`).

Same architecture every time — every improvement came from the training
data. The full story is in the report (Sections III-E/F, V-F/G, VI-E/F).

**The hybrid pipeline still can't do hand-held cards, and more data won't
help it.** Its classical detector thresholds on dark, low-saturation pixels
to find the card; a hand holding the card puts shadowed skin against the
card's dark material, and connected-component analysis fuses hand and card
into one blob. We confirmed this by ablating the detector's area/aspect/
border parameters on the same hand-held photo: the merge happens identically
under the original settings, so it is structural, not tunable. For the hybrid
pipeline, place the card flat on a plain surface.

`--loose` widens the classical detector's bounds. We tested it as a response
to "nothing detected" and found it does **not** fix the hand-shadow merge,
while adding false positives on background clutter and on the glyph's own
negative-space curves. It is kept only for a card that's unusually large or
close to the frame edge but genuinely *not* hand-held.

## Collecting more training data: `burst_capture.py`

```bash
python burst_capture.py --label G            # hold up the G card and move it around
python burst_capture.py --label G --roi      # first drag a box to fence out background clutter
```

Hands-free: hold one card up and slowly work through near, far and tilted
poses; it saves a clean frame roughly every 0.3 s (`--interval`), rejecting
blurry frames and any frame where the card can't be found unambiguously, and
writes auto-labelled images straight into `yolo/yolo_dataset/images/train`
as `webcam_<label>_NNNN`. `space` pauses, `q` quits. Nothing drawn on the
preview is ever saved into a frame.

Use `--roi` if anything dark sits behind the card. The final round's data
lost 51 frames to exactly that — a tinted window and a soft toy fusing with the
card under the classical detector's brightness threshold, producing a box
that included background — and no automatic check caught it. Check the
results by eye before training on them (`../yolo/scripts/extract_frame_folder.py
--preview` does the same auto-labelling for a folder of frames and writes
contact sheets for exactly this check).

## Measured performance

Measured over 25 live frames at 640×480 on an Intel Core i5-12450HX with an
NVIDIA RTX 3050 (6 GB):

| Pipeline | Device | Frame rate | Latency |
| --- | --- | --- | --- |
| YOLOv8n | RTX 3050 | 87.6 FPS | 11 ms |
| YOLOv8n | CPU | 15.8 FPS | 63 ms |
| Classical + CNN | CPU/GPU mix | 24.2 FPS | 41 ms |

All three are comfortably real-time at webcam frame rates. The hybrid
pipeline's cost is dominated by connected-component analysis on the CPU
rather than by the CNN, so it scales with image resolution more steeply
than YOLO does. All four YOLO checkpoints share the same architecture, so
they run at the same speed.

## Troubleshooting

**"Could not read from camera 0"** — another application (Teams, Zoom, the
Windows Camera app) may be holding the camera. Close it and retry, or pass a
different index with `--source 1`.

On Windows the demo tries several OpenCV capture backends and verifies each
with a real frame read before accepting it. This is deliberate: the default
MSMF backend on this machine reports the camera as opened and then fails
every grab, so checking `isOpened()` alone is not sufficient.

**Nothing detected when holding a card up** — you're almost certainly
running the default baseline weights. Pass
`--weights ../yolo/yolo_runs/cards_v2/weights/best.pt`. If you're on
`--pipeline hybrid`, see the hand-shadow note above — lay the card flat.

**Some cards are shakier than others** — on `cards_v2`, "1" in particular
reads less confidently than the rest. Fill a good part of the frame with the
card and keep it fairly still; `-` lowers the confidence threshold live.

The console prints a rate-limited warning
(`[hybrid] no card-shaped region found in N consecutive frames ...`) when the
hybrid pipeline's detector has gone a while without finding anything — that
message points back to this README.
