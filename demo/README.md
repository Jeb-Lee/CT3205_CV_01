# Real-Time Detection Demo

Live camera demonstration of the trained symbol detector. Shows the camera
feed with bounding boxes, class labels and confidence scores drawn on every
detected symbol, plus a HUD reporting device, frame rate, current confidence
threshold and detection count.

## Run it

```bash
cd demo
python webcam_demo.py                      # webcam, YOLO detector
python webcam_demo.py --pipeline hybrid    # webcam, classical detect + CNN
python webcam_demo.py --source clip.mp4    # from a video file instead
python webcam_demo.py --conf 0.1           # lower the confidence threshold (default 0.15)
python webcam_demo.py --cap-width 1920 --cap-height 1080   # request a capture resolution
python webcam_demo.py --pipeline hybrid --loose             # widen the classical detector's bounds (see below — usually not what you want)
```

Requires the packages in the repository's `requirements.txt` and
`requirements-yolo.txt`, plus OpenCV (`pip install opencv-python`).

`--cap-width`/`--cap-height` (default 1280×720) request a capture resolution
from the camera driver; the demo reports what it actually got, since drivers
don't always grant the exact size asked for.

## Controls

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `s` | save a screenshot to `captures/` |
| `r` | start / stop recording video to `captures/` |
| `+` / `-` | raise / lower the confidence threshold |
| `p` | switch pipeline (YOLO ↔ hybrid) live |

The `r` key is the easiest way to produce the short demonstration video the
assignment asks for — it writes an `.mp4` straight into `captures/`.

## Two pipelines

`--pipeline yolo` (default) runs the fine-tuned YOLOv8n detector, which does
localisation and classification in a single pass. This is the assignment's
required baseline.

`--pipeline hybrid` runs the classical detector (dark-card thresholding plus
convex-hull reconstruction) for localisation and the from-scratch CNN for
classification. Pressing `p` switches between them live, which makes the
comparison discussed in the report directly visible: the two disagree most
on unfamiliar backgrounds, where the classical stage tends to keep finding
cards that the fine-tuned detector misses.

Note that the CNN has no classes for the pictographic symbols (circle and
arrows), so in hybrid mode it will confidently mislabel those as letters.
That is expected, and is a documented limitation rather than a bug.

### Hold the card flat, not up

Every training and evaluation photo in this project — for both the CNN and
YOLO — shows cards resting flat on a surface, never hand-held. That single
difference from a live demo is the root cause of most "nothing detected"
reports we've chased down:

- **YOLO** was fine-tuned on photos where a card occupies roughly 4–8% of
  the frame (several cards spread on a table). A hand-held card typically
  fills 35–90%+ of the frame — a scale the detector never saw. This isn't a
  confidence-threshold problem: on real hand-held captures, YOLO returns
  zero boxes even at `--conf 0.01`, meaning it produces no signal at all,
  not just a low-confidence one.
- **The hybrid pipeline's classical detector** thresholds on dark, low-
  saturation pixels to find the card's silhouette. A hand holding the card
  puts skin and shadow directly against that dark material, and shadowed
  skin can pass the same threshold — connected-component analysis then
  fuses the hand and the card into one blob, which fails the aspect-ratio
  filter (or, worse, passes it with the wrong shape) and is reported as
  "no card" or a bad label. We confirmed this by ablating the detector's
  area/aspect/border parameters on the same hand-held photo: the merge
  happens identically under the original, unmodified settings, so it isn't
  something the parameters can tune away — cropping the hand out of the
  frame by hand, by contrast, classifies correctly every time.

**Practical fix: place the card flat on a plain surface in front of the
camera instead of holding it up.** That matches the conditions both models
were actually trained/tuned on, and is what the report's own measurements
are based on.

The hybrid pipeline's detector defaults to the same strict area/aspect/
border-margin bounds the training data was built with (`--loose` opts into
wider bounds instead). We tested loosening them as a first response to
"nothing detected" and found it does **not** fix the hand-shadow merge
above — that failure is structural, not a threshold issue — while it does
add false positives on background clutter (hair, curtains) and on the
glyph's own negative-space curves. `--loose` is kept as an opt-in for the
narrower case of a card that's unusually large or close to the frame edge
but genuinely *not* hand-held; it is not a general "more forgiving" switch.

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
than YOLO does.

## Troubleshooting

**"Could not read from camera 0"** — another application (Teams, Zoom, the
Windows Camera app) may be holding the camera. Close it and retry, or pass a
different index with `--source 1`.

On Windows the demo tries several OpenCV capture backends and verifies each
with a real frame read before accepting it. This is deliberate: the default
MSMF backend on this machine reports the camera as opened and then fails
every grab, so checking `isOpened()` alone is not sufficient.

**Few or no detections on a new surface** — expected for the YOLO pipeline,
and one of the report's main findings. The detector was fine-tuned on 18
photographs spanning three backgrounds, and generalises poorly beyond them.
Hold cards against a plain, light, matte surface for the best results, avoid
camera flash and strong specular reflections, and try `--pipeline hybrid`,
which does not depend on having seen the background during training.

**Nothing detected at all, on either pipeline** — check first whether the
card is flat on a surface or held up in your hand. See "Hold the card flat,
not up" above: this is the single most common cause, and it affects both
pipelines for different underlying reasons, neither of which a confidence
or parameter tweak can fix on its own. If the card is already flat and nothing
is detected, then fall back to the new-surface guidance above.

The console prints a rate-limited warning
(`[hybrid] no card-shaped region found in N consecutive frames ...`) when the
hybrid pipeline's detector has gone a while without finding anything — that
message points back to this section.
