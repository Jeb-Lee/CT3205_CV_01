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
```

Requires the packages in the repository's `requirements.txt` and
`requirements-yolo.txt`, plus OpenCV (`pip install opencv-python`).

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
