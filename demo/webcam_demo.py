"""
Real-time symbol detection demo.

Shows the live camera feed with bounding boxes, class labels and confidence
scores drawn on every detected symbol, plus a small HUD (FPS, device,
current confidence threshold, detection count).

Two pipelines are selectable:

  --pipeline yolo    (default) the fine-tuned YOLOv8n detector: one model
                     does localisation and classification together.
  --pipeline hybrid  the classical detector (dark-card thresholding +
                     convex hull) for localisation, with the from-scratch
                     CNN classifying each detected crop.

Both are the systems compared in the report; running them side by side on a
live feed is the most direct way to see the difference discussed there.

`--pipeline yolo --scale-tilt` swaps in a second fine-tune
(yolo/yolo_runs/cards_scaletilt) trained with extra synthetic scale/
rotation/perspective augmentation so it can actually produce a detection on
a hand-held card -- something the report's original baseline weights cannot
do at all (zero boxes at any confidence). It isn't a strict improvement:
its own classification is a little less reliable than the baseline's on the
flat-photo case the report measures. See yolo/scripts/eval_scaletilt.py and
demo/README.md's "Hold the card flat, not up" section.

Examples
--------
    python webcam_demo.py                          # webcam, YOLO
    python webcam_demo.py --pipeline hybrid        # webcam, classical + CNN
    python webcam_demo.py --scale-tilt             # webcam, hand-held-capable YOLO
    python webcam_demo.py --source clip.mp4        # from a video file
    python webcam_demo.py --source photo.jpg       # single image
    python webcam_demo.py --headless --source photo.jpg --output out.jpg

Keys (interactive mode)
-----------------------
    q / Esc   quit
    s         save a screenshot to captures/ (with boxes/labels/HUD drawn on)
    c         save a "clean" screenshot to captures/ -- no overlay, safe to
              reuse as future training data (see s vs. c note below)
    r         start / stop recording to captures/
    + / -     raise / lower the confidence threshold
    p         switch pipeline (yolo <-> hybrid)

s vs. c: 's' burns the detected box, label and HUD into the saved pixels --
good for showing the demo working, useless (actively misleading) as training
data, since a model would learn to key off the drawn box. 'c' saves the same
frame before any of that is drawn, precisely so it can be relabelled and fed
back into training later.
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CNN_DIR = os.path.join(REPO, "cnn")
sys.path.insert(0, CNN_DIR)
sys.path.insert(0, os.path.join(CNN_DIR, "scripts"))

DEFAULT_WEIGHTS = os.path.join(REPO, "yolo", "yolo_runs", "cards", "weights", "best.pt")
# The scale+tilt fine-tune (see yolo/scripts/augment_scale_tilt.py) -- trained
# on synthetic zoomed/rotated/perspective-warped crops of the same 18 source
# photos so it has actually seen a card at hand-held scale and angle, and is
# the only YOLO checkpoint in this project that produces any detection at
# all on a hand-held card. Opt into it with --scale-tilt; the plain "yolo"
# pipeline defaults to the original baseline so it matches the report's
# numbers out of the box. Its own classification is a little less reliable
# on the flat-photo case the report measures -- see eval_scaletilt.py.
SCALETILT_WEIGHTS = os.path.join(REPO, "yolo", "yolo_runs", "cards_scaletilt", "weights", "best.pt")
CAPTURES = os.path.join(HERE, "captures")

# Distinct BGR colours, cycled per class id.
PALETTE = [
    (66, 135, 245), (66, 245, 138), (245, 197, 66), (245, 66, 111),
    (188, 66, 245), (66, 233, 245), (140, 245, 66), (245, 129, 66),
]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
class YoloPipeline:
    """Fine-tuned YOLOv8n: detection and classification in one pass."""

    name = "YOLO (detect + classify)"

    def __init__(self, weights, device, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.names = self.model.names

    def __call__(self, frame, conf):
        res = self.model.predict(
            frame, conf=conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        out = []
        for box, cls, cf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            out.append((x0, y0, x1, y1, self.names[int(cls)], float(cf)))
        return out


class HybridPipeline:
    """Classical dark-card detection + from-scratch CNN classification."""

    name = "Hybrid (classical detect + CNN)"

    def __init__(self, device, loose=False):
        import torch
        from PIL import Image

        import cnn_char_reader as M
        from detect_cards import detect_cards, crop_with_padding

        self.torch = torch
        self.Image = Image
        self.M = M
        self.detect_cards = detect_cards
        self.crop_with_padding = crop_with_padding
        self.model, self.classes = M._load_model_for_inference()
        self.model.eval()

        # The classical detector's defaults (area/aspect/border tolerance)
        # were tuned on carefully-framed photos of the whole deck taken for
        # dataset construction. A card held up to a webcam by hand routinely
        # violates all three: it can easily fill more than the 25% frame-area
        # cap when held close, tilt further than 40% off-square from lens
        # distortion at short range, and sit near the edge of a narrow field
        # of view. `loose` widens all three for live use; it does not touch
        # detect_cards()'s own defaults, so the dataset-building scripts are
        # unaffected.
        self.loose = loose
        self.zero_streak = 0

    def __call__(self, frame, conf):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = self.Image.fromarray(rgb)
        if self.loose:
            boxes = self.detect_cards(
                pil, min_area_frac=0.0006, max_area_frac=0.65,
                min_aspect=0.5, max_aspect=2.0, border_margin_px=0,
            )
        else:
            boxes = self.detect_cards(pil)

        if not boxes:
            self.zero_streak += 1
            if self.zero_streak in (30, 90) or self.zero_streak % 300 == 0:
                print(f"[hybrid] no card-shaped region found in {self.zero_streak} "
                      f"consecutive frames — this pipeline never got past "
                      f"localisation, so no crop reached the classifier. See "
                      f"README 'Hold the card flat, not up' / Troubleshooting.")
        else:
            self.zero_streak = 0

        out = []
        for box in boxes:
            crop = self.crop_with_padding(pil, box)
            mask = self.M.extract_card_mask(crop)
            if mask is None:
                continue
            arr = np.asarray(mask).astype(np.float32) / 255.0
            t = self.torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(t), dim=1)[0]
            idx = int(probs.argmax())
            score = float(probs[idx])
            if score < conf:
                continue
            x0, y0, x1, y1 = (int(v) for v in box)
            out.append((x0, y0, x1, y1, self.classes[idx], score))
        return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_detections(frame, dets):
    for x0, y0, x1, y1, label, score in dets:
        colour = PALETTE[hash(label) % len(PALETTE)]
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)

        text = f"{label} {score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y0, th + 6)
        cv2.rectangle(frame, (x0, ty - th - 6), (x0 + tw + 6, ty + base - 2), colour, -1)
        cv2.putText(frame, text, (x0 + 3, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return frame


def draw_hud(frame, lines, recording=False):
    pad, lh = 8, 20
    h = pad * 2 + lh * len(lines)
    w = 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad, pad + lh * (i + 1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 25, 25), 9, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 78, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def open_source(source, cap_width=None, cap_height=None):
    """Returns (capture_or_None, still_image_or_None).

    `cap_width`/`cap_height` are only a request: cv2.VideoCapture silently
    keeps the current resolution if the device rejects the requested one, so
    the actual size actually granted is read back and reported either way.
    """
    if isinstance(source, str) and os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            img = cv2.imread(source)
            if img is None:
                raise SystemExit(f"Could not read image: {source}")
            return None, img
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {source}")
        return cap, None

    index = int(source)

    # Opening a camera successfully is not the same as being able to read from
    # it: on Windows the default MSMF backend will happily report isOpened()
    # and then fail every grab. So each backend is verified with a real read.
    backends = [(None, "default")]
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append((cv2.CAP_DSHOW, "DirectShow"))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append((cv2.CAP_MSMF, "MSMF"))

    for api, label in backends:
        cap = cv2.VideoCapture(index) if api is None else cv2.VideoCapture(index, api)
        if cap.isOpened():
            if cap_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
            if cap_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                note = "" if (w, h) == (cap_width, cap_height) else (
                    f" (requested {cap_width}x{cap_height}, camera gave {w}x{h})"
                    if cap_width else ""
                )
                print(f"camera {index} opened via {label} backend at {w}x{h}{note}")
                return cap, None
        cap.release()

    raise SystemExit(
        f"Could not read from camera {index}. Check that a webcam is connected "
        f"and not in use by another application (Teams, Zoom, the Camera app), "
        f"or pass --source with a different index or a video file."
    )


def build_pipeline(kind, args):
    if kind == "yolo":
        weights = SCALETILT_WEIGHTS if args.scale_tilt else args.weights
        return YoloPipeline(weights, args.device, args.imgsz)
    return HybridPipeline(args.device, loose=args.loose)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="camera index (default 0), or path to an image/video")
    ap.add_argument("--pipeline", choices=["yolo", "hybrid"], default="yolo")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="weights for --pipeline yolo (the original baseline "
                         "fine-tune, matching the report's numbers); "
                         "overridden by --scale-tilt")
    ap.add_argument("--scale-tilt", action="store_true",
                    help="pipeline yolo only: use the scale+tilt-augmented "
                         "fine-tune (yolo_runs/cards_scaletilt) instead of "
                         "--weights. The only YOLO checkpoint here that "
                         "produces a detection at all on a hand-held card "
                         "(the baseline returns zero boxes at any confidence "
                         "-- see yolo/scripts/eval_scaletilt.py). Its own "
                         "classification is a little less reliable than the "
                         "baseline's on the flat-photo case the report "
                         "measures, so this isn't a strict upgrade -- try it "
                         "specifically when holding a card up to the camera.")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="lower = more forgiving (more, shakier detections). "
                         "Adjustable live with +/-. Default lowered from "
                         "YOLO's usual 0.25 since a card on a novel "
                         "background often needs it.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--loose", action="store_true",
                    help="hybrid pipeline only: widen detect_cards()'s area/"
                         "aspect/border limits beyond their original, dataset- "
                         "tuned defaults. Tested and found NOT to fix a card "
                         "held in hand (that's a hand-shadow segmentation "
                         "merge, unaffected by these bounds) — it mainly adds "
                         "false positives on background clutter and on the "
                         "glyph's own negative-space curves. Off by default "
                         "for that reason; kept as an opt-in for cases where "
                         "it's genuinely useful, e.g. a card unusually large "
                         "or close to the frame edge but NOT hand-held.")
    ap.add_argument("--cap-width", type=int, default=1280,
                    help="requested camera capture width; falls back to the "
                         "camera's default if unsupported")
    ap.add_argument("--cap-height", type=int, default=720)
    ap.add_argument("--device", default=None,
                    help="'cpu', '0' for first GPU, or omit to auto-select")
    ap.add_argument("--headless", action="store_true",
                    help="no window; process frames and write --output")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = unlimited)")
    ap.add_argument("--output", default=None,
                    help="headless output path (image or .mp4)")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    if args.pipeline == "yolo":
        effective_weights = SCALETILT_WEIGHTS if args.scale_tilt else args.weights
        if not os.path.isfile(effective_weights):
            hint = (
                "Train it first: yolo/scripts/augment_scale_tilt.py then "
                "yolo/scripts/train_yolo.py --name cards_scaletilt"
                if args.scale_tilt else
                "Train the detector first (yolo/scripts/train_yolo.py) or pass --weights."
            )
            raise SystemExit(f"Weights not found: {effective_weights}\n{hint}")

    print(f"Loading pipeline '{args.pipeline}' on device '{args.device}' ...")
    pipe = build_pipeline(args.pipeline, args)
    print(f"Ready: {pipe.name}")

    cap, still = open_source(args.source, args.cap_width, args.cap_height)
    os.makedirs(CAPTURES, exist_ok=True)

    conf = args.conf
    fps_hist = deque(maxlen=30)
    writer = None
    recording = False
    frames = 0

    try:
        while True:
            if still is not None:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            t0 = time.time()
            dets = pipe(frame, conf)
            elapsed = time.time() - t0
            fps_hist.append(1.0 / elapsed if elapsed > 0 else 0.0)

            # draw_detections/draw_hud draw directly onto `frame` -- keep an
            # undrawn copy for 'c' (clean capture) before that happens, so
            # anything saved for future training data isn't burned in with a
            # box/label/HUD the model would otherwise learn to key off of.
            clean_frame = frame.copy()

            draw_detections(frame, dets)
            draw_hud(frame, [
                f"{pipe.name}",
                f"device: {args.device}    {np.mean(fps_hist):.1f} FPS "
                f"({elapsed*1000:.0f} ms/frame)",
                f"conf threshold: {conf:.2f}    detections: {len(dets)}",
                "q quit | s shot | c clean | r rec | +/- conf | p pipeline",
            ], recording)

            frames += 1

            if args.headless:
                if args.output and still is not None:
                    cv2.imwrite(args.output, frame)
                    print(f"wrote {args.output}")
                elif args.output:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            args.output, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                    writer.write(frame)
                for d in dets:
                    print(f"  frame {frames}: {d[4]} {d[5]:.2f} at {d[:4]}")
                if still is not None or (args.max_frames and frames >= args.max_frames):
                    break
                continue

            cv2.imshow("CT3205 - Symbol Detection Demo", frame)
            if recording and writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                path = os.path.join(CAPTURES, time.strftime("shot_%Y%m%d_%H%M%S.jpg"))
                cv2.imwrite(path, frame)
                print(f"saved {path}")
            elif key == ord("c"):
                path = os.path.join(CAPTURES, time.strftime("clean_%Y%m%d_%H%M%S.jpg"))
                cv2.imwrite(path, clean_frame)
                print(f"saved {path} (no box/label/HUD -- usable as future "
                      f"training data)")
            elif key == ord("r"):
                if recording:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print("recording stopped")
                else:
                    h, w = frame.shape[:2]
                    path = os.path.join(CAPTURES, time.strftime("clip_%Y%m%d_%H%M%S.mp4"))
                    writer = cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                    recording = True
                    print(f"recording to {path}")
            elif key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
            elif key in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
            elif key == ord("p"):
                kind = "hybrid" if args.pipeline == "yolo" else "yolo"
                print(f"switching to '{kind}' ...")
                args.pipeline = kind
                pipe = build_pipeline(kind, args)
                fps_hist.clear()

            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if fps_hist:
        mean_fps = float(np.mean(fps_hist))
        print(f"done: {frames} frames | {pipe.name} on '{args.device}' | "
              f"{mean_fps:.1f} FPS mean ({1000/mean_fps:.0f} ms/frame)")
    else:
        print(f"done: {frames} frames")


if __name__ == "__main__":
    main()
