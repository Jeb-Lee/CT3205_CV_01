"""
cnn_char_reader.py — a CNN that reads the character flashcards (digits and
letters), excluding the shape cards (circle, left arrow, right arrow).

Unrelated to the receipt tracker app and to identify_cards.py (the Azure
vision-LLM version) — this is a classic trained-from-scratch CNN, no external
AI API calls at inference time.

WHY IT'S BUILT THIS WAY
------------------------
We only have one real photograph per character (images/img1.jpg .. img25.jpg,
labelled below). A CNN trained directly on 25 raw photos would just memorize
them. Two choices compensate for that:

1. Color-invariant input. Each card is a colored glyph on a black square.
   We segment out the black card, then the glyph on it, and feed the network
   a binary silhouette (glyph vs. not) rather than raw color. Otherwise the
   network could cheat by learning "yellow -> A" (true only by coincidence in
   our one photo per class) instead of the actual shape.
2. Heavy synthetic augmentation. From each single template silhouette we
   generate many rotated / rescaled / shifted / thickened / noisy variants
   on the fly, approximating the range a new photo of the SAME physical card
   set would look like (different angle, distance, lighting). This is a
   closed-vocabulary classifier for this specific 25-card deck, not a
   general handwriting/font recognizer — it will not generalize to a
   different font or a card style it has never seen a real photo of.

Classes (25): digits 1-9, letters A-H, S-V, W-Z. Shapes (circle, left_arrow,
right_arrow — img26-28) are intentionally excluded, per the request.

Usage:
    python cnn_char_reader.py train                 # builds + trains the model
    python cnn_char_reader.py predict <image path>   # classifies one card image
"""

import os
import sys
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "images")
MANIFEST_PATH = os.path.join(IMAGES_DIR, "manifest.json")
MODEL_PATH = os.path.join(HERE, "char_cnn_model.pt")
LABELS_PATH = os.path.join(HERE, "char_cnn_labels.json")

MASK_SIZE = 64          # network input is MASK_SIZE x MASK_SIZE, 1 channel
SAMPLES_PER_CLASS = 400  # synthetic augmented samples generated per class, per epoch
EPOCHS = 40
BATCH_SIZE = 64
SEED = 42

# img<N>.jpg -> character. Shapes (26-28) deliberately omitted.
LABELS = {
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H",
    18: "S", 19: "T", 20: "U", 21: "V",
    22: "W", 23: "X", 24: "Y", 25: "Z",
}
CLASSES = [LABELS[i] for i in sorted(LABELS)]          # fixed, deterministic order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Segmentation: raw photo -> normalized binary glyph mask
# ---------------------------------------------------------------------------
def _dark_neutral_mask(rgb: np.ndarray, brightness_max=70, spread_max=12) -> np.ndarray:
    """True where a pixel is near-black AND low-saturation — the black card
    material, as distinct from colored/white content or a bluish background."""
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return (mx <= brightness_max) & ((mx - mn) <= spread_max)


def _largest_component_bbox(mask: np.ndarray):
    """Bounding box (x0, y0, x1, y1) of the largest True blob in a boolean mask."""
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == biggest)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _convex_hull_mask(component: np.ndarray) -> np.ndarray:
    """Fill in the convex hull of a boolean component — used to reconstruct
    a tilted card's full square footprint even where its outline is broken."""
    ys, xs = np.where(component)
    points = np.column_stack([xs, ys])
    hull = ConvexHull(points)
    polygon = [tuple(points[i]) for i in hull.vertices]

    canvas = Image.new("L", (component.shape[1], component.shape[0]), 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=255)
    return np.asarray(canvas) > 0


def extract_card_mask(image: "Image.Image", size: int = MASK_SIZE) -> "Image.Image | None":
    """
    Turn a photo of ONE card (with some background around it, like
    images/img1.jpg .. img25.jpg) into a centered, size x size binary glyph
    silhouette — independent of the glyph's color and largely independent of
    the card's scale/position/rotation in the source photo.

    Returns a PIL 'L' image (0 = background, 255 = glyph) or None if no card
    could be located.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)

    # Step 1: find the black card's footprint. A rotated card's AXIS-ALIGNED
    # bounding box always includes some background in its four corners
    # (think of the white paper triangles poking into the corners of a
    # tilted square's bbox), so we can't just crop to the bbox. And a glyph
    # stroke that reaches all the way to the card's physical edge (common —
    # e.g. the top of a "6") breaches the outline, so a hole-fill can leak
    # straight out through that gap rather than reconstructing the interior.
    #
    # The card itself is a (near-perfect) convex square, so instead of
    # relying on the black pixels forming a sealed ring, we take their
    # CONVEX HULL as the footprint — robust to both problems at once.
    black_mask = _dark_neutral_mask(rgb)
    labeled, n = ndimage.label(black_mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(black_mask, labeled, index=range(1, n + 1))
    card_id = int(np.argmax(sizes)) + 1
    card_component = labeled == card_id
    card_footprint = _convex_hull_mask(card_component)

    ys0, xs0 = np.where(card_footprint)
    x0, x1, y0, y1 = xs0.min(), xs0.max(), ys0.min(), ys0.max()  # bbox of the footprint itself
    footprint_crop = card_footprint[y0:y1 + 1, x0:x1 + 1]
    black_crop = black_mask[y0:y1 + 1, x0:x1 + 1]
    if footprint_crop.size == 0:
        return None

    # Step 2: the glyph is whatever is inside the card's footprint but not
    # part of the black frame — restricting to the footprint is what keeps
    # the white background corners of a tilted card out of the glyph mask.
    glyph_mask = footprint_crop & ~black_crop
    ys, xs = np.where(glyph_mask)
    if len(xs) == 0:
        return None
    # Tight bbox over ALL glyph pixels, padded a little so strokes aren't
    # clipped at the edge.
    pad = max(2, int(0.06 * max(footprint_crop.shape[:2])))
    gx0, gx1 = max(0, xs.min() - pad), min(footprint_crop.shape[1] - 1, xs.max() + pad)
    gy0, gy1 = max(0, ys.min() - pad), min(footprint_crop.shape[0] - 1, ys.max() + pad)

    glyph_crop = glyph_mask[gy0:gy1 + 1, gx0:gx1 + 1]
    if glyph_crop.size == 0:
        return None

    # Step 3: paste the tight glyph crop into a square canvas (preserving
    # aspect ratio, like MNIST-style centering) then resize to size x size.
    h, w = glyph_crop.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = glyph_crop.astype(np.uint8) * 255

    mask_img = Image.fromarray(canvas, mode="L").resize((size, size), Image.LANCZOS)
    return mask_img


# ---------------------------------------------------------------------------
# Augmentation: one template mask -> endless plausible variants
# ---------------------------------------------------------------------------
def augment_mask(mask_img: "Image.Image") -> "Image.Image":
    """Random rotation/scale/shift/thickness/noise applied to a glyph mask."""
    size = mask_img.size[0]

    # Random rotation + scale via affine transform, expand then re-crop so
    # nothing important gets pushed out of frame.
    angle = random.uniform(-25, 25)
    scale = random.uniform(0.82, 1.15)
    canvas_size = int(size * 1.6)
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    off = (canvas_size - size) // 2
    canvas.paste(mask_img, (off, off))
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, expand=False)

    new_side = max(1, int(size * scale))
    scaled = canvas.resize((new_side, new_side), Image.BILINEAR)

    # Random translation, then crop/pad back to canvas_size x canvas_size.
    out = Image.new("L", (canvas_size, canvas_size), 0)
    max_shift = int(size * 0.12)
    dx = off + (size - new_side) // 2 + random.randint(-max_shift, max_shift)
    dy = off + (size - new_side) // 2 + random.randint(-max_shift, max_shift)
    out.paste(scaled, (dx, dy))

    # Crop the center size x size region back out.
    left = (canvas_size - size) // 2
    result = out.crop((left, left, left + size, left + size))

    # Random stroke thickness (dilate/erode) to mimic font/marker variation.
    if random.random() < 0.5:
        result = result.filter(ImageFilter.MaxFilter(3))   # thicker
    elif random.random() < 0.3:
        result = result.filter(ImageFilter.MinFilter(3))   # thinner

    arr = np.asarray(result).astype(np.float32) / 255.0

    # Light salt-and-pepper / Gaussian noise.
    arr += np.random.normal(0, 0.03, arr.shape)
    arr = np.clip(arr, 0.0, 1.0)

    return arr  # float32 HxW in [0, 1]


# ---------------------------------------------------------------------------
# Dataset: generates augmented samples on the fly from the 25 templates
# ---------------------------------------------------------------------------
class CardCharDataset(Dataset):
    def __init__(self, templates: dict, samples_per_class: int):
        self.templates = templates          # label -> list of PIL mask images
        self.samples_per_class = samples_per_class
        self.labels = list(templates.keys())

    def __len__(self):
        return len(self.labels) * self.samples_per_class

    def __getitem__(self, idx):
        label = self.labels[idx % len(self.labels)]
        # Several real photos may exist per class now — pick one at random
        # each time, so every one of them gets its own share of augmentation
        # instead of always starting from a single fixed template.
        template = random.choice(self.templates[label])
        arr = augment_mask(template)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()  # (1, H, W)
        return tensor, CLASS_TO_IDX[label]


def load_templates() -> dict:
    """
    Load every real training photo and turn each into a glyph mask, grouped
    by class: label -> [mask, mask, ...].

    Reads images/manifest.json (relative path -> label) when present — this
    is how additional real photos beyond the original img1..img25 get
    included (see images/extra/ and prepare_dataset-style tooling). Falls
    back to the original one-photo-per-class scheme (img1.jpg..img25.jpg)
    if no manifest exists yet.
    """
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = {f"img{n}.jpg": label for n, label in LABELS.items()}

    templates: dict = {}
    for rel_path, label in manifest.items():
        path = os.path.join(IMAGES_DIR, rel_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Manifest entry not found on disk: {path}")
        mask = extract_card_mask(Image.open(path))
        if mask is None:
            raise RuntimeError(f"Could not segment a card/glyph out of {path}")
        templates.setdefault(label, []).append(mask)

    missing = set(CLASSES) - set(templates)
    if missing:
        raise RuntimeError(f"No training photo found for class(es): {sorted(missing)}")
    return templates


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CharCNN(nn.Module):
    """
    Three conv blocks, flattened into a small FC head.

    A global-average-pooling head was tried here (collapsing each of the
    final 64 channels to its mean activation before a small Linear layer —
    cutting parameters from ~551K to ~45K, aimed at the fact that 524,416
    of those 551K parameters sat in one Linear(4096, 128), a lot of surface
    area to overfit on a dataset built mostly from augmentation). Measured
    head-to-head on the same 5-seed, 72-example held-out evaluation, GAP
    scored 55.6% (+/-9.8%) against this FC head's 81.4% (+/-6.5%) — a large,
    clear regression, not noise. The likely reason: GAP discards *where* in
    the frame a feature fired, keeping only *whether* it fired anywhere,
    which is the right trade for a large object that could be anywhere in a
    scene. Here the glyphs are already centered and small, and telling
    "S" from "5" or "8" from "B" depends on the spatial arrangement of
    strokes relative to each other — exactly what GAP throws away. Kept as
    a documented negative result: the overfitting-surface argument for GAP
    was reasonable on paper and wrong in practice for this task.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 64 -> 32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    templates = load_templates()
    total = sum(len(v) for v in templates.values())
    counts = ", ".join(f"{c}:{len(templates[c])}" for c in CLASSES)
    print(f"Loaded {total} real training photos across {len(CLASSES)} classes ({counts})")

    full_train = CardCharDataset(templates, SAMPLES_PER_CLASS)
    full_val = CardCharDataset(templates, max(20, SAMPLES_PER_CLASS // 5))
    train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(full_val, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharCNN(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        print(
            f"epoch {epoch:2d}/{EPOCHS}  "
            f"train_loss={train_loss / train_total:.4f}  "
            f"train_acc={train_correct / train_total:.3f}  "
            f"val_acc={val_correct / val_total:.3f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(CLASSES, f)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved class list to {LABELS_PATH}")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def _load_model_for_inference():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            "No trained model found. Run `python cnn_char_reader.py train` first."
        )
    with open(LABELS_PATH) as f:
        classes = json.load(f)
    model = CharCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, classes


def predict(image_path: str, top_k: int = 3):
    model, classes = _load_model_for_inference()

    mask = extract_card_mask(Image.open(image_path))
    if mask is None:
        print("Could not find a card in that image.")
        return None

    arr = np.asarray(mask).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    ranked = sorted(zip(classes, probs.tolist()), key=lambda p: -p[1])[:top_k]
    best_label, best_prob = ranked[0]
    print(f"Predicted: {best_label}  (confidence {best_prob:.1%})")
    if top_k > 1:
        print("Top candidates:", ", ".join(f"{c} ({p:.1%})" for c, p in ranked))
    return best_label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        print("Usage:")
        print("  python cnn_char_reader.py train")
        print("  python cnn_char_reader.py predict <path-to-card-image>")
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    else:
        if len(sys.argv) != 3:
            print("Usage: python cnn_char_reader.py predict <path-to-card-image>")
            sys.exit(1)
        predict(sys.argv[2])
