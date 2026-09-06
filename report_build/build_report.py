"""
Generate the CT3205 Assignment 1 report as a .docx, reusing the provided
IEEE 2-column A4 template's own styles.xml / section layout so the output
stays valid Strict OOXML.

Approach: rebuild word/document.xml inside the unpacked template rather than
authoring a new file, so every style id (papertitle, Abstract, Heading1,
BodyText, tablecolhead, figurecaption, references ...) resolves.
"""
import os
import re
import shutil
import zipfile
from xml.sax.saxutils import escape

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
UNPACKED = os.path.join(HERE, "unpacked")
MEDIA = os.path.join(UNPACKED, "word", "media")
DOC_XML = os.path.join(UNPACKED, "word", "document.xml")
RELS = os.path.join(UNPACKED, "word", "_rels", "document.xml.rels")
CT = os.path.join(UNPACKED, "[Content_Types].xml")

EMU_PER_PT = 12700
COL_WIDTH_PT = 240.0          # single column width in a 2-col IEEE A4 layout
FULL_WIDTH_PT = 500.0         # across both columns

# --------------------------------------------------------------------------
# tiny XML builders
# --------------------------------------------------------------------------
_pid = [1000]


def _next_id():
    _pid[0] += 1
    return _pid[0]


def para(text="", style="BodyText", runs=None):
    """A paragraph. `runs` overrides `text` and is a list of (text, props)."""
    if runs is None:
        runs = [(text, {})] if text else []
    out = [f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>']
    for t, props in runs:
        rpr = ""
        if props.get("i"):
            rpr += "<w:i/>"
        if props.get("b"):
            rpr += "<w:b/>"
        if props.get("sub"):
            rpr += '<w:vertAlign w:val="subscript"/>'
        if props.get("sup"):
            rpr += '<w:vertAlign w:val="superscript"/>'
        rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>')
    out.append("</w:p>")
    return "".join(out)


def heading(text, level=1):
    return para(text, style=f"Heading{level}")


def figure(rid, img_path, caption, width_pt=COL_WIDTH_PT):
    """Inline image paragraph + caption paragraph."""
    with Image.open(img_path) as im:
        w_px, h_px = im.size
    cx = int(width_pt * EMU_PER_PT)
    cy = int(cx * h_px / w_px)
    did = _next_id()
    a_ns = "http://purl.oclc.org/ooxml/drawingml/main"
    pic_ns = "http://purl.oclc.org/ooxml/drawingml/picture"
    drawing = (
        f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{did}" name="Picture {did}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{a_ns}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="{a_ns}"><a:graphicData uri="{pic_ns}">'
        f'<pic:pic xmlns:pic="{pic_ns}">'
        f'<pic:nvPicPr><pic:cNvPr id="{did}" name="Picture {did}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>'
    )
    # NB: deliberately not styled "figurecaption" — that style carries the
    # automatic "Fig. n." numbering, so using it here would consume a figure
    # number for the image itself and double-number every caption.
    img_para = (
        '<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/>'
        '<w:spacing w:before="6pt" w:after="3pt"/></w:pPr>'
        f"<w:r>{drawing}</w:r></w:p>"
    )
    return img_para + para(caption, style="figurecaption")


def table(rows, widths_pt, caption=None, head_rows=1):
    """rows: list of list of str. widths_pt sums to table width."""
    out = []
    if caption:
        out.append(para(caption, style="tablehead"))
    grid = "".join(f'<w:gridCol w:w="{int(w*20)}"/>' for w in widths_pt)
    borders = (
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:start w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:end w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
    )
    out.append(
        '<w:tbl><w:tblPr><w:tblW w:w="0pt" w:type="auto"/><w:jc w:val="center"/>'
        + borders
        + '<w:tblLayout w:type="fixed"/><w:tblLook w:firstRow="1" w:lastRow="0" '
        'w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>"
    )
    for ri, row in enumerate(rows):
        style = "tablecolhead" if ri < head_rows else "tablecopy"
        trpr = '<w:trPr><w:cantSplit/><w:jc w:val="center"/>'
        if ri < head_rows:
            trpr += "<w:tblHeader/>"
        trpr += "</w:trPr>"
        cells = []
        for ci, cell in enumerate(row):
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths_pt[ci]}pt" w:type="dxa"/>'
                f'<w:vAlign w:val="center"/></w:tcPr>{para(cell, style=style)}</w:tc>'
            )
        out.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
    out.append("</w:tbl>")
    out.append(para("", style="BodyText"))
    return "".join(out)


# --------------------------------------------------------------------------
# media / relationships
# --------------------------------------------------------------------------
def register_images(figs):
    """figs: list of (src_path,). Returns {src: rId}. Copies into word/media."""
    os.makedirs(MEDIA, exist_ok=True)
    rels = open(RELS, encoding="utf-8").read()
    existing = set(re.findall(r'Id="(rId\d+)"', rels))
    next_n = max(int(x[3:]) for x in existing) + 1

    mapping = {}
    additions = []
    exts = set()
    for src in figs:
        ext = os.path.splitext(src)[1].lower().lstrip(".")
        ext = "jpeg" if ext == "jpg" else ext
        exts.add(ext)
        name = f"image{next_n}.{ext}"
        shutil.copy(src, os.path.join(MEDIA, name))
        rid = f"rId{next_n}"
        mapping[src] = rid
        additions.append(
            f'<Relationship Id="{rid}" '
            f'Type="http://purl.oclc.org/ooxml/officeDocument/relationships/image" '
            f'Target="media/{name}"/>'
        )
        next_n += 1

    rels = rels.replace("</Relationships>", "".join(additions) + "</Relationships>")
    open(RELS, "w", encoding="utf-8").write(rels)

    ct = open(CT, encoding="utf-8").read()
    for ext in exts:
        if f'Extension="{ext}"' not in ct:
            ct = ct.replace(
                "<Default Extension=\"rels\"",
                f'<Default Extension="{ext}" ContentType="image/{ext}"/><Default Extension="rels"',
            )
    open(CT, "w", encoding="utf-8").write(ct)
    return mapping


# --------------------------------------------------------------------------
# report content
# --------------------------------------------------------------------------
FIG_SRC = {
    "annot": r"C:\Users\jeble\AppData\Local\Temp\claude\C--Users-jeble-PycharmProjects-CT3079-Capstone\6309f598-2260-42e7-932e-2bd4acbbd104\scratchpad\yolo_check_train006_full.jpg",
    "curves": os.path.join(HERE, "..", "yolo", "yolo_runs", "cards", "results.png"),
    "confusion": os.path.join(HERE, "..", "yolo", "yolo_runs", "cards", "confusion_matrix_normalized.png"),
    "preds": os.path.join(HERE, "..", "yolo", "yolo_runs", "cards", "val_batch0_pred.jpg"),
    "glare": os.path.join(HERE, "..", "RngTest", "Test2.jpeg"),
}


def build_body(rid):
    B = []
    A = B.append

    # ---------------- Abstract / keywords ----------------
    A(para(
        "Abstract—This report presents a custom object detection system for "
        "text symbols, built on a YOLOv8n detector adapted by transfer learning "
        "from COCO-pretrained weights to a 29-class dataset of photographed "
        "symbol cards. The dataset was captured, annotated and verified as part "
        "of the work, comprising 27 multi-symbol photographs containing 323 "
        "annotated instances. The fine-tuned detector reaches mAP@0.5 of 0.914 "
        "and mAP@0.5:0.95 of 0.894 on a held-out split, with markedly uneven "
        "per-class recall. Beyond the required baseline, a second, independent "
        "pipeline was implemented for comparison: a classical segmentation stage "
        "(colour-invariant thresholding and convex-hull reconstruction) feeding a "
        "small convolutional classifier trained from scratch. Evaluating both on "
        "an identical held-out set, and separating detection from classification "
        "rather than reporting one blended figure, shows the classical detector "
        "recovering 100% of instances against YOLO's 84.7%, while the dedicated "
        "classifier reads correctly-detected symbols slightly more accurately "
        "than YOLO's own head (94.4% versus 90.2%). Testing under adverse "
        "conditions reverses expectations further: on three photographs taken "
        "against previously unseen backgrounds, the fine-tuned detector produced "
        "no detections at all, while the hand-specified geometric rule succeeded "
        "on two. The results are interpreted as a consequence of limited training "
        "diversity rather than of architecture, and an inappropriate default "
        "augmentation setting is identified as a likely contributor. Live use of "
        "the accompanying real-time demonstration surfaced a further failure—"
        "neither pipeline reliably detects a card held in the hand rather than "
        "resting on a table—and a synthetic scale, rotation and perspective "
        "augmentation of the existing photographs is shown to recover the "
        "fine-tuned detector's ability to localise such cards on most test "
        "cases, though not yet its ability to classify them reliably.",
        style="Abstract"))
    A(para(
        "Keywords—object detection, YOLO, transfer learning, convolutional "
        "neural networks, symbol recognition, data augmentation, failure analysis",
        style="Keywords"))

    # ---------------- I. Introduction ----------------
    A(heading("Introduction"))
    A(para(
        "Object detection differs from image classification in that a detector "
        "must answer two questions simultaneously: what objects are present, and "
        "where each one is. This assignment concerns the detection of text "
        "symbols—digits, letters and a small number of pictographic symbols—"
        "printed individually on black cards and photographed in ordinary "
        "environments. For every symbol in an image the system must report a "
        "class label, a confidence score, and bounding-box coordinates."))
    A(para(
        "The required baseline is a YOLO detector adapted by transfer learning "
        "from a pretrained checkpoint. That baseline was implemented and forms "
        "the core of this report. However, the assignment places explicit "
        "emphasis on experimental work and analysis rather than on peak accuracy, "
        "and the most informative results here came from comparing the learned "
        "detector against a deliberately different approach and from testing both "
        "outside the conditions they were trained on."))
    A(para(
        "The contributions of this work are as follows. First, a dataset of "
        "photographed symbol cards was captured, annotated, and verified, with "
        "the annotation pipeline itself made reproducible. Second, a YOLOv8n "
        "detector was fine-tuned and evaluated with standard detection metrics. "
        "Third, an independent comparison system—classical segmentation feeding a "
        "small convolutional classifier trained from scratch—was built and "
        "evaluated on the identical held-out data, with detection and "
        "classification performance reported separately so that the source of "
        "each error is visible. Fourth, several failure modes were diagnosed to "
        "root cause rather than merely observed, including one caused by camera "
        "flash and one caused by an augmentation setting that is inappropriate "
        "for symbol data. Fifth, a measurement methodology is adopted—repeated "
        "training runs under different random seeds—after single-run results "
        "were shown to be unreliable at this dataset scale."))

    # ---------------- II. Dataset ----------------
    A(heading("Dataset"))

    A(heading("Data Collection", 2))
    A(para(
        "The dataset consists of photographs of a physical set of symbol cards. "
        "Each card is a matte black square bearing a single symbol printed in one "
        "of five colours. The full symbol inventory comprises the digits 1 to 9, "
        "the letters A to H and S to Z, and four pictographic symbols: a filled "
        "circle and left-, right- and up-pointing arrows, giving 29 classes in "
        "total. This exceeds the required minimum of ten classes."))
    A(para(
        "Photographs were captured in three sessions using a mobile phone camera "
        "under ordinary indoor lighting. The sessions deliberately varied the "
        "conditions: cards were photographed on a clipboard, on two different "
        "marble surfaces, and on a woven mat, from a range of heights and "
        "viewing angles, with individual cards left rotated by up to roughly "
        "twenty degrees. A later session spaced the cards further apart after an "
        "earlier attempt with cards placed edge to edge proved problematic for "
        "annotation, as described below. Approximately ninety photographs were "
        "captured in total, of which twenty-seven are used for detector training "
        "and evaluation; the remainder were either reserved, used for the "
        "comparison classifier, or excluded for reasons documented in Section "
        "II-E."))

    A(heading("Annotation Methodology", 2))
    A(para(
        "Annotating bounding boxes by hand for several hundred instances would "
        "have been slow and error-prone. Instead, annotation was semi-automated. "
        "A deterministic classical detector locates candidate cards in a "
        "photograph by thresholding for pixels that are simultaneously dark and "
        "low in saturation—the signature of the black card material as distinct "
        "from coloured symbols or ordinary backgrounds—then grouping those pixels "
        "into connected components and filtering by area and aspect ratio. Each "
        "candidate is cropped and assembled into an indexed contact sheet, from "
        "which the true class of every candidate is read off visually and "
        "recorded. Because the detector is deterministic, re-running it on the "
        "same photograph reproduces the same candidates in the same order, so "
        "box coordinates can be regenerated from the recorded index-to-label maps "
        "without re-annotating anything."))
    A(para(
        "This has a useful property for reproducibility: the labelling decisions "
        "are stored as explicit, auditable maps in source control rather than as "
        "opaque coordinate files, and the script that converts them into "
        "YOLO-format annotations is included in the repository."))

    A(heading("Dataset Composition", 2))
    A(table(
        [["Property", "Value"],
         ["Classes", "29 (25 characters, 4 pictographs)"],
         ["Images (train)", "18"],
         ["Images (validation)", "9"],
         ["Annotated instances (train)", "251"],
         ["Annotated instances (val)", "72"],
         ["Instances per image", "6 to 28"],
         ["Image resolutions", "1200x1600, 1600x1200, 1536x2048, 2048x1536"],
         ["Annotation format", "YOLO (class, cx, cy, w, h; normalised)"],
         ["Split strategy", "By source photograph, not by instance"]],
        [110, 130],
        caption="Dataset composition"))
    A(para(
        "The split is by photograph rather than by instance. This matters: "
        "splitting individual boxes at random would place crops of the same "
        "physical card in the same lighting and pose on both sides of the split, "
        "which would inflate validation scores. Every validation photograph is "
        "one the detector has never seen in any form."))
    A(para(
        "Class distribution is uneven, and unavoidably so, because each "
        "photograph contains whichever subset of the deck was laid out at the "
        "time. In the validation split, classes such as T, U, V, W, X, Y and Z "
        "appear four to five times each, while several digits appear only once or "
        "twice. This imbalance is revisited in Section V when interpreting "
        "per-class recall, where a single missed instance in a one-instance class "
        "moves that class's recall from 1.0 to 0.0."))

    A(heading("Annotation Verification", 2))
    A(para(
        "The assignment requires that annotations be verified before training. "
        "Every generated annotation file was rendered back onto its source "
        "photograph and inspected. Figure 1 shows one such verification image."))
    A(figure(rid[FIG_SRC["annot"]], FIG_SRC["annot"],
             "Annotation verification. Generated YOLO boxes rendered back "
             "onto a training photograph. All twelve annotated instances in this "
             "image are correctly placed and labelled; unannotated cards in the "
             "frame belong to a different source-photograph subset."))
    A(para(
        "This step caught a real defect. An initial inspection at reduced "
        "resolution appeared to show missing boxes, which prompted a check of the "
        "underlying label files; the labels were in fact complete, and the "
        "apparent absence was an artefact of the preview scale. The lesson "
        "retained was that visual verification must be performed at a resolution "
        "where the smallest annotation is legible."))

    A(heading("Data Quality Defects Discovered", 2))
    A(para(
        "A more consequential defect surfaced later. The comparison classifier "
        "described in Section III-B is trained on individual card crops, and an "
        "audit of every crop's segmentation output—not merely its pixel "
        "statistics—revealed that nine of them were silently corrupted. In some, "
        "a neighbouring card had fused with the target; in others, a shadowed "
        "surface adjacent to the card had merged with it during thresholding, so "
        "that the reconstructed card outline extended well beyond the physical "
        "card and swept background into the symbol mask."))
    A(para(
        "Four of these came from the very first batch of crops, and had gone "
        "undetected through several rounds of training. The reason is "
        "instructive: they had been checked only by pixel count, a statistic that "
        "cannot distinguish a correct symbol from an incorrect shape of similar "
        "area. They also could not be caught by disagreement, because at that "
        "stage each class had exactly one example, so a corrupt example simply "
        "became that class's definition. The defect only became visible once "
        "multiple genuine examples per class existed and could contradict one "
        "another. Two further sources were excluded wholesale: a photograph taken "
        "on a dark, patterned mat, whose background repeatedly defeated the "
        "black-card threshold, and a recurring oversized crop of one letter."))

    # ---------------- III. Methodology ----------------
    A(heading("Methodology"))

    A(heading("YOLO Baseline with Transfer Learning", 2))
    A(para(
        "The baseline detector is YOLOv8n, the smallest model in the YOLOv8 "
        "family, implemented with the Ultralytics framework [2]. Training starts "
        "from the publicly released COCO-pretrained checkpoint rather than from "
        "random initialisation, as required. The rationale for expecting transfer "
        "to help is specific: the low-level features a detector needs—edges, "
        "corners, closed contours, figure-ground separation—are largely "
        "domain-independent, so a backbone pretrained on a large natural-image "
        "detection corpus [3] begins with a substantial head start on locating a "
        "small rectangular object, even though no COCO class resembles a symbol "
        "card. Only the detection head must learn the new 29-class vocabulary."))
    A(table(
        [["Setting", "Value"],
         ["Model", "YOLOv8n (COCO-pretrained)"],
         ["Parameters", "3,011,303 (fused), 72 layers"],
         ["Compute", "8.1 GFLOPs"],
         ["Input image size", "640 x 640"],
         ["Classes", "29"],
         ["Epochs", "150 (patience 40)"],
         ["Batch size", "8"],
         ["Optimizer", "AdamW (auto-selected)"],
         ["Initial learning rate", "0.000303"],
         ["Momentum", "0.9"],
         ["Weight decay", "0.0005"],
         ["Random seed", "42"],
         ["Hardware", "Intel Core i5-12450HX (CPU)"],
         ["Training time", "744 s (12.4 min)"]],
        [110, 130],
        caption="YOLO training configuration"))
    A(para(
        "One configuration detail is worth recording because it was not "
        "intended. Although an NVIDIA RTX 3050 with 6 GB of memory was available, "
        "training ran on CPU. Installing the Ultralytics package into the shared "
        "environment replaced the CUDA build of PyTorch with a CPU-only build "
        "without warning, and the substitution was noticed only when the training "
        "log reported the device. Training was fast enough at this dataset size "
        "that the run was not repeated, but the incident is documented because it "
        "is exactly the kind of silent environment change that invalidates timing "
        "comparisons."))

    A(heading("Comparison System: Segmentation and a From-Scratch CNN", 2))
    A(para(
        "The second pipeline separates the two questions that YOLO answers "
        "jointly. Localisation is performed by the classical detector already "
        "described. Classification is performed by a small convolutional network "
        "trained from scratch on individual card crops."))
    A(para(
        "The segmentation stage is worth describing because two of its design "
        "decisions were forced by failures. A card photographed at an angle is a "
        "rotated square, and the axis-aligned bounding box of a rotated square "
        "necessarily contains background in its four corners; an early version "
        "that treated everything non-black inside that box as symbol therefore "
        "swept those corners into the mask and severely distorted tilted cards. "
        "Filling the interior of the black region instead fails differently: some "
        "symbols, such as the digit 6, have a stroke that reaches the physical "
        "edge of the card, so the interior is not an enclosed region and nothing "
        "is filled. The resolution used throughout is to take the convex hull of "
        "the card's black pixels. A physical card is convex, so its hull is its "
        "true footprint regardless of whether its outline happens to be closed, "
        "and both failure modes disappear at once."))
    A(para(
        "The classifier receives a 64x64 binary mask—symbol versus not-symbol—"
        "rather than the colour crop. This is deliberate. In the source deck each "
        "symbol happens to be printed in a consistent colour, so a network given "
        "colour could reach high accuracy by learning the colour-to-class mapping "
        "while learning nothing about shape, and would fail immediately on a "
        "differently coloured card. Discarding colour before training removes the "
        "shortcut. The network itself has three convolutional blocks (16, 32 and "
        "64 channels, each with batch normalisation, ReLU and 2x2 max pooling) "
        "followed by a fully connected head, totalling approximately 551,000 "
        "parameters. It is trained on synthetic variants generated on the fly "
        "from the real crops by random rotation, scaling, translation, stroke "
        "thickening or thinning, and additive noise."))

    A(heading("Evaluation Protocol", 2))
    A(para(
        "Detection performance is reported with precision, recall, mAP@0.5 and "
        "mAP@0.5:0.95. For the pipeline comparison, a further distinction is "
        "drawn that a single accuracy figure would hide. A prediction is matched "
        "to a ground-truth instance greedily by intersection over union at a "
        "threshold of 0.5. Detection recall is then the proportion of real "
        "symbols matched by any prediction; classification accuracy given "
        "detection is the proportion of matched symbols whose predicted class is "
        "also correct; and overall accuracy requires both. A system can fail by "
        "not finding a symbol or by finding it and misreading it, and these "
        "demand different remedies."))
    A(para(
        "For the classifier, every reported figure is the mean over five "
        "independent training runs with different random seeds, with the "
        "standard deviation quoted. The reason is given in Section IV-E."))

    A(heading("Real-Time Demonstration", 2))
    A(para(
        "Both pipelines were packaged into an interactive real-time "
        "demonstration (webcam_demo.py) to satisfy the assignment's live "
        "demonstration requirement. It reads a live camera feed or a recorded "
        "video, draws each detected box, its predicted class and its "
        "confidence score on the frame, and overlays frame rate and the "
        "active confidence threshold. Either pipeline can be selected at "
        "start-up and swapped at run time without restarting the process, "
        "and screenshots or short clips can be captured directly from the "
        "running demonstration. This tool is also how the failure described "
        "in Section VI-E was found: it surfaced during ordinary use of the "
        "demonstration, not from a planned experiment."))

    # ---------------- IV. Experimental Design ----------------
    A(heading("Experimental Design"))
    A(para(
        "Five experiments were conducted. Each varies one factor and measures its "
        "effect on the same held-out data."))

    A(heading("Experiment 1: Model Comparison", 2))
    A(para(
        "Three pipelines are compared on an identical held-out split of nine "
        "photographs containing 72 instances: the classical detector with the "
        "from-scratch classifier; the fine-tuned YOLO detector using its own "
        "classification head; and a hybrid using YOLO for localisation and the "
        "from-scratch classifier for reading. The hybrid isolates which component "
        "is responsible for any difference between the first two. Hypothesis: the "
        "learned detector will localise more robustly, and the dedicated "
        "classifier will read more accurately."))

    A(heading("Experiment 2: Training Data Volume", 2))
    A(para(
        "The classifier was trained with progressively more real photographs per "
        "class—one, then approximately four, then approximately eight—holding "
        "architecture fixed. Hypothesis: accuracy is limited by the number of "
        "genuinely distinct real examples rather than by model capacity, so it "
        "should rise with data volume even though the architecture is unchanged."))

    A(heading("Experiment 3: Classifier Head Architecture", 2))
    A(para(
        "Ninety-five percent of the classifier's parameters lie in a single "
        "fully connected layer that maps the flattened 8x8x64 feature map to 128 "
        "units. Global average pooling is the standard remedy for such a "
        "parameter concentration [7], reducing the model to approximately 45,000 "
        "parameters. Hypothesis: with a training set dominated by synthetic "
        "augmentation of a small number of real examples, reducing this "
        "overfitting surface should improve generalisation."))

    A(heading("Experiment 4: Training Duration", 2))
    A(para(
        "Training length was varied between 15 and 40 epochs at two dataset "
        "sizes, monitoring training accuracy alongside held-out accuracy to "
        "distinguish underfitting from overfitting. Hypothesis: as the real "
        "dataset grows, the training distribution becomes harder to fit and a "
        "schedule tuned on the smaller dataset will leave the model underfitted."))

    A(heading("Experiment 5: Measurement Reliability", 2))
    A(para(
        "This experiment concerns the measurement instrument rather than the "
        "model. The same configuration was trained five times under different "
        "random seeds and evaluated on the same held-out data, in order to "
        "quantify how much of an observed difference between configurations could "
        "be attributed to training stochasticity alone."))

    # ---------------- V. Results ----------------
    A(heading("Results"))

    A(heading("Detector Performance", 2))
    A(table(
        [["Metric", "Value"],
         ["Precision", "0.729"],
         ["Recall", "0.794"],
         ["mAP@0.5", "0.914"],
         ["mAP@0.5:0.95", "0.894"]],
        [110, 130],
        caption="YOLOv8n detection performance on the held-out split"))
    A(para(
        "The aggregate figures are respectable for a model fine-tuned on "
        "eighteen photographs. mAP@0.5:0.95 of 0.894 sitting close to mAP@0.5 of "
        "0.914 indicates that when the detector does find a card, it localises it "
        "tightly—unsurprising, since the objects are high-contrast quadrilaterals "
        "with unambiguous boundaries. The interesting information is in the "
        "spread across classes rather than in the mean."))
    A(figure(rid[FIG_SRC["curves"]], FIG_SRC["curves"],
             "YOLOv8n training and validation curves over 150 epochs. "
             "Box, classification and distribution-focal losses decrease "
             "throughout; precision, recall and mAP rise steeply after "
             "approximately epoch 30 and plateau by epoch 120."))
    A(para(
        "Per-class results are markedly uneven. Fourteen classes reach recall of "
        "1.0, but several do not. The letter F was never detected at all, giving "
        "recall of 0.0; T reaches precision 0.200 and recall 0.150; Z reaches "
        "recall 0.280; A reaches 0.333; and U and V reach 0.488 and 0.455 "
        "respectively. The classes that fail are not obviously harder to see than "
        "those that succeed, which points to training coverage rather than "
        "intrinsic difficulty—those classes appear in fewer training photographs "
        "and in fewer distinct positions within them."))
    A(figure(rid[FIG_SRC["confusion"]], FIG_SRC["confusion"],
             "Normalised confusion matrix for the fine-tuned detector on "
             "the held-out split. Off-diagonal mass is concentrated in a small "
             "number of visually similar pairs and in the background row, the "
             "latter corresponding to missed detections."))
    A(figure(rid[FIG_SRC["preds"]], FIG_SRC["preds"],
             "Qualitative detections on a held-out photograph, with "
             "predicted class and confidence. Most symbols are detected and read "
             "correctly; missed and misread instances are discussed in the text."))

    A(heading("Pipeline Comparison", 2))
    A(table(
        [["Pipeline", "Recall", "Prec.", "Cls|det", "Overall"],
         ["Classical detect + CNN", "100%", "76.6%", "94.4%", "94.4%"],
         ["YOLO (detect + classify)", "84.7%", "71.8%", "90.2%", "76.4%"],
         ["Hybrid: YOLO + CNN", "84.7%", "71.8%", "93.4%", "79.2%"]],
        [95, 36, 36, 36, 37],
        caption="Three-pipeline comparison on the identical held-out "
                "split (72 instances). Cls|det is classification accuracy given "
                "a correct detection."))
    A(para(
        "The hypothesis was half right and half wrong. The dedicated classifier "
        "does read more accurately than YOLO's own head: comparing rows two and "
        "three, which share identical detections, accuracy rises from 90.2% to "
        "93.4% purely by substituting the classifier. That is a modest but real "
        "advantage, and it is what one would expect from a network specialised "
        "for a single 25-way discrimination against a lightweight head trained "
        "jointly with localisation."))
    A(para(
        "The prediction about localisation was wrong, and decisively so. The "
        "classical detector found every one of the 72 instances; YOLO missed "
        "eleven. Because the hybrid inherits YOLO's localisation, that gap—not "
        "the classifier—determines the ranking. The lower precision of the "
        "classical detector, at 76.6%, reflects spurious candidate regions that "
        "survive its area and aspect filters; in a system where a downstream "
        "classifier can reject them, missing a symbol is the more costly error."))

    A(heading("Effect of Training Data Volume", 2))
    A(table(
        [["Real photos", "Epochs", "Test set", "Accuracy"],
         ["99 (cleaned)", "30", "25 inst.", "68.8% ± 1.6%"],
         ["104 (uncleaned)", "30", "25 inst.", "77.6% ± 6.5%"],
         ["211 (cleaned)", "40", "72 inst.", "81.4% ± 6.5%"]],
        [70, 40, 55, 75],
        caption="Classifier accuracy versus training data volume, mean "
                "and standard deviation over five seeds."))
    A(para(
        "Accuracy rose from roughly 69% to roughly 81% as the number of real "
        "photographs per class grew from one to about eight, with the "
        "architecture held constant throughout. The comparison must be qualified "
        "honestly: the final row is measured on a larger held-out set than the "
        "first two, and epoch count differs, so these rows are not a controlled "
        "ablation of data volume alone. The middle row is also anomalous—removing "
        "corrupted training examples appeared to reduce mean accuracy—which at "
        "that sample size is not a reliable signal and should not be read as "
        "evidence that corrupt data helps."))

    A(heading("Classifier Head Architecture", 2))
    A(table(
        [["Head", "Parameters", "Accuracy (5 seeds)"],
         ["Flatten + FC(4096→128)", "≈551,000", "81.4% ± 6.5%"],
         ["Global average pooling", "≈45,000", "55.6% ± 9.8%"]],
        [95, 65, 80],
        caption="Effect of replacing the fully connected head with "
                "global average pooling."))
    A(para(
        "The hypothesis was wrong by a wide margin. Removing 92% of the "
        "parameters cost 26 points of accuracy, and the smaller model was also "
        "more variable across seeds rather than less—the opposite of what a "
        "successful regularisation change produces. The explanation is that "
        "global average pooling discards where in the frame a feature activated, "
        "retaining only whether it activated. That is the correct trade for a "
        "large object appearing anywhere in a scene, but these symbols are "
        "already cropped and centred by the segmentation stage, and "
        "distinguishing S from 5, or 8 from B, depends precisely on the spatial "
        "arrangement of strokes relative to one another. The change was reverted; "
        "the reasoning and the measurement are retained in the source as a "
        "documented negative result."))

    A(heading("Training Duration and Measurement Reliability", 2))
    A(para(
        "Training accuracy, not held-out accuracy, revealed that the classifier "
        "was underfitting after the dataset grew. At 15 epochs training accuracy "
        "reached only 77%; extending to 40 epochs raised it to 95.6% on the same "
        "data. A schedule adequate for one example per class was inadequate once "
        "each class had eight genuinely different examples to reconcile."))
    A(para(
        "Experiment 5 produced the most methodologically important result in "
        "this work. Identical configurations, differing only in random seed, "
        "produced held-out scores ranging from 73.6% to 93.1%. Earlier in the "
        "project, single-run comparisons had appeared to show a configuration "
        "change moving accuracy from 76% to 60%; repeated under five seeds, that "
        "apparent effect was within the noise. On a held-out set of this size a "
        "single flipped instance moves the score by more than a percentage point, "
        "and training stochasticity moves it by considerably more. Every "
        "classifier figure in this report is therefore a five-seed mean, and "
        "single-run numbers are not treated as evidence."))

    # ---------------- VI. Challenging scenarios ----------------
    A(heading("Challenging Scenarios"))
    A(para(
        "Four adverse conditions were tested. Three were tested on purpose-taken "
        "photographs; the fourth emerged during dataset construction."))

    A(heading("Camera Flash on a Reflective Background", 2))
    A(para(
        "A card was photographed with flash on a black rubber mat with raised "
        "reflective dimples. The classifier reported the letter D, with full "
        "confidence, for a card bearing T. Inspecting the mask actually presented "
        "to the network explains why, and exonerates the classifier: the flash "
        "blew out roughly the lower half of the card, and each dimple on the mat "
        "returned enough light to register as foreground. The segmentation stage "
        "assumes background is darker than the symbol; flash on a glossy dark "
        "object near a reflective surface violates that assumption directly. What "
        "reached the classifier was largely noise, and D was simply its least-bad "
        "interpretation. This is a segmentation failure, not a classification "
        "failure, and the distinction matters because the remedies differ."))
    A(figure(rid[FIG_SRC["glare"]], FIG_SRC["glare"],
             "Failure case. Flash illumination on a glossy card against a "
             "reflective dimpled mat. Specular highlights on both the card and "
             "the background defeat the dark-background assumption on which the "
             "segmentation stage depends."))

    A(heading("Unseen Backgrounds", 2))
    A(para(
        "Three photographs were taken against backgrounds absent from all "
        "training data: a woven carpet, a household appliance with a tiled floor "
        "and a foot in frame, and the reflective mat above. The result was the "
        "reverse of expectation. The fine-tuned detector produced no detections "
        "on any of the three, even at a permissive confidence threshold, while "
        "the classical geometric rule located and correctly read two, and on the "
        "third produced no detection rather than a confident error—its area and "
        "aspect filters rejected the glare region that the single-card path had "
        "accepted."))
    A(para(
        "The interpretation is that the advantage usually attributed to a "
        "learned detector—that it recognises objects by appearance rather than by "
        "a brittle hand-specified rule—is contingent on having seen enough "
        "variation to generalise. Trained on eighteen photographs spanning three "
        "surfaces, the detector learned the appearance of cards on those "
        "surfaces. The threshold rule never specialised to any surface, because "
        "it encodes a property of the cards themselves, and so transferred "
        "without difficulty."))

    A(heading("Adjacent and Touching Symbols", 2))
    A(para(
        "An early photography session placed cards edge to edge. Because the "
        "classical detector groups pixels into connected components, touching "
        "cards merge into a single region and are annotated and detected as one "
        "object. This is a structural limitation of the classical approach that a "
        "learned detector does not share: YOLO's box-proposal mechanism is "
        "designed to separate adjacent instances. The practical mitigation "
        "adopted was to re-photograph with visible gaps, which is acceptable for "
        "dataset construction but would not be acceptable in deployment."))

    A(heading("Rotation and Visually Similar Classes", 2))
    A(para(
        "Rotation was constrained by design. Both systems assume symbols are "
        "approximately upright, within roughly twenty-five degrees, and the "
        "augmentation used to train the classifier reflects that range. Beyond "
        "it, an unavoidable ambiguity appears: a 6 rotated by 180 degrees is "
        "indistinguishable in shape from a 9. Because colour is deliberately "
        "discarded, no shape-based system can resolve that pair under free "
        "rotation; only the deck's colour convention could, at the cost of the "
        "very shortcut the design removes."))
    A(para(
        "Visually similar pairs account for most of the remaining classification "
        "errors in both systems, but the pairs differ. The from-scratch "
        "classifier confuses 1 with 2, V with W, 1 with 4 and 9 with S. The "
        "detector's own head confuses T with E repeatedly, and also 5 with F and "
        "F with 4. That the two systems fail on different pairs, given identical "
        "inputs, indicates the confusions arise from what each learned rather "
        "than from an intrinsic property of the symbols."))

    A(heading("Hand-Held Cards at Deployment Time", 2))
    A(para(
        "Live use of the real-time demonstration (Section III-D) surfaced a "
        "failure absent from every prior test: neither pipeline reliably "
        "detects a card held up to the camera in a hand, as opposed to one "
        "resting on a table. Diagnosis traced this to two distinct causes "
        "that happen to share one root."))
    A(para(
        "For the fine-tuned detector the cause is scale. Every training "
        "photograph shows several cards spread on a table, each occupying "
        "roughly 5-11% of the frame's linear dimension per side; a card held "
        "up to a webcam typically fills 35-90% or more of the frame instead. "
        "On real hand-held photographs the detector produced zero boxes at "
        "every confidence threshold tested, down to 0.01—not a low-confidence "
        "detection but no detection signal at all, which confidence-threshold "
        "tuning cannot fix. For the classical pipeline the cause is different: "
        "its threshold-and-connected-component stage groups shadowed skin "
        "against the card's dark material into a single region, merging hand "
        "and card into one blob that fails the aspect-ratio filter or passes "
        "it with the wrong shape. A controlled test that varied the "
        "detector's area, aspect and border parameters across four settings, "
        "including its original unmodified defaults, reproduced this merge "
        "identically in every case, showing it is not an artefact of "
        "parameter tuning but a structural property of thresholding on "
        "brightness and saturation alone. Both failures trace to the same "
        "underlying gap: no photograph anywhere in this project's training or "
        "evaluation data shows a card held in a hand."))
    A(para(
        "A targeted fix was attempted for the fine-tuned detector, since its "
        "failure—unlike the classical detector's—is a property of what it was "
        "shown rather than of its detection rule, and is therefore addressable "
        "with more representative data. Seventy-two synthetic training images "
        "were generated from the same eighteen real photographs already used: "
        "each is a crop centred on one existing annotated card, resized so the "
        "card fills 35-80% of the new frame, rotated in-plane by up to thirty "
        "degrees, and warped by a small random projective transform to "
        "approximate an off-axis viewing angle. Every box's label, including "
        "for cards other than the one a crop was centred on, was re-derived "
        "by mapping its corners through the same transform and taking their "
        "axis-aligned bounding box, rather than re-annotated by hand. No new "
        "photography was involved."))
    A(table(
        [["Measure", "Baseline", "Scale+tilt"],
         ["Held-out recall", "84.7%", "100%"],
         ["Held-out cls. given det.", "90.2%", "100%"],
         ["Hand-held photos, any detection", "0 / 6", "5 / 6"],
         ["Hand-held photos, correctly classified", "0 / 6", "2 / 6"]],
        [150, 45, 45],
        caption="Detector performance before and after scale+tilt "
                "augmentation, on the original held-out split and on six "
                "real hand-held photographs. Single run for each "
                "configuration; see Section VII-C."))
    A(para(
        "Retraining on the enlarged set left held-out performance unchanged "
        "or improved—recall and classification-given-detection both reached "
        "100% on the original nine-photograph split, though a jump this size "
        "on a 72-instance set from a single run should be read cautiously, "
        "consistent with this report's treatment of small-sample single-run "
        "figures elsewhere. On the six real hand-held photographs, where the "
        "baseline produced no detections whatsoever, the retrained detector "
        "produced a detection on five, two of which were also correctly "
        "classified, one at a confidence above the demonstration's default "
        "threshold. This is a genuine but partial result: three of the five "
        "detections were misclassified, and the underlying scale gap is "
        "narrowed rather than eliminated."))
    A(para(
        "One further attempt is reported because it failed instructively. "
        "Section V-B's pipeline comparison shows the from-scratch classifier "
        "reading correctly-detected symbols more accurately than YOLO's own "
        "head, so pairing the retrained detector's localisation with that "
        "classifier seemed a natural way to combine the fix above with the "
        "project's more accurate classifier. On three sampled hand-held "
        "photographs it did the opposite: none were correctly classified, "
        "against two of three for the detector's own head, and the wrong "
        "answers carried higher confidence, up to 0.99, than the correct "
        "answers they replaced. The cause is that the retrained detector's "
        "hand-held boxes, while real, are imprecise—one measured 1226 by 671 "
        "pixels on a roughly 1280 by 720 frame, closer to \"somewhere in this "
        "image\" than a tight box—and the classifier's convex-hull "
        "segmentation stage requires a tight single-card crop to locate the "
        "correct region. A loose box causes it to segment confidently and "
        "wrongly, an instance of the same confident-misclassification pattern "
        "as the flash case in Section VI-A, now triggered by an imprecise box "
        "rather than an imprecise photograph. The demonstration therefore "
        "uses the retrained detector's own classification, despite it being "
        "the less accurate classifier everywhere else in this project."))
    A(para(
        "This episode is a sharper version of the argument developed in "
        "Section VII-B: a detector shown insufficient training diversity does "
        "not merely perform worse, it can fail completely, and synthetic "
        "diversity constructed from data already on hand recovers a real "
        "fraction of that capability without any architectural change. It "
        "also shows the limit of that argument: recovering detection did not "
        "recover classification, and the classical detector's hand-shadow "
        "merge, being a property of a hand-specified rule rather than of "
        "training data, is not addressable by this kind of fix at all."))

    # ---------------- VII. Discussion ----------------
    A(heading("Discussion"))

    A(heading("An Inappropriate Default Augmentation", 2))
    A(para(
        "Reviewing the recorded training configuration after the fact revealed a "
        "setting that should not have been left at its default. Ultralytics "
        "enables horizontal flipping with probability 0.5, together with mosaic "
        "composition, random erasing, scaling, translation and HSV jitter. For "
        "natural objects horizontal flipping is sound: a mirrored dog is still a "
        "dog. For text symbols it is not. A mirrored S is not an S, and the "
        "consequence for the arrow classes is worse than merely unhelpful: "
        "mirroring a left-pointing arrow produces an image of a right-pointing "
        "arrow that is still labelled left arrow. Half of all arrow training "
        "examples were therefore label-corrupted by construction."))
    A(para(
        "Of the enabled augmentations, scaling, translation, mild HSV jitter and "
        "mosaic are defensible for this problem; random erasing is questionable "
        "for symbols whose identity depends on complete strokes; horizontal and "
        "vertical flipping should be disabled outright; and small-angle rotation, "
        "which is disabled by default, should arguably be enabled, since cards in "
        "these photographs are routinely rotated by ten to twenty degrees. This "
        "was diagnosed by inspection rather than by ablation, so it remains a "
        "well-founded hypothesis rather than a measured result. Re-training with "
        "flipping disabled is the single most immediate next experiment, and it "
        "is expected to improve both the arrow classes and overall "
        "classification."))

    A(heading("Data Volume Against Model Capacity", 2))
    A(para(
        "A consistent pattern runs through the experiments. Where performance "
        "improved, it improved because the model was shown more or better data: "
        "increasing real examples per class from one to eight, and removing "
        "corrupted examples, moved the classifier substantially. Where "
        "performance was pursued through architecture instead, it did not "
        "improve: reducing the head by 92% cost 26 points, and a detector with "
        "roughly six times the classifier's parameters and a large-scale "
        "pretrained backbone nonetheless failed to generalise to unseen "
        "backgrounds that a threshold rule handled without difficulty."))
    A(para(
        "This should not be read as an argument against learned detectors. It is "
        "an argument about matching model generality to data diversity. Eighteen "
        "photographs spanning three surfaces is not enough diversity for a "
        "detector to learn a surface-independent concept of the object, in the "
        "same way that one photograph per class was not enough for a classifier "
        "to learn a pose-independent concept of a symbol. The remedy in both "
        "cases is the same and is cheap here: roughly forty already-captured "
        "photographs remain unused, and adding them to the detector's training "
        "set requires no new photography."))

    A(heading("Limitations", 2))
    A(para(
        "Several limitations should be stated plainly. Only one training run "
        "was performed per detector configuration reported, baseline and "
        "scale+tilt augmented alike, so the detector figures are single-run "
        "and, by this report's own argument, should be treated with the "
        "caution applied to the classifier's single-run numbers—the jump to "
        "100% held-out recall and classification in Section VI-E is the "
        "figure most in need of that caution. The detector "
        "experiments are consequently thinner than the classifier experiments: "
        "the data-volume, duration and architecture experiments were conducted on "
        "the classifier, and equivalents for the detector remain to be run. The "
        "held-out set of nine photographs and 72 instances is small, and per-class "
        "conclusions drawn from one or two instances are weak. Finally, the "
        "classical pipeline's apparent superiority is specific to this deck and "
        "these conditions; it depends on an assumption—dark, matte cards on "
        "lighter backgrounds—that the flash experiment shows is easy to break."))

    A(heading("Future Work", 2))
    A(para(
        "Four directions follow directly from the results. First, retrain the "
        "detector with flipping disabled and rotation enabled, and measure the "
        "effect on the arrow and letter classes. Second, add the forty unused "
        "photographs and re-measure both recall and out-of-distribution "
        "behaviour, testing the diagnosis in Section VII-B. Third, replace the "
        "threshold-based segmentation with a learned segmenter such as U-Net [4] "
        "or a promptable model such as Segment Anything [5], which would not "
        "depend on the dark-background assumption that the flash case defeats. "
        "Fourth, run the detector across seeds so its figures carry the same "
        "reliability as the classifier's. Fifth, address the two remaining "
        "gaps from Section VI-E directly: constrain the scale+tilt "
        "augmentation to also produce tighter boxes on largely-empty "
        "backgrounds, since loose boxes are what currently rules out pairing "
        "it with the more accurate classifier, and address the classical "
        "detector's hand-shadow merge at the segmentation stage itself, for "
        "example by excluding skin-tone pixels before thresholding, since no "
        "amount of additional training data can fix a hand-specified rule."))

    # ---------------- VIII. Conclusion ----------------
    A(heading("Conclusion"))
    A(para(
        "A YOLOv8n detector was adapted by transfer learning to a 29-class "
        "symbol-detection task on a dataset captured, annotated and verified for "
        "this assignment, reaching mAP@0.5 of 0.914 and mAP@0.5:0.95 of 0.894 "
        "with uneven per-class recall. A second pipeline combining classical "
        "segmentation with a small from-scratch classifier was evaluated on "
        "identical data and, by separating detection from classification, showed "
        "that the classifier reads more accurately than the detector's own head "
        "while the detector localises less reliably than a hand-specified "
        "geometric rule—decisively so on unseen backgrounds, where it detected "
        "nothing at all."))
    A(para(
        "The most durable findings are methodological. Single training runs at "
        "this data scale are not measurements: identical configurations varied by "
        "nearly twenty points across random seeds. Corrupted training data can "
        "persist undetected when there is only one example per class to define "
        "it. Failures should be diagnosed to the stage that produced them, since "
        "a confident wrong answer from a classifier may in fact be a segmentation "
        "failure. And a default augmentation setting that is standard for natural "
        "images can silently corrupt labels for symbol data. Each of these cost "
        "more to learn than the accuracy figures did, and each generalises "
        "further."))
    A(para(
        "A final, unplanned test reinforced the same argument from a different "
        "angle. Live use of the real-time demonstration showed both pipelines "
        "fail to detect a card held in the hand rather than resting on a table, "
        "for two different low-level reasons that shared one root cause: no "
        "training or evaluation photograph in this project showed a hand-held "
        "card. A purely data-side fix—synthetic scale, rotation and perspective "
        "augmentation of photographs already collected, with no new "
        "photography—recovered detection on most hand-held test cases where "
        "none existed before, but did not recover reliable classification, and "
        "revealed that a detector's own imprecise localisation can defeat a "
        "more accurate classifier downstream of it. Both outcomes are further "
        "evidence for this report's central claim, that data diversity rather "
        "than architecture was the binding constraint throughout."))

    # ---------------- AI declaration ----------------
    A(heading("AI Usage Declaration"))
    A(para(
        "AI tools were used in this assignment and their contribution is "
        "declared here in full."))
    A(para(
        "Tool used: Anthropic Claude, accessed through the Claude Code "
        "command-line interface, used interactively throughout development."))
    A(para(
        "What it was used for: assistance in writing and debugging Python code "
        "for the segmentation pipeline, the classifier, the dataset-preparation "
        "and annotation-verification scripts, the YOLO training and evaluation "
        "scripts, the real-time demonstration, and the evaluation harness; "
        "assistance in diagnosing failures, including the tilted-card and "
        "edge-touching segmentation defects, the corrupted training crops, and "
        "the hand-held card detection failure described in Section VI-E, "
        "traced through live testing of the real-time demonstration to two "
        "distinct causes; implementation of a synthetic scale, rotation and "
        "perspective augmentation to address one of them; discussion of "
        "experimental design and of how to report results honestly, including "
        "the decision to report the unsuccessful detector-plus-classifier "
        "combination in Section VI-E rather than omit it; and drafting "
        "assistance for this report."))
    A(para(
        "Which parts were AI-assisted: the majority of the source code was "
        "written with AI assistance. The experimental direction, the decisions "
        "about what to investigate, the data collection and photography, and the "
        "final interpretation of results were directed by the author. This report "
        "was drafted with AI assistance from the experimental record and then "
        "reviewed and revised by the author."))
    A(para(
        "How the material was verified: every reported number in this document "
        "was produced by running the scripts in the accompanying repository, and "
        "the outputs are reproducible from it. Intermediate artefacts were "
        "inspected visually rather than accepted on trust—an audit of this kind "
        "is what revealed the nine corrupted training crops described in Section "
        "II-E, several of which had passed an automated numerical check. Claims "
        "that could not be verified by measurement are identified as hypotheses "
        "in the text, most notably the augmentation diagnosis in Section VII-A."))
    A(para(
        "A separate script in the repository, identify_cards.py, queries a "
        "hosted vision-language model through the Azure OpenAI API. It was "
        "written during exploration, is not part of either pipeline evaluated "
        "here, and contributes to none of the reported results. It is retained "
        "only as a point of reference and is declared for completeness."))

    # ---------------- References ----------------
    A(heading("References"))
    refs = [
        'J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look '
        'Once: Unified, Real-Time Object Detection," in Proc. IEEE Conf. '
        "Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779-788.",
        'G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," version 8, '
        "2023. [Online]. Available: https://github.com/ultralytics/ultralytics",
        'T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in '
        "Proc. European Conf. Computer Vision (ECCV), 2014, pp. 740-755.",
        'O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional '
        'Networks for Biomedical Image Segmentation," in Proc. MICCAI, 2015, '
        "pp. 234-241.",
        'A. Kirillov et al., "Segment Anything," in Proc. IEEE/CVF Int. Conf. '
        "Computer Vision (ICCV), 2023, pp. 4015-4026.",
        'S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep '
        'Network Training by Reducing Internal Covariate Shift," in Proc. Int. '
        "Conf. Machine Learning (ICML), 2015, pp. 448-456.",
        'M. Lin, Q. Chen, and S. Yan, "Network in Network," in Proc. Int. Conf. '
        "Learning Representations (ICLR), 2014.",
        'I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," '
        "in Proc. Int. Conf. Learning Representations (ICLR), 2019.",
        'C. B. Barber, D. P. Dobkin, and H. Huhdanpaa, "The Quickhull Algorithm '
        'for Convex Hulls," ACM Trans. Mathematical Software, vol. 22, no. 4, '
        "pp. 469-483, 1996.",
        'P. Virtanen et al., "SciPy 1.0: Fundamental Algorithms for Scientific '
        'Computing in Python," Nature Methods, vol. 17, pp. 261-272, 2020.',
    ]
    # The "references" style supplies the [n] numbering automatically.
    for r in refs:
        A(para(r, style="references"))

    return "".join(B)


def main():
    # Always start from the pristine template so the build is idempotent.
    if os.path.isdir(UNPACKED):
        shutil.rmtree(UNPACKED)
    with zipfile.ZipFile(os.path.join(HERE, "template.docx")) as z:
        z.extractall(UNPACKED)

    xml = open(DOC_XML, encoding="utf-8").read()

    root_m = re.search(r"^(.*?<w:body>)", xml, re.S)
    root_open = root_m.group(1)

    sectprs = re.findall(r"<w:sectPr.*?</w:sectPr>", xml, re.S)
    title_sect = sectprs[0]      # single column, title area
    body_sect = sectprs[3]       # two column, body

    rid = register_images(list(FIG_SRC.values()))

    title_block = (
        para("Custom Object Detection for Text Symbols: A YOLO Baseline and a "
             "From-Scratch CNN Comparison", style="papertitle")
        + para("[Your Name]", style="Author")
        + para("[Matriculation Number]", style="Author")
        + para("College of Computing and Data Science", style="Affiliation")
        + para("Nanyang Technological University, Singapore", style="Affiliation")
        + f"<w:p><w:pPr><w:pStyle w:val=\"Affiliation\"/>{title_sect.replace('<w:sectPr', '<w:sectPr')}</w:pPr></w:p>"
    )
    # the sectPr must live inside pPr
    title_block = (
        para("Custom Object Detection for Text Symbols: A YOLO Baseline and a "
             "From-Scratch CNN Comparison", style="papertitle")
        + para("[Your Name]", style="Author")
        + para("[Matriculation Number]", style="Author")
        + para("College of Computing and Data Science", style="Affiliation")
        + para("Nanyang Technological University, Singapore", style="Affiliation")
        + '<w:p><w:pPr><w:pStyle w:val="Affiliation"/>' + title_sect + "</w:pPr></w:p>"
    )

    body = build_body(rid)
    new_xml = root_open + title_block + body + body_sect + "</w:body></w:document>"
    open(DOC_XML, "w", encoding="utf-8").write(new_xml)

    out = os.path.join(HERE, "CT3205_Assignment1_Report_DRAFT.docx")
    if os.path.exists(out):
        os.remove(out)
    # [Content_Types].xml first, as Word expects.
    names = []
    for root, _, files in os.walk(UNPACKED):
        for f in files:
            full = os.path.join(root, f)
            names.append(os.path.relpath(full, UNPACKED).replace("\\", "/"))
    names.sort(key=lambda n: (n != "[Content_Types].xml", n))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(UNPACKED, n), n)
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
