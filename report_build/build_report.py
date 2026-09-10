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
    "annot": os.path.join(HERE, "figures", "annotation_verification.jpg"),
    "curves": os.path.join(HERE, "..", "yolo", "yolo_runs", "cards", "results.png"),
    "confusion": os.path.join(HERE, "..", "yolo", "yolo_runs", "cards", "confusion_matrix_normalized.png"),
    "preds": os.path.join(HERE, "..", "yolo", "yolo_runs", "cards", "val_batch0_pred.jpg"),
    "glare": os.path.join(HERE, "..", "RngTest", "Test2.jpeg"),
    "fusing": os.path.join(HERE, "figures", "fusing_neighbouring_cards.jpg"),
    "video": os.path.join(HERE, "figures", "deployment_video_frame.png"),
    "grabber": os.path.join(HERE, "figures", "frame_grabber_roi.jpg"),
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
        "cases, though not yet its ability to classify them reliably. A second "
        "round of testing finds a sharper fix for the same class of problem: "
        "re-training the detector on the 211 individually-cropped real "
        "photographs already used by the comparison classifier—material the "
        "detector had never seen, despite being drawn from the identical "
        "source photographs—raises correct real-world detection-and-"
        "classification from 1 of 8 to 7 of 8 sampled hand-held and "
        "novel-background test photographs, exceeding a synthetic-augmentation "
        "variant of the same experiment (2 of 8) without cost to held-out "
        "performance. This same round of live testing also finds that only "
        "the detector, never the comparison classifier, is able to recognise "
        "the dataset's pictographic symbols at all, and surfaces further, "
        "still-unconfirmed evidence for the augmentation hypothesis above. A "
        "third round closes the loop on the deployment gap directly: capturing "
        "training data in the deployment condition itself—cards held in the "
        "hand, filmed on an iPhone 16 Pro Max and extracted frame by "
        "frame at 0.5-second intervals—raises real-world correctness to 8 of 8, repairs the last "
        "remaining weak class from the previous round, and reaches mAP@0.5 of "
        "0.988 while extending the system to a thirtieth class. Across all "
        "three rounds not one architectural change was made: every gain came "
        "from what the detector was shown.",
        style="Abstract"))
    A(para(
        "Keywords—object detection, YOLO, transfer learning, convolutional "
        "neural networks, symbol recognition, data augmentation, failure analysis",
        style="Keywords"))
    A(para(runs=[
        ("Revision 4 note: ", {"b": True}),
        ("revision 2 added a dataset experiment for the detector (Sections "
         "III-E, V-F) and a second round of real-time testing (Section VI-F); "
         "revision 3 added deployment-condition data capture (Section III-F), "
         "its results (Section V-G), and links to the demonstration video and "
         "source repository (Section I). This revision incorporates the "
         "author's own revisions, corrects the class count and the account of "
         "how the final round's frames were captured, and adds figures of a "
         "data defect and of the capture process. Material from the original "
         "submission is otherwise unchanged.", {"i": True}),
    ]))

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
        "were shown to be unreliable at this dataset scale. Sixth, and added "
        "after the first two rounds of testing, the training set was extended "
        "with data captured in the deployment condition itself, which is what "
        "finally closed the gap between held-out scores and real-world "
        "behaviour."))
    A(para(runs=[
        ("Supporting materials. ", {"b": True}),
        ("All source code, trained weights, dataset-building scripts and "
         "evaluation harnesses are in the project repository at:", {}),
    ]))
    A(para("https://github.com/Jeb-Lee/CT3205_CV_01"))
    A(para(
        "A recorded demonstration of the real-time system described in "
        "Section III-D, running live on a webcam with the final detector, is "
        "at:"))
    A(para("https://youtu.be/7xptfqeyB-0"))

    # ---------------- II. Dataset ----------------
    A(heading("Dataset"))

    A(heading("Data Collection", 2))
    A(para(
        "The dataset consists of photographs of a physical set of symbol cards. "
        "Each card is a matte black square bearing a single symbol printed in one "
        "of five colours. The full symbol inventory comprises the digits 1 to 9, "
        "the letters A to H and S to Z, a filled circle, and two arrow cards: "
        "28 cards in all. An arrow's class is the direction it points, so the "
        "two arrow cards supply left-, right- and up-pointing arrows—and, in "
        "the final round, down (Section III-F)—giving 29 classes in total, 30 "
        "in the final round. This exceeds the required minimum of ten "
        "classes."))
    A(para(
        "Photographs were captured in three sessions using a mobile phone camera "
        "under ordinary indoor lighting. The sessions deliberately varied the "
        "conditions: cards were photographed on a clipboard, on two different "
        "marble surfaces with differing lighting conditions, and on a tarpaulin "
        "sheet with sub-optimal lighting, from a range of heights and "
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
         ["Classes", "29 (25 characters, circle, 3 arrow directions)"],
         ["Images (train)", "18"],
         ["Images (validation)", "9"],
         ["Annotated instances (train)", "251"],
         ["Annotated instances (val)", "72"],
         ["Instances per image", "6 to 28"],
         ["Image resolutions", "1200x1600, 1600x1200, 1536x2048, 2048x1536"],
         ["Annotation format", "YOLO (class, cx, cy, w, h; normalised)"],
         ["Split strategy", "By source photograph, not by instance"]],
        [110, 130],
        caption="Dataset composition as originally captured. Two later "
                "rounds add training material without altering the "
                "validation split; the final composition is given in "
                "Section III-F."))
    A(para(
        "This table describes the dataset as first captured, which is what "
        "the baseline detector and every comparison in Sections V-A to V-E "
        "are trained and measured on. Two later rounds of data collection, "
        "described in Sections III-E and III-F, add training images and one "
        "further class, bringing the final totals to 579 training images "
        "across 30 classes. The validation split of nine photographs and 72 "
        "instances is deliberately held fixed throughout, so that every "
        "figure in this report remains comparable with every other; the "
        "consequences of freezing it are examined in Section VII-C."))
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
        "on a dark, patterned tarpaulin sheet, whose background repeatedly "
        "defeated the "
        "black-card threshold, and a recurring oversized crop of one letter."))
    A(figure(rid[FIG_SRC["fusing"]], FIG_SRC["fusing"],
             "Fusing of neighbouring cards. Where cards touch (circled), "
             "thresholding merges them into a single region, one source of "
             "the corrupted crops described in this section.", width_pt=150))

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
    A(para(
        "A recording of the demonstration, running live on a webcam with the "
        "final detector of Section III-F, is available at "
        "https://youtu.be/7xptfqeyB-0. It shows the system being launched "
        "from the command line and cards being read at varying distance and "
        "angle, with class labels and confidence scores drawn in real time."))
    A(para(
        "Throughput was measured over 25 live frames at 640x480 on an Intel "
        "Core i5-12450HX with an NVIDIA RTX 3050. The detector runs at 87.6 "
        "frames per second on the GPU (11 ms per frame) and 15.8 on the CPU "
        "(63 ms); the classical-plus-classifier pipeline runs at 24.2 (41 "
        "ms), its cost dominated by connected-component analysis rather than "
        "by the network, which is why it scales more steeply with image "
        "resolution than the detector does. All three are comfortably "
        "real-time at webcam frame rates, so the choice between them is "
        "governed by accuracy rather than speed."))

    A(heading("Real-Photo Diversity for the Detector", 2))
    A(para(
        "Section VI-E's synthetic augmentation fixed part of the hand-held "
        "failure by manufacturing scale and angle variety from the same "
        "eighteen source photographs. A separate question is whether the "
        "detector benefits from real photographic diversity it has simply "
        "never been shown—specifically, the 211 individually-cropped card "
        "photographs the comparison classifier already trains on (Section "
        "II-E), built from the identical three photography sessions but "
        "never previously reformatted for detection. Each crop shows one "
        "card generously padded against its background, so it fills a much "
        "larger fraction of frame than any instance in the eighteen "
        "whole-photo training images; unlike the synthetic crops, the "
        "lighting, shadow and framing in each is genuinely distinct, not "
        "re-derived from the same pixels."))
    A(para(
        "Converting these into detection training data is not a direct "
        "relabelling, because several of them still show a sliver of a "
        "neighbouring card in the padding margin. Each crop is re-processed "
        "with the same classical detector used elsewhere in this project, "
        "with its area and aspect bounds widened to suit a single "
        "frame-filling card rather than a busy multi-card photograph; the "
        "largest detected region is kept as the card regardless of whether a "
        "second, smaller region is also found—inspection showed these "
        "secondary regions are usually a glyph's own enclosed loop (as in D, "
        "G, 6 or 9) registering as a separate dark island under the same "
        "threshold, not a second card—and the crop is re-cut tightly around "
        "that region so any genuine neighbour is excluded rather than left "
        "in frame and unlabelled. All 211 crops converted successfully by "
        "this method, verified by re-rendering a sample of the results with "
        "their boxes drawn on. Combined with the original eighteen "
        "whole-photo images, this yields 229 real training images—still "
        "using the same nine held-out photographs for validation as every "
        "other detector variant in this report—fine-tuned from the same "
        "COCO-pretrained checkpoint under an otherwise identical "
        "configuration to the baseline."))

    A(heading("Capturing Data in the Deployment Condition", 2))
    A(para(
        "Sections VI-E and V-F between them establish that the detector's "
        "real-world failures were a property of what it had been shown, and "
        "that supplying more genuinely distinct real photographs closed most "
        "of the gap. Both rounds, however, still drew on the same original "
        "photographs: cards resting flat on a surface. The remaining difference "
        "from the deployment condition was not background but the condition "
        "itself—a card held in a hand, at arm's length, filling much of the "
        "frame and turning as the hand moves. The final round of data "
        "collection targets that difference directly by filming cards held in "
        "exactly that way. The recording was made on a phone rather than on "
        "the demonstration webcam, so camera characteristics were not matched: "
        "what this round supplies is the pose and scale of use, not its "
        "camera."))
    A(para(
        "One continuous video, 4 minutes 44 seconds long, was recorded on an "
        "iPhone 16 Pro Max; the copy used here is 1024x576 at 30 frames per "
        "second. Each card of the deck was presented in turn, in deck order: "
        "brought close to the camera, drawn slowly back, then rotated a little "
        "clockwise and counter-clockwise, so that a single pass covers the "
        "range of scale and in-plane rotation a live demonstration actually "
        "produces. An attempt to segment this video into per-card clips "
        "automatically was abandoned: the classical detector finds a "
        "card-shaped region continuously, including through the swaps, so gaps "
        "in detection do not mark the boundaries, and frame-to-frame motion "
        "peaks were ambiguous because reversing the rotation direction "
        "mid-card produces a spike indistinguishable from a card change."))
    A(figure(rid[FIG_SRC["video"]], FIG_SRC["video"],
             "A frame from the recording used for the deployment-condition "
             "training data: a card held in the hand at arm's length, filmed "
             "on an iPhone 16 Pro Max."))
    A(para(
        "Frames were extracted with a purpose-written tool "
        "(tools/frame_grabber.py) that samples the recording every 0.5 "
        "seconds within a capture region dragged out by hand at the start of "
        "each run. The method is efficient, but placing the capture region "
        "proved hit-and-miss: a hand cannot hold a card in a pixel-accurate "
        "position, so no single region captured every card cleanly, and many "
        "frames showed a card with an indistinct or cut-off border, out of "
        "focus, blinded by glare, or outside the region altogether. The tool "
        "was therefore run several times, each time with a fresh region aimed "
        "at the cards the previous run had missed, and the bad frames from "
        "each run were deleted by hand as the rest were sorted into per-card "
        "sets. The card boundaries, which no automatic cue located reliably, "
        "were thus decided by a person looking at the frames. This is why the "
        "frames come in more than one size, and why the number of frames per "
        "card is uneven. The result was 446 frames across 30 classes—a set "
        "still marred in places by a dark window in the background, the cause "
        "of the rejections described in Section V-G."))
    A(figure(rid[FIG_SRC["grabber"]], FIG_SRC["grabber"],
             "Frame extraction with tools/frame_grabber.py, run as "
             "frame_grabber.py New_Vid.mp4 --roi --every 0.5: a capture "
             "region is dragged over the recording by hand, and one cropped "
             "frame is saved every 0.5 seconds."))
    A(para(
        "The thirtieth class is a down-pointing arrow. The card set supplies "
        "two physical arrow cards, and which directions they are taken to "
        "represent is left to the user; earlier rounds of this project had "
        "defined left, right and up, and this round adds down. Because the "
        "detection head's output width is fixed by the number of classes, "
        "adding one cannot be done by continuing to fine-tune an existing "
        "checkpoint—the final round is therefore trained afresh from the same "
        "COCO-pretrained weights, and the earlier 29-class checkpoints are "
        "retained unchanged rather than superseded in place."))
    A(para(
        "Labelling the 446 frames reuses the classical detector once more, "
        "with the same widened bounds as Section III-E, plus two filters. A "
        "frame is discarded if its Laplacian variance falls below a blur "
        "threshold, which removes the motion-blurred frames that the pull-in "
        "and rotation phases inevitably produce; and a frame is discarded if "
        "the area of its detected region is an outlier against the median for "
        "that same class, since each class was filmed in one continuous pass "
        "at broadly consistent distance. That left 401 frames, with the class "
        "imbalance still unresolved. Visual review "
        "of all thirty classes then rejected a further 51, for a reason worth "
        "recording because no automatic check caught it, described in Section "
        "V-G. The verified set is 350 frames, which combined with the 18 "
        "whole-photo images and 211 solo crops gives 579 real training "
        "images—every one of them a genuine photograph, none synthesised."))

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

    A(heading("Effect of Real Photographic Diversity on the Detector", 2))
    A(table(
        [["Measure", "Baseline", "Scale+tilt", "All-real"],
         ["Held-out recall", "84.7%", "100%", "98.6%"],
         ["Held-out cls. given det.", "90.2%", "100%", "98.6%"],
         ["Held-out overall", "76.4%", "100%", "97.2%"]],
        [90, 50, 50, 50],
        caption="Held-out performance of the baseline detector against "
                "the two data-diversity variants (Section III-E and Section "
                "VI-E). All three use the identical nine-photograph "
                "validation split."))
    A(table(
        [["Measure", "Baseline", "Scale+tilt", "All-real"],
         ["Sampled real-world photos correct", "1 / 8", "2 / 8", "7 / 8"],
         ["Confidence on a correct hit", "0.40", "0.26-0.42", "0.41-0.98"]],
        [90, 50, 50, 50],
        caption="Detection-and-classification correctness on eight sampled "
                "real-world photographs (six hand-held, two on a novel "
                "background—see Section VI-F), at the demonstration's "
                "default 0.25 confidence threshold. Single run for each "
                "configuration; see Section VII-C."))
    A(para(
        "The all-real variant matches the scale+tilt variant's near-perfect "
        "held-out performance—one miss, S misread as 3, against the "
        "earlier variant's zero—while using a comparable number of "
        "training images (229 against ninety) and none of it synthetically. "
        "On the real-world sample the difference is not incremental. Two "
        "hand-held V-card photographs that the scale+tilt detector barely "
        "registered (confidence 0.23 and 0.26, one misclassified) are read "
        "by the all-real detector at 0.98 each; a novel-background A-card "
        "photograph goes from a wrong guess (baseline) to 0.42 (scale+tilt) "
        "to 0.96 (all-real). The one miss at the default threshold, a G-card "
        "photograph, still recovers correctly at a slightly lower threshold "
        "(0.22 at confidence 0.1)."))
    A(para(
        "The comparison between the two diversity variants is informative "
        "beyond the headline numbers. Both add real photographic material "
        "from the same three sessions; the difference is that scale+tilt "
        "re-derives synthetic examples from eighteen source images by "
        "geometric transform, while all-real supplies 211 genuinely distinct "
        "real photographs, each with its own lighting and shadow. That the "
        "latter generalises further, at a comparable training-set size, is "
        "evidence that a fixed pool of source pixels stretched further by "
        "synthesis is not a full substitute for additional real capture—"
        "consistent with the caution already given for the augmentation "
        "hypothesis in Section VII-A, and revisited in Section VII-B."))

    A(heading("Effect of Deployment-Condition Training Data", 2))
    A(para(
        "The final detector adds the 350 verified hand-held phone-camera frames of "
        "Section III-F to the 229 images of the previous round, and extends "
        "the class list to thirty. Held-out performance is given below "
        "against the previous best; the same nine-photograph validation split "
        "is used throughout, so the two rows are directly comparable."))
    A(table(
        [["Measure", "All-real", "Deployment"],
         ["Held-out precision", "0.895", "0.931"],
         ["Held-out recall", "0.973", "0.990"],
         ["mAP@0.5", "0.995", "0.988"],
         ["mAP@0.5:0.95", "0.988", "0.983"],
         ["Worst per-class recall", "0.54 (class 4)", "0.79 (class E)"],
         ["Sampled real-world photos correct", "7 / 8", "8 / 8"]],
        [120, 60, 60],
        caption="The all-real detector of Section V-F against the "
                "deployment-condition detector of Section III-F, on the "
                "identical held-out split and the identical eight real-world "
                "photographs. Single run for each; see Section VII-C."))
    A(para(
        "The aggregate held-out figures barely move, and the two mAP columns "
        "even tip marginally the other way—at 0.988 against 0.995 they are "
        "well inside the range a different random seed would produce, and "
        "neither difference should be read as meaningful. What does move is "
        "the distribution underneath them. In the previous round one class "
        "stood out as genuinely weak: class 4, at recall 0.54, the single "
        "measurable defect left after that round's improvement. It now sits "
        "at 1.00. Of the twenty-three classes with held-out instances, "
        "twenty-one reach recall 1.00, and the weakest is E at 0.79. The "
        "improvement is not in the mean but in the floor."))
    A(para(
        "The real-world sample is where the effect is plainest. On the same "
        "eight photographs used in Section V-F, the deployment-condition "
        "detector reads all eight correctly at the demonstration's default "
        "0.25 confidence threshold, against seven for its predecessor. The "
        "photograph that previously required the threshold to be lowered to "
        "0.1 before it registered at all is now detected outright. "
        "Confidence on correct detections rises across most of the sample—"
        "0.41 to 0.97, 0.58 to 0.85, 0.60 to 0.89, 0.54 to 0.75—while two "
        "of the eight fall, from 0.98 to 0.75 and from 0.98 to 0.83, both "
        "still correct and both still far above threshold. The pattern is "
        "one of raising the weakest cases rather than sharpening the "
        "strongest, which is what the held-out per-class figures show as "
        "well."))
    A(para(
        "One finding from the labelling process belongs here rather than in "
        "the methodology, because it is a failure of automatic verification "
        "and not merely a processing step. The 51 frames rejected by visual "
        "review after the automatic filters had passed them shared a single "
        "cause: a dark object in the room behind the card—a tinted window, a soft "
        "toy—fell within the same brightness and saturation bounds as the "
        "black card material, and being adjacent to the card in the image, "
        "was absorbed into the same connected component. The resulting box "
        "was a plausible-looking rectangle that included a piece of the "
        "background. This is the hand-shadow merge of Section VI-E in "
        "another guise, and like it, it defeats the dark-background "
        "assumption the classical stage rests on. It resisted every "
        "automatic test attempted: box area was not anomalous, because in "
        "the affected classes enough frames shared the condition to shift "
        "the class median itself; aspect ratio was not anomalous, because "
        "the merged region happened to remain roughly square; and mask "
        "solidity was not anomalous, because a card's own glyph already "
        "punches a large hole in the mask, so the metric is dominated by the "
        "symbol rather than by the merge. Only inspection separated them. "
        "The rejection was uneven across classes—three of the sixteen frames "
        "for class 3 survived, against twenty-five of twenty-six for class "
        "Y—entirely according to what happened to be behind the operator's "
        "hand at that point in the recording."))

    # ---------------- VI. Challenging scenarios ----------------
    A(heading("Challenging Scenarios"))
    A(para(
        "Four adverse conditions were tested. Three were tested on purpose-taken "
        "photographs; the fourth emerged during dataset construction."))

    A(heading("Camera Flash on a Reflective Background", 2))
    A(para(
        "A card was photographed with flash on a black Ikea footstool with raised "
        "reflective dimples. The classifier reported the letter D, with full "
        "confidence, for a card bearing T. Inspecting the mask actually presented "
        "to the network explains why, and exonerates the classifier: the flash "
        "blew out roughly the lower half of the card, and each dimple on the footstool "
        "returned enough light to register as foreground. The segmentation stage "
        "assumes background is darker than the symbol; flash on a glossy dark "
        "object near a reflective surface violates that assumption directly. What "
        "reached the classifier was largely noise, and D was simply its least-bad "
        "interpretation. This is a segmentation failure, not a classification "
        "failure, and the distinction matters because the remedies differ."))
    A(figure(rid[FIG_SRC["glare"]], FIG_SRC["glare"],
             "Failure case. Flash illumination on a glossy card against a "
             "reflective dimpled footstool. Specular highlights on both the card and "
             "the background defeat the dark-background assumption on which the "
             "segmentation stage depends."))

    A(heading("Unseen Backgrounds", 2))
    A(para(
        "Three photographs were taken against backgrounds absent from all "
        "training data: a woven carpet, a household appliance with a tiled floor "
        "and a foot in frame, and the footstool above. The result was the "
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

    A(heading("A Second Round of Real-Time Testing", 2))
    A(para(
        "Further live use of the demonstration, after the all-real detector "
        "of Section III-E was trained, tested two more conditions the static "
        "held-out set does not exercise: a card placed on a background never "
        "seen in training, and the four pictographic classes that the "
        "comparison classifier cannot represent at all."))
    A(para(
        "A card was stuck to a plain whiteboard—a background absent from "
        "every training and evaluation photograph, distinct from the "
        "hand-held scale problem in Section VI-E since the card itself sat "
        "flat and roughly frame-filling as in training. The baseline "
        "detector produced a wrong label at low confidence; the all-real "
        "detector produced the correct label at confidence 0.96—consistent "
        "with the finding in Section V-F that this variant's improvement "
        "extends beyond hand-held scale specifically to backgrounds absent "
        "from the whole-photo training images."))
    A(para(
        "The comparison classifier has no output classes for the circle or "
        "the three arrow symbols; those four classes exist only in the "
        "detector's own vocabulary (Section II-A), so the fine-tuned "
        "detector is the only system in this report capable of recognising "
        "them under any condition, and the all-real variant did so live for "
        "the circle and both non-mirrored arrows. The right-pointing arrow "
        "was visibly weaker. This is notable because it is not a data-volume "
        "effect in the usual sense: the all-real training set contains eight "
        "instances of right_arrow against six of left_arrow, so the weaker "
        "class has more, not fewer, examples. That rules out scarcity and "
        "points instead at the augmentation hypothesis from Section VII-A, "
        "confirmed still active in this run's configuration (fliplr: 0.5, "
        "unchanged from the baseline). Left_arrow and right_arrow are the "
        "only pair of classes in this dataset that are exact mirror images "
        "of one another, so the label corruption that hypothesis describes—"
        "a flipped image keeps its pre-flip label—turns roughly half the "
        "training signal for each of these two classes specifically into a "
        "contradiction of the other, while leaving the rotationally "
        "symmetric circle and the up-pointing arrow largely unaffected. This "
        "is consistent with, but does not on its own confirm, the "
        "hypothesis: it remains, as in Section VII-A, an inference from "
        "inspection rather than a controlled ablation, since fliplr has "
        "still not been disabled and re-measured."))

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
    A(para(
        "Section VI-F adds supporting, still non-ablated, evidence rather "
        "than a confirmation. Right_arrow performed visibly worse than the "
        "dataset's other pictographic classes in a second round of live "
        "testing, despite having more training instances than left_arrow "
        "(eight against six), which rules out the ordinary data-scarcity "
        "explanation and points back at this same augmentation. Left_arrow "
        "and right_arrow are the only mirror-image pair in the dataset, so "
        "they are exactly the pair this hypothesis predicts should suffer "
        "most, and no other explanation for that specific asymmetry was "
        "identified. The hypothesis therefore remains open but is now better "
        "motivated; the fliplr ablation described above has still not been "
        "run."))
    A(para(
        "It has also now survived a third training round without being "
        "tested. The deployment-condition detector of Section III-F was "
        "trained under the same default configuration, horizontal flipping "
        "included, and adds real examples of all four arrow orientations "
        "including the new down-pointing class. Whether the extra data "
        "outweighs the label corruption for these classes cannot be read "
        "off the held-out figures, because none of the arrow classes has a "
        "single instance in the validation split. The ablation remains the "
        "cheapest outstanding experiment in this report and the one most "
        "likely to change a per-class result."))

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
    A(para(
        "Section V-F's real-photo-diversity experiment is the sharpest "
        "confirmation of this argument in the report. It changes no "
        "architecture, no hyperparameter and no held-out split—only what the "
        "detector is shown—and moves real-world correctness from one "
        "sampled photograph in eight to seven, a larger effect than any "
        "architectural change attempted anywhere in this work, including the "
        "global-average-pooling experiment above. It also refines the "
        "diversity argument: the same experiment run synthetically (Section "
        "VI-E) achieved less than half the improvement from a similarly "
        "sized training set, suggesting the relevant quantity is not merely "
        "how many training images exist but how much genuinely new visual "
        "information they carry."))
    A(para(
        "The third round sharpens that refinement into something more "
        "specific than diversity. Its 350 added frames are less diverse than "
        "the 211 crops of the previous round by most ordinary measures: they "
        "come from a single continuous recording, in one room, in one "
        "session, against a nearly constant background, and many are near "
        "neighbours of each other in time. What they have instead is "
        "correspondence with the deployment condition—hand-held, at "
        "arm's length, filling much of the frame, held the way the system is "
        "actually used—though not through the camera it is demonstrated on, "
        "since the recording was made on a phone. That narrow, repetitive set repaired the last measurable weak "
        "class and completed the real-world sample, which suggests the "
        "governing quantity is not diversity in the abstract but overlap with "
        "the distribution the model will be asked to work in. Taken across "
        "all three rounds the pattern is consistent and, by this point, hard "
        "to attribute to chance: synthetic variation on existing pixels "
        "helped least, genuinely new photographs of the same subject helped "
        "more, and photographs taken under the conditions of use helped most "
        "—with the architecture, the optimiser, the schedule and the "
        "validation split held fixed throughout."))

    A(heading("Limitations", 2))
    A(para(
        "Several limitations should be stated plainly. Only one training run "
        "was performed per detector configuration reported—baseline, "
        "scale+tilt, all-real and deployment-condition alike—so the detector "
        "figures are "
        "single-run and, by this report's own argument, should be treated "
        "with the caution applied to the classifier's single-run numbers; "
        "the near-100% held-out figures in Sections VI-E, V-F and V-G are those "
        "most in need of it, and the eight-photograph real-world sample used "
        "in Sections V-F and V-G is smaller still. The detector "
        "experiments are consequently thinner than the classifier experiments: "
        "the data-volume, duration and architecture experiments were conducted on "
        "the classifier, and equivalents for the detector remain to be run. The "
        "held-out set of nine photographs and 72 instances is small, and per-class "
        "conclusions drawn from one or two instances are weak. Finally, the "
        "classical pipeline's apparent superiority is specific to this deck and "
        "these conditions; it depends on an assumption—dark, matte cards on "
        "lighter backgrounds—that the flash experiment shows is easy to break."))
    A(para(
        "One limitation grew rather than shrank over the three rounds and "
        "should be stated in its own right. The held-out split was fixed "
        "early, before the pictographic classes and the letters Y and Z were "
        "well represented, and it was deliberately not changed afterwards so "
        "that figures from different rounds would stay comparable. The "
        "consequence is that seven of the final system's thirty classes—Y, Z, "
        "the circle, and all four arrows—have no instances at all in the "
        "validation split. Their held-out recall is not low; it is undefined. "
        "Everything this report says about those classes rests on live "
        "observation during demonstration, which is evidence of a weaker kind "
        "than the rest of the numbers here, and the down-pointing arrow added "
        "in the final round has never been measured at all. A held-out set "
        "extended to cover all thirty classes is a prerequisite for any "
        "stronger claim about them, and is the first thing a continuation of "
        "this work should build."))

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
    A(para(
        "Two further directions followed from Sections V-F and VI-F, and one "
        "of them has since been carried out. Sixth, run the fliplr ablation "
        "now more strongly motivated by the instance-count evidence in "
        "Section VII-A—the arrow-class asymmetry predicted by that hypothesis "
        "was observed, but a controlled before-and-after comparison has still "
        "not been made, and the final round adds arrow data without "
        "disentangling the two effects. Seventh, and no longer outstanding: "
        "the proposal to extend the all-real approach with real rather than "
        "synthetic hand-held examples, on the prediction that data captured "
        "on the deployment camera at the deployment scale would close more of "
        "the gap than synthesis had. Section III-F carried this out and "
        "Section V-G reports the result, which was in the predicted "
        "direction—8 of 8 on the real-world sample, and the last measurable "
        "weak class repaired."))
    A(para(
        "Three directions now follow from that final round. Eighth, and "
        "before any of the others, extend the held-out split to cover all "
        "thirty classes, since seven of them currently cannot be measured at "
        "all; without this, further per-class work on the pictographic "
        "classes cannot be evaluated even if it succeeds. Ninth, address the "
        "background-merge failure of Section V-G at the segmentation stage "
        "rather than by inspection: every automatic test tried—area, aspect "
        "ratio, mask solidity—failed to separate merged boxes from clean "
        "ones, and 51 frames had to be rejected by eye, which does not scale "
        "and is not reproducible. A learned segmenter, as already proposed "
        "above, would sidestep the brightness assumption the merge exploits. "
        "Tenth, re-shoot the four classes left thin by that rejection—3, 4, 8 "
        "and 9, reduced to between three and seven usable frames each—against "
        "a plain background, which is a few minutes of work and would even "
        "out the per-class counts the final round otherwise leaves uneven."))

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
    A(para(
        "A second round of testing sharpened that claim further. Re-training "
        "the detector on 211 real card photographs already collected for the "
        "comparison classifier, but never previously shown to the detector, "
        "raised correct real-world detection-and-classification from one "
        "sampled photograph in eight to seven, exceeding the synthetic-"
        "augmentation variant's two, while matching its near-perfect "
        "held-out performance. The same round of live testing found that "
        "only the fine-tuned detector, never the comparison classifier, "
        "could recognise the dataset's pictographic classes at all—a "
        "consequence of what each system was built to represent, not of "
        "which performed better—and surfaced further, still unconfirmed, "
        "evidence for the augmentation hypothesis in Section VII-A: the "
        "one class pair in this dataset that hypothesis predicts should be "
        "confused with each other performed worse than the rest, despite "
        "having more training examples, not fewer. Taken together with the "
        "rest of this report, the pattern is consistent throughout: every "
        "improvement came from showing the model more or better data, and "
        "every remaining gap is one a controlled experiment has not yet been "
        "run to close."))
    A(para(
        "A third round settled the question the first two had circled. If the "
        "detector's real-world failures came from never having seen the "
        "deployment condition, the direct remedy was to capture training data "
        "in that condition: cards held in the hand at arm's length, filmed on "
        "a phone, extracted frame by frame and verified "
        "by inspection. Adding 350 such frames raised real-world correctness "
        "to 8 of 8, repaired class 4—the last measurable weak class, at "
        "recall 0.54—to 1.00, and extended the system to a thirtieth class, "
        "while held-out aggregates stayed level. That set was narrower than "
        "the material added in the round before it, not broader: one room, "
        "one session, one background. Its advantage was correspondence with "
        "the conditions of use, which suggests the operative quantity was "
        "never diversity for its own sake but overlap with the distribution "
        "the system is asked to work in."))
    A(para(
        "Across three rounds of improvement the architecture never changed. "
        "The detector that reads eight of eight real photographs is the same "
        "YOLOv8n, fine-tuned from the same COCO-pretrained checkpoint, under "
        "the same optimiser and schedule, as the one that read one of eight. "
        "What changed was the training data, and the most useful data was the "
        "data that looked like the problem. Set against that, the report's "
        "one unambiguous architectural experiment—replacing the classifier's "
        "fully connected head with global average pooling—cost 26 points. The "
        "lesson the project actually taught, as distinct from the accuracy it "
        "reached, is where to spend effort when a model underperforms: on "
        "what it is being shown, and on measuring honestly enough to tell the "
        "difference between a model that has improved and a benchmark that "
        "cannot see the failure."))

    # ---------------- AI declaration ----------------
    A(heading("AI Usage Declaration"))
    A(para(
        "AI tools were used in this assignment and their contribution is "
        "declared here in full."))
    A(para(
        "Tool used: Anthropic Claude, accessed through the Claude Code "
        "command-line interface, used interactively throughout development."))
    A(para(
        "What it was used for: assistance in drafting Python code "
        "for the segmentation pipeline, the classifier, the dataset-preparation "
        "and annotation-verification scripts, the YOLO training and evaluation "
        "scripts, the real-time demonstration, and the evaluation harness; "
        "assistance in diagnosing possible areas of failure, including the "
        "tilted-card and "
        "edge-touching segmentation defects, the corrupted training crops, and "
        "the hand-held card detection failure described in Section VI-E, "
        "traced through live testing of the real-time demonstration to two "
        "distinct causes; implementation of a synthetic scale, rotation and "
        "perspective augmentation to address one of them; implementation of "
        "the real-photo-diversity dataset conversion and detector variant "
        "described in Section III-E; diagnosis of the arrow-class result in "
        "Section VI-F, including checking the training configuration and "
        "per-class instance counts against the augmentation hypothesis rather "
        "than asserting it from the observation alone; the labelling, "
        "filtering and verification pipeline for the deployment-condition "
        "frames in Section III-F, including the visual review of all thirty "
        "classes that rejected 51 frames the automatic filters had passed; "
        "discussion of experimental design and of "
        "how to report results honestly, including the decision to report "
        "the unsuccessful detector-plus-classifier combination in Section "
        "VI-E rather than omit it; and drafting assistance for this report."))
    A(para(
        "Work in the final round that was not AI-assisted should be "
        "distinguished, since it is the round that produced the largest "
        "improvement. The decision to capture training data in the "
        "deployment condition was the author's, as was the filming, the "
        "repeated frame extraction and hand-sorting of frames into per-card "
        "sets after automatic segmentation was found unreliable, and the "
        "frame-extraction tool (tools/frame_grabber.py), which was written by "
        "the author independently of this assistance."))
    A(para(
        "Which parts were AI-assisted: the majority of the source code was "
        "written with AI assistance. The experimental direction, the options "
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
        + para("Lee Tze Hian", style="Author")
        + para("U2423241F", style="Author")
        + para("College of Computing and Data Science", style="Affiliation")
        + para("Nanyang Technological University, Singapore", style="Affiliation")
        + f"<w:p><w:pPr><w:pStyle w:val=\"Affiliation\"/>{title_sect.replace('<w:sectPr', '<w:sectPr')}</w:pPr></w:p>"
    )
    # the sectPr must live inside pPr
    title_block = (
        para("Custom Object Detection for Text Symbols: A YOLO Baseline and a "
             "From-Scratch CNN Comparison", style="papertitle")
        + para("Lee Tze Hian", style="Author")
        + para("U2423241F", style="Author")
        + para("College of Computing and Data Science", style="Affiliation")
        + para("Nanyang Technological University, Singapore", style="Affiliation")
        + '<w:p><w:pPr><w:pStyle w:val="Affiliation"/>' + title_sect + "</w:pPr></w:p>"
    )

    body = build_body(rid)
    new_xml = root_open + title_block + body + body_sect + "</w:body></w:document>"
    open(DOC_XML, "w", encoding="utf-8").write(new_xml)

    out = os.path.join(HERE, "CT3205_Assignment1_Report_rev04.docx")
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
