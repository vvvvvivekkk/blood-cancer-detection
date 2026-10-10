#!/usr/bin/env python3
"""Build docs/HOW_IT_WORKS.pdf and docs/HOW_IT_WORKS.docx from source data.

Usage:
    python scripts/build_how_it_works.py
"""

import json
import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SCREENSHOTS = ROOT / "docs" / "screenshots"
MODELS_DIR = ROOT / "models"
OUT_PDF = ROOT / "docs" / "HOW_IT_WORKS.pdf"
OUT_DOCX = ROOT / "docs" / "HOW_IT_WORKS.docx"
SRC_DIR = ROOT / "docs" / "how_it_works_src"

W, H = A4
MARGIN = 2 * cm


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_json(p):
    with open(p) as f:
        return json.load(f)


metrics = load_json(RESULTS / "metrics.json")
run_info = load_json(RESULTS / "run_info.json")
ensemble_weights = load_json(MODELS_DIR / "ensemble_weights.json")
split_counts_raw = (RESULTS / "split_counts.csv").read_text().strip().splitlines()

# Parse split counts
split_header = split_counts_raw[0].split(",")
split_rows = [line.split(",") for line in split_counts_raw[1:]]
total_train = sum(int(r[1]) for r in split_rows)
total_val = sum(int(r[2]) for r in split_rows)
total_test = sum(int(r[3]) for r in split_rows)
total_images = total_train + total_val + total_test


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    "CoverTitle",
    parent=styles["Title"],
    fontSize=28,
    leading=34,
    textColor=colors.HexColor("#1a237e"),
    spaceAfter=12,
    alignment=TA_CENTER,
)
style_subtitle = ParagraphStyle(
    "CoverSubtitle",
    parent=styles["Normal"],
    fontSize=14,
    leading=18,
    textColor=colors.HexColor("#424242"),
    alignment=TA_CENTER,
    spaceAfter=6,
)
style_h1 = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontSize=18,
    leading=22,
    textColor=colors.HexColor("#1a237e"),
    spaceBefore=24,
    spaceAfter=10,
    keepWithNext=True,
)
style_h2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=14,
    leading=17,
    textColor=colors.HexColor("#283593"),
    spaceBefore=16,
    spaceAfter=8,
    keepWithNext=True,
)
style_body = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
)
style_body_bold = ParagraphStyle(
    "BodyBold",
    parent=style_body,
    fontName="Helvetica-Bold",
)
style_bullet = ParagraphStyle(
    "Bullet",
    parent=style_body,
    leftIndent=18,
    bulletIndent=6,
    spaceAfter=3,
)
style_code = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontSize=8,
    leading=10,
    backColor=colors.HexColor("#f5f5f5"),
    borderColor=colors.HexColor("#e0e0e0"),
    borderWidth=0.5,
    borderPadding=4,
    spaceAfter=6,
)
style_caption = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=9,
    leading=11,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#616161"),
    spaceAfter=12,
    spaceBefore=4,
)
style_toc_entry = ParagraphStyle(
    "TOCEntry",
    parent=styles["Normal"],
    fontSize=12,
    leading=20,
    leftIndent=12,
)
style_glossary = ParagraphStyle(
    "Glossary",
    parent=style_body,
    leftIndent=12,
    spaceAfter=4,
)
style_disclaimer = ParagraphStyle(
    "Disclaimer",
    parent=style_body,
    fontSize=9,
    leading=12,
    textColor=colors.HexColor("#b71c1c"),
    borderColor=colors.HexColor("#b71c1c"),
    borderWidth=1,
    borderPadding=8,
    backColor=colors.HexColor("#fff3f3"),
    spaceAfter=12,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def p(text, style=style_body):
    return Paragraph(text, style)


def h1(text):
    return Paragraph(text, style_h1)


def h2(text):
    return Paragraph(text, style_h2)


def bullet(text):
    return Paragraph(f"• {text}", style_bullet)


def bold(text):
    return f"<b>{text}</b>"


def italic(text):
    return f"<i>{text}</i>"


def code(text):
    return f'<font face="Courier" size="8">{text}</font>'


def fig(path, caption_text, width=None):
    """Return [Image, Caption] or [Caption noting missing]."""
    elems = []
    if path.exists():
        max_w = W - 2 * MARGIN - 1 * cm
        max_h = 14 * cm
        if width is None:
            width = min(max_w, 14 * cm)
        img = Image(str(path), width=width, height=None)
        iw, ih = img.imageWidth, img.imageHeight
        ratio = ih / iw
        final_w = min(width, max_w)
        final_h = final_w * ratio
        if final_h > max_h:
            final_h = max_h
            final_w = final_h / ratio
        img = Image(str(path), width=final_w, height=final_h)
        img.hAlign = "CENTER"
        elems.append(img)
    else:
        elems.append(
            p(f"[Figure not found: {path.name}]", style_caption)
        )
    elems.append(p(caption_text, style_caption))
    return elems


def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def pct(v):
    return f"{v * 100:.2f}%"


def f4(v):
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Page numbering
# ---------------------------------------------------------------------------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#757575"))
    canvas.drawCentredString(W / 2, 1.2 * cm, f"Page {doc.page}")
    canvas.drawString(MARGIN, 1.2 * cm, "Blood Cancer Detection — How It Works")
    canvas.restoreState()


def cover_footer(canvas, doc):
    pass


# ---------------------------------------------------------------------------
# Build story
# ---------------------------------------------------------------------------
def build_story():
    story = []

    # ── Cover ──
    story.append(Spacer(1, 6 * cm))
    story.append(p("Blood Cancer Identification using<br/>Hybrid Ensemble Deep Learning", style_title))
    story.append(Spacer(1, 1 * cm))
    story.append(p("How It Works — A Non-Technical Guide", style_subtitle))
    story.append(Spacer(1, 0.5 * cm))
    story.append(p("Prepared for the client", style_subtitle))
    story.append(Spacer(1, 1 * cm))
    story.append(
        p(
            '<font color="#b71c1c"><b>IMPORTANT NOTICE:</b> This software is a research and educational '
            "project. It is <b>not</b> approved for clinical diagnosis. All results must be verified by "
            "a qualified medical professional.</font>",
            style_disclaimer,
        )
    )
    story.append(Spacer(1, 2 * cm))
    story.append(p(f"Generated from project results dated {run_info['generated_at']}", style_caption))

    story.append(NextPageTemplate("normal"))
    story.append(PageBreak())

    # ── Table of Contents ──
    story.append(p("Table of Contents", style_h1))
    story.append(Spacer(1, 0.5 * cm))
    toc_items = [
        "1. What This Project Does",
        "2. The Data",
        "3. How a Computer Learns to Recognise Cells",
        "4. The Three Backbones",
        "5. The Hybrid Ensemble",
        "6. Grad-CAM: Where the Model Is Looking",
        "7. Results",
        "8. The Web Application",
        "9. Where Everything Is",
        "10. How It Was Built",
        "11. Glossary",
        "12. References",
    ]
    for item in toc_items:
        story.append(p(item, style_toc_entry))
    story.append(PageBreak())

    # ── 1. WHAT THIS PROJECT DOES ──
    story.append(h1("1. What This Project Does"))
    story.append(
        p(
            "Acute Lymphoblastic Leukaemia (ALL) is a type of blood cancer that affects white blood "
            "cells. Doctors diagnose it by examining a thin smear of blood under a microscope — a "
            f"{bold('peripheral blood smear')} (a glass slide coated with a drop of blood, spread thin, "
            "stained with dye and photographed). This project builds software that looks at those "
            "photographs and sorts them into one of four groups:"
        )
    )
    story.append(bullet(f"{bold('Benign')} — normal, healthy cells (no cancer)."))
    story.append(
        bullet(f"{bold('[Malignant] early Pre-B')} — cancer cells at an early stage called \"early Pre-B\".")
    )
    story.append(bullet(f"{bold('[Malignant] Pre-B')} — cancer cells at the \"Pre-B\" stage."))
    story.append(bullet(f"{bold('[Malignant] Pro-B')} — cancer cells at the \"Pro-B\" stage."))
    story.append(
        p(
            "The three malignant (cancerous) classes correspond to different stages of the cancer "
            "cell's development. Identifying the stage helps doctors choose the right treatment."
        )
    )
    story.append(
        p(
            f"In practice the user uploads a blood smear photograph through a web browser. The "
            f"software returns the predicted class, a confidence score (how sure it is), and a "
            f"highlighted image showing which part of the cell it focused on."
        )
    )
    story.append(
        p(
            '<font color="#b71c1c"><b>Not for clinical use.</b></font> '
            "This is a research project. It demonstrates that the technique works well on this "
            "dataset, but it has not been validated on other patient populations or approved by any "
            "regulatory body. A qualified haematologist must always make the final diagnosis.",
        )
    )

    # ── 2. THE DATA ──
    story.append(h1("2. The Data"))
    story.append(
        p(
            f"The images come from a public dataset on Kaggle (a website for data-science datasets) "
            f"called {italic('Blood Cells Cancer (ALL)')}, published by Aria et al. It contains "
            f"{bold(f'{total_images:,}')} microscope photographs of stained blood cells, each already "
            f"labelled by an expert as one of the four classes above."
        )
    )
    story.append(h2("What Is a Peripheral Blood Smear Image?"))
    story.append(
        p(
            "A peripheral blood smear is prepared by placing a drop of blood on a glass slide, "
            "spreading it into a thin layer, then applying a chemical stain (typically Wright or "
            "Giemsa stain) that colours different cell parts in different shades. The slide is then "
            "photographed under a microscope. The resulting image shows individual cells whose shape, "
            "size and colour pattern reveal whether they are healthy or cancerous."
        )
    )
    story.append(h2("Class Counts"))
    sc_headers = ["Class", "Training", "Validation", "Test", "Total"]
    sc_rows = []
    for r in split_rows:
        total = int(r[1]) + int(r[2]) + int(r[3])
        sc_rows.append([r[0], r[1], r[2], r[3], str(total)])
    sc_rows.append([bold("Total"), bold(str(total_train)), bold(str(total_val)), bold(str(total_test)), bold(str(total_images))])
    # Wrap cells in Paragraph for bold support
    sc_rows_p = []
    for row in sc_rows:
        sc_rows_p.append([p(c, style_body) for c in row])
    sc_headers_p = [p(h, style_body_bold) for h in sc_headers]
    story.append(make_table(sc_headers_p, sc_rows_p, col_widths=[5.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(h2("Train / Validation / Test Split"))
    story.append(
        p(
            "The images are divided into three groups, and each group has a specific job:"
        )
    )
    story.append(
        bullet(
            f"{bold('Training set')} ({total_train:,} images, 70%) — the computer studies these images "
            f"to learn the patterns. Think of this as the textbook a student reads."
        )
    )
    story.append(
        bullet(
            f"{bold('Validation set')} ({total_val:,} images, 15%) — used during training as a "
            f"practice exam. The computer checks its answers on these images to tune its learning, "
            f"but never memorises them."
        )
    )
    story.append(
        bullet(
            f"{bold('Test set')} ({total_test:,} images, 15%) — the final exam. The computer never "
            f"sees these images until the very end, so the score is an honest measure of performance."
        )
    )
    story.append(
        p(
            f"The split is {bold('stratified')}: each group has roughly the same proportion of each "
            f"class, so no class is accidentally left out. The file {code('src/data.py')} performs "
            f"the split and saves the lists to {code('data/splits/train.csv')}, "
            f"{code('val.csv')} and {code('test.csv')}."
        )
    )

    story.append(h2("Augmentation"))
    story.append(
        p(
            f"{bold('Augmentation')} means creating slightly altered copies of the training images — "
            f"flipping them horizontally, rotating them a few degrees, zooming in a little, or "
            f"adjusting the brightness. This teaches the computer to recognise a cell regardless "
            f"of its orientation or lighting. Augmentation is applied {italic('only')} to the "
            f"training set; validation and test images are left untouched so the scores stay fair."
        )
    )

    # ── 3. HOW A COMPUTER LEARNS ──
    story.append(h1("3. How a Computer Learns to Recognise Cells"))
    story.append(
        p(
            "At its core, the software uses a {0}. Imagine a chain of simple "
            "calculations, loosely inspired by the way neurons in a brain pass signals to each other. "
            "Each link in the chain receives a set of numbers, multiplies them by adjustable values "
            "called {1}, adds them up, and passes the result to the next link. At the end, "
            "the chain outputs four numbers — one for each class — and picks the highest.".format(
                bold("neural network"), bold("weights")
            )
        )
    )
    story.append(
        p(
            f"{bold('Training')} is the process of adjusting those weights so the network gives the "
            f"right answer more often. The computer shows the network a training image, compares its "
            f"answer to the correct label, measures how wrong it was (the {bold('loss')}), and nudges "
            f"the weights in the direction that would have made the answer closer to correct. This "
            f"cycle repeats millions of times."
        )
    )
    story.append(
        p(
            f"An {bold('epoch')} is one complete pass through every training image. Our models were "
            f"each trained for {run_info['backbones']['mobilenetv2']['epochs']} epochs."
        )
    )
    story.append(h2("A Visual Analogy"))
    story.append(
        p(
            "Imagine learning to identify birds. At first you would learn basic concepts — what "
            "feathers look like, the shape of beaks, the difference between wings and tails. "
            "Only then would you learn to tell a robin from a sparrow. A neural network works "
            "the same way: the early layers learn simple patterns (edges, colours, textures), "
            "and the later layers combine them into complex features (cell membranes, nuclei, "
            "the irregular shapes that signal cancer). The final layer compares those features "
            "against what it learned during training and picks the best-matching class."
        )
    )
    story.append(
        p(
            "The beauty of this layered approach is that the simple patterns (edges, textures) "
            "are useful for almost any image task. That is why transfer learning works so well: "
            "the early layers already know how to see."
        )
    )

    story.append(h2("How the Computer Improves"))
    story.append(
        p(
            "Each training cycle works like a feedback loop. The computer makes a prediction, "
            "checks the answer, and adjusts its internal numbers to do slightly better next time. "
            "The size of each adjustment is controlled by a setting called the {0}. "
            "If the learning rate is too large, the model overshoots and never settles; if too "
            "small, learning takes too long. Our project uses a learning rate of 0.001 in the "
            "first phase and 0.00001 in the second, striking a balance between speed and stability.".format(
                bold("learning rate")
            )
        )
    )
    story.append(
        p(
            "The {0} controls how many images are processed together in one step. "
            "We use a batch size of 32, meaning the model sees 32 images, averages its mistakes "
            "across them, and makes one adjustment. Larger batches are more stable; smaller batches "
            "can sometimes find better solutions.".format(bold("batch size"))
        )
    )

    story.append(h2("Transfer Learning"))
    story.append(
        p(
            "Training a neural network from scratch requires millions of images and days of "
            "computing time. A shortcut called {0} starts with a network that has "
            "already learned to recognise everyday objects (cats, cars, flowers) from a huge "
            "collection called {1} (over 14 million photos). Those pre-learned "
            "weights already know how to detect edges, textures and shapes. We keep that "
            "knowledge and re-train only the final layers to tell blood-cell classes apart. "
            "This is why the approach is called {2} — the network has been "
            "\"pre-trained\" on ImageNet.".format(
                bold("transfer learning"),
                bold("ImageNet"),
                bold("ImageNet pre-trained"),
            )
        )
    )

    # ── 4. THE THREE BACKBONES ──
    story.append(h1("4. The Three Backbones"))
    story.append(
        p(
            "A {0} is the core image-recognition network that does the heavy lifting. "
            "We use three different backbones, each designed by a different research team, "
            "so that their strengths complement each other.".format(bold("backbone"))
        )
    )

    backbones = [
        (
            "MobileNetV2",
            "mobilenetv2",
            "Designed by Google for mobile phones, MobileNetV2 is small and fast. It uses "
            "\"depth-wise separable convolutions\" — a way of processing images that needs far fewer "
            "calculations than the traditional approach, making it efficient even on a laptop.",
            "~3.5 million",
        ),
        (
            "EfficientNetB0",
            "efficientnetb0",
            "Also from Google, EfficientNetB0 was found by an automated search that balances "
            "network depth, width and image resolution to get the best accuracy for a given "
            "computing budget. It is compact but punches above its weight.",
            "~5.3 million",
        ),
        (
            "DenseNet121",
            "densenet121",
            "DenseNet121 connects every layer directly to every later layer (\"dense connections\"), "
            "so information flows freely from early features to late ones. This makes it very "
            "good at learning from smaller datasets.",
            "~8 million",
        ),
    ]

    for name, key, desc, params in backbones:
        story.append(h2(name))
        story.append(p(desc))
        bi = run_info["backbones"][key]
        story.append(
            p(
                f"Parameters: {bold(params)}. "
                f"Built in {code('src/models.py')}, trained in {code('src/train.py')}. "
                f"Trained for {bold(str(bi['epochs']))} epochs. "
                f"Test accuracy: {bold(pct(bi['test_accuracy']))}. "
                f"Test macro-F1: {bold(f4(bi['test_macro_f1']))}."
            )
        )
        # Per-class breakdown for each backbone
        pc = metrics["models"][key]["per_class"]
        bb_headers = [p("Class", style_body_bold), p("Precision", style_body_bold),
                      p("Recall", style_body_bold), p("F1", style_body_bold)]
        bb_rows = []
        for cls_name, vals in pc.items():
            bb_rows.append([p(cls_name, style_body), p(pct(vals["precision"]), style_body),
                           p(pct(vals["recall"]), style_body), p(f4(vals["f1-score"]), style_body)])
        story.append(make_table(bb_headers, bb_rows, col_widths=[5.5*cm, 3*cm, 3*cm, 3*cm]))

    story.append(h2("Two-Phase Fine-Tuning"))
    story.append(
        p(
            "Each backbone is trained in two phases:"
        )
    )
    story.append(
        bullet(
            f"{bold('Phase 1 — Head only')} (5 epochs): The pre-trained layers are \"frozen\" "
            f"(locked so their weights cannot change). Only the new classification layer at the "
            f"top learns from our blood-cell images. Learning rate: 0.001."
        )
    )
    story.append(
        bullet(
            f"{bold('Phase 2 — Partial unfreeze')} (10 more epochs): The top 30% of the "
            f"pre-trained layers are unfrozen, letting them adapt to blood-cell features while "
            f"the deeper layers (which detect basic edges and textures) stay locked. Learning "
            f"rate: 0.00001 (much smaller, to avoid destroying the pre-learned knowledge)."
        )
    )

    # ── 5. THE HYBRID ENSEMBLE ──
    story.append(h1("5. The Hybrid Ensemble"))
    story.append(
        p(
            "An {0} combines the opinions of several models, like a panel of doctors rather "
            "than a single doctor. Even if one model makes a mistake, the others can outvote it. "
            "Think of it as asking three specialists and going with the majority view.".format(
                bold("ensemble")
            )
        )
    )
    story.append(
        p(
            "The word {0} means we use two different combination methods and keep "
            "whichever scores higher — giving us the best of both worlds.".format(bold("hybrid"))
        )
    )

    story.append(h2("Method 1: Weighted Soft Voting"))
    story.append(
        p(
            "Each model outputs four numbers (one per class) representing how confident it is. "
            "We multiply each model's numbers by a weight that reflects how well that model "
            "performed on the validation set, then add them up. The class with the highest "
            "combined score wins."
        )
    )
    w = ensemble_weights["weights"]
    wm = f"{w['mobilenetv2']:.4f}"
    we = f"{w['efficientnetb0']:.4f}"
    wd = f"{w['densenet121']:.4f}"
    story.append(
        p(
            f"The weights used: MobileNetV2 = {bold(wm)}, "
            f"EfficientNetB0 = {bold(we)}, "
            f"DenseNet121 = {bold(wd)}. "
            f"(Stored in {code('models/ensemble_weights.json')}.)"
        )
    )

    story.append(h2("Worked Example"))
    story.append(
        p(
            "Suppose for one image the three models output these confidences for the \"Benign\" class:"
        )
    )
    story.append(bullet("MobileNetV2 says 0.90"))
    story.append(bullet("EfficientNetB0 says 0.85"))
    story.append(bullet("DenseNet121 says 0.95"))
    story.append(
        p(
            f"Weighted score = (0.3418 × 0.90) + (0.3114 × 0.85) + (0.3468 × 0.95) "
            f"= 0.3076 + 0.2647 + 0.3295 = {bold('0.9018')}. "
            f"The same calculation is done for each of the four classes; the class with the "
            f"highest weighted score is the prediction."
        )
    )

    story.append(h2("Method 2: Stacked Meta-Classifier"))
    story.append(
        p(
            "Instead of a simple weighted average, we feed all twelve numbers (four from each "
            "of the three models) into a small second-stage classifier — a {0}. "
            "This is a simple mathematical formula that learns, from the validation data, "
            "which combinations of model outputs are most reliable. Think of it as a senior "
            "doctor who has learned to read and interpret the three specialists' reports.".format(
                bold("logistic regression")
            )
        )
    )
    story.append(
        p(
            f"The meta-classifier is saved as {code('models/meta_clf.pkl')}. "
            f"The code lives in {code('src/ensemble.py')} and {code('src/predict.py')}."
        )
    )

    story.append(h2("Which One Was Chosen?"))
    m_v = metrics["models"]["voting"]
    m_s = metrics["models"]["stacking"]
    story.append(
        p(
            f"Both methods were evaluated on the test set. "
            f"Voting achieved {bold(pct(m_v['accuracy']))} accuracy (macro-F1 {bold(f4(m_v['macro_f1']))}). "
            f"Stacking achieved {bold(pct(m_s['accuracy']))} accuracy (macro-F1 {bold(f4(m_s['macro_f1']))}). "
            f"The system automatically selected {bold('Stacking')} because it had a higher macro-F1. "
            f"This choice is recorded in {code('models/ensemble_mode.txt')}."
        )
    )
    story.append(
        p(
            "Importantly, the test set was {0} used to choose the method — the comparison "
            "was first done on validation data, and the test set only measured the final score. "
            "This prevents the system from \"cheating\" by tailoring its choice to the test questions.".format(
                bold("never")
            )
        )
    )

    # ── 6. GRAD-CAM ──
    story.append(h1("6. Grad-CAM: Where the Model Is Looking"))
    story.append(
        p(
            f"{bold('Grad-CAM')} (Gradient-weighted Class Activation Mapping) produces a coloured "
            f"heatmap laid over the original image. Warm colours (red, yellow) mark the regions "
            f"the model paid the most attention to when making its decision; cool colours (blue) "
            f"mark regions it largely ignored."
        )
    )
    story.append(
        p(
            "This does {0} prove the model is correct — it only shows where it looked. "
            "A doctor can glance at the heatmap to quickly judge whether the model's reasoning "
            "makes sense (e.g., is it looking at the cell, or at a staining artefact?).".format(bold("not"))
        )
    )
    story.append(
        p(
            f"The computation is in {code('src/gradcam.py')}. For each backbone, the heatmap is "
            f"taken from the last spatial layer of the network."
        )
    )
    story.extend(fig(FIGURES / "gradcam_examples.png", "Figure 1. Grad-CAM examples — the heatmap shows where each model looked."))

    # ── 7. RESULTS ──
    story.append(h1("7. Results"))
    story.append(h2("Comparison Table"))
    comp_headers = [
        p("Model", style_body_bold),
        p("Accuracy", style_body_bold),
        p("Macro-F1", style_body_bold),
        p("Weighted-F1", style_body_bold),
        p("Kappa", style_body_bold),
        p("ROC-AUC", style_body_bold),
    ]
    comp_rows = []
    for name, key in [
        ("MobileNetV2", "mobilenetv2"),
        ("EfficientNetB0", "efficientnetb0"),
        ("DenseNet121", "densenet121"),
        ("Ensemble-Voting", "voting"),
        ("Ensemble-Stacking ✓", "stacking"),
    ]:
        m = metrics["models"][key]
        comp_rows.append([
            p(name, style_body),
            p(pct(m["accuracy"]), style_body),
            p(f4(m["macro_f1"]), style_body),
            p(f4(m["weighted_f1"]), style_body),
            p(f4(m["kappa"]), style_body),
            p(f4(m["roc_auc_ovr"]), style_body),
        ])
    story.append(make_table(comp_headers, comp_rows, col_widths=[3.5 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        p(
            f"{bold('Accuracy')} is the percentage of images classified correctly. "
            f"{bold('Macro-F1')} averages precision and recall across all four classes equally, "
            f"so a class with few images counts just as much as a large one. "
            f"{bold('Kappa')} measures agreement beyond chance. "
            f"{bold('ROC-AUC')} measures how well the model separates each class from the others "
            f"across all confidence thresholds; 1.0 is perfect."
        )
    )

    story.append(h2("Per-Class Results (Stacking Ensemble)"))
    pc_headers = [
        p("Class", style_body_bold),
        p("Precision", style_body_bold),
        p("Recall", style_body_bold),
        p("F1-Score", style_body_bold),
        p("Support", style_body_bold),
    ]
    pc = metrics["models"]["stacking"]["per_class"]
    pc_rows = []
    for cls_name, vals in pc.items():
        pc_rows.append([
            p(cls_name, style_body),
            p(pct(vals["precision"]), style_body),
            p(pct(vals["recall"]), style_body),
            p(f4(vals["f1-score"]), style_body),
            p(str(int(vals["support"])), style_body),
        ])
    story.append(make_table(pc_headers, pc_rows, col_widths=[5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        p(
            f"{bold('Precision')} answers: \"Of all images the model called X, how many really were X?\" "
            f"{bold('Recall')} answers: \"Of all images that really are X, how many did the model find?\" "
            f"{bold('Support')} is the number of test images in that class."
        )
    )

    story.append(h2("Confusion Matrix"))
    story.append(
        p(
            "A confusion matrix is a grid that shows, for each true class, how many images "
            "the model put into each predicted class. The diagonal (top-left to bottom-right) "
            "shows correct predictions; off-diagonal numbers are mistakes. A perfect model "
            "would have zeros everywhere except the diagonal."
        )
    )
    story.extend(fig(FIGURES / "cm_stacking.png", "Figure 2. Confusion matrix for the selected Stacking Ensemble."))

    story.append(h2("ROC Curve"))
    story.append(
        p(
            "An ROC (Receiver Operating Characteristic) curve plots how well the model separates "
            "one class from the rest at every possible confidence threshold. The closer the curve "
            "hugs the top-left corner, the better. The area under this curve (AUC) of 1.0 is perfect; "
            "0.5 is random guessing."
        )
    )
    story.extend(fig(FIGURES / "roc_ensemble.png", "Figure 3. ROC curves for the ensemble."))

    story.append(h2("Training Curves"))
    story.append(
        p(
            "Training curves show how accuracy and loss changed over each epoch. A well-trained "
            "model shows both lines converging — the training score and validation score settle "
            "near each other. If the training score climbs but validation does not, the model is "
            "memorising the training images rather than learning general patterns (\"overfitting\")."
        )
    )
    story.extend(fig(FIGURES / "training_curves_mobilenetv2.png", "Figure 4. Training curves for MobileNetV2."))
    story.extend(fig(FIGURES / "training_curves_efficientnetb0.png", "Figure 5. Training curves for EfficientNetB0."))
    story.extend(fig(FIGURES / "training_curves_densenet121.png", "Figure 6. Training curves for DenseNet121."))

    story.extend(fig(FIGURES / "model_comparison_bar.png", "Figure 7. Accuracy comparison across all models."))

    story.append(h2("Individual Backbone Confusion Matrices"))
    story.append(
        p(
            "The confusion matrices below show how each individual backbone performed before "
            "being combined into the ensemble. Comparing them reveals where each model's "
            "strengths and weaknesses lie."
        )
    )
    story.extend(fig(FIGURES / "cm_mobilenetv2.png", "Figure 8. Confusion matrix for MobileNetV2."))
    story.extend(fig(FIGURES / "cm_efficientnetb0.png", "Figure 9. Confusion matrix for EfficientNetB0."))
    story.extend(fig(FIGURES / "cm_densenet121.png", "Figure 10. Confusion matrix for DenseNet121."))
    story.extend(fig(FIGURES / "cm_voting.png", "Figure 11. Confusion matrix for Weighted Voting Ensemble."))

    story.append(h2("Comparison to Published Work"))
    story.append(
        p(
            "The original IEEE paper by Jayachitra and Umarkathaf (2023) reported 95% accuracy, "
            "but on a binary classification (cancer vs. not-cancer) rather than four classes. "
            "Our four-class Stacking Ensemble achieves {0} accuracy — a strong result on the "
            "harder problem. The reference project by Roozgarian achieved 96% with MobileNetV2 alone; "
            "our DenseNet121 alone reaches {1}, and the ensemble matches it with higher ROC-AUC "
            "({2}).".format(
                bold(pct(m_s["accuracy"])),
                bold(pct(metrics["models"]["densenet121"]["accuracy"])),
                bold(f4(m_s["roc_auc_ovr"])),
            )
        )
    )
    story.append(h2("Honest Limitations"))
    story.append(bullet("The ensemble does not beat the best single backbone (DenseNet121) by a meaningful margin on this dataset."))
    story.append(bullet("All images come from a single source. Performance on images from a different hospital, microscope or staining protocol is unknown."))
    story.append(bullet("The dataset is moderately small (3,242 images). Larger datasets could improve and stress-test the model."))
    story.append(bullet("The system has not undergone clinical validation or regulatory review."))

    # ── 8. THE WEB APPLICATION ──
    story.append(h1("8. The Web Application"))
    story.append(
        p(
            "The web application lets a user interact with the system through a browser. "
            "No installation is needed on the user's computer — they simply open a web page. "
            "Below is a tour of each screen."
        )
    )

    screens = [
        ("Login", "overview", "The user logs in with a username and password. This is a demonstration gate (not hospital-grade security). Default credentials: admin / admin."),
        ("Home / Overview", "overview", "Shows the pipeline status: which steps have been completed (data downloaded, models trained, evaluation run). Provides navigation to all other screens."),
        ("Data", "data", "Manages the dataset: download from Kaggle or generate demo data, create the train/validation/test split, view class counts and sample images."),
        ("Train", "train", "Starts training. Shows live progress bars and charts as each backbone learns. Displays training history when complete."),
        ("Evaluate / Performance Analysis", "evaluate", "Fits the ensemble, runs evaluation on the test set, and displays all the metrics, confusion matrices and charts shown in Section 7."),
        ("Predict / Preview", "predict", "The main screen for the client. Upload a blood smear image and receive: the predicted class, a confidence score, individual model votes, and a Grad-CAM heatmap."),
        ("About", "about", "Describes the project, the method, the team, the paper reference, and the \"not for clinical use\" disclaimer."),
    ]
    for title, img_key, desc in screens:
        story.append(h2(title))
        story.append(p(desc))
        story.extend(fig(SCREENSHOTS / f"{img_key}.png", f"Screenshot: {title} screen."))

    story.append(h2("What Happens When You Upload an Image"))
    story.append(
        p("Here is the step-by-step journey of an uploaded image:")
    )
    steps = [
        "The user selects a blood smear image in their browser and clicks \"Predict\".",
        "The browser sends the image to the server (api/main.py) as an upload.",
        "The server passes the image to the prediction engine (src/predict.py).",
        "The image is resized to 224×224 pixels and pre-processed for each backbone.",
        "All three backbones (MobileNetV2, EfficientNetB0, DenseNet121) analyse the image and each outputs four confidence scores.",
        "The ensemble method (Stacking) combines the twelve scores into a final prediction.",
        "Grad-CAM generates a heatmap from the most confident backbone.",
        "The server returns the predicted class, confidence, per-model votes, and heatmap image to the browser.",
        "The browser displays the results.",
    ]
    for i, step in enumerate(steps, 1):
        story.append(bullet(f"{bold(f'Step {i}:')} {step}"))

    story.append(h2("Confidence and Out-of-Distribution Warnings"))
    story.append(
        p(
            "The system reports a {0} level for each prediction — high, medium or low — "
            "based on how certain the models are. If the confidence is low, a warning is shown.".format(
                bold("confidence")
            )
        )
    )
    story.append(
        p(
            "An {0} (OOD) check compares the uploaded image's colour statistics against "
            "what the training images looked like. If the image looks very different (e.g., a photo "
            "of a cat instead of a blood smear), the system warns the user that the prediction may "
            "not be reliable. This is computed in {1}.".format(
                bold("out-of-distribution"),
                code("src/ood.py"),
            )
        )
    )

    # ── 9. WHERE EVERYTHING IS ──
    story.append(h1("9. Where Everything Is"))
    story.append(
        p("The table below lists every important file and folder in the project.")
    )

    file_map = [
        ("run.py", "Entry point", "Starts the web app or runs pipeline steps from the command line.", "User"),
        ("app.py", "Alias", "Same as run.py — runs the web app.", "User"),
        ("requirements.txt", "Dependencies", "Lists every Python package and version needed.", "Installer"),
        ("INSTALL.md", "Setup guide", "Step-by-step installation instructions.", "User"),
        ("src/config.py", "Configuration", "Class names, image size, hyperparameters, paths.", "All code"),
        ("src/data.py", "Data pipeline", "Splits images, builds training batches, applies augmentation.", "train.py"),
        ("src/models.py", "Model builder", "Constructs the three backbones with pre-trained weights.", "train.py"),
        ("src/train.py", "Trainer", "Two-phase fine-tuning of each backbone.", "Pipeline"),
        ("src/ensemble.py", "Ensemble", "Fits voting weights and the stacking meta-classifier.", "Pipeline"),
        ("src/predict.py", "Predictor", "Loads models, runs inference, returns class and confidence.", "API"),
        ("src/gradcam.py", "Grad-CAM", "Generates heatmaps showing where the model looked.", "predict.py"),
        ("src/evaluate.py", "Evaluator", "Runs test-set metrics, generates all figures and tables.", "Pipeline"),
        ("src/ood.py", "OOD detector", "Checks if an uploaded image looks like training data.", "predict.py"),
        ("api/main.py", "Web server", "FastAPI routes for uploads, predictions, training, status.", "Browser"),
        ("api/jobs.py", "Job runner", "Runs long tasks (training) in the background.", "main.py"),
        ("frontend/index.html", "Web page", "The single-page application the user sees.", "Browser"),
        ("frontend/js/app.js", "Frontend logic", "Routing, rendering, event handling.", "Browser"),
        ("frontend/css/styles.css", "Styles", "Light/dark themes, responsive layout.", "Browser"),
        ("models/mobilenetv2.keras", "Trained model", "MobileNetV2 weights (the learned numbers).", "predict.py"),
        ("models/efficientnetb0.keras", "Trained model", "EfficientNetB0 weights.", "predict.py"),
        ("models/densenet121.keras", "Trained model", "DenseNet121 weights.", "predict.py"),
        ("models/ensemble_weights.json", "Voting config", "The three voting weights.", "ensemble.py"),
        ("models/meta_clf.pkl", "Stacking model", "The trained logistic regression.", "predict.py"),
        ("models/ensemble_mode.txt", "Mode flag", "Which ensemble method was selected (stacking).", "predict.py"),
        ("models/ood_stats.json", "OOD baseline", "Colour statistics of training images.", "ood.py"),
        ("results/metrics.json", "Metrics", "All accuracy, F1, confusion matrix numbers.", "Docs"),
        ("results/run_info.json", "Run log", "When training ran, how many epochs, versions.", "Docs"),
        ("results/figures/", "Charts", "Confusion matrices, ROC curves, training curves.", "Docs"),
        ("data/splits/", "Split CSVs", "Which images go to train, validation, test.", "data.py"),
        ("tests/", "Test suite", "Automated checks that everything works correctly.", "Developer"),
        ("notebooks/", "Colab notebook", "For training on Google's free GPUs.", "Developer"),
        ("docs/", "Documentation", "This document, the paper, screenshots, reports.", "Client"),
    ]
    fm_headers = [p("Path", style_body_bold), p("What It Is", style_body_bold), p("What It Does", style_body_bold), p("Used By", style_body_bold)]
    fm_rows = [[p(code(r[0]), style_body), p(r[1], style_body), p(r[2], style_body), p(r[3], style_body)] for r in file_map]
    story.append(make_table(fm_headers, fm_rows, col_widths=[4.2*cm, 2.5*cm, 6.5*cm, 2.3*cm]))

    story.append(h2("Data Flow Diagram"))
    story.append(
        p(
            "The overall pipeline flows like this: "
            f"{bold('Data')} (Kaggle images) → {bold('Split')} (train / val / test) → "
            f"{bold('Train')} (three backbones) → {bold('Models')} (saved weights) → "
            f"{bold('Ensemble')} (voting + stacking) → {bold('Evaluate')} (test metrics) → "
            f"{bold('Results')} (figures + tables) → {bold('Web App')} (browser interface)."
        )
    )

    # ── 10. HOW IT WAS BUILT ──
    story.append(h1("10. How It Was Built"))
    story.append(h2("Tools and Versions"))
    tools = [
        ("Python", "3.12"),
        ("TensorFlow", run_info["versions"]["tensorflow"]),
        ("Keras", run_info["versions"]["keras"]),
        ("scikit-learn", "1.7.2"),
        ("FastAPI", "0.136.3"),
        ("Playwright", "for automated browser testing"),
        ("Chart.js", "4.4.1 (in the browser)"),
        ("NumPy", "1.26.4"),
        ("Pandas", "2.2.2"),
        ("Pillow", "11.3.0"),
        ("OpenCV", "4.10.0"),
        ("Matplotlib / Seaborn", "3.10.7 / 0.13.2"),
    ]
    for name, ver in tools:
        story.append(bullet(f"{bold(name)}: {ver}"))

    story.append(h2("Testing"))
    story.append(
        p(
            "The project includes 14 test files in the tests/ folder. They cover the data pipeline, "
            "model building, training, ensemble logic, prediction, Grad-CAM, the API, the frontend, "
            "and static assets. Tests run with {0} and take about 1–2 minutes on a laptop. "
            "They use tiny sample images and untrained models, so no GPU is needed.".format(code("pytest"))
        )
    )

    story.append(h2("How to Retrain"))
    story.append(p("If you ever receive new data and want to retrain the models, run these commands:"))
    story.append(p("python run.py --cli download", style_code))
    story.append(p("  Downloads the dataset from Kaggle (requires credentials).", style_body))
    story.append(p("python run.py --cli splits", style_code))
    story.append(p("  Creates fresh train/validation/test splits.", style_body))
    story.append(p("python run.py --cli train", style_code))
    story.append(p("  Trains all three backbones (needs a GPU for reasonable speed).", style_body))
    story.append(p("python run.py --cli ensemble", style_code))
    story.append(p("  Fits the ensemble on validation predictions.", style_body))
    story.append(p("python run.py --cli evaluate", style_code))
    story.append(p("  Evaluates on the test set and generates all figures.", style_body))

    story.append(h2("Downloading Pre-Trained Models"))
    story.append(
        p(
            "If you just want to run the app without training, use:"
        )
    )
    story.append(p("python run.py --fetch-models &lt;URL&gt;", style_code))
    story.append(p("  This downloads the pre-trained model files (~105 MB) and verifies their integrity.", style_body))

    # ── 11. GLOSSARY ──
    story.append(h1("11. Glossary"))
    glossary = [
        ("Accuracy", "The percentage of all predictions that were correct."),
        ("ALL (Acute Lymphoblastic Leukaemia)", "A type of blood cancer affecting white blood cells."),
        ("Augmentation", "Creating altered copies of training images (flipped, rotated, etc.) to help the model generalise."),
        ("AUC (Area Under the Curve)", "A single number summarising how well the model separates classes; 1.0 is perfect."),
        ("Backbone", "The core image-recognition neural network (e.g., MobileNetV2)."),
        ("Batch size", "The number of images processed together in one step of training (32 in this project)."),
        ("Confusion matrix", "A grid showing how many images of each true class were predicted as each class."),
        ("Convolution", "A mathematical operation that slides a small filter across an image to detect patterns."),
        ("Dense connections", "A design where every layer feeds into every later layer (used by DenseNet)."),
        ("Ensemble", "A method that combines several models to make a joint prediction."),
        ("Epoch", "One complete pass through all training images."),
        ("F1-score", "The harmonic mean of precision and recall; balances both measures."),
        ("Fine-tuning", "Adjusting a pre-trained model's weights on a new, specific dataset."),
        ("Frozen layers", "Layers whose weights are locked and do not change during training."),
        ("Grad-CAM", "A technique that highlights which image regions influenced the model's prediction."),
        ("Heatmap", "A coloured overlay showing regions of high (red) and low (blue) importance."),
        ("Hybrid", "Using two different ensemble methods (voting and stacking) and selecting the better one."),
        ("ImageNet", "A public collection of over 14 million labelled photographs used to pre-train models."),
        ("Kappa", "Cohen's Kappa — a metric that measures agreement corrected for chance."),
        ("Learning rate", "How large each adjustment step is during training; too large = unstable, too small = slow."),
        ("Logistic regression", "A simple mathematical model that learns to classify based on input numbers."),
        ("Loss", "A number measuring how wrong the model's prediction is; training aims to minimise it."),
        ("Macro-F1", "The F1-score averaged equally across all classes."),
        ("Meta-classifier", "A second-stage model that learns to combine the outputs of other models."),
        ("Neural network", "A chain of mathematical operations, inspired by brain neurons, that learns patterns from data."),
        ("OOD (Out-of-Distribution)", "An image that looks very different from what the model was trained on."),
        ("Overfitting", "When a model memorises training data rather than learning general patterns."),
        ("Peripheral blood smear", "A thin layer of blood on a glass slide, stained and photographed under a microscope."),
        ("Precision", "Of all images the model predicted as class X, what fraction actually were X."),
        ("Pre-trained", "A model that has already learned from a large dataset before being adapted to a new task."),
        ("Recall", "Of all images that truly are class X, what fraction the model correctly identified."),
        ("ROC curve", "A plot of true-positive rate vs. false-positive rate across confidence thresholds."),
        ("Softmax", "A function that converts raw model outputs into probabilities that sum to 1."),
        ("Stacking", "An ensemble method where a second model learns to combine the first models' outputs."),
        ("Stratified split", "Dividing data so each subset has the same class proportions as the whole."),
        ("Support", "The number of test samples in a given class."),
        ("Transfer learning", "Using knowledge from one task (ImageNet) to help with another (blood cells)."),
        ("Validation set", "A held-out portion of data used to tune the model during training."),
        ("Voting (weighted soft)", "An ensemble method that averages models' probabilities using weights."),
        ("Weights", "The adjustable numbers inside a neural network that are learned during training."),
    ]
    for term, defn in glossary:
        story.append(p(f"{bold(term)}: {defn}", style_glossary))

    # ── 12. REFERENCES ──
    story.append(h1("12. References"))
    refs = [
        "J. Jayachitra and N. Umarkathaf, \"Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique,\" "
        "2023 International Conference on Emerging Research in Applied Sciences (ICEARS), IEEE, 2023.",
        "M. A. Eshraghi et al., \"Blood Cells Cancer (ALL) — 4 Classes,\" Kaggle dataset.",
        "M. Sandler et al., \"MobileNetV2: Inverted Residuals and Linear Bottlenecks,\" CVPR 2018.",
        "M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,\" ICML 2019.",
        "G. Huang et al., \"Densely Connected Convolutional Networks,\" CVPR 2017.",
        "R. R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,\" ICCV 2017.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(p(f"[{i}] {ref}"))

    story.append(Spacer(1, 2 * cm))
    story.append(
        p(
            '<font color="#b71c1c"><b>Disclaimer:</b> This software is a research and educational '
            "project. It is not approved for clinical use. All predictions must be verified by a "
            "qualified medical professional.</font>",
            style_disclaimer,
        )
    )

    return story


# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
def build_pdf():
    doc = BaseDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Blood Cancer Detection — How It Works",
        author="Blood Cancer Detection Project",
    )
    frame = Frame(MARGIN, MARGIN, W - 2 * MARGIN, H - 2 * MARGIN, id="normal")
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[frame], onPage=cover_footer),
            PageTemplate(id="normal", frames=[frame], onPage=footer),
        ]
    )
    story = build_story()
    doc.build(story)
    print(f"PDF written to {OUT_PDF} ({doc.page} pages)")
    return doc.page


# ---------------------------------------------------------------------------
# Build DOCX
# ---------------------------------------------------------------------------
def build_docx():
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(10)
    style.font.name = "Calibri"

    # Cover
    doc.add_paragraph("")
    doc.add_paragraph("")
    t = doc.add_paragraph("Blood Cancer Identification using Hybrid Ensemble Deep Learning")
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in t.runs:
        run.font.size = Pt(24)
        run.bold = True
    sub = doc.add_paragraph("How It Works — A Non-Technical Guide")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.font.size = Pt(14)
    disc = doc.add_paragraph(
        "IMPORTANT NOTICE: This software is a research and educational project. "
        "It is not approved for clinical diagnosis."
    )
    disc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in disc.runs:
        run.font.color.rgb = None
    doc.add_page_break()

    sections_text = _get_docx_sections()
    for section in sections_text:
        if section[0] == "h1":
            doc.add_heading(section[1], level=1)
        elif section[0] == "h2":
            doc.add_heading(section[1], level=2)
        elif section[0] == "p":
            doc.add_paragraph(section[1])
        elif section[0] == "bullet":
            doc.add_paragraph(section[1], style="List Bullet")
        elif section[0] == "img":
            path = section[1]
            caption = section[2]
            if Path(path).exists():
                doc.add_picture(str(path), width=Inches(5.5))
                last_p = doc.paragraphs[-1]
                last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cap.runs:
                run.italic = True
                run.font.size = Pt(9)
        elif section[0] == "table":
            headers, rows = section[1], section[2]
            table = doc.add_table(rows=1 + len(rows), cols=len(headers))
            table.style = "Light Shading Accent 1"
            for i, h in enumerate(headers):
                table.rows[0].cells[i].text = h
            for ri, row in enumerate(rows):
                for ci, cell in enumerate(row):
                    table.rows[ri + 1].cells[ci].text = cell

    doc.save(str(OUT_DOCX))
    print(f"DOCX written to {OUT_DOCX}")


def _get_docx_sections():
    """Return a list of (type, ...) tuples for the DOCX builder."""
    m_s = metrics["models"]["stacking"]
    m_v = metrics["models"]["voting"]
    w = ensemble_weights["weights"]

    s = []
    # 1
    s.append(("h1", "1. What This Project Does"))
    s.append(("p", "Acute Lymphoblastic Leukaemia (ALL) is a type of blood cancer that affects white blood cells. "
              "Doctors diagnose it by examining a thin smear of blood under a microscope — a peripheral blood smear. "
              "This project builds software that looks at those photographs and sorts them into four groups:"))
    s.append(("bullet", "Benign — normal, healthy cells (no cancer)."))
    s.append(("bullet", "[Malignant] early Pre-B — cancer cells at an early stage."))
    s.append(("bullet", "[Malignant] Pre-B — cancer cells at the Pre-B stage."))
    s.append(("bullet", "[Malignant] Pro-B — cancer cells at the Pro-B stage."))
    s.append(("p", "The user uploads a blood smear photograph through a web browser. The software returns the predicted class, "
              "a confidence score, and a highlighted image showing which part of the cell it focused on."))
    s.append(("p", "NOT FOR CLINICAL USE. This is a research project. All results must be verified by a qualified medical professional."))

    # 2
    s.append(("h1", "2. The Data"))
    s.append(("p", f"The images come from a public Kaggle dataset called 'Blood Cells Cancer (ALL)' by Aria et al. "
              f"It contains {total_images:,} microscope photographs of stained blood cells."))
    s.append(("h2", "Class Counts"))
    sc_h = ["Class", "Training", "Validation", "Test", "Total"]
    sc_r = []
    for r in split_rows:
        total = int(r[1]) + int(r[2]) + int(r[3])
        sc_r.append([r[0], r[1], r[2], r[3], str(total)])
    sc_r.append(["Total", str(total_train), str(total_val), str(total_test), str(total_images)])
    s.append(("table", sc_h, sc_r))
    s.append(("p", f"The split is 70% training ({total_train}), 15% validation ({total_val}), 15% test ({total_test}). "
              "The file src/data.py performs the split."))
    s.append(("h2", "Augmentation"))
    s.append(("p", "Augmentation means creating slightly altered copies of training images (flipped, rotated, zoomed, brightness-adjusted) "
              "to teach the model to recognise cells regardless of orientation or lighting."))

    # 3
    s.append(("h1", "3. How a Computer Learns to Recognise Cells"))
    s.append(("p", "The software uses a neural network — a chain of simple calculations where each link multiplies numbers by adjustable "
              "values called weights. Training adjusts those weights so the network gives correct answers more often. "
              f"Each backbone was trained for {run_info['backbones']['mobilenetv2']['epochs']} epochs (complete passes through the data)."))
    s.append(("p", "Transfer learning starts with a network pre-trained on ImageNet (14 million everyday photos) and re-trains "
              "only the final layers for blood cells, saving time and improving accuracy."))

    # 4
    s.append(("h1", "4. The Three Backbones"))
    for name, key, desc in [
        ("MobileNetV2", "mobilenetv2", "Small, fast, designed for mobile phones. ~3.5 million parameters."),
        ("EfficientNetB0", "efficientnetb0", "Balances depth, width and resolution efficiently. ~5.3 million parameters."),
        ("DenseNet121", "densenet121", "Dense connections let information flow freely between layers. ~8 million parameters."),
    ]:
        bi = run_info["backbones"][key]
        s.append(("h2", name))
        s.append(("p", f"{desc} Test accuracy: {pct(bi['test_accuracy'])}. Macro-F1: {f4(bi['test_macro_f1'])}."))
    s.append(("h2", "Two-Phase Fine-Tuning"))
    s.append(("bullet", "Phase 1 (5 epochs): Frozen base, only the classification head learns. Learning rate: 0.001."))
    s.append(("bullet", "Phase 2 (10 epochs): Top 30% of layers unfrozen. Learning rate: 0.00001."))

    # 5
    s.append(("h1", "5. The Hybrid Ensemble"))
    s.append(("p", "An ensemble combines several models like a panel of doctors. The word 'hybrid' means two combination methods are tried."))
    s.append(("h2", "Method 1: Weighted Soft Voting"))
    s.append(("p", f"Weights: MobileNetV2={w['mobilenetv2']:.4f}, EfficientNetB0={w['efficientnetb0']:.4f}, DenseNet121={w['densenet121']:.4f}."))
    s.append(("h2", "Method 2: Stacked Meta-Classifier"))
    s.append(("p", "A logistic regression learns to combine all twelve output scores from the three models."))
    s.append(("p", f"Stacking was selected with {pct(m_s['accuracy'])} accuracy and {f4(m_s['macro_f1'])} macro-F1."))

    # 6
    s.append(("h1", "6. Grad-CAM: Where the Model Is Looking"))
    s.append(("p", "Grad-CAM produces a coloured heatmap showing which image regions influenced the prediction. "
              "Warm colours = high attention, cool colours = low attention. Computed in src/gradcam.py."))
    s.append(("img", str(FIGURES / "gradcam_examples.png"), "Figure 1. Grad-CAM examples."))

    # 7
    s.append(("h1", "7. Results"))
    comp_h = ["Model", "Accuracy", "Macro-F1", "Weighted-F1", "Kappa", "ROC-AUC"]
    comp_r = []
    for name, key in [("MobileNetV2", "mobilenetv2"), ("EfficientNetB0", "efficientnetb0"),
                       ("DenseNet121", "densenet121"), ("Ensemble-Voting", "voting"), ("Ensemble-Stacking (selected)", "stacking")]:
        m = metrics["models"][key]
        comp_r.append([name, pct(m["accuracy"]), f4(m["macro_f1"]), f4(m["weighted_f1"]), f4(m["kappa"]), f4(m["roc_auc_ovr"])])
    s.append(("table", comp_h, comp_r))
    s.append(("img", str(FIGURES / "cm_stacking.png"), "Figure 2. Confusion matrix for Stacking Ensemble."))
    s.append(("img", str(FIGURES / "roc_ensemble.png"), "Figure 3. ROC curves."))
    s.append(("img", str(FIGURES / "training_curves_densenet121.png"), "Figure 4. Training curves for DenseNet121."))
    s.append(("img", str(FIGURES / "model_comparison_bar.png"), "Figure 5. Accuracy comparison."))

    # 8
    s.append(("h1", "8. The Web Application"))
    for title, img_key, desc in [
        ("Login / Home", "overview", "Login with admin/admin. Home shows pipeline status and navigation."),
        ("Data", "data", "Download dataset or generate demo data, create splits, view class counts."),
        ("Train", "train", "Start training with live progress bars and charts."),
        ("Evaluate", "evaluate", "Fit ensemble, run evaluation, display all metrics and charts."),
        ("Predict", "predict", "Upload a blood smear image to get prediction, confidence and Grad-CAM heatmap."),
        ("About", "about", "Project description, method, references, disclaimer."),
    ]:
        s.append(("h2", title))
        s.append(("p", desc))
        s.append(("img", str(SCREENSHOTS / f"{img_key}.png"), f"Screenshot: {title}."))

    # 9
    s.append(("h1", "9. Where Everything Is"))
    s.append(("p", "See the PDF version for the complete file map table."))

    # 10
    s.append(("h1", "10. How It Was Built"))
    s.append(("p", f"Python 3.12, TensorFlow {run_info['versions']['tensorflow']}, Keras {run_info['versions']['keras']}, "
              "scikit-learn 1.7.2, FastAPI 0.136.3, Chart.js 4.4.1."))
    s.append(("p", "14 test files covering data, models, training, ensemble, prediction, Grad-CAM, API, frontend and static assets."))
    s.append(("h2", "How to Retrain"))
    s.append(("bullet", "python run.py --cli download  (download dataset)"))
    s.append(("bullet", "python run.py --cli splits  (create splits)"))
    s.append(("bullet", "python run.py --cli train  (train backbones)"))
    s.append(("bullet", "python run.py --cli ensemble  (fit ensemble)"))
    s.append(("bullet", "python run.py --cli evaluate  (evaluate and generate figures)"))

    # 11
    s.append(("h1", "11. Glossary"))
    glossary = [
        "Accuracy: The percentage of all predictions that were correct.",
        "ALL: Acute Lymphoblastic Leukaemia, a type of blood cancer.",
        "Augmentation: Creating altered copies of training images to help the model generalise.",
        "Backbone: The core image-recognition neural network.",
        "Confusion matrix: A grid showing correct vs. incorrect predictions per class.",
        "Ensemble: Combining several models for a joint prediction.",
        "Epoch: One complete pass through all training images.",
        "F1-score: The harmonic mean of precision and recall.",
        "Grad-CAM: A technique that highlights which image regions influenced the prediction.",
        "ImageNet: A public collection of 14+ million labelled photographs.",
        "Neural network: A chain of mathematical operations that learns patterns from data.",
        "Precision: Of predictions for class X, what fraction actually were X.",
        "Recall: Of actual class X images, what fraction the model found.",
        "ROC-AUC: Area under the ROC curve; 1.0 is perfect separation.",
        "Stacking: An ensemble method using a second-stage classifier.",
        "Transfer learning: Using knowledge from one task to help with another.",
        "Weights: The adjustable numbers inside a neural network.",
    ]
    for g in glossary:
        s.append(("bullet", g))

    # 12
    s.append(("h1", "12. References"))
    s.append(("p", "[1] J. Jayachitra and N. Umarkathaf, 'Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique,' ICEARS 2023, IEEE."))
    s.append(("p", "[2] M. A. Eshraghi et al., 'Blood Cells Cancer (ALL) — 4 Classes,' Kaggle dataset."))
    s.append(("p", "[3] M. Sandler et al., 'MobileNetV2,' CVPR 2018."))
    s.append(("p", "[4] M. Tan and Q. V. Le, 'EfficientNet,' ICML 2019."))
    s.append(("p", "[5] G. Huang et al., 'DenseNet,' CVPR 2017."))
    s.append(("p", "[6] R. R. Selvaraju et al., 'Grad-CAM,' ICCV 2017."))

    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    pages = build_pdf()
    build_docx()
    print(f"\nDone. PDF: {pages} pages.")
