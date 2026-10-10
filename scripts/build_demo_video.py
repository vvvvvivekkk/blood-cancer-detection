"""Build an explainer video from real-data screenshots and result figures.
Output: demo_walkthrough.mp4 (1920x1080, 30 fps) + narration.srt + narration.md
"""
import json, subprocess, textwrap, pathlib
from PIL import Image, ImageDraw, ImageFont, ImageFilter

REPO = pathlib.Path('/home/claude/blood-cancer-detection')
OUT = REPO / 'docs/video'
FR = OUT / 'frames'; FR.mkdir(exist_ok=True)
W, H = 1920, 1080
NAVY, TEAL, LIGHT, INK, MUTED = (20, 33, 61), (14, 116, 144), (246, 248, 250), (20, 33, 61), (106, 113, 121)
F_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F_REG = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F_SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'

m = json.load(open(REPO / 'results/metrics.json'))
st = m['models']['stacking']; dn = m['models']['densenet121']; mb = m['models']['mobilenetv2']; ef = m['models']['efficientnetb0']; vo = m['models']['voting']
pct = lambda x: f'{x*100:.2f} %'

def font(path, size): return ImageFont.truetype(path, size)

def card(title, lines, step=None, dark=True):
    img = Image.new('RGB', (W, H), NAVY if dark else LIGHT)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 24, H], fill=TEAL)
    fg = (246, 248, 250) if dark else INK
    sub = (191, 216, 230) if dark else (74, 85, 104)
    y = 150
    if step:
        d.text((128, y), step.upper(), font=font(F_BOLD, 30), fill=(125, 211, 232) if dark else TEAL); y += 70
    for ln in textwrap.wrap(title, 34):
        d.text((128, y), ln, font=font(F_SERIF, 84), fill=fg); y += 100
    y += 30
    for ln in lines:
        for sl in textwrap.wrap(ln, 70):
            d.text((128, y), sl, font=font(F_REG, 40), fill=sub); y += 58
        y += 14
    return img

def shot(path, caption, step, crop=None, focus=None):
    """Screenshot/figure fitted into a 1920x1080 frame with caption bar."""
    img = Image.new('RGB', (W, H), LIGHT)
    d = ImageDraw.Draw(img)
    src = Image.open(path).convert('RGB')
    if crop: src = src.crop(crop)
    box_w, box_h = W - 2 * 96, H - 96 - 250
    s = min(box_w / src.width, box_h / src.height)
    src = src.resize((int(src.width * s), int(src.height * s)), Image.LANCZOS)
    x = (W - src.width) // 2; y = 72
    shadow = Image.new('RGB', (src.width + 40, src.height + 40), LIGHT)
    ImageDraw.Draw(shadow).rounded_rectangle([20, 20, src.width + 20, src.height + 20], 18, fill=(205, 212, 220))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    img.paste(shadow, (x - 20, y - 12))
    img.paste(src, (x, y))
    d.rectangle([x - 1, y - 1, x + src.width, y + src.height], outline=(217, 225, 232), width=2)
    if focus:  # highlight box in source coords (after crop), scaled
        fx0, fy0, fx1, fy1 = focus
        d.rounded_rectangle([x + fx0 * s, y + fy0 * s, x + fx1 * s, y + fy1 * s], 10, outline=(194, 65, 12), width=5)
    # caption bar
    d.rectangle([0, H - 190, W, H], fill=NAVY)
    d.rectangle([0, H - 190, 24, H], fill=TEAL)
    d.text((96, H - 160), step.upper(), font=font(F_BOLD, 26), fill=(125, 211, 232))
    yy = H - 120
    for ln in textwrap.wrap(caption, 95)[:2]:
        d.text((96, yy), ln, font=font(F_REG, 38), fill=(246, 248, 250)); yy += 50
    return img

A = REPO / 'docs/audit'; FG = REPO / 'results/figures'
scenes = [  # (image, duration, narration)
 (card('Blood Cancer Identification using Hybrid Ensemble Deep Learning',
       ['Automatic screening of Acute Lymphoblastic Leukemia subtypes from peripheral blood smear images',
        'G. Revathi · G. Chandana · G. Prashanth  —  Guide: A. Divya Reddy',
        'Dept. of Artificial Intelligence & ML, Mallareddy University'], 'Major project'),
  6, 'This is a hybrid ensemble deep learning system that identifies blood cancer subtypes from microscope images of blood smears.'),
 (card('The problem', ['Acute Lymphoblastic Leukemia (ALL) is the most common childhood cancer.',
       'Diagnosis starts with a blood smear under a microscope — slow, and the three malignant stages look alike.',
       'Goal: classify an image as Benign, early Pre-B, Pre-B or Pro-B automatically, and show why.'], 'Step 1 of 6 · Why'),
  9, 'Acute Lymphoblastic Leukemia is the most common childhood cancer. Diagnosis starts with a blood smear under a microscope, which is slow, and the three malignant stages look very similar. Our goal is to classify an image into Benign, early Pre-B, Pre-B or Pro-B automatically, and to show why.'),
 (shot(REPO / 'docs/architecture.png', 'One image goes to three fine-tuned CNNs — MobileNetV2, EfficientNetB0, DenseNet121. Their twelve probabilities are combined by weighted voting or a stacked meta-classifier.', 'Step 2 of 6 · How it works'),
  9, 'Every image is sent to three separately fine-tuned convolutional networks: MobileNetV2, EfficientNetB0 and DenseNet121. Each outputs four probabilities. The twelve numbers are combined in two ways, weighted soft voting and a stacked logistic-regression meta-classifier. That pair is the hybrid ensemble.'),
 (shot(A / 'preview_Benign.png', 'Preview page: choose a smear image and press Submit. The ensemble returns the class and a confidence — here Benign at 97 %.', 'Step 3 of 6 · Preview', focus=(760, 90, 1410, 230)),
  8, 'On the Preview page the user chooses a blood smear image and presses Submit. Within about two seconds the ensemble returns the predicted class and a confidence. This image is Benign at 97 percent.'),
 (shot(A / 'preview_early_PreB.png', 'Probability bars show how the 100 % is shared across the four classes, so a close call is visible at a glance.', 'Step 3 of 6 · Preview', focus=(760, 240, 1410, 430)),
  7, 'The probability bars show how the confidence is shared across all four classes, so a close call is visible at a glance. This image is an early Pre-B malignant cell.'),
 (shot(A / 'preview_PreB.png', 'Each backbone\'s own vote is listed underneath — if the three disagree, the result is flagged for manual review.', 'Step 3 of 6 · Preview', focus=(760, 920, 1410, 1100)),
  7, 'Underneath, each backbone\'s own vote and confidence are listed. If the three networks disagree, the app flags the result for manual review.'),
 (shot(A / 'preview_ProB.png', 'Grad-CAM heatmap: red marks the regions that drove the decision — the large nucleated blast cells, not the background.', 'Step 4 of 6 · Explainability', focus=(770, 440, 1150, 820)),
  8, 'The Grad-CAM heatmap explains the decision. Red marks the regions that most increased the predicted class score: the large nucleated blast cells, not the background.'),
 (shot(FG / 'gradcam_examples.png', 'Top row: correct predictions, one per class. Bottom row: the misclassified cases — the heat is diffuse, a warning sign a clinician can see.', 'Step 4 of 6 · Explainability'),
  7, 'Across the test set, correct predictions show focused heat on the cell; the few misclassified cases show diffuse heat, which is itself a useful warning sign.'),
 (shot(FG / 'model_comparison_bar.png', f'On 487 unseen test images: MobileNetV2 {pct(mb["accuracy"])}, EfficientNetB0 {pct(ef["accuracy"])}, DenseNet121 {pct(dn["accuracy"])}, voting {pct(vo["accuracy"])}, stacking {pct(st["accuracy"])}.', 'Step 5 of 6 · Results'),
  9, f'On 487 unseen test images, MobileNetV2 reached {pct(mb["accuracy"])}, EfficientNetB0 {pct(ef["accuracy"])} and DenseNet121 {pct(dn["accuracy"])}. The stacked ensemble reached {pct(st["accuracy"])} accuracy with a macro F1 of {st["macro_f1"]:.3f}.'),
 (shot(FG / 'cm_stacking.png', 'Confusion matrix of the selected ensemble: only five errors in 487 images, and no Benign image was ever called malignant.', 'Step 5 of 6 · Results'),
  7, 'The confusion matrix shows only five errors in 487 images. Every class is recalled above 98 percent, and no benign image was ever called malignant.'),
 (shot(FG / 'roc_ensemble.png', f'One-vs-rest ROC curves: area under curve {st["roc_auc_ovr"]:.4f} — the ensemble separates all four classes almost perfectly.', 'Step 5 of 6 · Results'),
  6, f'The ROC curves confirm it: an area under the curve of {st["roc_auc_ovr"]:.3f}, meaning the ensemble separates all four classes almost perfectly.'),
 (card('Summary', [f'Stacked hybrid ensemble: {pct(st["accuracy"])} accuracy, {st["macro_f1"]:.3f} macro-F1 on held-out test data.',
       'Grad-CAM explanations and per-backbone votes on every prediction.',
       'Complete web application: Login → Preview → Performance Analysis → Chart.',
       'Research / educational project — not for clinical use.'], 'Step 6 of 6 · Summary'),
  9, f'In summary: a stacked hybrid ensemble reaching {pct(st["accuracy"])} accuracy on held-out data, with explanations on every prediction, delivered as a complete web application. This is a research project and is not for clinical use. Thank you.'),
]

# write frames + per-scene clips with gentle zoom
clips = []
for i, (img, dur, _) in enumerate(scenes):
    p = FR / f'scene{i:02d}.png'; img.save(p)
    c = FR / f'scene{i:02d}.mp4'
    z = f"zoompan=z='min(1.0+0.0008*in,1.06)':d={dur*30}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps=30"
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-loop', '1', '-i', str(p), '-vf', z, '-t', str(dur),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', str(c)], check=True)
    clips.append((c, dur))

# concat with crossfades
inputs = []; fc = []; offset = 0; prev = '[0:v]'
for i, (c, dur) in enumerate(clips):
    inputs += ['-i', str(c)]
for i in range(1, len(clips)):
    offset += clips[i-1][1] - 0.6
    out = f'[v{i}]'
    fc.append(f'{prev}[{i}:v]xfade=transition=fade:duration=0.6:offset={offset:.2f}{out}')
    prev = out
final = OUT / 'demo_walkthrough.mp4'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *inputs, '-filter_complex', ';'.join(fc), '-map', prev,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '20', '-movflags', '+faststart', str(final)], check=True)

# narration script + srt
t = 0.0; srt = []; md = ['# Narration script — demo_walkthrough.mp4\n']
def ts(x): h=int(x//3600); m_=int(x%3600//60); s=x%60; return f'{h:02d}:{m_:02d}:{int(s):02d},{int((s%1)*1000):03d}'
for i, (_, dur, text) in enumerate(scenes):
    start = max(t - (0.6 if i else 0), 0); end = start + dur - (0.3 if i else 0)
    srt.append(f'{i+1}\n{ts(start)} --> {ts(end)}\n' + '\n'.join(textwrap.wrap(text, 80)) + '\n')
    md.append(f'**[{ts(start)[:8]}] Scene {i+1}** — {text}\n')
    t = start + dur
(OUT / 'narration.srt').write_text('\n'.join(srt))
(OUT / 'narration.md').write_text('\n'.join(md))
print('done', final, f'{t:.1f}s')
