"""Draft of the Q4 framework figure as an editable .pptx (python-pptx). Usage: make_framework_pptx.py OUT.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

GRAY, BLUE, ORANGE = RGBColor(0xE7, 0xE9, 0xED), RGBColor(0xC9, 0xDD, 0xF5), RGBColor(0xFB, 0xDD, 0xB8)
LINE, TEXT = RGBColor(0x55, 0x5B, 0x66), RGBColor(0x1F, 0x23, 0x2B)
FONT = "DejaVu Sans"

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])

def text(shape, lines, size=12, bold_first=True):
    tf = shape.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run(); run.text = line
        run.font.size = Pt(size if i == 0 else size - 2); run.font.bold = bool(bold_first and i == 0)
        run.font.name = FONT; run.font.color.rgb = TEXT

def box(x, y, w, h, lines, color, size=13):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.color.rgb = LINE; s.line.width = Pt(1)
    s.shadow.inherit = False; text(s, lines, size); return s

def label(x, y, w, h, content, size=12, bold=False, align=PP_ALIGN.LEFT):
    t = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = t.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = content
    r.font.size = Pt(size); r.font.bold = bold; r.font.name = FONT; r.font.color.rgb = TEXT; return t

def arrow(a, b, elbow=False):
    kind = MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT
    c = slide.shapes.add_connector(kind, 0, 0, 0, 0)
    if elbow: c.begin_connect(a, 2); c.end_connect(b, 0)
    else: c.begin_connect(a, 3); c.end_connect(b, 1)
    c.line.color.rgb = LINE; c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln(); tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"}); ln.append(tail)

label(0.4, 0.15, 12.5, 0.5, "Relational deep learning pipeline: database → time-respecting subgraph → user vector → next 7 days", 20, True)

W, H = 2.15, 1.25
# Row 1: data
y1 = 1.0
label(0.4, y1 - 0.05, 12, 0.3, "1. Data: turn the database into a graph", 12, True)
r1 = [box(0.5, y1 + 0.3, W, H, ["GH Archive hourly files", "official preprocess.py"], GRAY),
      box(3.6, y1 + 0.3, 3.2, H, ["8 tables", "Users, Repositories, PullRequests, Issues, IssueComments, PRReviews, PRReviewComments, Pushes"], GRAY, 13),
      box(7.9, y1 + 0.3, 4.9, H, ["One large graph (row = node, foreign key = edge)", "19 FKs + reverse = 38 edge types; every row has a timestamp"], GRAY)]
arrow(r1[0], r1[1]); arrow(r1[1], r1[2])
# Row 2: encoder
y2 = 3.2
label(0.4, y2 - 0.05, 12, 0.3, "2. For every (user, cutoff t): sample a time-respecting subgraph and encode it", 12, True)
r2 = [box(0.5, y2 + 0.3, W, H, ["Temporal sampling", "only rows before t; 3 hops: 128 / 64 / 32"], GRAY),
      box(3.3, y2 + 0.3, 2.9, H, ["Column encoders", "categorical, numeric, time, GloVe text; + relative time (t − row time)"], BLUE),
      box(6.9, y2 + 0.3, 2.9, H, ["GraphSAGE × 3 layers", "one transform per edge type, sum of neighbors"], BLUE),
      box(10.5, y2 + 0.3, 2.3, H, ["User vector", "projected to 64 dims"], BLUE)]
for a, b in zip(r2, r2[1:]): arrow(a, b)
arrow(r1[2], r2[0], elbow=True)
label(3.3, y2 + 1.6, 6.5, 0.5, "Pretraining: predict the next day's activities from the user vector (BCE), then freeze the whole encoder", 11)
# Row 3: forecast
y3 = 5.35
label(0.4, y3 - 0.05, 12, 0.3, "3. Forecast the next 7 days and convert to 0/1", 12, True)
w3 = 2.2
r3 = [box(0.5, y3 + 0.3, w3, H, ["Initial state", "shared vector + user vector + recent statistics"], ORANGE),
      box(3.1, y3 + 0.3, w3, H, ["GRU unrolled 7 days", "6 activity probabilities per day"], ORANGE),
      box(5.7, y3 + 0.3, w3, H, ["Weekly probability", "max over the 7 days"], ORANGE),
      box(8.3, y3 + 0.3, w3, H, ["Per-activity threshold", "chosen on validation week → 0/1"], ORANGE),
      box(10.9, y3 + 0.3, 1.9, H, ["Output CSV", "4,465 users × 6 columns"], GRAY)]
for a, b in zip(r3, r3[1:]): arrow(a, b)
arrow(r2[3], r3[0], elbow=True)
# legend
for i, (c, t) in enumerate([(GRAY, "data / fixed"), (BLUE, "pretrained, frozen"), (ORANGE, "stage-2 trained")]):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.4 + i * 1.3 - 0.0), Inches(0.72), Inches(0.2), Inches(0.2))
    s.fill.solid(); s.fill.fore_color.rgb = c; s.line.color.rgb = LINE
    label(9.65 + i * 1.3, 0.62, 1.1, 0.35, t, 10)
prs.save(sys.argv[1])
