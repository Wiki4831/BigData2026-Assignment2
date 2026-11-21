"""Draft of the Q4 framework figure as an editable .pptx (python-pptx). Usage: make_framework_pptx.py OUT.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

GRAY, BLUE, ORANGE = RGBColor(0xE7, 0xE9, 0xED), RGBColor(0xC9, 0xDD, 0xF5), RGBColor(0xFB, 0xDD, 0xB8)
LINE, TEXT = RGBColor(0x55, 0x5B, 0x66), RGBColor(0x1F, 0x23, 0x2B)
FONT = "Noto Sans CJK TC"

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

label(0.4, 0.15, 12.5, 0.5, "關係型深度學習（RDL）預測管線：資料庫 → 時間子圖 → 使用者向量 → 未來 7 天活動", 20, True)

W, H = 2.15, 1.25
# Row 1: data
y1 = 1.0
label(0.4, y1 - 0.05, 12, 0.3, "① 資料：把資料庫變成圖", 12, True)
r1 = [box(0.5, y1 + 0.3, W, H, ["GH Archive 每小時檔", "官方 preprocess.py"], GRAY),
      box(3.6, y1 + 0.3, 3.2, H, ["8 張表", "Users、Repositories、PullRequests、Issues、IssueComments、PRReviews、PRReviewComments、Pushes"], GRAY, 13),
      box(7.9, y1 + 0.3, 4.9, H, ["一張大圖（每列＝節點、外鍵＝邊）", "19 條外鍵＋反向＝38 種邊；每列帶時間戳"], GRAY)]
arrow(r1[0], r1[1]); arrow(r1[1], r1[2])
# Row 2: encoder
y2 = 3.2
label(0.4, y2 - 0.05, 12, 0.3, "② 每個（使用者, 切點 t）：即時取樣時間子圖並編碼", 12, True)
r2 = [box(0.5, y2 + 0.3, W, H, ["時間取樣", "只取 t 之前的列；3 hop：128／64／32"], GRAY),
      box(3.3, y2 + 0.3, 2.9, H, ["欄位編碼", "類別、數字、時間、GloVe 文字；加相對時間（t－列時間）"], BLUE),
      box(6.9, y2 + 0.3, 2.9, H, ["GraphSAGE × 3 層", "每種邊各自轉換、鄰居求和"], BLUE),
      box(10.5, y2 + 0.3, 2.3, H, ["使用者向量", "投影到 64 維"], BLUE)]
for a, b in zip(r2, r2[1:]): arrow(a, b)
arrow(r1[2], r2[0], elbow=True)
label(3.3, y2 + 1.6, 6.5, 0.5, "預訓練：用使用者向量預測「切點後一天」做了哪些活動（BCE），之後整組凍結", 11)
# Row 3: forecast
y3 = 5.35
label(0.4, y3 - 0.05, 12, 0.3, "③ 預測未來 7 天，轉成 0／1", 12, True)
w3 = 2.2
r3 = [box(0.5, y3 + 0.3, w3, H, ["初始狀態", "共用向量＋個人向量＋近期統計特徵"], ORANGE),
      box(3.1, y3 + 0.3, w3, H, ["GRU 展開 7 天", "每天輸出 6 個活動機率"], ORANGE),
      box(5.7, y3 + 0.3, w3, H, ["週機率", "7 天取最大值"], ORANGE),
      box(8.3, y3 + 0.3, w3, H, ["每活動門檻", "在驗證週選，轉成 0／1"], ORANGE),
      box(10.9, y3 + 0.3, 1.9, H, ["輸出 CSV", "4,465 人 × 6 欄"], GRAY)]
for a, b in zip(r3, r3[1:]): arrow(a, b)
arrow(r2[3], r3[0], elbow=True)
# legend
for i, (c, t) in enumerate([(GRAY, "資料／固定"), (BLUE, "預訓練後凍結"), (ORANGE, "第二階段訓練")]):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.4 + i * 1.3 - 0.0), Inches(0.72), Inches(0.2), Inches(0.2))
    s.fill.solid(); s.fill.fore_color.rgb = c; s.line.color.rgb = LINE
    label(9.65 + i * 1.3, 0.62, 1.1, 0.35, t, 10)
prs.save(sys.argv[1])
