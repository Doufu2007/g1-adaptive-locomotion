"""Extract all text content from two PPTX files for summarization."""
import io
from pptx import Presentation

FILES = [
    r"C:\Users\郑傅韬\Desktop\pre v6.0.pptx",
    r"C:\Users\郑傅韬\Desktop\学术研究海报v6.pptx",
]

OUT = r"C:\Users\郑傅韬\Desktop\大学生活\剑桥暑期项目\unitree_rl_mjlab\unitree_rl_mjlab\scripts\_pptx_content.txt"

buf = io.StringIO()

def shape_text(shape):
    txts = []
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            line = "".join(run.text for run in para.runs)
            if line.strip():
                txts.append(line)
    if shape.has_table:
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells]
            txts.append(" | ".join(cells))
    return txts

def collect(s, lines):
    if s.shape_type == 6:  # group
        for ss in s.shapes:
            collect(ss, lines)
    else:
        for t in shape_text(s):
            lines.append(t)

for f in FILES:
    buf.write("=" * 80 + "\n")
    buf.write("FILE: " + f + "\n")
    buf.write("=" * 80 + "\n")
    prs = Presentation(f)
    for i, slide in enumerate(prs.slides, 1):
        buf.write(f"\n----- SLIDE {i} -----\n")
        for shape in slide.shapes:
            lines = []
            collect(shape, lines)
            for ln in lines:
                buf.write("  " + ln + "\n")
    buf.write("\n")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(buf.getvalue())

print("Done ->", OUT)
