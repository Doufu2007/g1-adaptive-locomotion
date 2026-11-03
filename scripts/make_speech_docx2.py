"""Generate the trimmed bilingual speech script as a Word document."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\郑傅韬\Desktop\PHASOR_演讲稿_中英对照.docx"

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor(0x15, 0x65, 0xC0)
    return p


def subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    return p


def heading(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0x2E, 0x7D, 0x32)
    p.paragraph_format.space_before = Pt(10)
    return p


def zh(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(11)
    return p


def en(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    r.italic = True
    p.paragraph_format.space_after = Pt(8)
    return p


# ===== Header =====
title("SECTION 3-3  Adaptive Walking Gaits")
subtitle("AI+ Academic Conference 2026")
subtitle("精简版 · 约 175 词 · 正常语速约 1 分 25 秒（2 分钟内）")

# ===== 1 =====
heading("1. 开场 Opening")
zh("今天介绍自适应步态。传统机器人用固定 0.6 秒的 CPG 时钟走路，节律僵硬，一到崎岖地面就容易摔。")
en("Today I'll present our work on adaptive walking gaits. Conventional robots walk on a fixed 0.6-second CPG clock — the rhythm is rigid, so they fall easily on uneven ground.")

# ===== 2 =====
heading("2. 方法 Methodology")
zh("思路很简单：用一个可学习的相位编码器，替换这个固定时钟。")
en("The idea is simple: replace this fixed clock with a learnable phase encoder.")

zh("左边是基线：固定正余弦相位进 PPO，输出 29 个关节目标，但无法适配地形。")
en("On the left is the baseline — fixed sine/cosine phase feeds PPO, 29 joint targets, but no terrain adaptation.")

zh("右边是 PHASOR：一个 1D-CNN，读最近 10 帧、29 个关节位置，输出单位圆上的相位，用人体动捕数据预训练。")
en("On the right is PHASOR: a 1D-CNN reads the last 10 frames of 29 joint positions and outputs a phase on the unit circle, pre-trained on human mocap.")

zh("学习相位进 PPO，奖励对齐足部接触，节律随地形自适应，又稳又高效。")
en("This learned phase feeds PPO, with reward aligned to foot contact — the rhythm adapts to terrain, more stable and efficient.")

# ===== 3 =====
heading("3. 结果 Results")
zh("结果：训练收敛更快更稳，跌倒率大幅下降。")
en("Results: faster, more stable convergence, with a large drop in fall rate.")

zh("崎岖地形跌倒率降 61%，平坦降 48%，角度跟踪升 17%。")
en("Fall rate down 61% on rough, 48% on flat; angular tracking up 17%.")

zh("模型仅 407KB，推理 0.1 毫秒，几乎零额外开销。")
en("The model is only 407KB with 0.1ms inference — almost no extra cost.")

# ===== 4 =====
heading("4. 收尾 Conclusion")
zh("总结：我们用学习相位编码器替代固定 CPG 时钟，大幅提升崎岖地形鲁棒性，模型又小又快。谢谢。")
en("To sum up: we replace the fixed CPG clock with a learned phase encoder, greatly improving rough-terrain robustness with a small, fast model. Thank you.")

doc.save(OUT)
print("Saved ->", OUT)
