"""Generate the 2-minute bilingual speech script as a Word document."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\郑傅韬\Desktop\PHASOR_演讲稿_两分钟中英对照.docx"

doc = Document()

# Base font
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0x15, 0x65, 0xC0)
    return p


def subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)
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
title("PHASOR 演讲稿 · 两分钟版（中英对照）")
subtitle("SECTION 3-3  Adaptive Walking Gaits  |  AI+ Academic Conference 2026")

# ===== 1 =====
heading("1. 开场")
zh("我们讲的是自适应步态。传统机器人靠一个固定 0.6 秒的节拍器走路，节奏是死的，地面一变就容易踩错、摔倒。")
en("We're talking about adaptive walking gaits. Traditional robots walk on a fixed 0.6-second metronome — the rhythm never changes, so on uneven ground they misplace their steps and fall.")

# ===== 2 =====
heading("2. 方法（第一页 · Methodology）")
zh("我们的办法很简单：把这个固定时钟，换成一个会学习的时钟。左边是基线，固定相位信号送进 PPO，输出 29 个关节目标，但它没法适应地形。")
en("Our idea is simple: replace that fixed clock with a learned one. On the left is the baseline — a fixed phase signal feeds into PPO, which outputs 29 joint targets, but it cannot adapt to terrain.")

zh("右边是我们的 PHASOR：一个一维卷积网络，读入最近 10 帧、29 个关节的位置，输出单位圆上的相位，而且是用人体动捕数据预训练的。")
en("On the right is PHASOR: a 1D-CNN reads the last 10 frames of 29 joint positions and outputs a phase on the unit circle, pre-trained on human motion-capture data.")

zh("这个学习到的相位再送进 PPO，奖励也改成让编码器的相位去匹配足部接触。于是节律会随地形自适应，更稳、更高效。")
en("This learned phase then feeds PPO, and the reward now matches the encoder's phase to foot contact. The rhythm adapts to terrain — more stable and more efficient.")

# ===== 3 =====
heading("3. 结果（第二页 · Results）")
zh("那效果怎么样？看结果。训练上，它收敛更快更稳定，跌倒率大幅下降。")
en("So what are the results? In training, it converges faster and more stably, with a large drop in fall rate.")

zh("最关键的数字：崎岖地形跌倒率下降 61%，平坦地形下降 48%，角度跟踪提升 17%。")
en("The key numbers: fall rate down 61% on rough terrain, 48% on flat, and angular tracking up 17%.")

zh("而且模型只有 407KB，推理只要 0.1 毫秒，几乎没有额外成本。")
en("And the model is only 407KB, with 0.1ms inference — almost no extra cost.")

# ===== 4 =====
heading("4. 收尾")
zh("一句话总结：我们把固定 CPG 时钟换成了学习相位编码器，让 G1 在崎岖地形上跌倒率下降 61%，模型又小又快。谢谢。")
en("To summarize: we replaced the fixed CPG clock with a learned phase encoder, cutting fall rate by 61% on rough terrain, with a model that's small and fast. Thank you.")

doc.save(OUT)
print(f"Saved -> {OUT}")
