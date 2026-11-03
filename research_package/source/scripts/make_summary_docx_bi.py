"""Generate a bilingual (Chinese-English) summary as a Word document."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\郑傅韬\Desktop\PHASOR_项目摘要_中英对照.docx"

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
    p.paragraph_format.space_after = Pt(6)
    return p


def bullet_zh(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.size = Pt(11)
    return p


def bullet_en(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    r.italic = True
    p.paragraph_format.space_after = Pt(6)
    return p


# ===== Header =====
title("项目摘要：改进的强化学习机器人行走训练策略")
subtitle("Project Summary: Improved Reinforcement Learning Strategy for Robot Walking Training")
subtitle("CS1 Group 5  |  AI+ Academic Conference 2026  |  2026-08-13")
zh("小组成员：孙赫、张景皓、李涵鸿、秦涣然、郑傅韬、陈天澜")
en("Team members: He Sun, Jinghao Zhang, Hanhong Li, Huanran Qing, Futao Zheng, Tianlan Chen")

# ===== 1 =====
heading("一、研究背景与目标  |  1. Background & Aims")
zh("人形机器人要在动态环境中安全运行，必须具备稳定可靠的行走能力。在运动需求变化和外部扰动下保持行走稳定，是 AI 机器人控制领域的重要挑战。")
en("Humanoid robots require stable and reliable locomotion to operate safely in dynamic environments. Maintaining walking stability while responding to changes in motion demands and external disturbances remains an important challenge in AI-based robot control.")
zh("本研究采用 AI 方法，目标是提升人形机器人的行走稳定性与鲁棒性，确保在不同条件下可靠运动。整体技术路线为：机器人状态感知 → AI 策略网络 → 运动控制（力矩级跟踪）→ 稳定行走（闭环鲁棒步态）。")
en("This study adopts AI-based methods with the goal of enhancing humanoid walking stability and robustness. The overall pipeline is: robot state sensing -> AI policy network -> motion control (torque-level tracking) -> stable walking (closed-loop robust gait).")

# ===== 2 =====
heading("二、核心研究问题  |  2. Research Questions")
zh("总问题：在鲁棒行走中，如何分离传感数据、增强抗干扰能力、实现自适应步态？")
en("Main question: For robust locomotion, how can we separate sensor data, enhance anti-interference ability, and enable adaptive gaits?")
bullet_zh("① 分离传感数据：基线 PPO 策略用单一 MLP 联合编码本体感觉与外部感知观测，早期特征提取时产生跨模态干扰，导致容量利用效率低、泛化能力受限。")
bullet_en("(1) Separate Sensor Data - The baseline PPO policy uses a single MLP to jointly encode proprioceptive and exteroceptive observations, causing cross-modal interference and limited generalization.")
bullet_zh("② 抗干扰能力：复杂地形上的机器人常遇到不可预测的推力、摩擦突变、质心偏移，传统策略对这些动态变化高度敏感，跌倒率高。")
bullet_en("(2) Anti-interference Ability - Robots face unpredictable push forces, friction changes, and center-of-mass offsets; traditional policies are highly sensitive, resulting in high fall rates.")
bullet_zh("③ 自适应步态：传统步态用固定 0.6 秒周期的 CPG 时钟协调双腿节律，像僵硬节拍器，平地上尚可、崎岖地形无法自适应步频。")
bullet_en("(3) Adaptive Walking Gaits - Traditional gait uses a fixed 0.6s CPG clock, like a rigid metronome; adequate on flat terrain but unable to adapt to rough terrain.")

# ===== 3 =====
heading("三、方法（三条技术路线）  |  3. Methods")

zh("路线 1｜分离传感数据 —— 双流编码器")
en("Approach 1 - Separate Sensor Data: Dual-Stream (Dual-Encoder)")
bullet_zh("将单一 MLP 头替换为两个独立编码器：本体感觉流（480 维）处理关节角度与身体姿态；外部感知流（32 维）处理环境参数。")
bullet_en("Replaces the single MLP head with two encoders: a 480-dim proprioceptive stream and a 32-dim exteroceptive stream.")
bullet_zh("用一个小型全连接层融合两路特征，不增加原始参数量（演员网络参数减少约 30%）。")
bullet_en("A compact FC layer fuses the streams without increasing parameter count (actor parameters reduced ~30%).")

zh("路线 2｜抗干扰能力 —— 故障注入训练")
en("Approach 2 - Anti-interference Ability: Fault-Injection Training")
bullet_zh("训练时随机对关节目标施加 dropout，让策略学会在部分关节失效或扰动下仍能稳健恢复。")
bullet_en("Random joint-target dropout is applied during training, so the policy learns robust recovery under partial joint failure.")
bullet_zh("隐式适应，恢复稳健且不损失基线性能，动作更平滑（均方根值下降约 2.2%）而不牺牲跟踪精度。")
bullet_en("Unlocks implicit adaptation: robust recovery without losing baseline performance, with ~2.2% lower action RMS.")

zh("路线 3｜自适应步态 —— 学习相位编码器（PHASOR）")
en("Approach 3 - Adaptive Walking Gaits: PHASOR Learned Phase Encoder")
bullet_zh("用一维卷积网络（1D-CNN）从近期关节位置（10 帧 × 29 关节）学习相位，替换固定 0.6 秒 CPG 时钟；基于人体动捕数据预训练。")
bullet_en("A 1D-CNN learns phase from recent joint positions (10 frames x 29 joints), replacing the fixed CPG clock; pre-trained on human mocap data.")
bullet_zh("学习相位随地形与实时步态动态变化，奖励对齐足部接触，实现地形感知的自适应步态。")
bullet_en("The learned phase evolves with terrain and real-time gait, with rewards aligned to foot contact, enabling adaptive gait.")

# ===== 4 =====
heading("四、关键结果  |  4. Key Results")
bullet_zh("分离传感数据：双编码器消除跨模态干扰，收敛快速稳定、动作更平滑，演员参数减少约 30%。")
bullet_en("Separate Sensor Data: dual encoders eliminate interference, giving stable convergence and smoother actions, with ~30% fewer parameters.")
bullet_zh("自适应步态（PHASOR）：崎岖地形跌倒率下降 61%，平坦地形下降 48%，平均奖励提升约 51%，角度跟踪提升 17%。")
bullet_en("PHASOR: rough-terrain fall rate down 61%, flat-terrain down 48%, mean reward up ~51%, angular tracking up 17%.")
bullet_zh("抗干扰能力：故障注入训练带来隐式自适应，稳健恢复且不损失基线性能，动作均方根值下降约 2.2%。")
bullet_en("Anti-interference: fault-injection training enables robust recovery without losing baseline performance, with ~2.2% lower action RMS.")

# ===== 5 =====
heading("五、结论  |  5. Conclusion")
zh("本研究围绕表征、步态时序与容错三个维度改进 Unitree G1 的 PPO 行走策略：双流编码器分离感知、学习相位替代固定时钟、鲁棒训练应对关节扰动。三者共同实现了更稳定、更高效、更容错的人形机器人行走。")
en("This work improves the PPO walking policy of the Unitree G1 humanoid along three dimensions - representation, gait timing, and fault tolerance - delivering more stable, more efficient, and more fault-tolerant humanoid locomotion.")

# ===== 6 =====
heading("六、参考文献  |  6. References")
bullet_zh("SCHWARKE C, et al. RSL-RL: A Learning Library for Robotics Research. arXiv:2509.10771, 2025.")
bullet_en("SCHWARKE C, et al. RSL-RL: A Learning Library for Robotics Research. arXiv:2509.10771, 2025.")
bullet_zh("STARKE S, MASON I, KOMURA T. DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds. ACM TOG, 2022, 41(4).")
bullet_en("STARKE S, MASON I, KOMURA T. DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds. ACM TOG, 2022, 41(4).")
bullet_zh("LUO Z, XIAO E, LU P. FT-Net: Learning Failure Recovery and Fault-Tolerant Locomotion for Quadruped Robots. IEEE RA-L, 2023, 8(12).")
bullet_en("LUO Z, XIAO E, LU P. FT-Net: Learning Failure Recovery and Fault-Tolerant Locomotion for Quadruped Robots. IEEE RA-L, 2023, 8(12).")

doc.save(OUT)
print("Saved ->", OUT)
