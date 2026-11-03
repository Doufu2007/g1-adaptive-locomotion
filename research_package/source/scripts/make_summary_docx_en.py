"""Generate an English summary of the two PPT decks as a Word document."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = r"C:\Users\郑傅韬\Desktop\PHASOR_Project_Summary_EN.docx"

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


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


def body(text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(11)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.size = Pt(11)
    return p


# ===== Header =====
title("Project Summary: Improved Reinforcement Learning Strategy for Robot Walking Training")
subtitle("CS1 Group 5  |  AI+ Academic Conference 2026  |  Presented 2026-08-13")

body("Team members: He Sun, Jinghao Zhang, Hanhong Li, Huanran Qing, Futao Zheng, Tianlan Chen")

# ===== 1 Background & Aims =====
heading("1. Background & Aims")
body("Humanoid robots require stable and reliable locomotion to operate safely in dynamic environments. Maintaining walking stability while responding to changes in motion demands and external disturbances remains an important challenge in AI-based robot control.")
body("This study adopts AI-based methods with the goal of enhancing humanoid walking stability and robustness, ensuring reliable locomotion under varying conditions. The overall pipeline is: robot state sensing -> AI policy network -> motion control (torque-level tracking) -> stable walking (closed-loop robust gait).")

# ===== 2 Research Questions =====
heading("2. Research Questions")
body("Main question: For robust locomotion, how can we separate sensor data, enhance anti-interference ability, and enable adaptive gaits?", bold=True)
body("Three research axes:")
bullet("(1) Separate Sensor Data - The baseline PPO policy uses a single MLP to jointly encode proprioceptive and exteroceptive observations, causing cross-modal interference during early feature extraction, leading to inefficient capacity utilization and limited generalization.")
bullet("(2) Anti-interference Ability - Robots on complex terrains face unpredictable push forces, sudden friction changes, and center-of-mass offsets; traditional policies are highly sensitive to these variations, resulting in high fall rates.")
bullet("(3) Adaptive Walking Gaits - Traditional gait uses a fixed 0.6s-period CPG clock to coordinate leg rhythms, like a rigid metronome; adequate on flat terrain but unable to adapt gait frequency to rough terrain.")

# ===== 3 Methods =====
heading("3. Methods (Three Technical Approaches)")

body("Approach 1 - Separate Sensor Data: Dual-Stream (Dual-Encoder)", bold=True)
bullet("Replaces the single MLP head with two independent encoders: a 480-dim proprioceptive stream for joint angles and body poses, and a 32-dim exteroceptive stream for environmental parameters.")
bullet("A compact fully-connected layer fuses the two streams without increasing the original parameter count (actor parameters reduced by ~30%).")

body("Approach 2 - Anti-interference Ability: Fault-Injection Training", bold=True)
bullet("Random joint-target dropout is applied during training, so the policy learns to recover robustly under partial joint failure or disturbance.")
bullet("Unlocks implicit adaptation: robust recovery without sacrificing baseline performance, with smoother actions (~2.2% lower RMS) and no loss in tracking accuracy.")

body("Approach 3 - Adaptive Walking Gaits: PHASOR Learned Phase Encoder", bold=True)
bullet("A 1D-CNN learns phase from recent joint positions (10 frames x 29 joints), replacing the fixed 0.6s CPG clock; pre-trained on human motion-capture data.")
bullet("The learned phase evolves with terrain and real-time gait, with rewards aligned to foot contact, enabling terrain-aware adaptive gait.")

# ===== 4 Key Results =====
heading("4. Key Results")
bullet("Separate Sensor Data: dual encoders eliminate cross-modal interference, giving fast, stable convergence and smoother actions, with ~30% fewer actor parameters.")
bullet("Adaptive Walking Gaits (PHASOR): rough-terrain fall rate down 61%, flat-terrain fall rate down 48%, mean reward up ~51%, angular tracking up 17%.")
bullet("Anti-interference Ability: fault-injection training enables implicit adaptation and robust recovery without losing baseline performance, with ~2.2% lower action RMS.")

# ===== 5 Conclusion =====
heading("5. Conclusion")
body("This work improves the PPO walking policy of the Unitree G1 humanoid along three dimensions - representation, gait timing, and fault tolerance - via a dual-stream encoder for perception, a learned phase encoder replacing the fixed clock, and robust training against joint disturbances. Together they deliver more stable, more efficient, and more fault-tolerant humanoid locomotion.")

# ===== 6 References =====
heading("6. References")
bullet("SCHWARKE C, MITTAL M, RUDIN N, HOELLER D, HUTTER M. RSL-RL: A Learning Library for Robotics Research. arXiv:2509.10771, 2025.")
bullet("STARKE S, MASON I, KOMURA T. DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds. ACM Transactions on Graphics, 2022, 41(4).")
bullet("LUO Z, XIAO E, LU P. FT-Net: Learning Failure Recovery and Fault-Tolerant Locomotion for Quadruped Robots. IEEE Robotics and Automation Letters, 2023, 8(12).")

doc.save(OUT)
print("Saved ->", OUT)
