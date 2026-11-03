"""Generate vertical flowchart for poster."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 22), facecolor='white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')

BLUE = '#1565c0';   BLUE_BG = '#e3f2fd'
RED = '#c62828';    RED_BG = '#ffebee'
GREEN = '#2e7d32';  GREEN_BG = '#e8f5e9'
PURPLE = '#6a1b9a'; PURPLE_BG = '#f3e5f5'
DARK = '#1a1a1a'

def box(x, y, w, h, text, bg, border, fs=24, fc='#333'):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3',
                       facecolor=bg, edgecolor=border, linewidth=3)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color=fc, weight='bold')

def arrow(x1, y1, x2, y2, c='#555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=5))

# ===== TITLE =====
ax.text(7, 21.2, 'PHASOR: Learned Phase Encoder for Humanoid Locomotion',
        ha='center', fontsize=30, weight='bold', color=DARK)

# ===== Step 1 =====
box(1.5, 19.0, 11, 1.5, 'Human Mocap Data  >>  Phase Encoder', BLUE_BG, BLUE, 24, BLUE)
arrow(7, 19.0, 7, 17.8, BLUE)

# ===== Step 2 =====
box(1.5, 16.5, 11, 1.5, '1D-CNN Autoencoder  >>  Unit-Circle Phase', BLUE_BG, BLUE, 24, BLUE)
arrow(7, 16.5, 7, 15.2, '#555')

# ===== Step 3 =====
box(1.5, 14.0, 11, 1.5, 'RL Training — Unitree G1, PPO, 4096 Envs', PURPLE_BG, PURPLE, 24, PURPLE)

# Fork
arrow(3.5, 14.0, 3.5, 12.8, RED)
arrow(10.5, 14.0, 10.5, 12.8, GREEN)

# ===== Step 4 =====
box(0.5, 11.5, 6, 1.5, 'BASELINE\nFixed CPG Clock', RED_BG, RED, 24)
box(7.5, 11.5, 6, 1.5, 'PHASOR (ours)\nLearned Encoder Phase', GREEN_BG, GREEN, 24)

# Merge
arrow(3.5, 11.5, 7, 10.2, RED)
arrow(10.5, 11.5, 7, 10.2, GREEN)

# ===== Step 5 =====
box(1.5, 9.0, 11, 1.5, 'Evaluation — Flat & Rough Procedural Terrain', '#f5f5f5', '#bbb', 24)
arrow(7, 9.0, 7, 7.8, '#555')

# ===== Bottom =====
box(0.8, 6.5, 12.4, 1.5,
    'Learned phase adapts gait timing to terrain,\nreducing falls without sacrificing speed.',
    GREEN_BG, GREEN, 22, DARK)

fig.tight_layout(pad=1)
out = 'results/final_report/project_structure.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved -> {out}')
plt.close()
