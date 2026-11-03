"""Generate a polished poster visualization — bar chart + metric cards."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
metrics = [
    ("Fall Rate",              0.0163, 0.0085,  0.5578, 0.2156, True),
    ("Velocity Tracking",      0.6611, 0.6845,  0.2882, 0.2902, False),
    ("Angular Tracking",       0.5773, 0.6238,  0.3945, 0.4608, False),
    ("Velocity Error XY",      0.8317, 0.7927,  1.9435, 2.1140, True),
    ("Yaw Error",              1.3542, 1.2148,  1.8264, 1.6970, True),
    ("Posture",                0.6665, 0.6887,  0.5503, 0.6098, False),
]

def imp(b, p, lower):
    return (b - p) / abs(b) * 100 if lower else (p - b) / abs(b) * 100

flat_imps  = [imp(d[1], d[2], d[5]) for d in metrics]
rough_imps = [imp(d[3], d[4], d[5]) for d in metrics]
names = [d[0] for d in metrics]

# ── Colors ────────────────────────────────────────────────────────────
GREEN   = "#2e7d32"
LGREEN  = "#66bb6a"
RED     = "#c62828"
GRAY    = "#9e9e9e"
LGRAY   = "#bdbdbd"
BLUE    = "#1565c0"
DARK    = "#212121"

# ── Figure ────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(18, 8), facecolor="white",
                                  gridspec_kw={"wspace": 0.6})

for ax, title, imps_list, color in [
    (ax_f, "Flat Terrain",   flat_imps,  GREEN),
    (ax_r, "Rough Terrain",  rough_imps, BLUE),
]:
    bars = ax.barh(names, imps_list, height=0.55, color=[
        GREEN if v > 0 else RED for v in imps_list
    ], edgecolor="white", linewidth=0.8, zorder=3)

    # ── Value labels on bars ──────────────────────────────────────
    for bar, val in zip(bars, imps_list):
        sign = "+" if val > 0 else ""
        xpos = bar.get_width()
        offset = 2.5 if val >= 0 else -8.5
        ha = "left" if val >= 0 else "right"
        ax.text(xpos + offset, bar.get_y() + bar.get_height()/2,
                f"{sign}{val:.0f}%", va="center", ha=ha,
                fontsize=15, fontweight="bold",
                color=GREEN if val > 0 else RED)

    # ── Zero line ─────────────────────────────────────────────────
    ax.axvline(0, color="#ddd", linewidth=1.5, zorder=1)

    # ── Style ─────────────────────────────────────────────────────
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20, color=DARK)
    ax.set_xlabel("Improvement over Baseline (%)", fontsize=12, color="#555")
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlim(min(imps_list) - 15, max(imps_list) + 15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    ax.spines["bottom"].set_color("#ddd")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2, linestyle="--", zorder=0)

# ── Big-number cards at bottom ───────────────────────────────────────
card_y = -0.35
fig.text(0.15, card_y,
         "–48%\nFalls", ha="center", fontsize=22, fontweight="bold",
         color=GREEN, linespacing=1.3)
fig.text(0.15, card_y - 0.12,
         "vs baseline on flat", ha="center", fontsize=10, color="#888")

fig.text(0.35, card_y,
         "–61%\nFalls", ha="center", fontsize=22, fontweight="bold",
         color=BLUE, linespacing=1.3)
fig.text(0.35, card_y - 0.12,
         "vs baseline on rough", ha="center", fontsize=10, color="#888")

fig.text(0.55, card_y,
         "+17%\nAngular", ha="center", fontsize=22, fontweight="bold",
         color=GREEN, linespacing=1.3)
fig.text(0.55, card_y - 0.12,
         "turning accuracy on rough", ha="center", fontsize=10, color="#888")

fig.text(0.75, card_y,
         "407KB\nModel", ha="center", fontsize=22, fontweight="bold",
         color=DARK, linespacing=1.3)
fig.text(0.75, card_y - 0.12,
         "encoder weight — 0.1ms inference", ha="center", fontsize=10, color="#888")

fig.suptitle("PHASOR: Learned Phase Encoder vs Fixed CPG Clock",
             fontsize=24, fontweight="bold", y=1.02, color=DARK)
fig.text(0.5, 0.93, "Unitree G1 Humanoid — PPO Locomotion",
         ha="center", fontsize=14, color="#888")

# ── Save ──────────────────────────────────────────────────────────────
out = "results/final_report/poster_barchart.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved → {out}")
plt.close()
