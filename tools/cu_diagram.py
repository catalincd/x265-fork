"""
Generate a diagram illustrating H.265 CU quad-tree partitioning.
Output: cu_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
DEPTH_COLORS = {
    0: "#4E79A7",   # 64×64 – deep blue
    1: "#F28E2B",   # 32×32 – orange
    2: "#59A14F",   # 16×16 – green
    3: "#E15759",   # 8×8   – red
}
DEPTH_ALPHA = {0: 0.18, 1: 0.22, 2: 0.28, 3: 0.35}
EDGE_COLOR  = "#333333"
BG_COLOR    = "#F8F8F8"
GRID_COLOR  = "#CCCCCC"

# ── figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
fig.suptitle("H.265 / HEVC  –  Coding Unit (CU) Quad-Tree Partitioning",
             fontsize=17, fontweight="bold", y=0.97, color="#222222")

# Three panels: frame overview | single CTU zoom | depth legend + explanation
ax_frame  = fig.add_axes([0.01, 0.08, 0.30, 0.82])
ax_ctu    = fig.add_axes([0.34, 0.08, 0.40, 0.82])
ax_legend = fig.add_axes([0.76, 0.08, 0.22, 0.82])

for ax in (ax_frame, ax_ctu, ax_legend):
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

# ════════════════════════════════════════════════════════════════════════════
# PANEL 1 – frame divided into CTUs (simplified 8×4 grid = 512×256 px shown)
# ════════════════════════════════════════════════════════════════════════════
FRAME_W, FRAME_H = 8, 5        # in CTU units
CTU_PX = 1.0                   # 1 unit = 1 CTU (64 px in real life)

ax_frame.set_xlim(-0.1, FRAME_W + 0.1)
ax_frame.set_ylim(-0.5, FRAME_H + 0.6)
ax_frame.set_title("Video Frame  →  CTU Grid\n(each cell = 64×64 Coding Tree Unit)",
                   fontsize=10, color="#444444", pad=6)

# draw faint image-like background
grad = np.linspace(0.75, 0.92, 100).reshape(1, -1)
ax_frame.imshow(np.tile(grad, (50, 1)), extent=[0, FRAME_W, 0, FRAME_H],
                cmap="Blues", vmin=0, vmax=1, aspect="auto", zorder=0)

# CTU grid
for r in range(FRAME_H):
    for c in range(FRAME_W):
        rect = patches.Rectangle((c, r), 1, 1,
                                  linewidth=0.8, edgecolor=GRID_COLOR,
                                  facecolor="none", zorder=2)
        ax_frame.add_patch(rect)

# highlight one CTU that we'll zoom into
HI_C, HI_R = 3, 2
hi = patches.Rectangle((HI_C, HI_R), 1, 1,
                        linewidth=2.5, edgecolor="#E15759",
                        facecolor="#E15759", alpha=0.25, zorder=3)
ax_frame.add_patch(hi)
ax_frame.text(HI_C + 0.5, HI_R + 0.5, "zoom →",
              ha="center", va="center", fontsize=7.5,
              color="#C0392B", fontweight="bold", zorder=4)

ax_frame.text(FRAME_W / 2, -0.3,
              f"1920×1080 frame  =  {30*17} CTUs",
              ha="center", va="top", fontsize=8.5, color="#555555")

# ════════════════════════════════════════════════════════════════════════════
# PANEL 2 – CTU quad-tree (one example partition)
# ════════════════════════════════════════════════════════════════════════════
S = 8.0   # CTU side length in axis units (represents 64 px)

ax_ctu.set_xlim(-0.3, S + 0.3)
ax_ctu.set_ylim(-0.7, S + 0.9)
ax_ctu.set_title("Single CTU  –  Quad-Tree Split Example\n"
                 "(64×64 → 32×32 → 16×16 → 8×8)",
                 fontsize=10, color="#444444", pad=6)

def draw_cu(ax, x, y, size, depth, label=None):
    """Draw one CU rectangle."""
    c = DEPTH_COLORS[depth]
    a = DEPTH_ALPHA[depth]
    rect = patches.Rectangle((x, y), size, size,
                              linewidth=1.4 if depth > 0 else 2.2,
                              edgecolor=c, facecolor=c, alpha=a, zorder=depth + 1)
    ax.add_patch(rect)
    if label:
        fs = max(5.5, 8.5 - depth * 1.2)
        ax.text(x + size / 2, y + size / 2, label,
                ha="center", va="center", fontsize=fs,
                color=DEPTH_COLORS[depth], fontweight="bold",
                zorder=depth + 5, alpha=0.9)

# depth-0: entire CTU (64×64) – draw as border only
draw_cu(ax_ctu, 0, 0, S, 0, "64×64\n(depth 0)")

# top-left quadrant split → depth-1 (32×32) – kept as-is
draw_cu(ax_ctu, 0, S/2, S/2, 1, "32×32\n(depth 1)")

# top-right depth-1 further split into four 16×16
half = S / 2
qtr  = S / 4
for (qx, qy) in [(half, half), (half + qtr, half),
                 (half, half + qtr), (half + qtr, half + qtr)]:
    draw_cu(ax_ctu, qx, qy, qtr, 2)
# label one of them
ax_ctu.text(half + qtr / 2, half + qtr / 2, "16×16\n(depth 2)",
            ha="center", va="center", fontsize=6.5,
            color=DEPTH_COLORS[2], fontweight="bold", zorder=8)

# bottom-left depth-1 kept as 32×32
draw_cu(ax_ctu, 0, 0, S/2, 1, "32×32\n(depth 1)")

# bottom-right: split depth-1 → four 16×16, one of which splits to 8×8
draw_cu(ax_ctu, half, 0,      qtr, 2)   # top-left 16×16
draw_cu(ax_ctu, half, qtr,    qtr, 2)   # bottom-left 16×16
# top-right 16×16 split into four 8×8
eighth = S / 8
for (ex, ey) in [(half + qtr,        qtr),
                 (half + qtr + eighth, qtr),
                 (half + qtr,         qtr + eighth),
                 (half + qtr + eighth, qtr + eighth)]:
    draw_cu(ax_ctu, ex, ey, eighth, 3)
ax_ctu.text(half + qtr + eighth / 2, qtr + eighth / 2,
            "8×8\n(d3)", ha="center", va="center", fontsize=5.5,
            color=DEPTH_COLORS[3], fontweight="bold", zorder=10)
# bottom-right 16×16 kept
draw_cu(ax_ctu, half + qtr, 0, qtr, 2, "16×16\n(depth 2)")

# pixel-size annotation arrows
def dim_arrow(ax, x0, y0, x1, y1, label, color, vert=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if vert:
        ax.text(mx - 0.18, my, label, ha="right", va="center",
                fontsize=7, color=color)
    else:
        ax.text(mx, my - 0.18, label, ha="center", va="top",
                fontsize=7, color=color)

dim_arrow(ax_ctu, 0, -0.35, S, -0.35, "64 px", DEPTH_COLORS[0])
dim_arrow(ax_ctu, -0.25, 0, -0.25, S, "64 px", DEPTH_COLORS[0], vert=True)

# ════════════════════════════════════════════════════════════════════════════
# PANEL 3 – legend + textual explanation
# ════════════════════════════════════════════════════════════════════════════
ax_legend.set_xlim(0, 4)
ax_legend.set_ylim(0, 10)
ax_legend.set_title("Depth & Size Reference", fontsize=10,
                    color="#444444", pad=6)

depths = [
    (0, "64 × 64", "CTU root\n(always starts here)"),
    (1, "32 × 32", "1st split level"),
    (2, "16 × 16", "2nd split level"),
    (3, "8 × 8",   "3rd split level\n(smallest CU)"),
]

for i, (d, size_lbl, desc) in enumerate(depths):
    y = 8.8 - i * 2.1
    c = DEPTH_COLORS[d]
    sq_size = 0.55 + d * 0.0   # same visual size for legend
    rect = patches.FancyBboxPatch((0.1, y - 0.3), 0.7, 0.65,
                                  boxstyle="round,pad=0.04",
                                  linewidth=1.8, edgecolor=c,
                                  facecolor=c, alpha=0.35)
    ax_legend.add_patch(rect)
    ax_legend.text(0.45, y + 0.02, f"d{d}", ha="center", va="center",
                   fontsize=9, fontweight="bold", color=c)
    ax_legend.text(1.0, y + 0.18, size_lbl, va="center",
                   fontsize=9, fontweight="bold", color="#222222")
    ax_legend.text(1.0, y - 0.12, desc, va="center",
                   fontsize=7.5, color="#555555")

# explanation box
explanation = (
    "How the encoder decides:\n\n"
    "1.  Start at depth 0 (64×64 CTU).\n"
    "2.  Encode the block as-is and measure\n"
    "     RD cost (rate + distortion).\n"
    "3.  Split into 4 children, encode each,\n"
    "     sum their RD costs.\n"
    "4.  Keep whichever is cheaper.\n"
    "5.  Repeat recursively to depth 3.\n\n"
    "Skip heuristics (--rskip):\n"
    "  Stop early when splitting is unlikely\n"
    "  to help, saving RD evaluations.\n\n"
    "NN predictor (--rskip 3):\n"
    "  MLP trained on encoder ground-truth\n"
    "  decides whether to skip the recursion\n"
    "  based on 8 CU features."
)

ax_legend.text(0.1, 0.05, explanation, va="bottom",
               fontsize=7.8, color="#333333",
               fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#EAEAEA",
                         edgecolor="#BBBBBB", linewidth=1))

plt.savefig("cu_diagram.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
print("Saved cu_diagram.png")
