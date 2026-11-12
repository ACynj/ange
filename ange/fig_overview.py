#!/usr/bin/env python
"""Overview figure v2 — conservative layout: short lines, no inset axes,
all text width-audited to fit panels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/home/ynj/new-pro2/paper/figs"
plt.rcParams.update({"font.size": 7.0, "pdf.fonttype": 42,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
OI = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
      "purple": "#CC79A7", "grey": "#9AA5AD", "dark": "#37474F"}

fig, ax = plt.subplots(figsize=(7.16, 2.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")


def box(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))


def arrow(x1, y1, x2, y2, c):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=c,
                 lw=1.6, mutation_scale=11))


# ---- Panel A (x 0.5..30.5) : problem
box(0.5, 1.5, 30.0, 39, "#FAFAFA", OI["grey"])
ax.text(2.0, 38.6, "① THE PROBLEM", fontsize=8, weight="bold", color=OI["dark"])
ax.text(2.0, 35.8, "Positional encodings", fontsize=6.8, color=OI["dark"])
ax.text(2.0, 33.6, "$\\mathcal{C}_{pos}$: role = column position", fontsize=6.2, color=OI["grey"])
walls = [
    ("INFORMATION WALL", "lossy digest: divergence 0.00", "GPT-4o nbr F1 0.027", OI["red"]),
    ("EXPRESSIVITY WALL", "isomorphism: 4 families fail", "$C_6$ vs $2C_3$: solved by none", OI["orange"]),
    ("REALITY WALL", "synthetic 0.99 → real 0.10–0.34", "prompt ceiling, all vendors", OI["purple"]),
]
for i, (t, d1, d2, c) in enumerate(walls):
    y = 28.4 - i * 8.6
    box(1.8, y - 3.0, 27.4, 7.2, "white", c)
    ax.text(2.8, y + 2.8, t, fontsize=6.6, weight="bold", color=c)
    ax.text(2.8, y + 0.4, d1, fontsize=5.7, color=OI["dark"])
    ax.text(2.8, y - 1.9, d2, fontsize=5.7, color=OI["grey"])

# ---- Panel B (x 34..67) : construction
box(34.0, 1.5, 33.0, 39, "#FDF7F2", OI["red"])
ax.text(35.5, 38.6, "② THE CONSTRUCTION", fontsize=8, weight="bold", color=OI["red"])
ax.text(35.5, 35.8, "Pointer grammar $\\Gamma_{PG}$, proven native", fontsize=6.8, color=OI["dark"])
axioms = [("A1 lossless", "1-WL exact"), ("A2 local", "1 lookup"),
          ("A3 compose", "concat")]
for i, (t, d) in enumerate(axioms):
    x = 35.5 + i * 10.6
    box(x, 28.6, 10.0, 5.2, "white", OI["red"])
    ax.text(x + 0.7, 32.2, t, fontsize=5.4, weight="bold", color=OI["red"])
    ax.text(x + 0.8, 29.9, d, fontsize=5.4, color=OI["dark"])
ax.text(35.5, 26.0, "record v2 (1.3× edge list):", fontsize=6.0, color=OI["dark"])
ax.text(35.5, 23.2, "n3 -r1-> n17", family="monospace", fontsize=7.4, color=OI["red"],
        bbox=dict(boxstyle="round,pad=0.28", fc="#FDECEA", ec=OI["red"]))
ax.text(46.5, 23.2, "Q: n3 -r1-> ?", family="monospace", fontsize=7.4, color=OI["blue"],
        bbox=dict(boxstyle="round,pad=0.28", fc="#E3F2FD", ec=OI["blue"]))
results = [
    ("Thm. 1  no positional encoding is", "readable at bounded resources", OI["dark"]),
    ("Thm. 3  k attention layers ≡", "k rounds of message passing", OI["dark"]),
    ("Prop. 1  extrapolation penalty =", "f(length c, capacity κ); c≈1.3 ⇒ none", OI["dark"]),
]
for i, (l1, l2, c) in enumerate(results):
    y = 19.0 - i * 5.6
    ax.text(35.5, y, l1, fontsize=6.0, color=c)
    ax.text(35.5, y - 2.3, l2, fontsize=6.0, color=c)

# ---- Panel C (x 70.5..99.5) : validation
box(70.5, 1.5, 29.0, 39, "#F3F8FB", OI["blue"])
ax.text(72.0, 38.6, "③ VALIDATION", fontsize=8, weight="bold", color=OI["blue"])
ax.text(72.0, 35.8, "12 models · 4 families", fontsize=6.4, color=OI["dark"])
ax.text(72.0, 33.6, "frozen: pointer error flat 0.06–0.12", fontsize=5.8, color=OI["grey"])
ax.text(72.0, 31.7, "as n: 10→100; positional 0.18→0.55", fontsize=5.8, color=OI["grey"])
rows = [
    ("training repairs UTILIZATION", "nbr 0.21→0.63 (0.3% params)", "multi-hop 0.19→0.80 (7B)", OI["green"]),
    ("…never INFORMATION", "lossy arm hallucinates", "neighbors: F1 0.085", OI["red"]),
    ("reality wall is prompt-only", "in-domain LoRA: 0.17→0.52", "(5× chance; prompt ≤0.34)", OI["blue"]),
]
for i, (t, d1, d2, c) in enumerate(rows):
    y = 26.6 - i * 8.2
    box(71.8, y - 3.0, 26.4, 7.0, "white", c)
    ax.text(72.8, y + 2.7, t, fontsize=6.3, weight="bold", color=c)
    ax.text(72.8, y + 0.35, d1, fontsize=5.7, color=OI["dark"])
    ax.text(72.8, y - 1.9, d2, fontsize=5.7, color=OI["grey"])

arrow(30.9, 21, 33.6, 21, OI["dark"])
arrow(67.4, 21, 70.1, 21, OI["dark"])
fig.savefig(f"{OUT}/fig_overview.pdf")
print("overview v2 done")
