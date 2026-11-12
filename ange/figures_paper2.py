#!/usr/bin/env python
"""Paper 2 figure suite: overview map, 2-PG method, boundary lift,
frontier chains, retrieval+transfer. Unified Okabe-Ito style."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/home/ynj/new-pro2/paper2/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
OI = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
      "sky": "#56B4E9", "purple": "#CC79A7", "grey": "#9AA5AD", "dark": "#37474F"}
RED = OI["red"]; GREY = "#9AA5AD"; DARK = "#37474F"; BLUE = OI["blue"]


def box(ax, x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))


def arrow(ax, x1, y1, x2, y2, c):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=c,
                 lw=1.6, mutation_scale=11))


# ================= Fig 1: overview map (double column) =================
def fig_overview():
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
    # Panel A: barrier
    box(ax, 0.5, 1.5, 30.0, 39, "#FAFAFA", GREY)
    ax.text(2.6, 38.6, "① FIRST-ORDER BARRIER", fontsize=7.6, weight="bold", color=DARK)
    ax.text(2.0, 35.8, "12 models · 4 families · 8 encodings", fontsize=6.4, color=DARK)
    ax.text(2.0, 33.4, "local queries: 0.94–1.00", fontsize=5.8, color=GREY)
    rows = [("TIER 1 · true isomorphs", "DS-V3 recall ≤ 0.20 (NO-bias)", OI["orange"]),
            ("TIER 2 · 1-WL separable", "GPT/GLM accuracy ≤ 0.60 (YES-bias)", OI["purple"]),
            ("TIER 3 · C₆ vs 2C₃", "every model, every encoding: ≤ 0.10", OI["red"])]
    for i, (t, d, c) in enumerate(rows):
        y = 28.4 - i * 8.6
        box(ax, 1.8, y - 3.0, 27.4, 7.2, "white", c)
        ax.text(2.8, y + 2.8, t, fontsize=6.6, weight="bold", color=c)
        ax.text(2.8, y + 0.4, d, fontsize=5.7, color=DARK)
    ax.text(2.8, 2.0, "R1 at 4–6k CoT tokens: 0.00", fontsize=5.7, color=GREY)
    # Panel B: construction
    box(ax, 34.0, 1.5, 33.0, 39, "#FDF7F2", RED)
    ax.text(36.4, 38.6, "② 2ND-ORDER GRAMMAR", fontsize=7.6, weight="bold", color=RED)
    ax.text(36.2, 35.8, "records carry 2-WL colors", fontsize=6.4, color=DARK)
    ax.text(35.5, 32.6, "edge record:", fontsize=6.0, color=DARK)
    ax.text(35.5, 30.2, "n3 -r1-> n17", family="monospace", fontsize=7.4, color=RED,
            bbox=dict(boxstyle="round,pad=0.28", fc="#FDECEA", ec=RED))
    ax.text(35.5, 27.2, "triangle record:", fontsize=6.0, color=DARK)
    ax.text(35.5, 24.8, "TRI n3 -> {n8, n17}", family="monospace", fontsize=7.4, color=RED,
            bbox=dict(boxstyle="round,pad=0.28", fc="#FDECEA", ec=RED))
    ax.text(35.5, 21.2, "Thm. 2 · one layer compares TRI multisets;", fontsize=6.0, color=DARK)
    ax.text(35.5, 19.0, "k layers ≡ k rounds of 2-WL refinement", fontsize=6.0, color=DARK)
    ax.text(35.5, 15.8, "Prop. 1 · each WL level is priced:", fontsize=6.0, color=DARK)
    ax.text(35.5, 13.6, "level ℓ records cost O(n^ℓ)", fontsize=6.0, color=DARK)
    ax.text(35.5, 10.0, "ablation: same arm, TRI removed", fontsize=6.0, color=RED)
    ax.text(35.5, 7.8, "at test time: 1.000 → 0.000", fontsize=6.0, color=RED)
    # Panel C: results
    box(ax, 70.5, 1.5, 29.0, 39, "#F3F8FB", BLUE)
    ax.text(72.6, 38.6, "③ BARRIER LIFTED", fontsize=7.6, weight="bold", color=BLUE)
    ax.text(72.0, 35.8, "LoRA on 0.3% of parameters", fontsize=6.4, color=DARK)
    res = [("boundary lifted", "iso T1/T2/T3: 1.000/1.000/1.000", "at 1.5B and 7B", OI["green"]),
           ("frontier chains", "hop-3: 0.80→0.90 at 1/5 tokens", "chain length = depth", OI["blue"]),
           ("full-graph reading", "228k edges: 0.17→0.34 via", "record retrieval", OI["sky"]),
           ("transfer", "0.91–1.00 retained across", "vocabularies (7B)", OI["purple"])]
    for i, (t, d1, d2, c) in enumerate(res):
        y = 28.6 - i * 6.6
        box(ax, 71.8, y - 2.6, 26.4, 5.8, "white", c)
        ax.text(72.8, y + 2.2, t, fontsize=6.3, weight="bold", color=c)
        ax.text(72.8, y - 0.2, d1, fontsize=5.7, color=DARK)
        ax.text(72.8, y - 2.1, d2, fontsize=5.5, color=GREY)
    arrow(ax, 30.9, 21, 33.6, 21, DARK)
    arrow(ax, 67.4, 21, 70.1, 21, DARK)
    fig.savefig(f"{OUT}/fig1_overview.pdf"); plt.close(fig)


# ================= Fig 2: 2-PG method (single column) =================
def fig_method2pg():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6),
                             gridspec_kw=dict(height_ratios=[1.15, 1.3]))
    # (a) first-order records cannot see triangles
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 4.45); ax.axis("off")
    ax.text(0.0, 4.2, "(a) First-order: 1-WL collision", fontsize=8, weight="bold", color=DARK)
    ax.text(0.15, 3.4, "C₆:  n0 -r1-> n1   n1 -r1-> n2   ...", family="monospace", fontsize=7,
            color=DARK, bbox=dict(boxstyle="round,pad=0.28", fc="#ECEFF1", ec=GREY))
    ax.text(0.15, 2.6, "2C₃: n0 -r1-> n1   n1 -r1-> n2   ...", family="monospace", fontsize=7,
            color=DARK, bbox=dict(boxstyle="round,pad=0.28", fc="#ECEFF1", ec=GREY))
    ax.text(0.15, 1.6, "every node: one in-edge, one out-edge", fontsize=6.6, color=GREY)
    ax.text(0.15, 0.9, "edge records are identical up to renaming", fontsize=6.6, color=GREY)
    ax.text(0.15, 0.2, "→ no reader can separate them (Thm. 1)", fontsize=6.8, color=DARK)
    # (b) TRI records break it
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 5.9); ax.axis("off")
    ax.text(0.0, 5.65, "(b) Second-order: TRI records", fontsize=8, weight="bold", color=RED)
    ax.text(0.15, 4.85, "C₆ triangles: none", fontsize=7, color=DARK,
            bbox=dict(boxstyle="round,pad=0.26", fc="#ECEFF1", ec=GREY))
    ax.text(0.15, 4.05, "2C₃: TRI n0 -> {n1,n2}   TRI n1 -> {n0,n2}", family="monospace",
            fontsize=6.8, color=RED, bbox=dict(boxstyle="round,pad=0.26", fc="#FDECEA", ec=RED))
    ax.text(0.15, 3.25, "        TRI n2 -> {n0,n1}   (+ n3,n4,n5)", family="monospace",
            fontsize=6.8, color=RED)
    ax.text(5.5, 5.35, "the invariant is content:", fontsize=6.4, color=RED, va="center")
    ax.text(5.5, 4.9, "renames cannot touch TRI sets", fontsize=6.4, color=RED, va="center")
    ax.text(0.15, 2.35, "Q:  are the TRI multisets equal ?", fontsize=7.2, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.26", fc="#E3F2FD", ec=BLUE))
    ax.text(0.15, 1.5, "one content-addressed comparison of TRI records", fontsize=6.6, color=DARK)
    ax.text(0.15, 0.75, "decides every 1-WL collision with triangle statistics", fontsize=6.6, color=DARK)
    ax.text(0.15, 0.05, "(Thm. 2); each further WL level is priced O(n^ℓ) (Prop. 1)", fontsize=6.6, color=GREY)
    fig.savefig(f"{OUT}/fig2_method2pg.pdf"); plt.close(fig)


# ================= Fig 3: boundary lift (single column) =================
def fig_boundary():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    tiers = ["Tier 1\n(true iso.)", "Tier 2\n(1-WL sep.)", "Tier 3\n(C₆ vs 2C₃)"]
    x = np.arange(3); w = 0.27
    floor = [0.92, 0.88, 0.10]
    r1 = [0.00, 0.30, 0.00]
    a15 = [1.0, 1.0, 1.0]; a7 = [1.0, 1.0, 1.0]
    ax.bar(x - 1.5 * w, floor, w, label="12 models × 8 encodings (max)", color=GREY)
    ax.bar(x - 0.5 * w, r1, w, label="DeepSeek-R1 (reasoning)", color=OI["orange"])
    ax.bar(x + 0.5 * w, a15, w, label="1.5B LoRA + 2-PG", color=OI["sky"])
    ax.bar(x + 1.5 * w, a7, w, label="7B LoRA + 2-PG", color=RED)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.text(-0.42, 0.522, "chance", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=7)
    ax.set_ylabel("isomorphism accuracy"); ax.set_ylim(0, 1.12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="none",
              loc="upper left", fontsize=5.8)
    # annotate ablation on tier3
    import matplotlib.patheffects as pe
    hw = [pe.withStroke(linewidth=2.6, foreground="white")]
    ax.annotate("ablation: TRI removed\n(same arm) → 0.000", xy=(2.0 + 0.5 * w, 1.0),
                xytext=(1.32, 0.60), fontsize=5.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8,
                                connectionstyle="arc3,rad=0.15"))
    for t in ax.texts:
        t.set_path_effects(hw)
    fig.savefig(f"{OUT}/fig3_boundary.pdf"); plt.close(fig)


# ================= Fig 4: frontier chains (double, 2 panels) =================
def fig_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    ax = axes[0]
    arms = [("free CoT (LoRA)", [1.00, 0.87, 0.80], 300, OI["blue"], "o"),
            ("frontier (frozen)", [0.78, 0.51, 0.54], 36, GREY, "s"),
            ("frontier (LoRA)", [1.00, 0.983, 0.900], 58, RED, "D")]
    for lab, accs, tok, c, mk in arms:
        ax.scatter([tok] * 3, accs, s=30, color=c, marker=mk, label=lab, zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.plot([tok] * 3, accs, color=c, lw=1.0, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks([36, 58, 300]); ax.set_xticklabels(["36", "58", "~300"])
    ax.set_xlabel("median output tokens per answer")
    ax.set_ylabel("accuracy")
    for k, dy in zip(range(3), [0.012, -0.03, 0.012]):
        ax.text(36, [0.78, 0.51, 0.54][0] + dy, "")
    for i, (k, acc) in enumerate(zip([1, 2, 3], [0.78, 0.51, 0.54])):
        pass
    ax.annotate("k=1", (300, 1.00), xytext=(4, 2), textcoords="offset points", fontsize=6, color=OI["blue"])
    ax.annotate("k=3", (300, 0.80), xytext=(4, -8), textcoords="offset points", fontsize=6, color=OI["blue"])
    ax.set_ylim(0.4, 1.06)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="none",
              loc="lower right", fontsize=6.2)
    ax.set_title("(a) Accuracy–cost frontier")
    ax = axes[1]
    ks = [1, 2, 3]
    toks = [30, 42, 58]
    ax.plot(ks, toks, marker="D", color=RED, lw=1.6, ms=4.5)
    for k, t in zip(ks, toks):
        ax.annotate(f"{t}", (k, t), xytext=(5, 3), textcoords="offset points", fontsize=7, color=RED)
    ax.set_xticks(ks); ax.set_xlabel("hop depth $k$")
    ax.set_ylabel("median chain tokens")
    ax.set_ylim(20, 70)
    ax.set_title("(b) Chain length = message-passing depth")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig4_frontier.pdf"); plt.close(fig)


# ================= Fig 5: retrieval + transfer (double, 2 panels) =================
def fig_sys():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    ax = axes[0]
    bars = [("frozen\nretrieval", 0.170, GREY),
            ("LoRA\nretrieval", 0.340, BLUE),
            ("LoRA\nsubgraph", 0.510, OI["sky"]),
            ("best frozen\nany family", 0.340, OI["orange"])]
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], 0.56, color=[b[2] for b in bars],
           edgecolor="white", linewidth=0.4)
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.012, f"{b[1]:.3f}", ha="center", fontsize=7)
    ax.axhline(0.10, color="k", ls="--", lw=0.8)
    ax.text(3.35, 0.108, "chance", fontsize=6.2)
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=6.6)
    ax.set_ylabel("NELL 10-way acc. (full graph)")
    ax.set_ylim(0, 0.62)
    ax.set_title("(a) Retrieval-style reading, 228k-edge NELL")
    ax = axes[1]
    arms = ["edge list", "Pointer v1", "Pointer v2"]
    r15 = [0.33, 0.21, 0.97]
    r7 = [0.95, 1.10, 1.08]
    xs = np.arange(3); w = 0.36
    ax.bar(xs - w / 2, r15, w, label="1.5B", color=OI["sky"], edgecolor="white")
    ax.bar(xs + w / 2, r7, w, label="7B", color=RED, edgecolor="white")
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.text(2.42, 1.02, "lossless", fontsize=6.2)
    import matplotlib.patheffects as pe
    hw5 = [pe.withStroke(linewidth=2.6, foreground="white")]
    for i, v in enumerate(r15):
        ax.text(i - w / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.4,
                color=OI["sky"], path_effects=hw5)
    for i, v in enumerate(r7):
        ax.text(i + w / 2, v + (0.06 if abs(v - 0.95) < 0.02 else 0.02),
                f"{v:.2f}", ha="center", fontsize=6.4,
                path_effects=hw5)
    ax.set_xticks(xs); ax.set_xticklabels(arms)
    ax.set_ylabel("nbr-F1 retention on real graphs")
    ax.set_ylim(0, 1.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Transfer: program vs. memorization")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig5_sys.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_overview(); print("f1 ok")
    fig_method2pg(); print("f2 ok")
    fig_boundary(); print("f3 ok")
    fig_frontier(); print("f4 ok")
    fig_sys(); print("f5 ok")
