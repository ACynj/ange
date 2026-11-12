#!/usr/bin/env python
"""Publication-quality figure suite v2: improved existing + new figures."""
import json, os, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

R = "/mnt/ssd/ange/results"
OUT = "/home/ynj/new-pro2/paper/figs"
os.makedirs(OUT, exist_ok=True)

# ─── Global style: clean IEEE journal quality ───
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.2,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "pdf.fonttype": 42,
})

# ─── Color palette (Okabe-Ito + custom) ───
C = {
    "red": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
    "sky": "#56B4E9", "purple": "#CC79A7", "grey": "#8B95A0", "dark": "#2C3E50",
    "light_grey": "#E8ECF0", "yellow": "#F0E442", "black": "#000000",
}
RED = C["red"]; BLUE = C["blue"]; GREEN = C["green"]; DARK = C["dark"]
GREY = C["grey"]; ORANGE = C["orange"]; SKY = C["sky"]; PURPLE = C["purple"]

HW = [pe.withStroke(linewidth=2.5, foreground="white")]  # white halo

ENCS = ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]
NICE = {"edge_list": "Edge\nlist", "adjacency_list": "Adjacency\ngrp.", "incident_list": "Incidence\ngrp.",
        "nl": "Prose\n(NL)", "json": "JSON", "sog": "SOG\ndigest", "dfs_serial": "DFS\nserial.",
        "pointer": "Pointer\nv1", "pointer2": "Pointer\nv2"}
NICE1 = {k: v.replace("\n", " ") for k, v in NICE.items()}
C_ENC = {e: GREY for e in ENCS}
C_ENC["sog"] = C["dark"]; C_ENC["pointer"] = RED; C_ENC["dfs_serial"] = "#7B8B95"
C_ENC["adjacency_list"] = BLUE; C_ENC["edge_list"] = "#95A5A6"

# Custom diverging colormap: red (bad) → white → green (good)
cmap_div = LinearSegmentedColormap.from_list("rg", ["#E74C3C", "#F5B7B1", "#FDF2F2", "#D5F5E3", "#82E0AA", "#229954"])
cmap_heat = LinearSegmentedColormap.from_list("heat", ["#C0392B", "#E67E22", "#F1C40F", "#82E0AA", "#229954"])


def rows_of(path):
    try:
        return [json.loads(l) for l in open(path)]
    except OSError:
        return []


def clean(path):
    best = {}
    for r in rows_of(path):
        if "ERROR" in r.get("raw", ""): continue
        pid = r.get("pid", r.get("arm", "") + str(r.get("n", ""))).split("@")[0]
        best[(r.get("model", r.get("arm", "?")), r["task"], r.get("enc", "?"), pid)] = r["score"]
    return best


def acc(d, model, task, enc):
    v = [s for (m, t, e, i), s in d.items() if m == model and t == task and e == enc]
    return (sum(v) / len(v)) if v else np.nan


TASKS5 = ["reach1", "reach2", "reach3", "nbr", "ambig"]
TNICE = {"reach1": "1-hop", "reach2": "2-hop", "reach3": "3-hop", "nbr": "nbr F1", "ambig": "ambig"}


# ════════════════════════════════════════════════════════════════════
# NEW FIGURE A: Capacity Flip — encoding ranking vs model scale
# ══════════════════════════════════════════════════════════════════
def fig_capacity_flip():
    """Shows the key novel finding: best encoding changes with model size."""
    datasets = {
        "1.5B": ("e1_qwen1.5b.jsonl", "Qwen2.5-1.5B-Instruct"),
        "3B": ("e1_qwen3b.jsonl", "Qwen2.5-3B-Instruct"),
        "7B": ("e1_qwen7b.jsonl", "Qwen2.5-7B-Instruct"),
        "14B": ("e1_qwen14b.jsonl", "Qwen2.5-14B-Instruct"),
        "72B": ("e1_qwen72b.jsonl", "Qwen/Qwen2.5-72B-Instruct"),
        "GPT-4o": ("e1_gpt4o.jsonl", "gpt-4o"),
    }
    # focus on nbr F1 (the clean metric)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.3))

    # Panel (a): nbr F1 across model sizes for key encodings
    sizes = list(datasets.keys())
    x = np.arange(len(sizes))
    show = ["adjacency_list", "pointer", "edge_list", "sog"]
    labels = ["Adjacency grp.", "Pointer v1", "Edge list", "SOG digest"]
    colors = [BLUE, RED, GREY, C["dark"]]
    markers = ["s", "D", "o", "X"]

    for e, lab, col, mk in zip(show, labels, colors, markers):
        vals = []
        for sz in sizes:
            f, m = datasets[sz]
            d = clean(f"{R}/{f}")
            vals.append(acc(d, m, "nbr", e))
        ax1.plot(x, vals, marker=mk, color=col, label=lab, lw=1.6, ms=4.5,
                 markeredgecolor="white", markeredgewidth=0.4)

    ax1.set_xticks(x); ax1.set_xticklabels(sizes, fontsize=6.5)
    ax1.set_ylabel("nbr F1")
    ax1.set_xlabel("Model capacity →")
    ax1.set_ylim(0, 1.05)
    ax1.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.4)
    ax1.set_title("(a) Encoding ranking flips with capacity", fontsize=7.5, pad=4)

    # annotate the flip point
    ax1.annotate("adjacency peaks\nat 14B", xy=(3, 0.88), xytext=(1.5, 0.95),
                fontsize=5.5, color=BLUE, ha="center", path_effects=HW,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))
    ax1.annotate("pointer wins\nat 72B+", xy=(5, 0.84), xytext=(4.2, 0.65),
                fontsize=5.5, color=RED, ha="center", path_effects=HW,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))

    # Panel (b): per-encoding bars at 3 capacity levels
    ax2.set_visible(False)  # keep for spacing; we'll use a different approach

    plt.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig_capacity_flip.pdf")
    plt.close()
    print("capacity_flip done")


# ══════════════════════════════════════════════════════════════════
# NEW FIGURE B: Multi-model per-task heatmap
# ══════════════════════════════════════════════════════════════════
def fig_heatmap_grid():
    """Side-by-side heatmaps for 3 key models showing task-specific patterns."""
    models = [
        ("Qwen2.5-7B", "e1_qwen7b.jsonl", "Qwen2.5-7B-Instruct"),
        ("DeepSeek-V3", "e1_deepseek.jsonl", "deepseek-chat"),
        ("GPT-4o-mini", "e1_gpt4omini.jsonl", "gpt-4o-mini"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), constrained_layout=True)

    for ax, (label, f, mname) in zip(axes, models):
        d = clean(f"{R}/{f}")
        M = np.full((len(ENCS), len(TASKS5)), np.nan)
        for i, e in enumerate(ENCS):
            for j, t in enumerate(TASKS5):
                M[i, j] = acc(d, mname, t, e)

        im = ax.imshow(M, cmap=cmap_heat, vmin=0, vmax=1, aspect="auto")

        # cell values
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] == M[i, j]:
                    v = M[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                            color="black" if 0.25 < v < 0.75 else "white")

        ax.set_xticks(range(len(TASKS5)))
        ax.set_xticklabels([TNICE[t] for t in TASKS5], rotation=38, ha="right", fontsize=5.2)
        ax.set_yticks(range(len(ENCS)))
        ax.set_yticklabels([NICE1[e] for e in ENCS], fontsize=5.2)
        ax.axhline(4.5, color="white", lw=1.5)
        ax.set_title(label, fontsize=7, pad=3)
        ax.tick_params(length=1.5)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.015, aspect=25)
    cbar.set_label("score", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    fig.savefig(f"{OUT}/fig_heatmap_grid.pdf")
    plt.close()
    print("heatmap_grid done")


# ══════════════════════════════════════════════════════════════════
# IMPROVED FIG 0: Method schematic (cleaner, more professional)
# ══════════════════════════════════════════════════════════════════
def fig_method():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6),
                             gridspec_kw=dict(height_ratios=[1.0, 1.35], hspace=0.28))

    # (a) Positional — role by column
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 4.8); ax.axis("off")
    ax.text(0.0, 4.6, "(a) Positional record", fontsize=8, weight="bold", color=DARK)
    ax.text(0.0, 4.25, "role = column position", fontsize=6.5, color=GREY, style="italic")

    for y, txt in [(3.5, "n3   r1   n17"), (2.7, "n8   r1   n3")]:
        ax.text(0.15, y, txt, family="monospace", fontsize=8.5, color=DARK,
                bbox=dict(boxstyle="round,pad=0.35", fc=C["light_grey"], ec=GREY, lw=0.5))
    # column annotations
    ax.annotate("col 1\n= source", xy=(0.5, 3.5), xytext=(4.0, 3.55),
                fontsize=5.8, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    ax.annotate("col 3\n= target", xy=(2.8, 3.5), xytext=(4.0, 2.85),
                fontsize=5.8, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))

    ax.text(0.15, 1.7, "Q: is n17 an r1-neighbor of n3?", fontsize=7.2, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB", ec=BLUE, lw=0.5))
    ax.text(0.15, 0.6, "Must scan & re-parse every record;\nerror grows with graph size.", fontsize=6.2, color=GREY)

    # (b) Pointer — role in content
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis("off")
    ax.text(0.0, 5.8, "(b) Pointer record", fontsize=8, weight="bold", color=RED)
    ax.text(0.0, 5.45, "role = content token", fontsize=6.5, color=GREY, style="italic")

    recs = [("n5  -r2->  n9", C["light_grey"], DARK),
            ("n3  -r1->  n17", "#FDEBD0", RED),
            ("n3  -r1->  n8", "#FDEBD0", RED)]
    for i, (t, fc, cc) in enumerate(recs):
        ax.text(0.15, 4.85 - i * 0.82, t, family="monospace", fontsize=8, color=cc,
                bbox=dict(boxstyle="round,pad=0.32", fc=fc, ec=cc, lw=0.5))

    # annotation for direction token
    ax.annotate("direction\nis content", xy=(3.2, 4.03), xytext=(5.5, 4.85),
                fontsize=5.8, color=RED, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))

    ax.text(0.15, 1.9, "Q:  n3  -r1->  ?", family="monospace", fontsize=8, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.32", fc="#EBF5FB", ec=BLUE, lw=0.5))
    ax.text(4.5, 1.9, "one content match\nwindow O(1) in m", fontsize=5.8, color=RED, va="center")

    ax.text(0.15, 0.6, "k attention layers ≡ k rounds of message passing\n(Thm. 4); lossless (Thm. 3).",
            fontsize=6.2, color=GREY)

    fig.savefig(f"{OUT}/fig0_method.pdf")
    plt.close()
    print("fig0_method improved")


# ══════════════════════════════════════════════════════════════════
# IMPROVED FIG 3: Ambiguity divergence (more visual impact)
# ══════════════════════════════════════════════════════════════════
def fig_ambig():
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    files = [("Qwen-7B", "e1_qwen7b.jsonl", BLUE),
             ("DS-V3", "e1_deepseek.jsonl", ORANGE),
             ("GLM-4F", "e1_glm4flash.jsonl", PURPLE),
             ("GPT-4om", "e1_gpt4omini.jsonl", DARK)]
    encs = ["edge_list", "nl", "dfs_serial", "pointer", "sog"]
    w = 0.18

    for mi, (m, f, c) in enumerate(files):
        pair = defaultdict(dict)
        for r in rows_of(f"{R}/{f}"):
            if r["task"] == "ambig" and "ERROR" not in r.get("raw", ""):
                s = set(r["pred"].split(",")) if r["pred"] not in ("NONE", "NA") else set()
                pair[(r["enc"], r["meta"]["pair"])][r["meta"]["side"]] = s
        byenc = defaultdict(list)
        for (e, p), v in pair.items():
            if len(v) == 2: byenc[e].append(0.0 if v["A"] == v["B"] else 1.0)
        xs = np.arange(len(encs)) + (mi - 1.5) * (w + 0.012)
        ys = [np.mean(byenc[e]) if byenc.get(e) else np.nan for e in encs]
        bars = ax.bar(xs, ys, w, label=m, color=c, edgecolor="white", linewidth=0.4)

    # add "structure-blind zone" shading (no in-plot label to avoid bar collision)
    ax.axhspan(0, 0.15, color="#FADBD8", alpha=0.3, zorder=0)

    ax.set_xticks(range(len(encs)))
    ax.set_xticklabels([NICE1[e] for e in encs], rotation=26, ha="right", fontsize=5.5)
    ax.set_ylabel("answer divergence\non ambiguous pairs\n(shaded = blind zone)", fontsize=6.2)
    ax.set_ylim(0, 1.15); ax.set_xlim(-0.65, len(encs) - 0.35)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              fontsize=5.2, columnspacing=0.5, handletextpad=0.3)
    ax.axhline(0.15, color="#C0392B", ls=":", lw=0.5, alpha=0.5)

    fig.savefig(f"{OUT}/fig3_ambig.pdf")
    plt.close()
    print("fig3_ambig improved")


# ══════════════════════════════════════════════════════════════════
# IMPROVED FIG 5: E3 Training repairs (dumbbell + bars)
# ══════════════════════════════════════════════════════════════════
def fig_e3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.4),
                                    gridspec_kw=dict(width_ratios=[1.2, 1]))

    # (a) Dumbbell chart: frozen → LoRA for each encoding
    tasks = ["reach1", "reach2", "reach3", "nbr"]
    arms_data = {}
    for arm, enc in [("frozen", "edge_list"), ("lora_edge", "edge_list"),
                     ("lora_pointer", "pointer"), ("lora_sog", "sog")]:
        vals = defaultdict(list)
        for r in rows_of(f"{R}/e3_results.jsonl"):
            if r["arm"] == arm and r["enc"] == enc and "ERROR" not in r.get("raw", ""):
                vals[r["task"]].append(r["score"])
        arms_data[arm] = {t: np.mean(vals[t]) if vals[t] else np.nan for t in tasks}

    y_pos = np.arange(len(tasks))
    # frozen → LoRA edge list
    for t, y in zip(tasks, y_pos):
        f = arms_data["frozen"][t]; l = arms_data["lora_edge"][t]
        ax1.plot([f, l], [y, y], color=GREEN, lw=1.8, alpha=0.6, zorder=1)
        ax1.scatter(f, y, color=GREY, s=28, zorder=2, edgecolor="white", lw=0.5)
        ax1.scatter(l, y, color=GREEN, s=35, zorder=3, edgecolor="white", lw=0.5)
    # LoRA sog (the hallucination point)
    for t, y in zip(tasks, y_pos):
        s = arms_data["lora_sog"][t]
        if s == s:
            ax1.scatter(s, y, color=C["dark"], marker="X", s=40, zorder=3,
                       edgecolor="white", lw=0.5)

    ax1.set_yticks(y_pos); ax1.set_yticklabels([TNICE[t] for t in tasks], fontsize=6.5)
    ax1.set_xlabel("score", fontsize=6.5)
    ax1.set_xlim(0, 1.0)
    ax1.scatter([], [], color=GREY, s=28, label="Frozen")
    ax1.scatter([], [], color=GREEN, s=35, label="LoRA (edge list)")
    ax1.scatter([], [], color=C["dark"], marker="X", s=40, label="LoRA (digest)")
    ax1.legend(frameon=False, fontsize=5.5, loc="lower right", handletextpad=0.3)
    ax1.set_title("(a) Training repairs utilization,\nnot information (1.5B)", fontsize=7, pad=4)
    ax1.axvline(0.1, color="grey", ls=":", lw=0.4, alpha=0.3)

    # annotate the hallucination
    ax1.annotate("hallucinated", xy=(0.09, 3), xytext=(0.15, 3.42),
                fontsize=5.5, color=C["dark"], style="italic", path_effects=HW,
                arrowprops=dict(arrowstyle="->", color=C["dark"], lw=0.6,
                               connectionstyle="arc3,rad=0.2"))

    # (b) Real-KG repair with error bars
    vals = defaultdict(list)
    for r in rows_of(f"{R}/e3_real.jsonl"):
        if "ERROR" not in r.get("raw", ""): vals[r["arm"]].append(r["score"])
    ks = ["frozen7b_edge", "lora_edge", "lora_pointer2"]
    labs = ["Frozen 7B", "LoRA\nedge list", "LoRA\nPointer v2"]
    cs = [GREY, BLUE, ORANGE]
    ys = [np.mean(vals[k]) for k in ks]
    es = [1.96 * np.std(vals[k]) / np.sqrt(len(vals[k])) for k in ks]
    xs = np.arange(3)
    ax2.bar(xs, ys, 0.52, color=cs, edgecolor="white", lw=0.5,
           yerr=es, capsize=2.8, error_kw=dict(lw=0.7, ecolor=DARK))
    for i, v in enumerate(ys):
        ax2.text(i, v + es[i] + 0.025, f"{v:.3f}", ha="center", fontsize=6.5,
                fontweight="bold", color=cs[i])
    ax2.axhline(0.10, color="black", ls="--", lw=0.7, zorder=0)
    ax2.text(-0.38, 0.115, "chance", fontsize=5.5, style="italic")
    ax2.axhline(0.34, color=GREY, ls=":", lw=0.8, zorder=0)
    ax2.text(1.5, 0.355, "best prompt-only", fontsize=5.5, color=GREY, style="italic")
    ax2.set_xticks(xs); ax2.set_xticklabels(labs, fontsize=6.2)
    ax2.set_ylabel("NELL 10-way acc.", fontsize=6.5)
    ax2.set_ylim(0, 0.68); ax2.set_xlim(-0.55, 2.75)
    ax2.set_title("(b) Real-KG repair (7B, 3k probes)", fontsize=7, pad=4)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(f"{OUT}/fig5_e3.pdf")
    plt.close()
    print("fig5_e3 improved")


# ══════════════════════════════════════════════════════════════════
# IMPROVED FIG 4: Real-KG wall (more polished)
# ═════════════════════════════════════════════════════════════════
def fig_e2():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rows_data = [
        ("Qwen-7B", "e2_nell_qwen7b.jsonl", "e2_fb_qwen7b.jsonl"),
        ("Qwen-14B", "e2_nell_qwen14b.jsonl", "e2_fb_qwen14b.jsonl"),
        ("Qwen-72B", "e2_nell_qwen72b.jsonl", "e2_fb_qwen72b.jsonl"),
        ("DS-V3", "e2_nell_deepseek.jsonl", None),
        ("GPT-4om", "e2_nell_gpt4omini.jsonl", "e2_fb_gpt4omini.jsonl"),
    ]
    def mean(p):
        v = [r["score"] for r in rows_of(f"{R}/{p}") if "ERROR" not in r.get("raw", "")]
        return sum(v) / len(v) if v else np.nan

    names = [r[0] for r in rows_data]
    nell = [mean(r[1]) for r in rows_data]
    fbv = [mean(r[2]) if r[2] else np.nan for r in rows_data]
    xs = np.arange(len(names))

    ax.bar(xs - 0.18, nell, 0.34, label="NELL-995\n(readable IDs)", color=BLUE,
           edgecolor="white", lw=0.4)
    fbx = [x + 0.18 for x, v in zip(xs, fbv) if not np.isnan(v)]
    fby = [v for v in fbv if not np.isnan(v)]
    ax.bar(fbx, fby, 0.34, label="FB15k-237\n(opaque IDs)", color=SKY,
           edgecolor="white", lw=0.4)
    if np.isnan(fbv[3]):
        ax.text(xs[3] + 0.18, 0.015, "n/a", ha="center", fontsize=6, color=DARK)

    ax.axhline(0.10, color="black", ls="--", lw=0.7, zorder=0)
    ax.text(-0.58, 0.125, "chance", fontsize=6, style="italic", path_effects=HW)

    # shade the "prompt ceiling" region
    ax.axhspan(0.34, 0.70, color="#FADBD8", alpha=0.25, zorder=0)
    ax.text(2.0, 0.42, "prompt ceiling\n(≤0.34 for all\nfamilies)", fontsize=5.2,
            color="#C0392B", ha="center", style="italic")

    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=24, ha="right", fontsize=6)
    ax.set_ylabel("10-way accuracy", fontsize=6.5)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=5.2, ncol=2, labelspacing=0.3, columnspacing=1.0)
    ax.set_ylim(0, 0.55); ax.set_xlim(-0.65, len(names) - 0.35)

    fig.savefig(f"{OUT}/fig4_e2.pdf")
    plt.close()
    print("fig4_e2 improved")


# ══════════════════════════════════════════════════════════════════
# IMPROVED FIG 2: Scale invariance + extrapolation
# ═════════════════════════════════════════════════════════════════
def fig_scale():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.3))

    # (a) Frozen reading: error vs graph size
    series = {10: f"{R}/ab_n10_qwen7b.jsonl", 20: f"{R}/e1_qwen7b.jsonl",
              30: f"{R}/ab_n30_qwen7b.jsonl", 50: f"{R}/big_n50_qwen7b.jsonl",
              100: f"{R}/big_n100_qwen7b.jsonl"}
    ds = {n: clean(f) for n, f in series.items()}
    def val(n, e):
        v = acc(ds[n], "Qwen2.5-7B-Instruct", "reach1", e) if n in (10,20,30) else np.nan
        if np.isnan(v):
            s = [x for (m,t,ee,i), x in ds[n].items() if t=="reach1" and ee==e]
            v = sum(s)/len(s) if s else np.nan
        return v

    styles = {"pointer": (RED, "D", "-", "Pointer v1"),
              "edge_list": (GREY, "o", "--", "Edge list"),
              "adjacency_list": (BLUE, "s", "-.", "Adjacency"),
              "json": (PURPLE, "^", ":", "JSON"),
              "sog": (C["dark"], "X", "-", "SOG digest")}
    for e, (c, mk, ls, lab) in styles.items():
        pts = [(n, 1 - val(n, e)) for n in (10, 20, 30, 50, 100)]
        pts = [(n, y) for n, y in pts if y == y]
        ax1.plot(*zip(*pts), color=c, marker=mk, ls=ls, label=lab, ms=3.8,
                markeredgecolor="white", markeredgewidth=0.3)

    # shade the flat zone
    ax1.axhspan(0, 0.15, color="#D5F5E3", alpha=0.3, zorder=0)
    ax1.text(65, 0.06, "pointer: flat", fontsize=5.5, color=GREEN,
            ha="center", style="italic", path_effects=HW)

    ax1.set_xscale("log"); ax1.set_xlim(9, 115)
    ax1.set_xticks([10, 20, 30, 50, 100]); ax1.set_xticklabels([10,20,30,50,100])
    ax1.minorticks_off()
    ax1.set_xlabel("graph size $n$ (log scale)", fontsize=6.5)
    ax1.set_ylabel("1-hop error rate", fontsize=6.5)
    ax1.legend(frameon=False, ncol=2, loc="upper left", fontsize=5.2,
              columnspacing=0.5, handletextpad=0.3)
    ax1.set_title("(a) Frozen: locality = flat error curve (7B)", fontsize=7, pad=4)

    # (b) Trained extrapolation with aligned labels
    p2rows = [r for r in rows_of(f"{R}/e3_p2.jsonl") if "ERROR" not in r.get("raw", "")]
    scrows = [r for r in rows_of(f"{R}/e3_scale.jsonl") if "ERROR" not in r.get("raw", "")]
    series_b = [
        ("1.5B·Ptr v1", [r for r in scrows if r["arm"]=="lora_pointer"], "#E8A7A5", "^", "--"),
        ("1.5B·Ptr v2", [r for r in p2rows if r["arm"]=="lora_pointer2" and r["base"]=="1.5B"], "#C62828", "s", "-"),
        ("1.5B·edge", [r for r in scrows if r["arm"]=="lora_edge"], GREY, "o", ":"),
        ("7B·Ptr v2", [r for r in p2rows if r["arm"]=="lora_pointer2" and r["base"]=="7B"], "#7B1D1D", "D", "-"),
        ("7B·edge", [r for r in p2rows if r["arm"]=="lora_edge" and r["base"]=="7B"], "#455A64", "o", ":")]
    yslot = {s[0]: y for s, y in zip(series_b, [0.680, 0.615, 0.550, 0.485, 0.420])}
    for lab, rr, c, mk, ls in series_b:
        vals = defaultdict(list)
        for r in rr:
            if r["task"] == "nbr": vals[r["n"]].append(r["score"])
        xs = [n for n in (20,30,40) if vals.get(n)]
        ys = [sum(vals[n])/len(vals[n]) for n in xs]
        if not xs: continue
        ax2.plot(xs, ys, color=c, marker=mk, ls=ls, ms=3.8,
                markeredgecolor="white", markeredgewidth=0.3)
        # place label at yslot y-value (not data y) to guarantee separation
        y_pos = yslot.get(lab, ys[-1])
        ax2.annotate("", xy=(41.0, ys[-1]), xytext=(43.5, y_pos),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.6))
        ax2.text(44.0, y_pos, lab, fontsize=5.5, color=c, va="center",
                path_effects=HW)

    # highlight the v1 collapse (arrow stops short of text)
    ax2.annotate("v1 collapses\nat 1.5B", xy=(40, 0.375), xytext=(32, 0.44),
                fontsize=5.5, color="#E8A7A5", style="italic", path_effects=HW,
                arrowprops=dict(arrowstyle="->", color="#E8A7A5", lw=0.6,
                               connectionstyle="arc3,rad=0.15"))

    ax2.set_xticks([20, 30, 40]); ax2.set_xlim(19, 66)
    ax2.set_xlabel("test size (trained at $n{=}20$ only)", fontsize=6.5)
    ax2.set_ylabel("nbr F1", fontsize=6.5); ax2.set_ylim(0.30, 0.74)
    ax2.set_title("(b) Trained extrapolation: addressing vs. economy", fontsize=7, pad=4)

    fig.tight_layout(w_pad=1.3)
    fig.savefig(f"{OUT}/fig2_scale.pdf")
    plt.close()
    print("fig2_scale improved")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_method()
    fig_ambig()
    fig_e3()
    fig_e2()
    fig_scale()
    fig_capacity_flip()
    fig_heatmap_grid()
    print("\n=== All 7 figures done ===")
