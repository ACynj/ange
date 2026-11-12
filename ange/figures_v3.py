#!/usr/bin/env python
"""Paper-1 figure suite v3 — systematic redesign per strict audit:
(a) reference lines BEHIND bars (zorder); (b) axes trimmed to data;
(c) real gaps between bars (width+edge); (d) legends outside data;
(e) deterministic dodge in scatters; (f) gridded schematic layouts."""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = "/mnt/ssd/ange/results"
OUT = "/home/ynj/new-pro2/paper/figs"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
OI = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
      "sky": "#56B4E9", "purple": "#CC79A7", "grey": "#9AA5AD", "dark": "#37474F"}
RED = OI["red"]; GREY = "#9AA5AD"; DARK = "#37474F"; BLUE = OI["blue"]
ENCS = ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]
NICE = {"edge_list": "Edge list", "adjacency_list": "Adjacency", "incident_list": "Incident",
        "nl": "Prose (NL)", "json": "JSON", "sog": "SOG digest", "dfs_serial": "DFS serial.",
        "pointer": "Pointer v1", "pointer2": "Pointer v2"}


def rows_of(path):
    try:
        return [json.loads(l) for l in open(path)]
    except OSError:
        return []


def clean(path):
    best = {}
    for r in rows_of(path):
        if "ERROR" in r.get("raw", ""):
            continue
        pid = r.get("pid", r.get("arm", "") + str(r.get("n", ""))).split("@")[0]
        best[(r.get("model", r.get("arm", "?")), r["task"], r.get("enc", "?"), pid)] = r["score"]
    return best


def acc(d, model, task, enc):
    v = [s for (m, t, e, i), s in d.items() if m == model and t == task and e == enc]
    return (sum(v) / len(v)) if v else np.nan


# ================= fig0 method: gridded, padded, uniform =================
def fig_method():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.7),
                             gridspec_kw=dict(height_ratios=[1.0, 1.42], hspace=0.18))
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(0.0, 4.35, "(a) Positional record — role is a column", fontsize=8, weight="bold", color=DARK)
    for y, txt in [(3.45, "n3   r1   n17"), (2.6, "n8   r1   n3")]:
        ax.text(0.15, y, txt, family="monospace", fontsize=8.5, color=DARK,
                bbox=dict(boxstyle="round,pad=0.38", fc="#ECEFF1", ec=GREY))
    ax.text(4.7, 3.15, "role of each token = its column", fontsize=6.9, color=OI["orange"], va="center")
    ax.text(4.7, 2.6, "(1st: source, 3rd: target)", fontsize=6.9, color=OI["orange"], va="center")
    ax.text(0.15, 1.6, "Q:  is n17 an r1-neighbor of n3 ?", fontsize=7.8, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.36", fc="#E3F2FD", ec=BLUE))
    ax.text(0.15, 0.55, "Every record must be scanned and re-parsed;\nerror grows with the number of records m.",
            fontsize=7.0, color=DARK)
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6.1); ax.axis("off")
    ax.text(0.0, 5.85, "(b) Pointer record — role is content", fontsize=8, weight="bold", color=RED)
    ax.text(0.15, 5.30, "compact form v2; each record is self-contained", fontsize=6.2, color=DARK)
    recs = [("n5  -r2->  n9", "#ECEFF1", DARK),
            ("n3  -r1->  n17", "#FDECEA", RED),
            ("n3  -r1->  n8", "#FDECEA", RED)]
    for i, (t, fc, cc) in enumerate(recs):
        ax.text(0.15, 4.55 - i * 0.82, t, family="monospace", fontsize=8, color=cc,
                bbox=dict(boxstyle="round,pad=0.34", fc=fc, ec=cc))
    # right column: notes start BELOW the header line, aligned at x=6.1
    ax.text(6.1, 4.55, "the invariant is content:", fontsize=6.6, color=RED, va="center")
    ax.text(6.1, 4.05, "renames cannot touch it", fontsize=6.6, color=RED, va="center")
    ax.text(0.15, 1.78, "Q:  n3  -r1->  ?", family="monospace", fontsize=8, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.34", fc="#E3F2FD", ec=BLUE))
    ax.text(6.1, 1.78, "query = record template\nwith a hole: one\nlookup, window\n$O(1)$ in $m$",
            fontsize=6.4, color=RED, va="center")
    ax.text(0.15, 0.42, "k attention layers emulate k rounds of relational\nmessage passing (Thm. 3); lossless (Thm. 2).",
            fontsize=7.0, color=DARK)
    fig.savefig(f"{OUT}/fig0_method.pdf"); plt.close(fig)


# ================= fig_overview: gridded map =================
def fig_overview():
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis("off")
    L, W1, G, W2, W3 = 0.5, 30.5, 3.2, 33.0, 29.0
    X2 = L + W1 + G; X3 = X2 + W2 + G
    box = lambda x, y, w, h, fc, ec: ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))
    TY = 39.2  # common title baseline
    # Panel A
    box(L, 1.5, W1, 40.5, "#FAFAFA", GREY)
    ax.text(L + 1.5, TY, "① FIRST-ORDER BARRIER", fontsize=7.6, weight="bold", color=DARK)
    ax.text(L + 1.5, 36.3, "12 models · 4 families · 8 encodings", fontsize=6.4, color=DARK)
    ax.text(L + 1.5, 34.0, "local queries: 0.94–1.00", fontsize=5.8, color=GREY)
    walls = [("INFORMATION WALL", "lossy digest: divergence 0.00", OI["red"]),
             ("EXPRESSIVITY WALL", "isomorphism: 4 families fail", OI["orange"]),
             ("REALITY WALL", "synthetic 0.99 → real 0.10–0.34", OI["purple"])]
    for i, (t, d, c) in enumerate(walls):
        y = 28.8 - i * 8.8
        box(L + 1.3, y - 2.9, W1 - 2.6, 6.9, "white", c)
        ax.text(L + 2.2, y + 2.6, t, fontsize=6.6, weight="bold", color=c)
        ax.text(L + 2.2, y + 0.2, d, fontsize=5.7, color=DARK)
        ax.text(L + 2.2, y - 1.9, {"INFORMATION WALL": "GPT-4o nbr F1 0.027",
                                   "EXPRESSIVITY WALL": "$C_6$ vs $2C_3$: solved by none",
                                   "REALITY WALL": "prompt ceiling, all vendors"}[t],
                fontsize=5.7, color=GREY)
    ax.text(L + 2.2, 2.2, "R1 at 4–6k CoT tokens: 0.00", fontsize=5.7, color=GREY)
    # Panel B
    box(X2, 1.5, W2, 40.5, "#FDF7F2", RED)
    ax.text(X2 + 1.5, TY, "② 2ND-ORDER GRAMMAR", fontsize=7.6, weight="bold", color=RED)
    ax.text(X2 + 1.5, 36.3, "records carry 2-WL colors", fontsize=6.4, color=DARK)
    axioms = [("A1 lossless", "1-WL exact"), ("A2 local", "1 lookup"), ("A3 compose", "concat")]
    bw = (W2 - 2.4 - 2 * 0.7) / 3
    for i, (t, d) in enumerate(axioms):
        x = X2 + 1.2 + i * (bw + 0.7)
        box(x, 29.4, bw, 5.0, "white", RED)
        ax.text(x + 0.55, 32.9, t, fontsize=5.3, weight="bold", color=RED)
        ax.text(x + 0.55, 30.7, d, fontsize=5.0, color=DARK)
    ax.text(X2 + 1.5, 27.0, "record v2 (1.3× edge list):", fontsize=6.0, color=DARK)
    ax.text(X2 + 1.5, 24.2, "n3 -r1-> n17", family="monospace", fontsize=7.4, color=RED,
            bbox=dict(boxstyle="round,pad=0.28", fc="#FDECEA", ec=RED))
    ax.text(X2 + 12.0, 24.2, "TRI n3 -> {n8}", family="monospace", fontsize=7.4, color=RED,
            bbox=dict(boxstyle="round,pad=0.34", fc="#FDECEA", ec=RED))
    ax.text(X2 + 1.5, 20.6, "Q: n3 -r1-> ?", family="monospace", fontsize=7.4, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.28", fc="#E3F2FD", ec=BLUE))
    results = [("Thm. 2 — ", "one layer compares TRI multisets"),
               ("Thm. 3 — ", "k layers ≡ k rounds of 2-WL"),
               ("Prop. 1 — ", "each WL level is priced (level ℓ)")]
    for i, (a, b) in enumerate(results):
        y = 16.6 - i * 4.6
        ax.text(X2 + 1.2, y, a + b, fontsize=5.7, color=DARK)
    ax.text(X2 + 1.5, 4.6, "ablation: same arm, TRI removed", fontsize=6.0, color=RED)
    ax.text(X2 + 1.5, 2.5, "at test time: 1.000 → 0.000", fontsize=6.0, color=RED)
    # Panel C
    box(X3, 1.5, W3, 40.5, "#F3F8FB", BLUE)
    ax.text(X3 + 1.5, TY, "③ BARRIER LIFTED", fontsize=7.6, weight="bold", color=BLUE)
    ax.text(X3 + 1.5, 36.3, "LoRA on 0.3% of parameters", fontsize=6.4, color=DARK)
    rows = [("boundary lifted", "iso T1/T2/T3: 1.000/1.000/1.000", OI["green"]),
            ("frontier chains", "hop-3: 0.80→0.90 at 1/5 tokens", OI["blue"]),
            ("full-graph reading", "228k edges: 0.17→0.34 via retrieval", OI["sky"]),
            ("transfer", "0.91–1.00 kept across vocabularies", OI["purple"])]
    for i, (t, d, c) in enumerate(rows):
        y = 29.4 - i * 7.0
        box(X3 + 1.3, y - 2.6, W3 - 2.6, 5.9, "white", c)
        ax.text(X3 + 2.2, y + 2.2, t, fontsize=6.3, weight="bold", color=c)
        ax.text(X3 + 2.2, y - 0.1, d, fontsize=5.7, color=DARK)
    a1 = FancyArrowPatch((L + W1 + 0.5, 21.5), (X2 - 0.35, 21.5), arrowstyle="-|>",
                         color=DARK, lw=1.6, mutation_scale=11)
    a2 = FancyArrowPatch((X2 + W2 + 0.5, 21.5), (X3 - 0.35, 21.5), arrowstyle="-|>",
                         color=DARK, lw=1.6, mutation_scale=11)
    ax.add_patch(a1); ax.add_patch(a2)
    fig.savefig(f"{OUT}/fig_overview.pdf"); plt.close(fig)


# ================= fig1 walls =================
def fig_walls():
    import matplotlib.patheffects as pe
    global _HW
    _HW = [pe.withStroke(linewidth=2.6, foreground="white")]
    d = clean(f"{R}/e1_gpt4omini.jsonl")
    tasks = ["reach1", "reach2", "reach3", "nbr", "ambig"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6),
                             gridspec_kw=dict(width_ratios=[1.22, 1, 1]))
    M = np.full((len(ENCS), len(tasks)), np.nan)
    for i, e in enumerate(ENCS):
        for j, t in enumerate(tasks):
            M[i, j] = acc(d, "gpt-4o-mini", t, e)
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(["1-hop", "2-hop", "3-hop", "nbr", "ambig"], rotation=38, ha="right")
    ax.set_yticks(range(len(ENCS)))
    ax.set_yticklabels([NICE[e] + ("\n(lossy)" if e == "sog" else "") for e in ENCS], fontsize=6.8)
    ax.set_ylabel("encoding")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == M[i, j]:
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.4,
                        color="white" if v < 0.35 else "black")
    ax.axhline(4.5, color="k", lw=1.0, ls="--")
    ax.set_title("(a) Information wall\n(GPT-4o-mini)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("score", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    # (b) iso bars: legend ABOVE axes, chance label in left gutter, gaps via edges
    ax = axes[1]
    fams = [("DS-V3", f"{R}/e1_deepseek.jsonl", "deepseek-chat"),
            ("GLM-4F", f"{R}/e1_glm4flash.jsonl", "glm-4-flash"),
            ("Q-14B", f"{R}/e1_qwen14b.jsonl", "Qwen2.5-14B-Instruct"),
            ("GPT-4om", f"{R}/e1_gpt4omini.jsonl", "gpt-4o-mini")]
    data = {tag: clean(p) for tag, p, _ in fams}
    xpos = np.arange(len(fams)); w = 0.30
    for k, (tier, lab, c) in enumerate([("iso_pos", "tier 1: isomorphic", "#B0BEC5"),
                                        ("iso_sep", "tier 2: 1-WL sep.", OI["orange"])]):
        vals = []
        for tag, _, mname in fams:
            v = [acc(data[tag], mname, tier, e) for e in ENCS]
            vals.append(float(np.nanmean(v)) if not all(np.isnan(x) for x in v) else 0.0)
        ax.bar(xpos + (k - 0.5) * (w + 0.04), vals, w, label=lab, color=c,
               edgecolor="white", linewidth=0.8)
    ax.axhline(0.5, color="k", ls=":", lw=0.8, zorder=0.5)

    ax.set_xticks(xpos); ax.set_xticklabels([t for t, _, _ in fams], rotation=30, ha="right")
    ax.set_xlim(-0.55, len(fams) + 0.55)
    ax.text(len(fams) - 0.5 + 0.62, 0.5, "chance", fontsize=6, ha="left", va="center", color="k")
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy"); ax.set_xlabel("model family")
    hw1 = _HW
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="none",
              loc="upper center", fontsize=5.8, ncol=1, borderpad=0.3,
              handlelength=1.2, bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 1.24)
    ax.set_title("(b) Expressivity wall:\nisomorphism", pad=6)
    # (c) capacity scatter: deterministic dodge by encoding index, smaller markers
    ax = axes[2]
    d72 = clean(f"{R}/e1_qwen72b.jsonl"); d14 = clean(f"{R}/e1_qwen14b.jsonl"); dg = clean(f"{R}/e1_gpt4o.jsonl")
    groups = [("14B", d14, "Qwen2.5-14B-Instruct", ENCS),
              ("72B", d72, "Qwen/Qwen2.5-72B-Instruct", ENCS),
              ("GPT-4o", dg, "gpt-4o", ["edge_list", "adjacency_list", "sog", "pointer"])]
    for gi, (lab, dd, mname, encs) in enumerate(groups):
        others = [e for e in encs if e not in ("pointer", "sog")]
        for si, e in enumerate(others):  # deterministic ladder for others
            v = acc(dd, mname, "nbr", e)
            if np.isnan(v):
                continue
            ax.scatter(gi - 0.11 + si * (0.22 / max(len(others) - 1, 1)), v, s=13,
                       color=GREY, alpha=0.85, zorder=2, edgecolor="white", linewidth=0.3)
        for e, c, mk in [("pointer", RED, "D"), ("sog", DARK, "X")]:
            v = acc(dd, mname, "nbr", e)
            if np.isnan(v):
                continue
            ax.scatter(gi + (0.20 if mk == "D" else 0.34), v, s=30 if mk == "D" else 24, color=c, marker=mk,
                       zorder=3, edgecolor="white", linewidth=0.4)
    ax.scatter([], [], s=30, color=RED, marker="D", label="Pointer")
    ax.scatter([], [], s=26, color=DARK, marker="X", label="SOG (lossy)")
    ax.scatter([], [], s=16, color=GREY, label="other encodings")
    ax.set_xticks(range(3)); ax.set_xticklabels(["14B", "72B", "GPT-4o"])
    ax.set_xlim(-0.55, 2.75); ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("reader capacity"); ax.set_ylabel("nbr F1")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.00),
              ncol=3, fontsize=5.8, columnspacing=0.8, handletextpad=0.3)
    ax.set_title("(c) Capacity flips\nthe encoding ranking", pad=15)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{OUT}/fig1_walls.pdf"); plt.close(fig)


# ================= fig2 scale =================
def fig_scale():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    ax = axes[0]
    series = {10: f"{R}/ab_n10_qwen7b.jsonl", 20: f"{R}/e1_qwen7b.jsonl", 30: f"{R}/ab_n30_qwen7b.jsonl",
              50: f"{R}/big_n50_qwen7b.jsonl", 100: f"{R}/big_n100_qwen7b.jsonl"}
    ds = {n: clean(f) for n, f in series.items()}
    def val(n, e):
        v = acc(ds[n], "Qwen2.5-7B-Instruct", "reach1", e) if n in (10, 20, 30) else np.nan
        if np.isnan(v):
            s = [x for (m, t, ee, i), x in ds[n].items() if t == "reach1" and ee == e]
            v = sum(s) / len(s) if s else np.nan
        return v
    styles = {"pointer": (RED, "D", "-"), "edge_list": (GREY, "o", "--"),
              "adjacency_list": (OI["blue"], "s", "-."), "json": (OI["purple"], "^", ":"),
              "sog": (DARK, "X", "-")}
    xoff = {"pointer": 0.0, "edge_list": 0.10, "adjacency_list": -0.10,
            "json": 0.20, "sog": -0.10}
    for e, (c, mk, ls) in styles.items():
        pts = [(n, 1 - val(n, e)) for n in (10, 20, 30, 50, 100)]
        pts = [(n * (1 + xoff[e] * 0.3), y) for n, y in pts if y == y]
        if e == "adjacency_list":
            ax.plot(*zip(*pts), color=c, marker=mk, ls=ls, lw=1.4, ms=4.2,
                    label=NICE[e], markerfacecolor="none", markeredgewidth=1.2)
        else:
            ax.plot(*zip(*pts), color=c, marker=mk, ls=ls, lw=1.4, ms=3.6, label=NICE[e])
    ax.set_xscale("log"); ax.set_xlim(9, 115)
    ax.set_xticks([10, 20, 30, 50, 100]); ax.set_xticklabels([10, 20, 30, 50, 100])
    ax.minorticks_off()
    ax.set_xlabel("graph size $n$"); ax.set_ylabel("1-hop error rate")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              columnspacing=0.7, handlelength=1.4, fontsize=5.8)
    ax.set_title("(a) Frozen reading: locality = flat curve (7B)")
    ax = axes[1]
    p2rows = [r for r in rows_of(f"{R}/e3_p2.jsonl") if "ERROR" not in r.get("raw", "")]
    scrows = [r for r in rows_of(f"{R}/e3_scale.jsonl") if "ERROR" not in r.get("raw", "")]
    series_b = [
        ("1.5B·Pointer v1", [r for r in scrows if r["arm"] == "lora_pointer"], "#E8A7A5", "^", "--"),
        ("1.5B·Pointer v2", [r for r in p2rows if r["arm"] == "lora_pointer2" and r["base"] == "1.5B"], "#C62828", "s", "-"),
        ("1.5B·edge list", [r for r in scrows if r["arm"] == "lora_edge"], GREY, "o", ":"),
        ("7B·Pointer v2", [r for r in p2rows if r["arm"] == "lora_pointer2" and r["base"] == "7B"], "#7B1D1D", "D", "-"),
        ("7B·edge list", [r for r in p2rows if r["arm"] == "lora_edge" and r["base"] == "7B"], "#455A64", "o", ":")]
    for lab, rr, c, mk, ls in series_b:
        vals = defaultdict(list)
        for r in rr:
            if r["task"] == "nbr":
                vals[r["n"]].append(r["score"])
        xs = [n for n in (20, 30, 40) if vals.get(n)]
        ys = [sum(vals[n]) / len(vals[n]) for n in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=c, marker=mk, ls=ls, lw=1.4, ms=3.6)
    # aligned label column at right, no marker contact
    yslot = {lab: y for lab, y in zip([s[0] for s in series_b],
                                      [0.655, 0.615, 0.575, 0.535, 0.495])}
    for lab, rr, c, mk, ls in series_b:
        vals = defaultdict(list)
        for r in rr:
            if r["task"] == "nbr":
                vals[r["n"]].append(r["score"])
        xs = [n for n in (20, 30, 40) if vals.get(n)]
        ys = [sum(vals[n]) / len(vals[n]) for n in xs]
        if not xs:
            continue
        import matplotlib.patheffects as pe
        hw2 = [pe.withStroke(linewidth=2.6, foreground="white")]
        ax.annotate("", xy=(40.6, ys[-1]), xytext=(44.0, yslot[lab]),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.7))
        ax.text(44.4, yslot[lab], lab, fontsize=6.0, color=c, va="center",
                path_effects=hw2)
    ax.set_xticks([20, 30, 40]); ax.set_xlim(19, 58)
    ax.set_xlabel("test graph size (trained at $n{=}20$ only)")
    ax.set_ylabel("nbr F1"); ax.set_ylim(0.30, 0.74)
    ax.set_title("(b) Trained extrapolation: addressing vs. economy")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig2_scale.pdf"); plt.close(fig)


# ================= fig3 ambig =================
def fig_ambig():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    files = [("Qwen-7B", "e1_qwen7b.jsonl", OI["blue"]),
             ("DeepSeek-V3", "e1_deepseek.jsonl", OI["orange"]),
             ("GLM-4-Flash", "e1_glm4flash.jsonl", OI["purple"]),
             ("GPT-4o-mini", "e1_gpt4omini.jsonl", DARK)]
    encs = ["edge_list", "nl", "dfs_serial", "pointer", "sog"]
    w = 0.19
    for mi, (m, f, c) in enumerate(files):
        pair = defaultdict(dict)
        for r in rows_of(f"{R}/{f}"):
            if r["task"] == "ambig" and "ERROR" not in r.get("raw", ""):
                s = set(r["pred"].split(",")) if r["pred"] not in ("NONE", "NA") else set()
                pair[(r["enc"], r["meta"]["pair"])][r["meta"]["side"]] = s
        byenc = defaultdict(list)
        for (e, p), v in pair.items():
            if len(v) == 2:
                byenc[e].append(0.0 if v["A"] == v["B"] else 1.0)
        xs = np.arange(len(encs)) + (mi - 1.5) * (w + 0.015)
        ys = [np.mean(byenc[e]) if byenc.get(e) else np.nan for e in encs]
        ax.bar(xs, ys, w, label=m, color=c, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(encs))); ax.set_xticklabels([NICE[e] for e in encs], rotation=26, ha="right")
    ax.set_ylabel("answer divergence on\nambiguous pairs")
    ax.set_ylim(0, 1.30); ax.set_xlim(-0.65, len(encs) - 0.35)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=5.8, columnspacing=0.7)
    fig.savefig(f"{OUT}/fig3_ambig.pdf"); plt.close(fig)


# ================= fig4 e2 =================
def fig_e2():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    rows = [("Qwen-7B", "e2_nell_qwen7b.jsonl", "e2_fb_qwen7b.jsonl"),
            ("Qwen-14B", "e2_nell_qwen14b.jsonl", "e2_fb_qwen14b.jsonl"),
            ("Qwen-72B", "e2_nell_qwen72b.jsonl", "e2_fb_qwen72b.jsonl"),
            ("DeepSeek-V3", "e2_nell_deepseek.jsonl", None),
            ("GPT-4o-mini", "e2_nell_gpt4omini.jsonl", "e2_fb_gpt4omini.jsonl")]
    def mean(p):
        v = [r["score"] for r in rows_of(f"{R}/{p}") if "ERROR" not in r.get("raw", "")]
        return sum(v) / len(v) if v else np.nan
    names = [r[0] for r in rows]
    nell = [mean(r[1]) for r in rows]
    fbv = [mean(r[2]) if r[2] else np.nan for r in rows]
    xs = np.arange(len(names))
    ax.bar(xs - 0.185, nell, 0.34, label="NELL-995 (readable IDs)", color=OI["blue"],
           edgecolor="white", linewidth=0.5)
    fbx = [x + 0.185 for x, v in zip(xs, fbv) if not np.isnan(v)]
    fby = [v for v in fbv if not np.isnan(v)]
    ax.bar(fbx, fby, 0.34, label="FB15k-237 (opaque IDs)", color=OI["sky"],
           edgecolor="white", linewidth=0.5)
    if np.isnan(fbv[3]):
        ax.text(xs[3] + 0.185, 0.012, "n/a", ha="center", fontsize=6.4, color=DARK)
    ax.axhline(0.10, color="k", ls="--", lw=0.8, zorder=0.5)
    import matplotlib.patheffects as pe4
    hw4 = [pe4.withStroke(linewidth=2.8, foreground="white")]
    ax.annotate("chance", xy=(-0.1, 0.10), xytext=(-0.60, 0.165), fontsize=6.4,
                ha="left", path_effects=hw4,
                arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_xlim(-0.65, len(names) - 0.35)
    ax.set_ylabel("10-way tail prediction acc.")
    ax.legend(frameon=False, loc="upper left", fontsize=6.4)
    ax.set_ylim(0, 0.36)
    fig.savefig(f"{OUT}/fig4_e2.pdf"); plt.close(fig)


# ================= fig5 e3 =================
def fig_e3():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    ax = axes[0]
    arms = [("frozen / edge", "frozen", "edge_list", "#B0BEC5", ""),
            ("LoRA / edge", "lora_edge", "edge_list", OI["blue"], ""),
            ("LoRA / Pointer v1", "lora_pointer", "pointer", OI["orange"], ""),
            ("LoRA / SOG (lossy)", "lora_sog", "sog", DARK, "//")]
    tasks = ["reach1", "reach2", "reach3", "nbr"]
    xs = np.arange(len(tasks)); w = 0.19
    for k, (lab, arm, enc, c, h) in enumerate(arms):
        vals = defaultdict(list)
        for r in rows_of(f"{R}/e3_results.jsonl"):
            if r["arm"] == arm and r["enc"] == enc and "ERROR" not in r.get("raw", ""):
                vals[r["task"]].append(r["score"])
        ys = [np.mean(vals[t]) if vals[t] else np.nan for t in tasks]
        ax.bar(xs + (k - 1.5) * (w + 0.012), ys, w, label=lab, color=c, hatch=h,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(["1-hop", "2-hop", "3-hop", "nbr"])
    ax.set_ylim(0, 1.06); ax.set_ylabel("score"); ax.set_xlim(-0.6, 3.6)
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), columnspacing=0.8)
    ax.set_title("(a) Training repairs utilization,\nnot information (1.5B)", pad=14)
    ax = axes[1]
    vals = defaultdict(list)
    for r in rows_of(f"{R}/e3_real.jsonl"):
        if "ERROR" not in r.get("raw", ""):
            vals[r["arm"]].append(r["score"])
    ks = ["frozen7b_edge", "lora_edge", "lora_pointer2"]
    labs = ["frozen 7B", "LoRA / edge list", "LoRA / Pointer v2"]
    cs = ["#B0BEC5", OI["blue"], OI["orange"]]
    ys = [np.mean(vals[k]) for k in ks]
    es = [1.96 * np.std(vals[k]) / np.sqrt(len(vals[k])) for k in ks]
    ax.bar(range(3), ys, 0.5, color=cs, edgecolor="white",
           yerr=es, capsize=2.5, error_kw=dict(lw=0.8, ecolor=DARK))
    for i, v in enumerate(ys):
        ax.text(i - 0.38, v, f"{v:.3f}", ha="right", fontsize=6.6, va="center",
                color=cs[i])
    ax.axhline(0.10, color="k", ls="--", lw=0.8, zorder=0.5)
    ax.text(-0.5, 0.045, "chance", fontsize=6.2, ha="left")
    ax.axhline(0.34, color=GREY, ls=":", lw=0.9, zorder=0.5)
    ax.text(-0.5, 0.385, "best prompt-only", fontsize=6.2, ha="left", color=DARK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("NELL 10-way acc. (200 probes)")
    ax.set_ylim(0, 0.64); ax.set_xlim(-0.55, 2.75)
    ax.set_title("(b) Real-KG repair (7B, 3k probes)")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig5_e3.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_method(); fig_overview(); fig_walls(); fig_scale(); fig_ambig(); fig_e2(); fig_e3()
    print("v3 suite done")
