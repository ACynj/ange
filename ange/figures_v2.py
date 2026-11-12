#!/usr/bin/env python
"""Paper figure suite v2 — fixes from visual review (legends off data,
colorblind-safe palette, distinct markers, axis titles, clearance)."""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/mnt/ssd/ange/results"
OUT = "/home/ynj/new-pro2/paper/figs"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
# Okabe-Ito colorblind-safe
OI = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}
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


# ---------------------------------------------------------------- fig0 method
def fig_method():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5),
                             gridspec_kw=dict(height_ratios=[1.15, 1.3]))
    # (a) positional
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 4.45); ax.axis("off")
    ax.text(0.0, 3.95, "(a) Positional record — role is a column position", fontsize=8, weight="bold", color=DARK)
    for y, txt in [(3.1, "n3   r1   n17"), (2.35, "n8   r1   n3")]:
        ax.text(0.15, y, txt, family="monospace", fontsize=8.5, color=DARK,
                bbox=dict(boxstyle="round,pad=0.32", fc="#ECEFF1", ec=GREY))
    ax.text(4.6, 3.05, "role of each token = its column\n(1st: source, 3rd: target)", fontsize=6.9,
            color=OI["orange"], va="center")
    ax.text(0.15, 1.5, "Q:  is n17 an r1-neighbor of n3 ?", fontsize=7.8, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec=BLUE))
    ax.text(0.15, 0.42, "Every record must be scanned and re-parsed;\nerror grows with the number of records m.", fontsize=7.0, color=DARK)
    # (b) pointer
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 5.85); ax.axis("off")
    ax.text(0.0, 5.55, "(b) Pointer record — role is a content token", fontsize=8, weight="bold", color=RED)
    recs = [("n5  -r2->  n9", "#ECEFF1", DARK),
            ("n3  -r1->  n17", "#FDECEA", RED),
            ("n3  -r1->  n8", "#FDECEA", RED)]
    for i, (t, fc, cc) in enumerate(recs):
        ax.text(0.15, 4.55 - i * 0.95, t, family="monospace", fontsize=8, color=cc,
                bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec=cc))
    ax.text(0.15, 1.8, "Q:  n3  -r1->  ?", family="monospace", fontsize=8, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.25", fc="#E3F2FD", ec=BLUE))
    ax.text(0.15, 5.08, "(compact form v2; each record is self-contained)", fontsize=6.2, color=DARK)
    ax.text(4.4, 3.0, "the query uses the record\ntemplate with a hole (?):\nlookup is content-addressed,\nwindow $O(1)$ in $m$",
            fontsize=6.9, color=RED, va="center")
    ax.text(0.15, 0.5, "k attention layers emulate k rounds of relational\nmessage passing (Thm. 3); lossless (Thm. 2).", fontsize=7.0, color=DARK)
    fig.savefig(f"{OUT}/fig0_method.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig1 walls
def fig_walls():
    d = load_gpt()
    tasks = ["reach1", "reach2", "reach3", "nbr", "ambig"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    # (a) heatmap
    M = np.full((len(ENCS), len(tasks)), np.nan)
    for i, e in enumerate(ENCS):
        for j, t in enumerate(tasks):
            M[i, j] = acc(d, "gpt-4o-mini", t, e)
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(["1-hop", "2-hop", "3-hop", "nbr", "ambig"], rotation=38, ha="right")
    ax.set_xlabel("task")
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
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("score", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    # (b) iso tiers
    ax = axes[1]
    fams = [("DS-V3", f"{R}/e1_deepseek.jsonl", "deepseek-chat"),
            ("GLM-4F", f"{R}/e1_glm4flash.jsonl", "glm-4-flash"),
            ("Q-14B", f"{R}/e1_qwen14b.jsonl", "Qwen2.5-14B-Instruct"),
            ("GPT-4om", f"{R}/e1_gpt4omini.jsonl", "gpt-4o-mini")]
    data = {tag: clean(p) for tag, p, _ in fams}
    xpos = np.arange(len(fams)); w = 0.36
    for k, (tier, lab, c) in enumerate([("iso_pos", "tier 1: isomorphic", "#B0BEC5"),
                                        ("iso_sep", "tier 2: 1-WL separable", OI["orange"])]):
        vals = []
        for tag, _, mname in fams:
            v = [acc(data[tag], mname, tier, e) for e in ENCS]
            vals.append(float(np.nanmean(v)) if not all(np.isnan(x) for x in v) else 0.0)
        ax.bar(xpos + (k - 0.5) * w, vals, w, label=lab, color=c, edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.text(-0.42, 0.512, "chance", fontsize=6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([t for t, _, _ in fams], rotation=30, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy"); ax.set_xlabel("model family")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=6.2)
    ax.set_title("(b) Expressivity wall:\nisomorphism", pad=12)
    # (c) capacity flip
    ax = axes[2]
    d72 = clean(f"{R}/e1_qwen72b.jsonl"); d14 = clean(f"{R}/e1_qwen14b.jsonl"); dg = clean(f"{R}/e1_gpt4o.jsonl")
    groups = [("14B", d14, "Qwen2.5-14B-Instruct", ENCS),
              ("72B", d72, "Qwen/Qwen2.5-72B-Instruct", ENCS),
              ("GPT-4o", dg, "gpt-4o", ["edge_list", "adjacency_list", "sog", "pointer"])]
    rng = np.random.default_rng(3)
    for gi, (lab, dd, mname, encs) in enumerate(groups):
        for e in encs:
            v = acc(dd, mname, "nbr", e)
            if np.isnan(v):
                continue
            jm = rng.uniform(-0.13, 0.13)
            if e == "pointer":
                ax.scatter(gi + jm, v, s=34, color=RED, marker="D", zorder=3, edgecolor="white", linewidth=0.4)
            elif e == "sog":
                ax.scatter(gi + jm, v, s=30, color=DARK, marker="X", zorder=3, edgecolor="white", linewidth=0.3)
            else:
                ax.scatter(gi + jm, v, s=22, color=GREY, alpha=0.85, zorder=2)
    import matplotlib.patheffects as pe
    hw = [pe.withStroke(linewidth=2.4, foreground="white")]
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_path_effects(hw)
    leg = ax.legend(frameon=True, facecolor="white", framealpha=0.95,
                    edgecolor="none", loc="upper left", fontsize=6.3)
    for t in leg.get_texts():
        t.set_path_effects(hw)
    ax.set_ylim(0.0, 1.04)
    ax.set_xticks(range(3)); ax.set_xticklabels(["14B", "72B", "GPT-4o"])
    ax.set_xlabel("reader capacity"); ax.set_ylabel("nbr F1")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)
    ax.set_title("(c) Capacity flips\nthe encoding ranking")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{OUT}/fig1_walls.pdf"); plt.close(fig)


def load_gpt():
    return clean(f"{R}/e1_gpt4omini.jsonl")


# ---------------------------------------------------------------- fig2 scale
def fig_scale():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
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
    for e, (c, mk, ls) in styles.items():
        pts = [(n, 1 - val(n, e)) for n in (10, 20, 30, 50, 100)]
        pts = [(n, y) for n, y in pts if y == y]
        ax.plot(*zip(*pts), color=c, marker=mk, ls=ls, lw=1.4, ms=3.6, label=NICE[e])
    ax.set_xscale("log"); ax.set_xticks([10, 20, 30, 50, 100]); ax.set_xticklabels([10, 20, 30, 50, 100])
    ax.minorticks_off()
    ax.set_xlabel("graph size $n$"); ax.set_ylabel("1-hop error rate")
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="none",
              ncol=2, loc="upper left", columnspacing=0.9, handlelength=1.6, fontsize=6.4)
    ax.set_title("(a) Frozen reading: locality = flat curve (7B)")
    # (b) extrapolation
    ax = axes[1]
    # series spec: (label, rows-iterable, color, marker, linestyle)
    p2rows = [r for r in rows_of(f"{R}/e3_p2.jsonl") if "ERROR" not in r.get("raw", "")]
    scrows = [r for r in rows_of(f"{R}/e3_scale.jsonl") if "ERROR" not in r.get("raw", "")]
    series = [
        ("1.5B·Pointer v1", [r for r in scrows if r["arm"] == "lora_pointer"], "#E8A7A5", "^", "--"),
        ("1.5B·Pointer v2", [r for r in p2rows if r["arm"] == "lora_pointer2" and r["base"] == "1.5B"], "#C62828", "s", "-"),
        ("1.5B·edge list", [r for r in scrows if r["arm"] == "lora_edge"], GREY, "o", ":"),
        ("7B·Pointer v2", [r for r in p2rows if r["arm"] == "lora_pointer2" and r["base"] == "7B"], "#7B1D1D", "D", "-"),
        ("7B·edge list", [r for r in p2rows if r["arm"] == "lora_edge" and r["base"] == "7B"], "#455A64", "o", ":"),
    ]
    for lab, rr, c, mk, ls in series:
        vals = defaultdict(list)
        for r in rr:
            if r["task"] == "nbr":
                vals[r["n"]].append(r["score"])
        xs = [n for n in (20, 30, 40) if vals.get(n)]
        ys = [sum(vals[n]) / len(vals[n]) for n in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=c, marker=mk, ls=ls, lw=1.4, ms=3.6)
        ylab_off = {"1.5B·Pointer v1": -7, "1.5B·Pointer v2": 0,
                    "1.5B·edge list": -14, "7B·Pointer v2": 7, "7B·edge list": 14}[lab]
        ax.annotate(lab, (xs[-1], ys[-1]), xytext=(5, ylab_off), textcoords="offset points",
                    fontsize=6.2, color=c, va="center")
    ax.set_xticks([20, 30, 40]); ax.set_xlim(19, 56)
    ax.set_xlabel("test graph size (trained at $n{=}20$ only)")
    ax.set_ylabel("nbr F1"); ax.set_ylim(0.30, 0.74)
    ax.set_title("(b) Trained extrapolation: addressing vs. economy")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig2_scale.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig3 ambig
def fig_ambig():
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    files = [("Qwen-7B", "e1_qwen7b.jsonl", OI["blue"]),
             ("DeepSeek-V3", "e1_deepseek.jsonl", OI["orange"]),
             ("GLM-4-Flash", "e1_glm4flash.jsonl", OI["purple"]),
             ("GPT-4o-mini", "e1_gpt4omini.jsonl", DARK)]
    encs = ["edge_list", "nl", "dfs_serial", "pointer", "sog"]
    w = 0.2
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
        xs = np.arange(len(encs)) + (mi - 1.5) * w
        ys = [np.mean(byenc[e]) if byenc.get(e) else np.nan for e in encs]
        ax.bar(xs, ys, w, label=m, color=c, edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(encs))); ax.set_xticklabels([NICE[e] for e in encs], rotation=26, ha="right")
    ax.set_ylabel("answer divergence on\nambiguous pairs")
    ax.set_ylim(0, 1.35)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.0, columnspacing=0.8)
    fig.savefig(f"{OUT}/fig3_ambig.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig4 e2
def fig_e2():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
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
    ax.bar(xs - 0.19, nell, 0.38, label="NELL-995 (readable IDs)", color=OI["blue"], edgecolor="white", linewidth=0.4)
    fbx = [x + 0.19 for x, v in zip(xs, fbv) if not np.isnan(v)]
    fby = [v for v in fbv if not np.isnan(v)]
    ax.bar(fbx, fby, 0.38, label="FB15k-237 (opaque IDs)", color=OI["sky"], edgecolor="white", linewidth=0.4)
    if np.isnan(fbv[3]):
        ax.text(xs[3] + 0.19, 0.012, "n/a", ha="center", fontsize=6.4, color=DARK, rotation=90)
    ax.axhline(0.10, color="k", ls="--", lw=0.8)
    ax.text(len(names) - 0.55, 0.108, "chance", fontsize=6.4)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_ylabel("10-way tail prediction acc.")
    ax.legend(frameon=False, loc="upper left", fontsize=6.4)
    ax.set_ylim(0, 0.34)
    fig.savefig(f"{OUT}/fig4_e2.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig5 e3
def fig_e3():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    ax = axes[0]
    arms = [("frozen / edge", "frozen", "edge_list", "#B0BEC5", ""),
            ("LoRA / edge", "lora_edge", "edge_list", OI["blue"], ""),
            ("LoRA / Pointer v1", "lora_pointer", "pointer", OI["orange"], ""),
            ("LoRA / SOG (lossy)", "lora_sog", "sog", DARK, "//")]
    tasks = ["reach1", "reach2", "reach3", "nbr"]
    xs = np.arange(len(tasks)); w = 0.2
    for k, (lab, arm, enc, c, h) in enumerate(arms):
        vals = defaultdict(list)
        for r in rows_of(f"{R}/e3_results.jsonl"):
            if r["arm"] == arm and r["enc"] == enc and "ERROR" not in r.get("raw", ""):
                vals[r["task"]].append(r["score"])
        ys = [np.mean(vals[t]) if vals[t] else np.nan for t in tasks]
        ax.bar(xs + (k - 1.5) * w, ys, w, label=lab, color=c, hatch=h,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(["1-hop", "2-hop", "3-hop", "nbr"])
    ax.set_ylim(0, 1.0); ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper right")
    ax.set_title("(a) Training repairs utilization,\nnot information (1.5B)")
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
    ax.bar(range(3), ys, 0.55, color=cs, edgecolor="white",
           yerr=es, capsize=2.5, error_kw=dict(lw=0.8, ecolor=DARK))
    for i, v in enumerate(ys):
        ax.text(i, v + es[i] + 0.018, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0.10, color="k", ls="--", lw=0.8)
    ax.text(2.42, 0.112, "chance", fontsize=6.2)
    ax.axhline(0.34, color=GREY, ls=":", lw=0.9)
    ax.text(1.62, 0.352, "best prompt-only (any family)", fontsize=6.2, color=DARK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("NELL 10-way acc. (200 probes)")
    ax.set_ylim(0, 0.64)
    ax.set_title("(b) Real-KG repair (7B, 3k probes)")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig5_e3.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_method(); fig_walls(); fig_scale(); fig_ambig(); fig_e2(); fig_e3()
    print("all v2 figures done")
