#!/usr/bin/env python
"""Paper-grade figure suite for ANGE (IEEE single/double column, 8-9pt)."""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/mnt/ssd/ange/results"
OUT = "/home/ynj/new-pro2/paper/figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
RED = "#c62828"; GREY = "#90a4ae"; DARK = "#37474f"; BLUE = "#1565c0"
ENCS = ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]
NICE = {"edge_list": "Edge list", "adjacency_list": "Adjacency", "incident_list": "Incident",
        "nl": "Prose (NL)", "json": "JSON", "sog": "SOG digest", "dfs_serial": "DFS serial.",
        "pointer": "Pointer v1", "pointer2": "Pointer v2"}
C_ENC = {e: GREY for e in ENCS}
C_ENC["sog"] = DARK; C_ENC["pointer"] = RED; C_ENC["pointer2"] = "#e57373"


def load(path):
    best = {}
    try:
        for line in open(path):
            try: r = json.loads(line)
            except Exception: continue
            if "ERROR" in r.get("raw", ""): continue
            key = (r.get("model", r.get("arm", "?")), r["task"], r.get("enc", "?"), r.get("pid", r.get("arm","") + str(r.get("n",""))).split("@")[0])
            best[key] = r["score"]
    except OSError:
        pass
    return best


def acc(d, model, task, enc):
    v = [s for (m, t, e, i), s in d.items() if m == model and t == task and e == enc]
    return (sum(v) / len(v)) if v else np.nan


# ---------------------------------------------------------------- Fig 0: method
def fig_method():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.3))
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(0, 4.15, "(a) Positional record: role = column position", fontsize=8, weight="bold", color=DARK)
    ax.text(0.1, 3.55, "n3   r1   n17", family="monospace", fontsize=8.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.32", fc="#eceff1", ec=GREY))
    ax.annotate("role of n3 depends on\nits column (position)", xy=(1.15, 3.42), xytext=(3.4, 3.7),
                fontsize=6.8, color=DARK, arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax.text(0.1, 2.55, "n8   r1   n3", family="monospace", fontsize=8.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.32", fc="#eceff1", ec=GREY))
    ax.text(0.1, 1.15, "query:  is n17 an r1-neighbor of n3 ?", fontsize=7.6, color=BLUE)
    ax.text(0.1, 0.55, "reader must scan all records and parse\npositions every time  →  error grows with n",
            fontsize=7.2, color=DARK)

    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(0, 4.35, "(b) Pointer record: role = content token", fontsize=8, weight="bold", color=RED)
    rows = [("EDGE  rel=r2  src=n5  tgt=n9", "#eceff1", DARK),
            ("EDGE  rel=r1  src=n3  tgt=n17", "#ffebee", RED),
            ("EDGE  rel=r1  src=n3  tgt=n8", "#ffebee", RED)]
    for i, (t, fc, cc) in enumerate(rows):
        ax.text(0.1, 3.6 - i * 0.62, t, family="monospace", fontsize=8, color=cc,
                bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=cc))
    ax.text(0.1, 1.45, "QUERY:  rel=r1  src=n3  tgt=?", family="monospace", fontsize=8, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec=BLUE))
    ax.annotate("one content-addressed lookup\n(window indep. of n)", xy=(4.4, 1.45), xytext=(5.0, 2.4),
                fontsize=6.8, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.text(0.1, 0.5, "k attention layers ≡ k rounds of message passing (Thm. 3)", fontsize=7.2, color=DARK)
    fig.savefig(f"{OUT}/fig0_method.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 1: walls
def fig_walls():
    d = load(f"{R}/e1_gpt4omini.jsonl")
    tasks = ["reach1", "reach2", "reach3", "nbr", "ambig"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    # (a) information wall: GPT-4o-mini heatmap
    M = np.full((len(ENCS), len(tasks)), np.nan)
    for i, e in enumerate(ENCS):
        for j, t in enumerate(tasks):
            M[i, j] = acc(d, "gpt-4o-mini", t, e)
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(["1-hop", "2-hop", "3-hop", "nbr", "ambig"], rotation=30, ha="right")
    ax.set_yticks(range(len(ENCS))); ax.set_yticklabels([NICE[e] for e in ENCS])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.axhline(5 - .5, color="k", lw=1.0, ls="--")
    ax.text(4.35, 5.0, "lossy", fontsize=6.5, va="center")
    ax.set_title("(a) Information wall (GPT-4o-mini)")
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    # (b) expressivity wall: iso across families
    ax = axes[1]
    fams = {"DeepSeek-V3": f"{R}/e1_deepseek.jsonl", "GLM-4-Flash": f"{R}/e1_glm4flash.jsonl",
            "Qwen-14B": f"{R}/e1_qwen14b.jsonl", "GPT-4o-mini": f"{R}/e1_gpt4omini.jsonl"}
    data = {m: load(f) for m, f in fams.items()}
    xpos = np.arange(len(fams)); w = 0.27
    for k, (tier, lab) in enumerate([("iso_pos", "tier 1 (isomorphic)"), ("iso_sep", "tier 2 (1-WL sep.)")]):
        vals = []
        for m in fams:
            v = [acc(data[m], m, tier, e) for e in ENCS if not np.isnan(acc(data[m], m, tier, e))]
            vals.append(np.nanmean(v))
        ax.bar(xpos + (k - 1) * w, vals, w, label=lab, color=[GREY, "#455a64"][k])
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.text(3.42, 0.515, "chance", fontsize=6)
    ax.set_xticks(xpos); ax.set_xticklabels(["DS-V3", "GLM-4F", "Qwen-14B", "GPT-4o-m"], fontsize=7)
    ax.set_ylim(0, 1.05); ax.legend(loc="upper left", frameon=False)
    ax.set_title("(b) Expressivity wall: isomorphism")

    # (c) capacity flip: nbr at 14B vs 72B vs GPT-4o
    ax = axes[2]
    d72 = load(f"{R}/e1_qwen72b.jsonl"); d14 = load(f"{R}/e1_qwen14b.jsonl"); dg = load(f"{R}/e1_gpt4o.jsonl")
    pts = []
    for label, dd, m, encs in [("Qwen-14B", d14, "Qwen2.5-14B-Instruct", ENCS),
                                ("Qwen-72B", d72, "Qwen/Qwen2.5-72B-Instruct", ENCS)]:
        for e in ENCS:
            v = acc(dd, m, "nbr", e)
            if not np.isnan(v):
                pts.append((e, label, v))
    for e, lab, v in pts:
        ax.scatter(["14B", "72B"]["72B" in lab], v, s=26,
                   color=C_ENC.get(e, GREY), marker="o" if "pointer" not in e else "D",
                   zorder=3 if "pointer" in e else 2)
    # GPT-4o core subset encodings
    for e in ["edge_list", "adjacency_list", "sog", "pointer"]:
        v = acc(dg, "gpt-4o", "nbr", e)
        if not np.isnan(v):
            ax.scatter("GPT-4o", v, s=30, color=C_ENC.get(e, GREY), marker="o" if e != "pointer" else "D")
    for e, c in [("sog", DARK), ("pointer", RED)]:
        ax.scatter([], [], color=c, marker="D" if e == "pointer" else "o", s=24, label=NICE[e])
    ax.set_ylabel("nbr F1"); ax.legend(frameon=False, loc="lower right")
    ax.set_title("(c) Capacity flips the ranking")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(f"{OUT}/fig1_walls.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: scale
def fig_scale():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    # (a) frozen read regime n=10..100
    d7 = load(f"{R}/e1_qwen7b.jsonl")
    dn50 = load(f"{R}/big_n50_qwen7b.jsonl"); dn100 = load(f"{R}/big_n100_qwen7b.jsonl")
    ax = axes[0]
    series = {10: f"{R}/ab_n10_qwen7b.jsonl", 20: f"{R}/e1_qwen7b.jsonl",
              30: f"{R}/ab_n30_qwen7b.jsonl", 50: f"{R}/big_n50_qwen7b.jsonl",
              100: f"{R}/big_n100_qwen7b.jsonl"}
    ds = {n: load(f) for n, f in series.items()}
    for e in ["pointer", "edge_list", "adjacency_list", "json", "sog"]:
        xs, ys = [], []
        for n in (10, 20, 30, 50, 100):
            v = acc(ds[n], "Qwen2.5-7B-Instruct", "reach1", e) if n in (10, 20, 30) else np.nanmean(
                [s for (m, t, ee, i), s in ds[n].items() if t == "reach1" and ee == e])
            if v == v: xs.append(n); ys.append(1 - v)
        st = dict(color=RED, marker="D", lw=1.6) if e == "pointer" else dict(color=GREY, marker="o", lw=1.0)
        if e == "sog": st = dict(color=DARK, marker="s", lw=1.0)
        ax.plot(xs, ys, label=NICE[e], **st)
    ax.set_xscale("log"); ax.set_xticks([10, 20, 30, 50, 100]); ax.set_xticklabels([10, 20, 30, 50, 100])
    ax.set_xlabel("graph size n (log)"); ax.set_ylabel("1-hop error rate")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("(a) Frozen reading: locality = flat curve")
    # (b) trained extrapolation (nbr F1)
    ax = axes[1]
    p2 = load(f"{R}/e3_p2.jsonl")
    arms = [("1.5B + Pointer v1", "lora_pointer", "pointer", "1.5B", "#ef9a9a", "^"),
            ("1.5B + Pointer v2", "lora_pointer2", "pointer2", "1.5B", RED, "D"),
            ("1.5B + edge list", "lora_edge", "edge_list", "1.5B", GREY, "o"),
            ("7B + Pointer v2", "lora_pointer2", "pointer2", "7B", "#8e0000", "D"),
            ("7B + edge list", "lora_edge", "edge_list", "7B", "#546e7a", "o")]
    for lab, arm, enc, base, c, mk in arms:
        xs, ys = [], []
        for n in (20, 30, 40):
            v = [s for (m, t, e, i), s in p2.items()
                 if m == arm and t == "nbr" and e == enc and True] if False else None
            # e3_p2 stores base in 'base' field merged into model key? use raw rows instead
        # simpler: read raw
        vals = defaultdict(list)
        for line in open(f"{R}/e3_p2.jsonl"):
            r = json.loads(line)
            if r["arm"] == arm and r["task"] == "nbr" and r["enc"] == enc and r["base"] == base and "ERROR" not in r.get("raw", ""):
                vals[r["n"]].append(r["score"])
        xs = [n for n in (20, 30, 40) if vals.get(n)]
        ys = [sum(vals[n]) / len(vals[n]) for n in xs]
        ax.plot(xs, ys, marker=mk, color=c, lw=1.4, label=lab)
    ax.set_xticks([20, 30, 40]); ax.set_xlabel("test-time graph size (trained: n=20)")
    ax.set_ylabel("nbr F1"); ax.set_ylim(0.3, 0.75)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2)
    ax.set_title("(b) Trained extrapolation: economy trade-off")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(f"{OUT}/fig2_scale.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 3: ambig
def fig_ambig():
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    files = {"Qwen-7B": "e1_qwen7b.jsonl", "DeepSeek-V3": "e1_deepseek.jsonl",
             "GLM-4-Flash": "e1_glm4flash.jsonl", "GPT-4o-mini": "e1_gpt4omini.jsonl"}
    encs = ["edge_list", "nl", "dfs_serial", "pointer", "sog"]
    w = 0.2
    for mi, (m, f) in enumerate(files.items()):
        pair = defaultdict(dict)
        for line in open(f"{R}/{f}"):
            r = json.loads(line)
            if r["task"] == "ambig" and "ERROR" not in r.get("raw", ""):
                s = set(r["pred"].split(",")) if r["pred"] not in ("NONE", "NA") else set()
                pair[(r["enc"], r["meta"]["pair"])][r["meta"]["side"]] = s
        byenc = defaultdict(list)
        for (e, p), v in pair.items():
            if len(v) == 2:
                byenc[e].append(0.0 if v["A"] == v["B"] else 1.0)
        xs = np.arange(len(encs)) + (mi - 1.5) * w
        ys = [np.mean(byenc[e]) if byenc.get(e) else np.nan for e in encs]
        ax.bar(xs, ys, w, label=m, color=[GREY, "#78909c", "#546e7a", "#455a64", RED][mi])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(encs))); ax.set_xticklabels([NICE[e] for e in encs], rotation=28, ha="right")
    ax.set_ylabel("answer divergence\non ambiguous pairs")
    ax.legend(frameon=False, ncol=2, fontsize=6.4)
    fig.savefig(f"{OUT}/fig3_ambig.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 4: E2
def fig_e2():
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    rows = [("Qwen-7B", "e2_nell_qwen7b.jsonl", "e2_fb_qwen7b.jsonl"),
            ("Qwen-14B", "e2_nell_qwen14b.jsonl", "e2_fb_qwen14b.jsonl"),
            ("Qwen-72B", "e2_nell_qwen72b.jsonl", "e2_fb_qwen72b.jsonl"),
            ("DeepSeek-V3", "e2_nell_deepseek.jsonl", None),
            ("GPT-4o-mini", "e2_nell_gpt4omini.jsonl", "e2_fb_gpt4omini.jsonl")]
    names, nell, fb = [], [], []
    for nm, fn, ff in rows:
        def mean(p):
            try:
                v = [json.loads(l)["score"] for l in open(f"{R}/{p}") if "ERROR" not in json.loads(l).get("raw", "")]
                return sum(v) / len(v) if v else np.nan
            except OSError:
                return np.nan
        names.append(nm); nell.append(mean(fn)); fb.append(mean(ff) if ff else np.nan)
    xs = np.arange(len(names))
    ax.bar(xs - 0.19, nell, 0.38, label="NELL-995 (readable IDs)", color=BLUE)
    ax.bar(xs[:len(fb)] + 0.19, [0 if np.isnan(v) else v for v in fb], 0.38,
           label="FB15k-237 (opaque IDs)", color="#90caf9")
    ax.axhline(0.10, color="k", ls="--", lw=0.8)
    ax.text(len(names) - 0.5, 0.105, "chance", fontsize=6.4)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_ylabel("10-way lp accuracy")
    ax.legend(frameon=False, fontsize=6.6)
    fig.savefig(f"{OUT}/fig4_e2.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 5: E3
def fig_e3():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    # (a) four arms @ n=20
    ax = axes[0]
    arms = [("frozen / edge", "frozen", "edge_list", GREY),
            ("LoRA / edge", "lora_edge", "edge_list", "#546e7a"),
            ("LoRA / Pointer v1", "lora_pointer", "pointer", "#ef9a9a"),
            ("LoRA / sog (lossy)", "lora_sog", "sog", DARK)]
    tasks = ["reach1", "reach2", "reach3", "nbr"]
    xs = np.arange(len(tasks))
    w = 0.2
    for k, (lab, arm, enc, c) in enumerate(arms):
        vals = defaultdict(list)
        for line in open(f"{R}/e3_results.jsonl"):
            r = json.loads(line)
            if r["arm"] == arm and r["enc"] == enc and "ERROR" not in r.get("raw", ""):
                vals[r["task"]].append(r["score"])
        ys = [np.mean(vals[t]) if vals[t] else np.nan for t in tasks]
        ax.bar(xs + (k - 1.5) * w, ys, w, label=lab, color=c)
    ax.set_xticks(xs); ax.set_xticklabels(["1-hop", "2-hop", "3-hop", "nbr"])
    ax.set_ylim(0, 1.0); ax.legend(frameon=False, fontsize=6.2, ncol=2)
    ax.set_title("(a) Training repairs utilization,\nnot information (1.5B)")
    # (b) real-KG repair: NELL-995 and FB15k-237 (opaque MIDs)
    ax = axes[1]

    def load_arms(path):
        vals = {}
        for line in open(path):
            r = json.loads(line)
            if "ERROR" not in r.get("raw", ""):
                vals.setdefault(r["arm"], []).append(r["score"])
        return vals

    ks = ["frozen7b_edge", "lora_edge", "lora_pointer2"]
    rng = np.random.default_rng(0)

    def ci95(scores, B=10000):
        """Asymmetric bootstrap percentile CI (2.5/97.5) around the mean."""
        a = np.asarray(scores, float); n = len(a)
        means = a[rng.integers(0, n, size=(B, n))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return np.mean(a) - lo, hi - np.mean(a)

    groups = [("NELL-995", load_arms(f"{R}/e3_real.jsonl"), 0.0),
              ("FB15k-237 (MID)", load_arms(f"{R}/e3_real_fb.jsonl"), 3.9)]
    for gname, vals, x0 in groups:
        ys = [np.mean(vals[k]) for k in ks]
        yerr = np.array([ci95(vals[k]) for k in ks]).T
        ax.bar(x0 + np.arange(3), ys, 0.5, yerr=yerr,
               error_kw=dict(lw=0.8, capsize=2), color=[GREY, "#546e7a", RED])
        for i, v in enumerate(ys):
            ax.text(x0 + i, v + yerr[1, i] + 0.015, f"{v:.3f}", ha="center", fontsize=7)
        ax.text(x0 + 1, 0.75, gname, ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax.axhline(0.10, color="k", ls="--", lw=0.8); ax.text(2.72, 0.118, "chance", fontsize=6.4)
    ax.set_xticks(np.concatenate([np.arange(3), 3.9 + np.arange(3)]))
    ax.set_xticklabels(["frozen", "LoRA edge", "Ptr v2"] * 2, fontsize=6.2)
    ax.set_ylabel("Real-KG lp accuracy"); ax.set_ylim(0, 0.80)
    ax.set_xlim(-0.6, 6.6)
    ax.set_title("(b) Real-KG repair (7B, 3k probes)")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(f"{OUT}/fig5_e3.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_method(); print("fig0 ok")
    fig_walls(); print("fig1 ok")
    fig_scale(); print("fig2 ok")
    fig_ambig(); print("fig3 ok")
    fig_e2(); print("fig4 ok")
    fig_e3(); print("fig5 ok")
