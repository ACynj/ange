#!/usr/bin/env python
"""Nature/Science-level figure suite for ANGE paper.
Designed for IEEE TNNLS two-column format with print-quality output."""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

R = "/mnt/ssd/ange/results"
OUT = "/home/ynj/new-pro2/paper/figs"
os.makedirs(OUT, exist_ok=True)

# ═══ NATURE-LEVEL GLOBAL STYLE ═══
plt.rcParams.update({
    # Typography — large enough to read at print size
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.titlesize": 10,

    # Layout — tight, clean
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,

    # Axes — minimal, elegant
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "axes.titlecolor": "#222222",
    "axes.titleweight": "bold",
    "axes.titlepad": 6,

    # Ticks — inward, clean
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.color": "#555555",
    "ytick.color": "#555555",

    # Grid — subtle
    "axes.grid": True,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.25,
    "grid.color": "#CCCCCC",
    "grid.linestyle": "-",

    # Lines — confident
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "lines.solid_capstyle": "round",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ═══ COLOR PALETTE — vibrant, professional ═══
PAL = {
    "red": "#E63946",      # vivid red
    "blue": "#1D71B5",     # strong blue
    "green": "#2D9D78",    # emerald
    "orange": "#F77F00",   # bright orange
    "purple": "#7B2D8B",   # deep purple
    "teal": "#17A2B8",     # teal
    "dark": "#1A1A2E",     # near-black
    "grey": "#7F8C8D",     # medium grey
    "light": "#ECF0F1",    # light bg
    "gold": "#D4AC0D",     # gold
    "navy": "#1B2631",     # navy
}
RED = PAL["red"]; BLUE = PAL["blue"]; GREEN = PAL["green"]
ORANGE = PAL["orange"]; DARK = PAL["dark"]; GREY = PAL["grey"]
TEAL = PAL["teal"]; PURPLE = PAL["purple"]

HW = [pe.withStroke(linewidth=3, foreground="white")]

# Custom colormap: deep blue → white → deep red
cmap_heat = LinearSegmentedColormap.from_list(
    "professional", ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"])

ENCS = ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]
NICE = {"edge_list": "Edge list", "adjacency_list": "Adjacency", "incident_list": "Incidence",
        "nl": "Prose", "json": "JSON", "sog": "Digest", "dfs_serial": "DFS serial.",
        "pointer": "Pointer v1", "pointer2": "Pointer v2"}
TASKS5 = ["reach1", "reach2", "reach3", "nbr", "ambig"]
TN = {"reach1": "1-hop", "reach2": "2-hop", "reach3": "3-hop", "nbr": "nbr", "ambig": "ambig"}

# ═══ DATA HELPERS ═══
def rows_of(p):
    try: return [json.loads(l) for l in open(p)]
    except: return []

def clean(path):
    best = {}
    for r in rows_of(path):
        if "ERROR" in r.get("raw",""): continue
        pid = r.get("pid", r.get("arm","")+str(r.get("n",""))).split("@")[0]
        best[(r.get("model",r.get("arm","?")), r["task"], r.get("enc","?"), pid)] = r["score"]
    return best

def acc(d, m, t, e):
    v = [s for (mm,tt,ee,i),s in d.items() if mm==m and tt==t and ee==e]
    return (sum(v)/len(v)) if v else np.nan


# ═══════════════════════════════════════════════════════════════
# FIG A: Capacity Flip — the key novel finding
# ═══════════════════════════════════════════════════════════════
def fig_capacity():
    ds = {"1.5B":("e1_qwen1.5b.jsonl","Qwen2.5-1.5B-Instruct"), "3B":("e1_qwen3b.jsonl","Qwen2.5-3B-Instruct"),
          "7B":("e1_qwen7b.jsonl","Qwen2.5-7B-Instruct"), "14B":("e1_qwen14b.jsonl","Qwen2.5-14B-Instruct"),
          "72B":("e1_qwen72b.jsonl","Qwen/Qwen2.5-72B-Instruct"), "GPT-4o":("e1_gpt4o.jsonl","gpt-4o")}
    sizes = list(ds.keys())
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    series = [("adjacency_list", "Adjacency grp.", BLUE, "s", "-"),
              ("pointer", "Pointer v1", RED, "D", "-"),
              ("edge_list", "Edge list", GREY, "o", "--"),
              ("sog", "SOG digest", DARK, "X", ":")]
    for e, lab, c, mk, ls in series:
        vals = []
        for sz in sizes:
            f, m = ds[sz]
            d = clean(f"{R}/{f}")
            vals.append(acc(d, m, "nbr", e))
        ax.plot(x, vals, marker=mk, color=c, label=lab, ls=ls, ms=5,
               markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel("nbr F1", fontweight="bold")
    ax.set_xlabel("Model capacity", fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.5,
             labelspacing=0.3, borderpad=0)

    # Shaded annotation zones
    ax.axvspan(2.5, 3.5, color=BLUE, alpha=0.06, zorder=0)
    ax.axvspan(4.5, 5.5, color=RED, alpha=0.06, zorder=0)
    ax.text(3, 1.03, "adjacency\noptimal", fontsize=6, color=BLUE, ha="center",
            fontweight="bold", path_effects=HW)
    ax.text(5, 1.03, "pointer\noptimal", fontsize=6, color=RED, ha="center",
            fontweight="bold", path_effects=HW)

    fig.savefig(f"{OUT}/fig_capacity_flip.pdf")
    plt.close()
    print("✓ capacity_flip")


# ═══════════════════════════════════════════════════════════════
# FIG B: 3-model heatmap grid
# ═══════════════════════════════════════════════════════════════
def fig_heatmap():
    models = [("Qwen2.5-7B", "e1_qwen7b.jsonl", "Qwen2.5-7B-Instruct"),
              ("DeepSeek-V3", "e1_deepseek.jsonl", "deepseek-chat"),
              ("GPT-4o-mini", "e1_gpt4omini.jsonl", "gpt-4o-mini")]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), constrained_layout=True)

    for ax, (label, f, mname) in zip(axes, models):
        d = clean(f"{R}/{f}")
        M = np.full((len(ENCS), len(TASKS5)), np.nan)
        for i, e in enumerate(ENCS):
            for j, t in enumerate(TASKS5):
                M[i, j] = acc(d, mname, t, e)

        im = ax.imshow(M, cmap=cmap_heat, vmin=0, vmax=1, aspect="auto")

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i,j] == M[i,j]:
                    v = M[i,j]
                    txt_color = "white" if v < 0.25 or v > 0.85 else "#1A1A2E"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                           fontsize=6, fontweight="bold", color=txt_color)

        ax.set_xticks(range(len(TASKS5)))
        ax.set_xticklabels([TN[t] for t in TASKS5], rotation=35, ha="right")
        ax.set_yticks(range(len(ENCS)))
        ax.set_yticklabels([NICE[e] for e in ENCS], fontsize=6.5)
        ax.axhline(4.5, color="white", lw=2)
        ax.set_title(label, fontsize=8, pad=4)
        ax.grid(False)
        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, aspect=20)
    cbar.set_label("Score", fontsize=7, fontweight="bold")
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(f"{OUT}/fig_heatmap_grid.pdf")
    plt.close()
    print("✓ heatmap_grid")


# ═══════════════════════════════════════════════════════════════
# FIG C: Method schematic — clean, professional
# ═══════════════════════════════════════════════════════════════
def fig_method():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4),
                            gridspec_kw=dict(height_ratios=[1, 1.3], hspace=0.35))

    # (a) Positional
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 5.0); ax.axis("off")
    ax.text(0, 4.85, "(a) Positional record", fontsize=9, fontweight="bold", color=DARK)
    ax.text(0, 4.55, "role = column position", fontsize=7, color=GREY, style="italic")

    for y, txt in [(3.4, "n3   r1   n17"), (2.6, "n8   r1   n3")]:
        ax.text(0.15, y, txt, family="monospace", fontsize=9, color=DARK,
                bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA", ec=GREY, lw=0.6))

    ax.annotate("source", xy=(0.5, 3.4), xytext=(4.5, 3.55),
                fontsize=7, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
    ax.annotate("target", xy=(3.0, 3.4), xytext=(4.5, 2.9),
                fontsize=7, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))

    ax.text(0.15, 1.6, "Q: is n17 an r1-neighbor of n3?", fontsize=8, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB", ec=BLUE, lw=0.6))
    ax.text(0.15, 0.5, "Must scan every record\nerror grows with size", fontsize=7, color=GREY)

    # (b) Pointer
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis("off")
    ax.text(0, 5.85, "(b) Pointer record", fontsize=9, fontweight="bold", color=RED)
    ax.text(0, 5.55, "role = content token", fontsize=7, color=GREY, style="italic")

    recs = [("n5  -r2->  n9", "#F8F9FA", DARK),
            ("n3  -r1->  n17", "#FFF3E0", RED),
            ("n3  -r1->  n8", "#FFF3E0", RED)]
    for i, (t, fc, cc) in enumerate(recs):
        ax.text(0.15, 4.35 - i*0.85, t, family="monospace", fontsize=9, color=cc,
                bbox=dict(boxstyle="round,pad=0.33", fc=fc, ec=cc, lw=0.6))

    ax.annotate("direction\nis content", xy=(3.5, 3.5), xytext=(5.5, 4.3),
                fontsize=7, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))

    ax.text(0.15, 1.7, "Q:  n3  -r1->  ?", family="monospace", fontsize=9, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.33", fc="#EBF5FB", ec=BLUE, lw=0.6))
    ax.text(5.0, 1.7, "one match,\nO(1) window", fontsize=7, color=RED,
            va="center", fontweight="bold")
    ax.text(0.15, 0.5, "k layers ≡ k rounds message passing (Thm. 4)", fontsize=7, color=GREY)

    fig.savefig(f"{OUT}/fig0_method.pdf")
    plt.close()
    print("✓ method")


# ═══════════════════════════════════════════════════════════════
# FIG D: Ambiguity — clean bar chart
# ═══════════════════════════════════════════════════════════════
def fig_ambig():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    files = [("Qwen-7B","e1_qwen7b.jsonl",BLUE), ("DS-V3","e1_deepseek.jsonl",ORANGE),
             ("GLM-4F","e1_glm4flash.jsonl",PURPLE), ("GPT-4om","e1_gpt4omini.jsonl",DARK)]
    encs = ["edge_list","nl","dfs_serial","pointer","sog"]
    w = 0.19

    for mi, (m, f, c) in enumerate(files):
        pair = defaultdict(dict)
        for r in rows_of(f"{R}/{f}"):
            if r["task"]=="ambig" and "ERROR" not in r.get("raw",""):
                s = set(r["pred"].split(",")) if r["pred"] not in ("NONE","NA") else set()
                pair[(r["enc"],r["meta"]["pair"])][r["meta"]["side"]] = s
        byenc = defaultdict(list)
        for (e,p),v in pair.items():
            if len(v)==2: byenc[e].append(0.0 if v["A"]==v["B"] else 1.0)
        xs = np.arange(len(encs)) + (mi-1.5)*(w+0.012)
        ys = [np.mean(byenc[e]) if byenc.get(e) else np.nan for e in encs]
        ax.bar(xs, ys, w, label=m, color=c, edgecolor="white", lw=0.5)

    ax.axhspan(0, 0.12, color="#FADBD8", alpha=0.4, zorder=0)
    ax.set_xticks(range(len(encs)))
    ax.set_xticklabels([NICE[e] for e in encs], rotation=25, ha="right")
    ax.set_ylabel("Answer divergence", fontweight="bold")
    ax.set_title("Structural ambiguity diagnostic", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
             fontsize=6, columnspacing=0.5)

    # annotate the digest bar
    ax.text(3.5, 0.25, "blind\nzone", fontsize=6.5, color="#C0392B",
            fontweight="bold", style="italic", ha="center", path_effects=HW)

    fig.savefig(f"{OUT}/fig3_ambig.pdf")
    plt.close()
    print("✓ ambig")


# ═══════════════════════════════════════════════════════════════
# FIG E: E2 Real-KG — professional bars
# ═══════════════════════════════════════════════════════════════
def fig_e2():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    rd = [("Q-7B","e2_nell_qwen7b.jsonl","e2_fb_qwen7b.jsonl"),
          ("Q-14B","e2_nell_qwen14b.jsonl","e2_fb_qwen14b.jsonl"),
          ("Q-72B","e2_nell_qwen72b.jsonl","e2_fb_qwen72b.jsonl"),
          ("DS-V3","e2_nell_deepseek.jsonl",None),
          ("GPT-4om","e2_nell_gpt4omini.jsonl","e2_fb_gpt4omini.jsonl")]
    def mean(p):
        v=[r["score"] for r in rows_of(f"{R}/{p}") if "ERROR" not in r.get("raw","")]
        return sum(v)/len(v) if v else np.nan

    names=[r[0] for r in rd]
    nell=[mean(r[1]) for r in rd]
    fbv=[mean(r[2]) if r[2] else np.nan for r in rd]
    xs=np.arange(len(names))

    b1 = ax.bar(xs-0.18, nell, 0.34, label="NELL-999", color=BLUE,
               edgecolor="white", lw=0.5)
    fbx=[x+0.18 for x,v in zip(xs,fbv) if not np.isnan(v)]
    fby=[v for v in fbv if not np.isnan(v)]
    ax.bar(fbx, fby, 0.34, label="FB15k-237", color=TEAL, edgecolor="white", lw=0.5)
    if np.isnan(fbv[3]):
        ax.text(xs[3]+0.18, 0.015, "n/a", ha="center", fontsize=7, color=GREY, style="italic")

    ax.axhline(0.10, color=DARK, ls="--", lw=0.8, zorder=0)
    ax.text(-0.58, 0.118, "chance", fontsize=6.5, style="italic", color=DARK, path_effects=HW)

    # Ceiling annotation
    ax.axhspan(0.34, 0.60, color="#FADBD8", alpha=0.3, zorder=0)
    ax.text(2.0, 0.45, "prompt ceiling", fontsize=6.5, color="#C0392B",
            ha="center", style="italic", path_effects=HW)

    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("10-way accuracy", fontweight="bold")
    ax.set_title("Real-KG performance", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    ax.set_ylim(0, 0.55)

    fig.savefig(f"{OUT}/fig4_e2.pdf")
    plt.close()
    print("✓ e2")


# ═══════════════════════════════════════════════════════════════
# FIG F: Scale invariance + extrapolation
# ═══════════════════════════════════════════════════════════════
def fig_scale():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # (a) Frozen
    series = {10:f"{R}/ab_n10_qwen7b.jsonl", 20:f"{R}/e1_qwen7b.jsonl",
              30:f"{R}/ab_n30_qwen7b.jsonl", 50:f"{R}/big_n50_qwen7b.jsonl",
              100:f"{R}/big_n100_qwen7b.jsonl"}
    ds = {n:clean(f) for n,f in series.items()}
    def val(n,e):
        v = acc(ds[n],"Qwen2.5-7B-Instruct","reach1",e) if n in (10,20,30) else np.nan
        if np.isnan(v):
            s=[x for (m,t,ee,i),x in ds[n].items() if t=="reach1" and ee==e]
            v=sum(s)/len(s) if s else np.nan
        return v

    for e,c,mk,ls,lab in [("pointer",RED,"D","-","Pointer"),
                           ("edge_list",GREY,"o","--","Edge list"),
                           ("adjacency_list",BLUE,"s","-.","Adjacency"),
                           ("json",PURPLE,"^",":","JSON"),
                           ("sog",DARK,"X","-","Digest")]:
        pts=[(n,1-val(n,e)) for n in (10,20,30,50,100)]
        pts=[(n,y) for n,y in pts if y==y]
        ax1.plot(*zip(*pts), color=c, marker=mk, ls=ls, label=lab, ms=5,
                markeredgecolor="white", markeredgewidth=0.5)

    ax1.set_xscale("log"); ax1.set_xlim(9,115)
    ax1.set_xticks([10,20,30,50,100]); ax1.set_xticklabels([10,20,30,50,100])
    ax1.minorticks_off()
    ax1.set_xlabel("Graph size $n$", fontweight="bold")
    ax1.set_ylabel("1-hop error", fontweight="bold")
    ax1.set_title("(a) Frozen: flat = locality", fontweight="bold")
    ax1.legend(frameon=False, ncol=2, fontsize=6, loc="upper left",
              columnspacing=0.5)

    # (b) Extrapolation
    p2r = [r for r in rows_of(f"{R}/e3_p2.jsonl") if "ERROR" not in r.get("raw","")]
    scr = [r for r in rows_of(f"{R}/e3_scale.jsonl") if "ERROR" not in r.get("raw","")]
    sers = [("1.5B·v1",[r for r in scr if r["arm"]=="lora_pointer"],"#F5B7B1","^"),
            ("1.5B·v2",[r for r in p2r if r["arm"]=="lora_pointer2" and r["base"]=="1.5B"],RED,"s"),
            ("1.5B·edge",[r for r in scr if r["arm"]=="lora_edge"],GREY,"o"),
            ("7B·v2",[r for r in p2r if r["arm"]=="lora_pointer2" and r["base"]=="7B"],"#8B0000","D"),
            ("7B·edge",[r for r in p2r if r["arm"]=="lora_edge" and r["base"]=="7B"],"#34495E","o")]
    slots = [0.670, 0.600, 0.530, 0.460, 0.390]

    for (lab,rr,c,mk),slot in zip(sers,slots):
        vals=defaultdict(list)
        for r in rr:
            if r["task"]=="nbr": vals[r["n"]].append(r["score"])
        xs=[n for n in (20,30,40) if vals.get(n)]
        ys=[sum(vals[n])/len(vals[n]) for n in xs]
        if not xs: continue
        ax2.plot(xs,ys,color=c,marker=mk,ms=5,markeredgecolor="white",markeredgewidth=0.5)
        # leader line to fixed slot
        ax2.annotate("",xy=(41,ys[-1]),xytext=(43,slot),
                    arrowprops=dict(arrowstyle="-",color=c,lw=0.5,alpha=0.5))
        ax2.text(43.5,slot,lab,fontsize=6.5,color=c,va="center",
                fontweight="bold",path_effects=HW)

    ax2.set_xticks([20,30,40]); ax2.set_xlim(19,66)
    ax2.set_xlabel("Test size (trained at $n$=20)", fontweight="bold")
    ax2.set_ylabel("nbr F1", fontweight="bold")
    ax2.set_ylim(0.32, 0.72)
    ax2.set_title("(b) Trained: addressing vs. economy", fontweight="bold")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig2_scale.pdf")
    plt.close()
    print("✓ scale")


# ═══════════════════════════════════════════════════════════════
# FIG G: E3 Training — dumbbell + bars
# ═══════════════════════════════════════════════════════════════
def fig_e3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5),
                                   gridspec_kw=dict(width_ratios=[1.2,1]))

    tasks = ["reach1","reach2","reach3","nbr"]
    arms = {}
    for arm, enc in [("frozen","edge_list"),("lora_edge","edge_list"),
                     ("lora_sog","sog")]:
        vals = defaultdict(list)
        for r in rows_of(f"{R}/e3_results.jsonl"):
            if r["arm"]==arm and r["enc"]==enc and "ERROR" not in r.get("raw",""):
                vals[r["task"]].append(r["score"])
        arms[arm] = {t: np.mean(vals[t]) if vals[t] else np.nan for t in tasks}

    ypos = np.arange(len(tasks))
    for t, y in zip(tasks, ypos):
        f = arms["frozen"][t]; l = arms["lora_edge"][t]
        ax1.plot([f,l],[y,y],color=GREEN,lw=2,alpha=0.5,zorder=1)
        ax1.scatter(f,y,color=GREY,s=40,zorder=2,edgecolor="white",lw=0.8)
        ax1.scatter(l,y,color=GREEN,s=50,zorder=3,edgecolor="white",lw=0.8)
        s = arms["lora_sog"][t]
        if s==s:
            ax1.scatter(s,y,color=DARK,marker="X",s=55,zorder=3,edgecolor="white",lw=0.8)

    ax1.set_yticks(ypos); ax1.set_yticklabels([TN[t] for t in tasks])
    ax1.set_xlabel("Score", fontweight="bold")
    ax1.set_xlim(-0.05, 1.0)
    legend_el = [Line2D([0],[0],marker='o',color='w',markerfacecolor=GREY,markersize=7,label='Frozen'),
                Line2D([0],[0],marker='o',color='w',markerfacecolor=GREEN,markersize=7,label='LoRA (edge)'),
                Line2D([0],[0],marker='X',color='w',markerfacecolor=DARK,markersize=8,label='LoRA (digest)')]
    ax1.legend(handles=legend_el, frameon=False, fontsize=6.5, loc="lower right")
    ax1.set_title("(a) Training repairs utilization,\nnot information (1.5B)", fontweight="bold")

    # (b) Real-KG
    vals = defaultdict(list)
    for r in rows_of(f"{R}/e3_real.jsonl"):
        if "ERROR" not in r.get("raw",""): vals[r["arm"]].append(r["score"])
    ks=["frozen7b_edge","lora_edge","lora_pointer2"]
    labs=["Frozen","LoRA\nedge","LoRA\nPtr v2"]
    cs=[GREY,BLUE,ORANGE]
    ys=[np.mean(vals[k]) for k in ks]
    es=[1.96*np.std(vals[k])/np.sqrt(len(vals[k])) for k in ks]
    xs=np.arange(3)
    ax2.bar(xs,ys,0.5,color=cs,edgecolor="white",lw=0.5,
           yerr=es,capsize=3,error_kw=dict(lw=0.8,ecolor=DARK))
    for i,v in enumerate(ys):
        ax2.text(i,v+es[i]+0.02,f"{v:.3f}",ha="center",fontsize=7,fontweight="bold",color=cs[i])
    ax2.axhline(0.10,color=DARK,ls="--",lw=0.7,zorder=0)
    ax2.text(-0.35,0.115,"chance",fontsize=6,style="italic",color=DARK)
    ax2.set_xticks(xs); ax2.set_xticklabels(labs,fontsize=7)
    ax2.set_ylabel("NELL acc.", fontweight="bold")
    ax2.set_ylim(0,0.68)
    ax2.set_title("(b) Real-KG repair (7B)", fontweight="bold")

    fig.tight_layout(w_pad=1.4)
    fig.savefig(f"{OUT}/fig5_e3.pdf")
    plt.close()
    print("✓ e3")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_method()
    fig_ambig()
    fig_e2()
    fig_scale()
    fig_e3()
    fig_capacity()
    fig_heatmap()
    print("\n═══ All 7 figures generated ═══")
