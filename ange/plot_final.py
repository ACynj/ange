#!/usr/bin/env python
"""Paper-grade figures for ANGE E1/E2 results.

Fig 1 (hero): encoding x task heatmap at 7B with all-model supplement.
Fig 2: scale invariance of pointer (reach1 error vs graph size).
Fig 3: ambiguity diagnostic (divergence by encoding, sog = 0).
Fig 4: E2 real-KG bars (NELL vs FB, semantic channel on/off).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/mnt/ssd/ange/results"
OUT = f"{R}/figs"
os.makedirs(OUT, exist_ok=True)

ENCS = ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]
NICE = {"edge_list": "Edge list", "adjacency_list": "Adjacency grp.", "incident_list": "Incident grp.",
        "nl": "Natural lang.", "json": "JSON", "sog": "SOG digest (lossy)", "dfs_serial": "DFS serial.",
        "pointer": "Pointer grammar (ours)"}
TASKS = ["reach1", "reach2", "reach3", "nbr", "ambig"]
TNICE = {"reach1": "1-hop reach", "reach2": "2-hop reach", "reach3": "3-hop reach",
         "nbr": "Neighbor enum.", "ambig": "Ambiguity F1", "iso": "Isomorphism"}


def load(*fs):
    rows = []
    for f in fs:
        p = f if f.startswith("/") else f"{R}/{f}"
        if os.path.exists(p):
            for line in open(p):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def acc(rows):
    a = defaultdict(list)
    for r in rows:
        a[(r["task"], r["enc"])].append(r["score"])
    return {k: float(np.mean(v)) for k, v in a.items()}


def fig_heatmap():
    rows = load("e1_qwen7b.jsonl")
    iso = load("e1b_iso_qwen7b.jsonl")
    a = acc(rows)
    ai = acc(iso)
    M = np.full((len(ENCS), len(TASKS) + 1), np.nan)
    for i, e in enumerate(ENCS):
        for j, t in enumerate(TASKS):
            M[i, j] = a.get((t, e), np.nan)
        pos = ai.get(("iso_pos", e), np.nan); sep = ai.get(("iso_sep", e), np.nan)
        if pos == pos and sep == sep:
            M[i, -1] = (pos + sep) / 2
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TASKS) + 1))
    ax.set_xticklabels([TNICE[t] for t in TASKS] + ["Isomorphism"], rotation=20, ha="right")
    ax.set_yticks(range(len(ENCS)))
    ax.set_yticklabels([NICE[e] for e in ENCS])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Qwen2.5-7B: score by encoding x task")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_heatmap_7b.pdf")
    plt.close(fig)


def fig_scale():
    rows = {n: load(f"ab_n{n}_qwen7b.jsonl") if n != 20 else load("e1_qwen7b.jsonl") for n in (10, 20, 30)}
    a = {n: acc(rs) for n, rs in rows.items()}
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for e in ["pointer", "edge_list", "json", "nl"]:
        xs, ys = [], []
        for n in (10, 20, 30):
            v = a[n].get(("reach1", e))
            if v is not None:
                xs.append(n); ys.append(1 - v)
        style = dict(color="#c62828", marker="o", lw=2) if e == "pointer" else dict(color="#90a4ae", marker="s")
        ax.plot(xs, ys, label=NICE[e], **style)
    ax.set_xlabel("graph size (nodes)")
    ax.set_ylabel("1-hop error rate")
    ax.set_xticks([10, 20, 30])
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_scale_invariance.pdf")
    plt.close(fig)


def fig_ambig():
    rows = load("e1_qwen7b.jsonl")
    pair = defaultdict(dict)
    for r in rows:
        if r["task"] == "ambig":
            s = set(r["pred"].split(",")) if r["pred"] not in ("NONE", "NA") else set()
            pair[(r["enc"], r["meta"]["pair"])][r["meta"]["side"]] = s
    encs = ["edge_list", "nl", "sog", "dfs_serial", "pointer"]
    divs = {}
    by_enc = defaultdict(dict)
    for (e, p), v in pair.items():
        by_enc[e][p] = v
    for e in encs:
        vals = [0.0 if v.get("A") == v.get("B") else 1.0 for v in by_enc[e].values() if len(v) == 2]
        divs[e] = np.mean(vals) if vals else np.nan
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    xs = range(len(encs))
    ax.bar(xs, [divs.get(e, np.nan) for e in encs], color=["#90a4ae" if e != "sog" else "#c62828" for e in encs])
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[e].replace(" (lossy)", "\n(lossy)") for e in encs], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("answer divergence on\nstructurally ambiguous pairs")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_ambiguity.pdf")
    plt.close(fig)


def fig_e2():
    en = ["edge_list", "adjacency_list", "nl", "json", "dfs_serial", "pointer"]
    models = ["1.5b", "3b", "7b", "14b"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.12
    for di, ds in enumerate(["nell", "fb"]):
        for mi, m in enumerate(models):
            rows = load(f"e2_{ds}_qwen{m}.jsonl")
            if not rows:
                continue
            accs = [np.mean([r["score"] for r in rows if r["enc"] == e]) for e in en]
            x = np.arange(len(en)) + (di * (len(models) + 1) + mi) * w
            ax.bar(x, accs, w, label=f"Qwen {m}" if di == 0 else None,
                   color=plt.cm.viridis(mi / 4), alpha=1.0 if ds == "nell" else 0.45)
    ax.axhline(0.1, color="k", ls="--", lw=0.8)
    ax.text(5.4, 0.105, "chance", fontsize=7)
    ax.set_xticks(np.arange(len(en)) + (len(models) - 0.5) * w)
    ax.set_xticklabels([NICE[e] for e in en], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("lp_mc accuracy")
    ax.legend(fontsize=7, ncol=4)
    ax.set_title("NELL-995 (solid) vs FB15k-237 (faded, unreadable entity IDs)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_e2_realkg.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_heatmap(); print("fig1 ok")
    fig_scale(); print("fig2 ok")
    fig_ambig(); print("fig3 ok")
    fig_e2(); print("fig4 ok")
