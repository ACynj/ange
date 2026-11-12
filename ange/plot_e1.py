#!/usr/bin/env python
"""Paper figures for E1.

Fig 1 (hero): accuracy vs. encoding per task, Qwen2.5-7B panel (all sizes in supplement).
Fig 2: failure rate vs. hop k (reach1/2/3) -- the Thm-1 separation curve.
Fig 3: iso_insep bar (the predicted WL boundary: everyone ~0).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENCS = ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]
NICE = {"edge_list": "Edge list", "adjacency_list": "Adjacency", "incident_list": "Incident",
        "nl": "Natural lang.", "json": "JSON", "sog": "SOG digest", "dfs_serial": "DFS serial.",
        "pointer": "Pointer grammar (ours)"}
COL = {"pointer": "#c62828"}


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def acc_table(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[(r["model"], r["task"], r["enc"])].append(r["score"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main(paths, outdir="/mnt/ssd/ange/results/figs"):
    import os
    os.makedirs(outdir, exist_ok=True)
    rows = load(paths)
    acc = acc_table(rows)
    models = sorted({m for m, _, _ in acc})

    # Fig 1: grouped bars per task
    tasks = [t for t in ["reach1", "reach2", "reach3", "nbr", "iso_pos", "iso_sep", "iso_insep", "ambig"]
             if any(k[1] == t for k in acc)]
    for m in models:
        fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(tasks), 3.2))
        x = np.arange(len(tasks))
        w = 0.8 / len(ENCS)
        for i, e in enumerate(ENCS):
            vals = [acc.get((m, t, e), np.nan) for t in tasks]
            ax.bar(x + i * w, vals, w, label=NICE[e], color=COL.get(e, "#90a4ae"))
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(tasks, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(m)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{outdir}/fig1_{m.replace('/', '_')}.pdf")
        plt.close(fig)

    # Fig 2: failure vs k
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for e in ENCS:
        ks, fails = [], []
        for k in (1, 2, 3):
            v = acc.get((m, f"reach{k}", e)) if (m := models[-1]) else None
            if v is not None:
                ks.append(k)
                fails.append(1 - v)
        if ks:
            ax.plot(ks, fails, "o-", label=NICE[e], color=COL.get(e, None))
    ax.set_xlabel("hop k")
    ax.set_ylabel("error rate")
    ax.set_title("reachability error vs. depth")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_hop_failure.pdf")
    plt.close(fig)
    print("figures written to", outdir)


if __name__ == "__main__":
    main(sys.argv[1:])
