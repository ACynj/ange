#!/usr/bin/env python
"""Aggregate E1 jsonl results into a summary table + key separation stats."""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(paths):
    """Load rows; drop ERROR rows; deduplicate by (model,task,enc,pid)
    keeping the LAST valid occurrence (re-runs override earlier failures)."""
    best = {}
    for p in paths:
        try:
            lines = open(p).read().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ERROR" in r.get("raw", ""):
                continue
            best[(r["model"], r["task"], r["enc"], r["pid"])] = r
    return list(best.values())


def main(paths):
    rows = load(paths)
    acc = defaultdict(list)
    for r in rows:
        acc[(r["model"], r["task"], r["enc"])].append(r["score"])
    hdr = f"{'model':22s} {'task':10s} " + " ".join(f"{e:>11s}" for e in
         ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"])
    print(hdr)
    print("-" * len(hdr))
    models = sorted({m for m, _, _ in acc})
    tasks = sorted({t for _, t, _ in acc})
    for m in models:
        for t in tasks:
            cells = []
            for e in ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]:
                v = acc.get((m, t, e))
                cells.append(f"{sum(v)/len(v):11.3f}" if v else f"{'-':>11s}")
            print(f"{m:22s} {t:10s} " + " ".join(cells))

    # ambiguity diagnostic: answer divergence between pair members
    pair_ans = defaultdict(dict)
    for r in rows:
        if r["task"] == "ambig":
            pair_ans[(r["model"], r["enc"], r["meta"]["pair"])][r["meta"]["side"]] = set(
                r["pred"].split(",")) if r["pred"] not in ("NONE", "NA") else set()
    print("\nAmbiguity diagnostic (lower divergence = more label-blind):")
    for m in sorted({mm for mm, _, _ in pair_ans}):
        cells = []
        for e in ["edge_list", "nl", "sog", "dfs_serial", "pointer"]:
            divs = []
            for (mm, ee, pr), sides in pair_ans.items():
                if mm == m and ee == e and len(sides) == 2:
                    a, b = sides["A"], sides["B"]
                    divs.append(0.0 if a == b else 1.0)
            cells.append(f"{e}:{sum(divs)/len(divs):.2f}({len(divs)})" if divs else f"{e}:-")
        print(f"  {m:22s} " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
