#!/usr/bin/env python
"""Final table-fill: R1 complete row + seed-replicate std, printed as LaTeX-ready."""
import json, statistics, sys
from collections import defaultdict

R = "/mnt/ssd/ange/results"


def clean(path):
    best = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "ERROR" in r.get("raw", ""):
            continue
        pid = r.get("pid", r.get("arm", "") + str(r.get("n", ""))).split("@")[0]
        best[(r["task"], r["enc"], pid)] = r["score"]
    return best


tasks = ["reach1", "reach2", "reach3", "nbr", "ambig"]
encs = ["edge_list", "adjacency_list", "incident_list", "nl", "json", "sog", "dfs_serial", "pointer"]

d = clean(f"{R}/e1_r1.jsonl")
print("R1 complete row (5-task means):")
row = []
for e in encs:
    vals = []
    for t in tasks:
        v = [s for (tt, ee, i), s in d.items() if tt == t and ee == e]
        vals.append(sum(v) / len(v) if v else None)
    good = [x for x in vals if x is not None]
    m = sum(good) / len(good) if good else None
    row.append(f"{m:.2f}" if m is not None else "--")
print(" & ".join(row), "| overall", f"{sum(float(x) for x in row if x != '--')/len([x for x in row if x != '--']):.2f}")
print("n per enc:", {e: len([1 for (t, ee, i) in d if ee == e]) // 5 for e in encs})

# seed replicates
rows = [json.loads(l) for l in open(f"{R}/seed_replicates.jsonl")]
per = defaultdict(lambda: defaultdict(list))
for r in rows:
    per[r["arm"]][r["task"]].append(r["score"])
# original arms from e3_results
orig = [json.loads(l) for l in open(f"{R}/e3_results.jsonl")]
for r in orig:
    per[{"lora_edge": "edge_s1", "lora_sog": "sog_s1", "lora_pointer": "ptr_s1"}[r["arm"]]][r["task"]].append(r["score"])

print("\nSeed replicates (nbr F1 / reach1 / reach3, mean±std across seeds):")
for fam, arms in [("edge", ["edge_s1", "edge_s2", "edge_s3"]),
                  ("digest", ["sog_s1", "sog_s2"]),
                  ("pointer", ["ptr_s1", "ptr_s2"])]:
    for t in ["nbr", "reach1", "reach3"]:
        means = []
        for a in arms:
            v = per.get(a, {}).get(t)
            if v:
                means.append(sum(v) / len(v))
        if len(means) >= 2:
            print(f"  {fam:8s} {t:7s} {statistics.mean(means):.3f} ± {statistics.stdev(means):.3f}  (n_seeds={len(means)})")
