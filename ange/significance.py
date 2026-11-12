#!/usr/bin/env python
"""Paired bootstrap significance tests for the paper's core comparisons.

For each (model, task) we test encoding pairs on the SAME probe ids:
  pointer vs edge_list          (addressing benefit)
  sog vs best lossless          (information wall)
Reports mean diff, 95% CI, and two-sided p-value (10k resamples).
Also tests E3 LoRA-before/after on matched ids.
"""
import json
import sys
from collections import defaultdict
import random

R = "/mnt/ssd/ange/results"


def load_clean(path):
    best = {}
    try:
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ERROR" in r.get("raw", ""):
                continue
            key = (r["task"], r["enc"], r.get("pid", str(r.get("arm",""))+str(r.get("n",""))).split("@")[0])
            best[key] = r["score"]
    except OSError:
        pass
    return best


def paired_boot(a, b, n_boot=10000, seed=0):
    """a, b: aligned lists (same probes). Returns (mean_diff, lo, hi, p)."""
    assert len(a) == len(b) and len(a) > 0
    rng = random.Random(seed)
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    obs = sum(d) / n
    boots = []
    for _ in range(n_boot):
        boots.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    # two-sided p: fraction of bootstraps on the other side of 0
    p = 2 * min(sum(1 for x in boots if x <= 0), sum(1 for x in boots if x >= 0)) / n_boot
    return obs, lo, hi, min(1.0, p)


def test_file(fname, model_name, pairs, tasks=None):
    data = load_clean(f"{R}/{fname}")
    out = []
    for task in (tasks or ["reach1", "reach3", "nbr", "ambig"]):
        for e1, e2 in pairs:
            ids = sorted({k for k in data if k[0] == task and k[1] == e1} &
                         {k for k in data if k[0] == task and k[1] == e2})
            if len(ids) < 20:
                continue
            a = [data[(task, e1, i)] for i in ids]
            b = [data[(task, e2, i)] for i in ids]
            obs, lo, hi, p = paired_boot(a, b)
            out.append((model_name, task, f"{e1}>{e2}", len(ids), obs, lo, hi, p))
    return out


def main():
    rows = []
    WALLS = [("pointer", "edge_list"), ("adjacency_list", "edge_list")]
    for fname, name in [("e1_qwen7b.jsonl", "Qwen-7B"),
                        ("e1_qwen14b.jsonl", "Qwen-14B"),
                        ("e1_qwen72b.jsonl", "Qwen-72B"),
                        ("e1_gpt4omini.jsonl", "GPT-4o-mini"),
                        ("e1_deepseek.jsonl", "DeepSeek-V3")]:
        rows += test_file(fname, name, WALLS + [("sog", "json")])

    # E3 LoRA effect: two-sample bootstrap on arm means (pid differs across arms)
    e3_rows = [json.loads(l) for l in open(f"{R}/e3_results.jsonl")]
    by_arm = defaultdict(list)
    for r in e3_rows:
        by_arm[(r["arm"], r["task"])].append(r["score"])
    rng = random.Random(7)
    for task in ["reach1", "nbr"]:
        A = by_arm[("lora_edge", task)]; B = by_arm[("frozen", task)]
        if len(A) < 20 or len(B) < 20:
            continue
        obs = sum(A)/len(A) - sum(B)/len(B)
        boots = []
        for _ in range(10000):
            ma = sum(A[rng.randrange(len(A))] for _ in range(len(A)))/len(A)
            mb = sum(B[rng.randrange(len(B))] for _ in range(len(B)))/len(B)
            boots.append(ma - mb)
        boots.sort()
        lo, hi = boots[250], boots[9750]
        p = 2*min(sum(1 for x in boots if x <= 0), sum(1 for x in boots if x >= 0))/10000
        rows.append(("E3-1.5B", task, "LoRA vs frozen (2-sample)", len(A), obs, lo, hi, min(1.0, p)))

    print(f"{'model':14s}{'task':8s}{'comparison':26s}{'n':>5s}{'diff':>8s}{'95% CI':>18s}{'p':>10s}")
    print("-" * 90)
    for m, t, c, n, obs, lo, hi, p in rows:
        if obs != obs:
            continue
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"{m:14s}{t:8s}{c:26s}{n:5d}{obs:8.3f}  [{lo:+.3f},{hi:+.3f}]{p:10.4f}{star}")
    # summary counts
    sig = sum(1 for r in rows if r[7] == r[7] and r[7] < 0.05)
    tot = sum(1 for r in rows if r[4] == r[4])
    print(f"\n{sig}/{tot} comparisons significant at p<0.05")
    json.dump([list(r) for r in rows], open(f"{R}/significance.json", "w"), indent=1)
    print("saved", f"{R}/significance.json")


if __name__ == "__main__":
    main()
