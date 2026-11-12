#!/usr/bin/env python
"""E1 main runner: probe grid over (task, encoding, model).

Usage:
  python -m ange.run_e1 --model qwen1.5b --encodings edge_list,pointer \
      --tasks reach1,reach2,reach3,nbr,iso,ambig --n 50 --gpu 0 \
      --out /mnt/ssd/ange/results/e1_smoke.jsonl

Model aliases resolve to HF paths (pre-downloaded under /mnt/ssd/ange/models).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.probes import (Probe, gen_reach, gen_nbr, gen_iso, gen_iso_insep,
                         gen_ambig, reencode)
from ange.engine import LocalEngine, ApiEngine, parse_answer, score

MODELS = {
    "qwen1.5b": "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct",
    "qwen3b": "/mnt/ssd/ange/models/Qwen2.5-3B-Instruct",
    "qwen7b": "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct",
    "qwen14b": "/mnt/ssd/ange/models/Qwen2.5-14B-Instruct",
    "glm4p32:api": ("siliconflow", "THUDM/GLM-4-32B-0414"),
    "gpt4omini:api": ("dmxapi", "gpt-4o-mini"),
    "gpt4o:api": ("dmxapi", "gpt-4o"),
    "r1:api": ("siliconflow", "deepseek-ai/DeepSeek-R1"),
    "glm53:api": ("glm", "glm-5.3"),
    "qwen72b:api": ("siliconflow", "Qwen/Qwen2.5-72B-Instruct"),
    "glm4flash:api": ("glm", "glm-4-flash"),
    "glm4plus:api": ("glm", "glm-4-plus"),
    "deepseek:api": ("deepseek", "deepseek-chat"),
}


def build_probes(tasks, n, seed=11, graph_n=20):
    """Generate logical probe instances once (task level), then reencode."""
    base = []
    for t in tasks:
        if t.startswith("reach"):
            base += gen_reach(seed + int(t[5]), int(t[5]), n=graph_n, n_probes=n)
        elif t == "nbr":
            base += gen_nbr(seed, n=graph_n, n_probes=n)
        elif t == "iso":
            base += gen_iso(seed, n_probes=n)
        elif t == "iso_insep":
            base += gen_iso_insep(seed, n_probes=min(n, 30))
        elif t == "ambig":
            base += gen_ambig(seed, n_pairs=max(n // 2, 10))
        else:
            raise ValueError(t)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--encodings", default="edge_list,adjacency_list,incident_list,nl,json,sog,dfs_serial,pointer")
    ap.add_argument("--tasks", default="reach1,reach2,reach3,nbr,iso,iso_insep,ambig")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=320)
    ap.add_argument("--graph_n", type=int, default=20)
    ap.add_argument("--api_workers", type=int, default=12)
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    encodings = args.encodings.split(",")
    base = build_probes(tasks, args.n, args.seed, args.graph_n)

    is_api = args.model.endswith(":api")
    if is_api:
        prov, model = MODELS[args.model]
        eng = ApiEngine(prov, model)
    else:
        # NOTE: CUDA_VISIBLE_DEVICES is set by the launcher; do not override here
        eng = LocalEngine(MODELS[args.model], tp=args.tp)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a")
    t0 = time.time()
    for enc in encodings:
        probes = reencode(base, enc)
        # max_tokens: lists need more room than bools
        w = getattr(args, "api_workers", 12)
        outs = eng.run(probes, max_tokens=args.max_tokens, _workers_override=w) if is_api else eng.run(probes, max_tokens=args.max_tokens)
        for p, o in zip(probes, outs):
            pred = parse_answer(o, p.parse)
            s = score(pred, p.gold, p.parse)
            rec = {"model": eng.name, "task": p.task, "enc": enc,
                   "pid": p.pid, "gold": p.gold, "pred": pred, "score": s,
                   "raw": o[:400]}
            rec["meta"] = {k: v for k, v in p.meta.items()
                           if k not in ("graph", "ga", "gb", "qtext")}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{time.time()-t0:7.1f}s] {enc}: done", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
