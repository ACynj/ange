#!/usr/bin/env python
"""Experiment B: cross-graph transfer of LoRA arms (algorithm vs. memorization).

Arms (trained ONLY on synthetic n=20/R=4 graphs):
  1.5B: lora_edge, lora_pointer, lora_pointer2
  7B:   lora_edge, lora_pointer, lora_pointer2
Transfer targets:
  T1 shifted-synthetic: R=8 relations, avg_deg 2.4 (denser, richer vocab)
  T2 WikiTopics-real:   real relation names, aliased entities (e*) --
       both vocabulary and distribution shift
Tasks: reach1, reach2, nbr (same generators as training).
Metric: retention = transfer score / in-domain score (in-domain from e3_p2).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.graph import MRGraph, gen_graph
from ange.encoders import ENCODINGS, enc_pointer, enc_pointer2
from ange.e3_data import build_prompt
from ange.engine import parse_answer, score

WT = os.path.expanduser("~/git/ULTRA/query-datasets/WikiTopics_QE")


def load_wiki(topic: str, max_edges: int = 260) -> MRGraph:
    """Load a WikiTopics graph; alias entities e0..ek, keep real relation names."""
    edges_raw = []
    seen = set()
    for line in open(f"{WT}/{topic}/train_graph.txt"):
        h, r, t = line.split()
        if (h, r, t) in seen:
            continue
        seen.add((h, r, t))
        edges_raw.append((h, r, t))
    rng = random.Random(7)
    rng.shuffle(edges_raw)
    edges_raw = edges_raw[:max_edges]
    alias = {}
    rels = []
    def node(x):
        if x not in alias:
            alias[x] = f"n{len(alias)}"
        return alias[x]
    def rel(x):
        if x not in rels:
            rels.append(x)
        return x
    edges = [(node(h), rel(r), node(t)) for h, r, t in edges_raw]
    return MRGraph(len(alias), rels, edges)


def gen_probes_target(g: MRGraph, rng: random.Random, n_per_task=40):
    probes = []
    # reach1/2 balanced, nbr with 1-5 answers
    for k in (1, 2):
        want_yes = True
        got = 0
        budget = 400 * n_per_task
        while got < n_per_task and budget > 0:
            budget -= 1
            src = rng.choice(g.nodes()); r = rng.choice(g.rels)
            reach = g.reach(src, r, k)
            cands = [t for t in g.nodes() if t != src]
            pool = sorted(reach) if want_yes else [t for t in cands if t not in reach]
            if not pool:
                continue
            tgt = rng.choice(pool)
            q = (f"Starting from node {src} and following only {r} edges in their "
                 f"forward direction, is node {tgt} reachable in at most {k} step"
                 f"{'s' if k > 1 else ''}? ")
            probes.append({"graph": g, "task": f"reach{k}", "q": q,
                           "gold": "YES" if want_yes else "NO",
                           "meta": (src, r, tgt)})
            want_yes = not want_yes
            got += 1
    got = 0
    budget = 400 * n_per_task
    while got < n_per_task and budget > 0:
        budget -= 1
        src = rng.choice(g.nodes()); r = rng.choice(g.rels)
        ans = g.out_neighbors(src, r)
        if not (1 <= len(ans) <= 5):
            continue
        q = f"List ALL nodes that are targets of {r} edges starting from node {src}. "
        probes.append({"graph": g, "task": "nbr", "q": q,
                       "gold": ",".join(ans) if ans else "NONE",
                       "meta": (src, r, None)})
        got += 1
    return probes


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="/mnt/ssd/ange/results/transfer.jsonl")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rng = random.Random(11)
    t1 = [gen_graph(900 + i, 20, 8, 2.4) for i in range(3)]
    t2 = [load_wiki(t) for t in ("art", "sport", "health")]
    targets = []
    for g in t1:
        targets += [("T1_shifted", p) for p in gen_probes_target(g, rng, args.n)]
    for g in t2:
        targets += [("T2_wiki", p) for p in gen_probes_target(g, rng, args.n)]
    print(f"[data] {len(targets)} transfer probes", flush=True)

    ARMS = [
        ("1.5B", "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct", {
            "lora_edge": ("edge_list", "/mnt/ssd/ange/runs/e3_edge_list/adapter"),
            "lora_pointer": ("pointer", "/mnt/ssd/ange/runs/e3_pointer/adapter"),
            "lora_pointer2": ("pointer2", "/mnt/ssd/ange/runs/e3_pointer2/adapter")}),
        ("7B", "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct", {
            "lora_edge": ("edge_list", "/mnt/ssd/ange/runs/e3_7b_edge/adapter"),
            "lora_pointer": ("pointer", "/mnt/ssd/ange/runs/e3_7b_pointer/adapter"),
            "lora_pointer2": ("pointer2", "/mnt/ssd/ange/runs/e3_7b_pointer2/adapter")}),
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a")
    for tag, base_path, arms in ARMS:
        llm = LLM(model=base_path, dtype="bfloat16", max_model_len=8192,
                  gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
        sp = SamplingParams(temperature=0.0, max_tokens=300)
        for arm, (enc, adp) in arms.items():
            prompts = [build_prompt(p["graph"], p["q"], p["task"], enc, p["meta"])
                       for _, p in targets]
            outs = llm.chat([[{"role": "user", "content": pr}] for pr in prompts], sp,
                            lora_request=LoRARequest(arm, 1, adp))
            for (tgt_name, p), o in zip(targets, outs):
                txt = o.outputs[0].text
                parse = "bool" if p["task"].startswith("reach") else "list"
                pred = parse_answer(txt, parse)
                sc = score(pred, p["gold"], parse)
                fout.write(json.dumps({"base": tag, "arm": arm, "enc": enc,
                                       "target": tgt_name, "task": p["task"],
                                       "gold": p["gold"], "pred": pred,
                                       "score": sc}) + "\n")
            fout.flush()
            print(f"[{tag} {arm}] done", flush=True)
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    fout.close()
    print("TRANSFER DONE")


if __name__ == "__main__":
    main()
