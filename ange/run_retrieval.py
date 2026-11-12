#!/usr/bin/env python
"""Project C: retrieval-style pointer reading over the FULL NELL graph.

Uses the existing in-domain LoRA arm (e3_real_p2, trained on 160-edge
ego-subgraph prompts). At test time we do NOT build an ego subgraph;
instead we retrieve a compact record pack from the 228k-edge full graph:
  - all records whose source or target is the query head (1-hop evidence)
  - for each of the 10 answer candidates, its top relation records
    (evidence that discriminates candidates), capped to <=60 records.
This instantiates axiom (A2): records are self-contained retrievable units.
Compared against the same arm on ego-subgraph prompts (P3: 0.510).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.kgdata import KG, gen_lp_mc
from ange.e3_real import prompt_train
from ange.run_e2 import parse_letter

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"


def build_retrieval_prompt(kg: KG, p: dict, max_recs: int = 60):
    h, r, cands = p["head"], p["rel"], p["cands"]
    recs = []
    seen = set()
    def add(hh, rr, tt):
        if (hh, rr, tt) not in seen:
            seen.add((hh, rr, tt))
            recs.append((hh, rr, tt))
    # 1-hop evidence around head
    for rr, t in kg.out_.get(h, [])[:20]:
        add(h, rr, t)
    for rr, s in kg.in_.get(h, [])[:20]:
        add(s, rr, h)
    # candidate evidence: each candidate's most frequent relations (cap 4 each)
    for c in cands:
        added = 0
        for rr, t in kg.out_.get(c, []):
            if added >= 4:
                break
            add(c, rr, t); added += 1
        added2 = 0
        for rr, s in kg.in_.get(c, []):
            if added + added2 >= 6:
                break
            add(s, rr, c); added2 += 1
    recs = recs[:max_recs]
    # render as pointer records with aliases
    alias = {}
    rel_alias = {}
    def node(x):
        if x not in alias:
            alias[x] = f"n{len(alias)}"
        return alias[x]
    def rel(x):
        if x not in rel_alias:
            rel_alias[x] = kg.rel(x)
        return rel_alias[x]
    lines = [f"Nodes: " + ", ".join(f"n{i}" for i in range(min(len(alias) + len(cands) + 1, 80))),
             "Edges (u -rel-> v):"]
    lines += [f"{node(a)} -{rel(b)}-> {node(c)}" for a, b, c in recs]
    head_alias = node(kg.n(h)) if kg.n(h) in alias else None
    # ensure head got an alias even if it had no records
    if kg.n(h) not in alias:
        alias[kg.n(h)] = f"n{len(alias)}"
    lines.append(f"QUERY: {alias[kg.n(h)]} -{kg.rel(r)}-> ?")
    alias_tbl = "\n".join(f"{a} = {n2}" for a, n2 in alias.items())
    cand_lines = "\n".join(f"  {l}. {kg.n(c)}" for l, c in zip("ABCDEFGHIJ", cands))
    q = (f"Which entity is the correct target of the relation '{kg.rel(r)}' "
         f"starting from entity '{kg.n(h)}' (node {alias[kg.n(h)]})?\n"
         f"Choose exactly one letter among A-J.\nOptions:\n{cand_lines}\n"
         f"End your reply with: 'ANSWER: <letter>'.")
    hdr = ("You are given retrieved edge records from a large knowledge graph "
           "(a compact evidence pack, not the full graph). Nodes are n0..nk; "
           "the alias list gives their real names.\n\n")
    return hdr + "\n".join(lines) + "\nAliases:\n" + alias_tbl + "\n\nQuestion: " + q


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/ssd/ange/results/retrieval_full.jsonl")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    kg = KG("NELL-betae")
    tests = gen_lp_mc(kg, 100, seed=5) + gen_lp_mc(kg, 100, seed=6)
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    out = open(args.out, "a")

    def run(tag, prompts, adp):
        outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp,
                        lora_request=LoRARequest(tag, 1, adp) if adp else None)
        for p, pr, o in zip(tests, prompts, outs):
            txt = o.outputs[0].text
            pred = parse_letter(txt)
            out.write(json.dumps({"arm": tag, "gold": p["gold"], "pred": pred,
                                  "score": 1.0 if pred == p["gold"] else 0.0,
                                  "raw": txt[:150]}) + "\n")
        out.flush()
        print(f"[{tag}] done", flush=True)

    adp = "/mnt/ssd/ange/runs/e3_real_p2/adapter"
    prompts = [build_retrieval_prompt(kg, p) for p in tests]
    lens = [len(x) for x in prompts]
    print(f"[data] retrieval prompts: p50/p95/max chars = "
          f"{sorted(lens)[len(lens)//2]}/{sorted(lens)[int(len(lens)*0.95)]}/{max(lens)}")
    run("retrieval_full+lora_p2", prompts, adp)
    run("retrieval_full+frozen", prompts, None)
    out.close()
    print("RETRIEVAL DONE")


if __name__ == "__main__":
    main()
