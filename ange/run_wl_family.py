#!/usr/bin/env python
"""E-A: WL-family robustness of the 2-PG boundary lift (limitation ③).

Paper 2 lifted iso on C6 vs 2xC3 (k=3). Here we scale the 1-WL-indistinguishable
family: C_{2k} vs 2 x C_k for k = 3, 4, 5, plus a mixed-member variant.
Encodings: edge_list & pointer2 (frozen 7B, expect chance) vs pointer2pg
(frozen AND the trained fu_2pg_7B arm). If 2-PG holds across the family,
the lift is a property of triangle records, not of one toy pair.
"""
from __future__ import annotations
import json, random, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

from ange.graph import MRGraph
from ange.e3_data import build_prompt_iso
from ange.engine import parse_answer, score

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
ADP = "/mnt/ssd/ange/runs/fu_2pg_7b/adapter"
R = "/mnt/ssd/ange/results"


def cycle_edges(k: int, offset: int = 0):
    """Directed k-cycle edges on nodes n{offset}..n{offset+k-1}, relation r1."""
    nodes = [f"n{offset+i}" for i in range(k)]
    return [(nodes[i], "r1", nodes[(i+1) % k]) for i in range(k)]


def mr(edges, n, rels=("r1",)):
    return MRGraph(n, list(rels), list(edges))


def wl_pair(k: int, shuffle_seed: int):
    """C_{2k} (one 2k-cycle) vs 2 x C_k (two disjoint k-cycles)."""
    ga = mr(cycle_edges(2 * k), 2 * k)
    gb = mr(cycle_edges(k, 100) + cycle_edges(k, 100 + k), 2 * k)
    rng = random.Random(shuffle_seed)
    rng.shuffle(ga.edges)
    rng.shuffle(gb.edges)
    return ga, gb


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=400)

    out = open(f"{R}/wl_family.jsonl", "w")
    for k in (3, 4, 5):
        probes = []
        for s in range(20):
            ga, gb = wl_pair(k, 1000 + s)
            probes.append((k, ga, gb))
        for enc, arm in (("edge_list", "frozen"), ("pointer2", "frozen"),
                         ("pointer2pg", "frozen"), ("pointer2pg", "lora_2pg")):
            req = LoRARequest(arm, 1, ADP) if arm == "lora_2pg" else None
            prompts = [build_prompt_iso(ga, gb, enc) for _, ga, gb in probes]
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                            sp, lora_request=req)
            hits = 0
            for (kk, ga, gb), o in zip(probes, outs):
                pred = parse_answer(o.outputs[0].text, "bool")
                sc = score(pred, "NO", "bool")     # gold is always NO (non-iso)
                hits += sc
                out.write(json.dumps({"k": kk, "enc": enc, "arm": arm,
                                      "gold": "NO", "pred": pred,
                                      "score": sc}) + "\n")
            out.flush()
            print(f"[k={k} {enc}/{arm}] iso-NO acc = {hits/len(probes):.2f}",
                  flush=True)
    out.close()
    from collections import defaultdict
    acc = defaultdict(list)
    for line in open(f"{R}/wl_family.jsonl"):
        r = json.loads(line)
        acc[(r["k"], r["enc"], r["arm"])].append(r["score"])
    print("\n== WL family (C_2k vs 2xC_k), gold NO ==")
    for k in (3, 4, 5):
        for key in [(k, "edge_list", "frozen"), (k, "pointer2", "frozen"),
                    (k, "pointer2pg", "frozen"), (k, "pointer2pg", "lora_2pg")]:
            v = acc.get(key)
            if v:
                print(f"  k={k} {key[1]}/{key[2]}: {sum(v)/len(v):.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
