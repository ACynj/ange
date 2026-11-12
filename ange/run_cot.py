#!/usr/bin/env python
"""Experiment A: Controlled CoT as explicit message passing.

Three prompting regimes on reach_k (frozen Qwen-7B, few-shot):
  free     : model's natural chain of thought (baseline, matches E1 style)
  frontier : each step outputs ONLY the current BFS frontier set
             ("Step 1: {n2, n7}" ...) -- k steps == k rounds of message passing
  none     : answer immediately (early smoke-test style)

Measures accuracy per k and mean output tokens (cost).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.graph import gen_graph
from ange.encoders import ENCODINGS, enc_pointer2
from ange.engine import parse_answer

HEADER = ("You are given a directed multi-relational graph. Node names are n0..nk, "
          "relation names are r1..rm. They are arbitrary identifiers with no meaning.\n\n")


def frontier_trace(g, src, rel, k):
    """BFS trace: list of frontier sets per round."""
    frontier, seen, trace = {src}, {src}, []
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for s, r, t in g.edges:
                if s == u and r == rel and t not in seen:
                    nxt.add(t)
        seen |= nxt
        frontier = nxt
        trace.append(sorted(nxt))
    return trace


def build_prompt(g, src, rel, tgt, k, mode, enc_name="pointer2"):
    q = (f"Starting from node {src} and following only {rel} edges in their "
         f"forward direction, is node {tgt} reachable in at most {k} steps? ")
    if enc_name == "pointer2":
        gt = enc_pointer2(g, query=(src, rel))
        rec = "n3 -r1-> n17"
        qrec = "n3 -r1-> ?"
    else:
        gt = ENCODINGS[enc_name](g)
        rec = "n3 r1 n17"
        qrec = None
    # demo on a tiny 4-node graph
    dg = gen_graph(1, 4, 2, 1.0)
    # deterministic tiny demo graph: n0->n2, n0->n3 via r1; n1->n0 via r2
    from ange.graph import MRGraph
    demo = MRGraph(4, ["r1", "r2"], [("n0", "r1", "n2"), ("n0", "r1", "n3"),
                                     ("n1", "r2", "n0"), ("n2", "r1", "n3")])
    if enc_name == "pointer2":
        dt = enc_pointer2(demo, query=("n1", "r2"))
    else:
        dt = ENCODINGS[enc_name](demo)
    dq = ("Starting from node n1 and following only r2 edges in their forward "
          "direction, is node n2 reachable in at most 2 steps? ")
    if mode == "frontier":
        da = ("Step 1: r2 edges from n1 -> {n0}\n"
              "Step 2: r1 edges from n0 -> {n2, n3}\n"
              "n2 appears in step 2 (<=2 steps).\nANSWER: YES")
    elif mode == "free":
        da = ("n1 has an r2 edge to n0. From n0, r1 edges go to n2 and n3. "
              "So n1 -> n0 -> n2 is a 2-step path.\nANSWER: YES")
    else:  # none
        da = "ANSWER: YES"
    demo_block = (f"Example:\n{dt}\n\nQuestion: {dq}\n"
                  "Think step by step, briefly. Then end with 'ANSWER: YES' or "
                  "'ANSWER: NO'.\n" + da + "\n\nNow solve:\n\n")
    foot = ("\nThink step by step, briefly. Then end with 'ANSWER: YES' or "
            "'ANSWER: NO'.")
    if mode == "frontier":
        foot = ("\nWork in rounds: on each step output ONLY the frontier set as "
                "'Step i: {nodes}'. Use at most " + str(k) + " steps. Then end "
                "with 'ANSWER: YES' or 'ANSWER: NO'.")
    if mode == "none":
        foot = "\nEnd with 'ANSWER: YES' or 'ANSWER: NO'."
    return HEADER + demo_block + gt + "\n\nQuestion: " + q + foot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--base", default="/mnt/ssd/ange/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="/mnt/ssd/ange/results/cot_modes.jsonl")
    ap.add_argument("--n_probes", type=int, default=60)
    ap.add_argument("--enc", default="pointer2")
    args = ap.parse_args()

    import random
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.45)
    sp = SamplingParams(temperature=0.0, max_tokens=512)

    # build probes: balanced reach1/2/3 on fresh seeds (800+)
    probes = []
    rng = random.Random(99)
    gid = 800
    while len(probes) < args.n_probes * 3:
        g = gen_graph(gid, 20, 4, 1.2); gid += 1
        for k in (1, 2, 3):
            src = rng.choice(g.nodes()); rel = rng.choice(g.rels)
            reach = g.reach(src, rel, k)
            cands = [t for t in g.nodes() if t != src]
            if not cands:
                continue
            want = "YES" if len(probes) % 2 == 0 else "NO"
            pool = sorted(reach) if want == "YES" else [t for t in cands if t not in reach]
            if not pool:
                continue
            tgt = rng.choice(pool)
            probes.append({"k": k, "src": src, "rel": rel, "tgt": tgt,
                           "gold": want, "graph": g})
            if len(probes) >= args.n_probes * 3:
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a")
    for mode in ("free", "frontier", "none"):
        prompts = [build_prompt(p["graph"], p["src"], p["rel"], p["tgt"], p["k"],
                                mode, args.enc) for p in probes]
        outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp)
        for p, pr, o in zip(probes, prompts, outs):
            txt = o.outputs[0].text
            ntok = len(o.outputs[0].token_ids)
            pred = parse_answer(txt, "bool")
            fout.write(json.dumps({"mode": mode, "enc": args.enc, "k": p["k"],
                                   "gold": p["gold"], "pred": pred,
                                   "score": 1.0 if pred == p["gold"] else 0.0,
                                   "out_tokens": ntok, "raw": txt[:250]},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{mode}] done", flush=True)
    fout.close()
    print("COT MODES DONE")


if __name__ == "__main__":
    main()
