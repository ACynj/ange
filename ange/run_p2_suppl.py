#!/usr/bin/env python
"""Paper-2 supplementary: (a) 2-PG arm transfer to shifted/wiki targets;
(b) iso scale generalization (train n=12, test n=20/30)."""
from __future__ import annotations

import argparse, json, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ange.graph import gen_graph, isomorphic_pair, noniso_pair_wl_distinguishable
from ange.run_transfer import load_wiki, gen_probes_target
from ange.e3_data import build_prompt_iso
from ange.encoders import enc_pointer2pg
from ange.engine import parse_answer

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/ssd/ange/results/p2_suppl.jsonl")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.55, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=400)
    req = LoRARequest("arm", 1, "/mnt/ssd/ange/runs/fu_2pg_7b/adapter")
    out = open(args.out, "w")

    # (a) transfer of local tasks under 2-PG encoding
    rng = random.Random(11)
    t1 = [gen_graph(950 + i, 20, 8, 2.4) for i in range(3)]
    t2 = [load_wiki(t) for t in ("art", "sport", "health")]
    targets = []
    for g in t1:
        targets += [("T1", p) for p in gen_probes_target(g, rng, 40)]
    for g in t2:
        targets += [("T2", p) for p in gen_probes_target(g, rng, 40)]
    # render under pointer2pg: reuse build_prompt with encoding name
    from ange.e3_data import build_prompt
    prompts = [build_prompt(p["graph"], p["q"], p["task"], "pointer2pg", p["meta"])
               for _, p in targets]
    outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp, lora_request=req)
    from ange.engine import score as sc_fn
    for (tgt, p), o in zip(targets, outs):
        parse = "bool" if p["task"].startswith("reach") else "list"
        pred = parse_answer(o.outputs[0].text, parse)
        out.write(json.dumps({"block": "transfer", "target": tgt, "task": p["task"],
                              "gold": p["gold"], "pred": pred,
                              "score": sc_fn(pred, p["gold"], parse)}) + "\n")
    out.flush(); print("[transfer] done", flush=True)

    # (b) iso scale generalization: pairs at n=20 / n=30
    for n in (20, 30):
        rng2 = random.Random(31 + n)
        probes = []
        for i in range(25):
            ga, gb = isomorphic_pair(rng2.randrange(10**6), n=n)
            probes.append(("iso_pos", ga, gb, "YES"))
            got = noniso_pair_wl_distinguishable(rng2.randrange(10**6), n=n)
            if got:
                probes.append(("iso_sep", got[0], got[1], "NO"))
        prompts = [build_prompt_iso(ga, gb, "pointer2pg") for _, ga, gb, _ in probes]
        outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp, lora_request=req)
        for (tier, ga, gb, gold), o in zip(probes, outs):
            pred = parse_answer(o.outputs[0].text, "bool")
            out.write(json.dumps({"block": "iso_scale", "n": n, "tier": tier,
                                  "gold": gold, "pred": pred,
                                  "score": 1.0 if pred == gold else 0.0}) + "\n")
        out.flush(); print(f"[iso n={n}] done", flush=True)
    out.close(); print("SUPPL DONE")


if __name__ == "__main__":
    main()
