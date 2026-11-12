#!/usr/bin/env python
"""M9 fix: structure-corruption control on the real-KG repair result.
Take the P3 test probes; SHUFFLE the edges inside each subgraph (destroys
structure, preserves entity/label statistics). If the LoRA arm's accuracy
collapses toward chance, the 0.52 reflects structure reading, not priors."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ange.kgdata import KG, gen_lp_mc
from ange.e3_real import prompt_train
from ange.run_e2 import parse_letter

R = "/mnt/ssd/ange/results"
B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"


def shuffle_edges(edges, seed):
    rng = random.Random(seed)
    srcs = [e[0] for e in edges]
    tgts = [e[2] for e in edges]
    rng.shuffle(tgts)
    return [(s, edges[i][1], tgts[i]) for i, s in enumerate(srcs)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NELL-betae")
    ap.add_argument("--adp", default="/mnt/ssd/ange/runs/e3_real_edge/adapter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kg = KG(args.dataset)
    tests = gen_lp_mc(kg, 100, seed=5) + gen_lp_mc(kg, 100, seed=6)
    # corrupt: shuffle each probe's subgraph edges (seeded per probe)
    import copy
    corrupted = [dict(p, edges=shuffle_edges(p["edges"], 1000 + i)) for i, p in enumerate(tests)]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.5, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    out = open(args.out or f"{R}/shuffled_control.jsonl", "w")
    for tag, arm, adp in [("lora_edge_shuffled", "e3_real_edge", args.adp),
                          ("frozen_shuffled", None, None)]:
        prompts = [prompt_train(kg, p, "edge_list") for p in corrupted]
        outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp,
                        lora_request=LoRARequest(tag, 1, adp) if adp else None)
        for p, o in zip(corrupted, outs):
            pred = parse_letter(o.outputs[0].text)
            out.write(json.dumps({"arm": tag, "gold": p["gold"], "pred": pred,
                                  "score": 1.0 if pred == p["gold"] else 0.0}) + "\n")
        out.flush(); print(f"[{tag}] done", flush=True)
    out.close(); print("SHUFFLED CONTROL DONE")


if __name__ == "__main__":
    main()
