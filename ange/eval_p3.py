#!/usr/bin/env python
"""P3 evaluation: real-KG (NELL) repair by LoRA arms vs frozen 7B."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.kgdata import KG, gen_lp_mc, build_prompt
from ange.run_e2 import parse_letter, MODELS
from ange.e3_real import prompt_train

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/ssd/ange/results/e3_real.jsonl")
    ap.add_argument("--dataset", default="NELL-betae",
                    help="BetaE-suite KG name, e.g. NELL-betae / FB15k-237-betae")
    ap.add_argument("--adp_edge", default="/mnt/ssd/ange/runs/e3_real_edge/adapter")
    ap.add_argument("--adp_p2", default="/mnt/ssd/ange/runs/e3_real_p2/adapter")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    kg = KG(args.dataset)
    tests = gen_lp_mc(kg, 100, seed=5) + gen_lp_mc(kg, 100, seed=6)
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    out = open(args.out, "a")

    def run(tag, enc, adp):
        prompts = [prompt_train(kg, p, enc) for p in tests]
        req = LoRARequest(tag, 1, adp) if adp else None
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, lora_request=req)
        for p, pr, o in zip(tests, prompts, outs):
            txt = o.outputs[0].text
            pred = parse_letter(txt)
            out.write(json.dumps({"arm": tag, "enc": enc, "task": "lp_mc",
                                  "gold": p["gold"], "pred": pred,
                                  "score": 1.0 if pred == p["gold"] else 0.0,
                                  "raw": txt[:200]}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{tag} @ {enc}] done", flush=True)

    arms = {"frozen7b_edge": ("edge_list", None),
            "frozen7b_p2": ("pointer2", None),
            "lora_edge": ("edge_list", args.adp_edge),
            "lora_pointer2": ("pointer2", args.adp_p2)}
    for tag, (enc, adp) in arms.items():
        run(tag, enc, adp)
    out.close()
    print("P3 EVAL DONE")


if __name__ == "__main__":
    main()
