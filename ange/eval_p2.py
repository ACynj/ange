#!/usr/bin/env python
"""P2 evaluation: all new arms on held-out probes + size extrapolation.

Evaluates (single vLLM instance, LoRA switching):
  1.5B: lora_pointer2 @ n=20 (main) and n=30/40 (extrapolation)
  7B:   frozen(edge_list,pointer,pointer2) @ n=20; lora arms @ n=20 and n=30/40
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.e3_data import gen_split, build_prompt
from ange.engine import parse_answer, score
from ange.encoders import ENCODINGS, enc_pointer2

B15 = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"
B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


def run_arm(llm, sp, tag, test, nsize, enc, adp, out, base_name):
    from vllm.lora.request import LoRARequest
    prompts = [build_prompt(s["graph"], s["q"], s["task"], enc, s["meta"]) for s in test]
    req = LoRARequest(tag, 1, adp) if adp else None
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, lora_request=req)
    for s, pr, o in zip(test, prompts, outs):
        txt = o.outputs[0].text
        parse = "bool" if s["task"].startswith("reach") else "list"
        gold = ("YES" if s["a"].endswith("YES") else "NO") if parse == "bool" \
            else s["a"].split("ANSWER: ")[-1]
        pred = parse_answer(txt, parse)
        sc = score(pred, gold, parse)
        out.write(json.dumps({"arm": tag, "enc": enc, "base": base_name, "n": nsize,
                              "task": s["task"], "gold": gold, "pred": pred,
                              "score": sc, "raw": txt[:200]}, ensure_ascii=False) + "\n")
    out.flush()
    print(f"[{tag} @ {enc}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/ssd/ange/results/e3_p2.jsonl")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    tests = {"n20": (gen_split(300, 315, 30), 20),
             "n30": (gen_split(600, 610, 25, n=30), 30),
             "n40": (gen_split(620, 628, 25, n=40), 40)}
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    out = open(args.out, "a")

    # ---- 1.5B block
    llm = LLM(model=B15, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
    a = "/mnt/ssd/ange/runs/e3_pointer2/adapter"
    run_arm(llm, sp, "lora_pointer2", *tests["n20"], "pointer2", a, out, "1.5B")
    for tag in ("n30", "n40"):
        run_arm(llm, sp, "lora_pointer2", *tests[tag], "pointer2", a, out, "1.5B")
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    # ---- 7B block
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
    for enc in ("edge_list", "pointer", "pointer2"):
        run_arm(llm, sp, "frozen", *tests["n20"], enc, None, out, "7B")
    arms = {"lora_edge": ("edge_list", "/mnt/ssd/ange/runs/e3_7b_edge/adapter"),
            "lora_pointer": ("pointer", "/mnt/ssd/ange/runs/e3_7b_pointer/adapter"),
            "lora_pointer2": ("pointer2", "/mnt/ssd/ange/runs/e3_7b_pointer2/adapter")}
    for tag, (enc, adp) in arms.items():
        run_arm(llm, sp, tag, *tests["n20"], enc, adp, out, "7B")
        for t2 in ("n30", "n40"):
            run_arm(llm, sp, tag, *tests[t2], enc, adp, out, "7B")
    out.close()
    print("P2 EVAL DONE")


if __name__ == "__main__":
    main()
