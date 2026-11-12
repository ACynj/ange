#!/usr/bin/env python
"""Evaluate follow-up arms.

mode=2pg      : iso three tiers under pointer2pg (the boundary test) +
                reach/nbr regression on the e3 held-out set.
mode=frontier : reach1/2/3 accuracy AND output-token cost vs the free-CoT arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.e3_data import gen_split, build_prompt
from ange.encoders import enc_pointer2pg
from ange.engine import parse_answer, score
from ange.graph import isomorphic_pair, noniso_pair_wl_distinguishable, noniso_pair_wl_indistinguishable
from ange.e3_data import build_prompt_iso, ISO_Q
import random

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
B15 = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["2pg", "frontier"])
    ap.add_argument("--arm", required=True)          # adapter path
    ap.add_argument("--base", default=B7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=768)
    out = open(args.out, "a")
    req = LoRARequest("arm", 1, args.arm)

    def batch(tag, prompts, metas, parse):
        outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp,
                        lora_request=req)
        for m, pr, o in zip(metas, prompts, outs):
            txt = o.outputs[0].text
            pred = parse_answer(txt, parse)
            gold = m["gold"]
            sc = score(pred, gold, parse)
            out.write(json.dumps({"tag": tag, "task": m["task"], "gold": gold,
                                  "pred": pred, "score": sc,
                                  "out_tokens": len(o.outputs[0].token_ids),
                                  "raw": txt[:200]}) + "\n")
        out.flush()
        print(f"[{tag}] done", flush=True)

    if args.mode == "2pg":
        # iso three tiers under 2-PG encoding
        rng = random.Random(5)
        iso_probes = []
        for i in range(40):                       # tier1 pos
            ga, gb = isomorphic_pair(5000 + i * 2, n=12)
            iso_probes.append({"task": "iso_pos", "gold": "YES", "ga": ga, "gb": gb})
            got = noniso_pair_wl_distinguishable(5000 + i * 2 + 1, n=12)
            if got:
                ga2, gb2, _ = got
                iso_probes.append({"task": "iso_sep", "gold": "NO", "ga": ga2, "gb": gb2})
        got = noniso_pair_wl_indistinguishable(7)
        for i in range(30):                       # tier3 boundary
            ga, gb = got[0].permute(random.Random(77 + i)), got[1].permute(random.Random(88 + i))
            iso_probes.append({"task": "iso_insep", "gold": "NO", "ga": ga, "gb": gb})
        prompts = [build_prompt_iso(p["ga"], p["gb"], "pointer2pg") for p in iso_probes]
        batch("2pg_iso", prompts, iso_probes, "bool")
        # regression: reach/nbr on e3 held-out
        test = gen_split(300, 315, 30)
        for t in ("reach1", "reach2", "reach3", "nbr"):
            sub = [s for s in test if s["task"] == t][:40]
            prompts = [build_prompt(s["graph"], s["q"], s["task"], "pointer2pg", s["meta"]) for s in sub]
            metas = [{"task": t, "gold": ("YES" if s["a"].endswith("YES") else "NO")
                      if t.startswith("reach") else s["a"].split("ANSWER: ")[-1]} for s in sub]
            batch(f"2pg_{t}", prompts, metas, "bool" if t.startswith("reach") else "list")
    else:  # frontier
        test = gen_split(300, 315, 30)
        for t in ("reach1", "reach2", "reach3"):
            sub = [s for s in test if s["task"] == t][:60]
            prompts = [build_prompt(s["graph"], s["q"], s["task"], "pointer2", s["meta"]) for s in sub]
            metas = [{"task": t, "gold": "YES" if s["a"].endswith("YES") else "NO"} for s in sub]
            batch(f"frontier_{t}", prompts, metas, "bool")
    out.close()
    print("FOLLOWUP EVAL DONE")


if __name__ == "__main__":
    main()
