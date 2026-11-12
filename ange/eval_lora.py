#!/usr/bin/env python
"""E3 evaluation: LoRA arms + frozen baseline on the SAME held-out probes.

Usage: python -m ange.eval_lora --gpu 0 --out /mnt/ssd/ange/results/e3.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.e3_data import gen_split, make_chat, build_prompt
from ange.engine import parse_answer, score

BASE = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"
ARMS = {"frozen": None,
        "lora_edge": ("/mnt/ssd/ange/runs/e3_edge_list/adapter", "edge_list"),
        "lora_pointer": ("/mnt/ssd/ange/runs/e3_pointer/adapter", "pointer"),
        "lora_sog": ("/mnt/ssd/ange/runs/e3_sog/adapter", "sog")}


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="/mnt/ssd/ange/results/e3_results.jsonl")
    ap.add_argument("--n_test", type=int, default=450)
    args = ap.parse_args()

    test = gen_split(300, 300 + max(10, args.n_test // 30), 30)  # held-out seeds
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_lora=True,
              max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=200)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a")
    for arm, cfg in ARMS.items():
        enc = cfg[1] if cfg else "edge_list"   # frozen arm evaluated on both encodings
        encs = [enc] if cfg else ["edge_list", "pointer"]
        for e in encs:
            prompts = [build_prompt(s["graph"], s["q"], s["task"], e, s["meta"]) for s in test]
            msgs = [[{"role": "user", "content": p}] for p in prompts]
            lora_req = None
            if cfg:
                lora_req = LoRARequest(arm, 1, cfg[0])
            outs = llm.chat(msgs, sp, lora_request=lora_req)
            for s, pr, o in zip(test, prompts, outs):
                txt = o.outputs[0].text
                parse = "bool" if s["task"].startswith("reach") else "list"
                gold_bool = "YES" if s["a"].endswith("YES") else "NO"
                gold = gold_bool if parse == "bool" else s["a"].split("ANSWER: ")[-1]
                pred = parse_answer(txt, parse)
                sc = score(pred, gold, parse)
                fout.write(json.dumps({
                    "arm": arm, "enc": e, "task": s["task"], "gold": gold,
                    "pred": pred, "score": sc, "raw": txt[:300]},
                    ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{arm} @ {e}] done", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
