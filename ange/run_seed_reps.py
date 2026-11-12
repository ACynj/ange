#!/usr/bin/env python
"""Evaluate the seed-replicate arms on the SAME held-out probes as E3 main."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ange.e3_data import gen_split, build_prompt
from ange.engine import parse_answer, score
from collections import defaultdict

B15 = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"
R = "/mnt/ssd/ange/results"

ARMS = [
    ("edge_s2", "edge_list", "/mnt/ssd/ange/runs/e3_edge_s2/adapter"),
    ("edge_s3", "edge_list", "/mnt/ssd/ange/runs/e3_edge_s3/adapter"),
    ("sog_s2", "sog", "/mnt/ssd/ange/runs/e3_sog_s2/adapter"),
    ("ptr_s2", "pointer", "/mnt/ssd/ange/runs/e3_ptr_s2/adapter"),
]


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    test = gen_split(300, 315, 30)  # identical held-out set as E3 main
    llm = LLM(model=B15, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.5, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    out = open(f"{R}/seed_replicates.jsonl", "w")
    for arm, enc, adp in ARMS:
        if not os.path.exists(adp):
            print(f"[{arm}] SKIP (no adapter)"); continue
        prompts = [build_prompt(s["graph"], s["q"], s["task"], enc, s["meta"]) for s in test]
        outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp,
                        lora_request=LoRARequest(arm, 1, adp))
        for s, o in zip(test, outs):
            parse = "bool" if s["task"].startswith("reach") else "list"
            pred = parse_answer(o.outputs[0].text, parse)
            gold = ("YES" if s["a"].endswith("YES") else "NO") if parse == "bool" \
                else s["a"].split("ANSWER: ")[-1]
            out.write(json.dumps({"arm": arm, "task": s["task"],
                                  "score": score(pred, gold, parse)}) + "\n")
        out.flush(); print(f"[{arm}] done", flush=True)
    out.close(); print("REPLICATES DONE")


if __name__ == "__main__":
    main()
