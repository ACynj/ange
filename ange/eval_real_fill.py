#!/usr/bin/env python
"""Evaluate the four new real-KG arms on NELL + FB15k-237 (200 probes each)."""
from __future__ import annotations
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = [
    # (tag, enc, adapter, base)
    ("lora_15b_edge", "edge_list", "/mnt/ssd/ange/runs/e3_real15_edge/adapter",
     "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"),
    ("lora_15b_v1", "pointer", "/mnt/ssd/ange/runs/e3_real15_v1/adapter",
     "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"),
    ("lora_15b_v2", "pointer2", "/mnt/ssd/ange/runs/e3_real15_v2/adapter",
     "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"),
    ("lora_7b_v1", "pointer", "/mnt/ssd/ange/runs/e3_real_v1/adapter",
     "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    from ange.kgdata import KG, gen_lp_mc
    from ange.run_e2 import parse_letter
    from ange.e3_real import prompt_train

    tests = {
        "NELL": gen_lp_mc(KG("NELL-betae"), 100, seed=5) + gen_lp_mc(KG("NELL-betae"), 100, seed=6),
        "FB15k-237": gen_lp_mc(KG("FB15k-237-betae"), 100, seed=5) + gen_lp_mc(KG("FB15k-237-betae"), 100, seed=6),
    }
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    for tag, enc, adp, base in ARMS:
        for ds, probes in tests.items():
            out_path = f"/mnt/ssd/ange/results/e3_real_fill_{ds.replace('-','')}.jsonl"
            done = set()
            if os.path.exists(out_path):
                for line in open(out_path):
                    r = json.loads(line)
                    if r.get("arm") == tag and r.get("dataset") == ds:
                        done.add(r.get("pid", len(done)))
            llm = LLM(model=base, dtype="bfloat16", max_model_len=16384,
                      gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
            sp = SamplingParams(temperature=0.0, max_tokens=256)
            prompts = [prompt_train(KG(ds + "-betae") if ds == "FB15k-237" else KG("NELL-betae"),
                                    p, enc) for p in probes]
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp,
                            lora_request=LoRARequest(tag, 1, adp))
            with open(out_path, "a") as f:
                for i, (p, o) in enumerate(zip(probes, outs)):
                    txt = o.outputs[0].text
                    pred = parse_letter(txt)
                    f.write(json.dumps({"arm": tag, "dataset": ds, "pid": i,
                                        "task": "lp_mc", "gold": p["gold"],
                                        "pred": pred,
                                        "score": 1.0 if pred == p["gold"] else 0.0,
                                        "raw": txt[:200]}, ensure_ascii=False) + "\n")
            print(f"[{tag} @ {ds}] done", flush=True)
            del llm
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
    print("FILL EVAL DONE")


if __name__ == "__main__":
    main()
