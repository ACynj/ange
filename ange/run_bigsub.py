#!/usr/bin/env python
"""E-C: retrieval-style rendering at larger subgraph budgets (limitation ④).

E2 used <=160-edge entity-centric subgraphs. Here we scale the budget to
400 and 800 edges (the 'retrieved subgraph' regime), frozen Qwen-7B,
edge_list vs pointer2 rendering, NELL-995 tail prediction (10-way MC,
200 probes per cell). Question: does the pointer grammar's relative
position hold as the retrieved context grows 5x?
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

from ange.kgdata import KG, gen_lp_mc
from ange.run_e2 import parse_letter
from ange.e3_real import prompt_train

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=0.45)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    kg = KG("NELL-betae")
    out = open(f"{R}/bigsub.jsonl", "w")
    for budget in (160, 400, 800):
        probes = gen_lp_mc(kg, 100, seed=5, max_edges=budget) + \
                 gen_lp_mc(kg, 100, seed=6, max_edges=budget)
        for enc in ("edge_list", "pointer2"):
            prompts = [prompt_train(kg, p, enc) for p in probes]
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
            hits = 0
            import statistics
            toks = []
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(B7)
            for p, pr, o in zip(probes, prompts, outs):
                txt = o.outputs[0].text
                pred = parse_letter(txt)
                sc = 1.0 if pred == p["gold"] else 0.0
                hits += sc
                toks.append(len(tok(pr, add_special_tokens=False)["input_ids"]))
                out.write(json.dumps({"budget": budget, "enc": enc,
                                      "gold": p["gold"], "pred": pred,
                                      "score": sc}) + "\n")
            out.flush()
            print(f"[budget={budget} {enc}] acc={hits/len(probes):.3f} "
                  f"mean_prompt_tokens={int(statistics.mean(toks))}", flush=True)
    out.close()
    print("BIGSUB DONE")


if __name__ == "__main__":
    main()
