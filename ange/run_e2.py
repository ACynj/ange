#!/usr/bin/env python
"""E2 runner: real-KG multiple-choice queries under all encodings."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.kgdata import KG, gen_lp_mc, build_prompt
from ange.engine import LocalEngine, ApiEngine

MODELS = {
    "qwen1.5b": "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct",
    "qwen3b": "/mnt/ssd/ange/models/Qwen2.5-3B-Instruct",
    "qwen7b": "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct",
    "qwen14b": "/mnt/ssd/ange/models/Qwen2.5-14B-Instruct",
    "gpt4omini:api": ("dmxapi", "gpt-4o-mini"),
    "r1:api": ("siliconflow", "deepseek-ai/DeepSeek-R1"),
    "glm53:api": ("siliconflow", "zai-org/GLM-5.3"),
    "qwen72b:api": ("siliconflow", "Qwen/Qwen2.5-72B-Instruct"),
    "glm4flash:api": ("glm", "glm-4-flash"),
    "glm4plus:api": ("glm", "glm-4-plus"),
    "deepseek:api": ("deepseek", "deepseek-chat"),
}

DATASETS = {"nell": "NELL-betae", "fb": "FB15k-237-betae"}


def parse_letter(text: str):
    m = re.findall(r"ANSWER\s*[:：]?\s*\**([A-J])\b", text.upper())
    if m:
        return m[-1]
    m = re.findall(r"\b([A-J])\b", text.upper())
    return m[-1] if m else "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="nell", choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encodings", default="edge_list,adjacency_list,nl,json,dfs_serial,pointer")
    args = ap.parse_args()

    kg = KG(DATASETS[args.dataset])
    probes = gen_lp_mc(kg, args.n, seed=args.seed)
    print(f"[data] {len(probes)} lp_mc probes on {args.dataset}")

    is_api = args.model.endswith(":api")
    if is_api:
        prov, model = MODELS[args.model]
        eng = ApiEngine(prov, model)
    else:
        eng = LocalEngine(MODELS[args.model])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a")
    for enc in args.encodings.split(","):
        prompts = [build_prompt(kg, p, enc) for p in probes]
        outs = eng.run([_P(pr) for pr in prompts], max_tokens=320)
        for p, pr, o in zip(probes, prompts, outs):
            pred = parse_letter(o)
            fout.write(json.dumps({
                "model": eng.name, "task": "lp_mc", "ds": args.dataset, "enc": enc,
                "gold": p["gold"], "pred": pred,
                "score": 1.0 if pred == p["gold"] else 0.0,
                "raw": o[:200]}, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{enc}] done", flush=True)
    fout.close()


class _P:
    def __init__(self, prompt):
        self.prompt = prompt


if __name__ == "__main__":
    main()
