#!/usr/bin/env python
"""Reviewer fix (tokenizer idealization): forced single-token-per-entity ablation.

Maps every node id n<i> to a unique single-token CJK char and every relation
r<j> to another unique char, then evaluates frozen Qwen-7B on nbr probes
under (edge_list, pointer2) x (fragmented default, single-token) — 4 arms.
If the pointer advantage is unchanged under single-tokenization, the
idealized token model is not load-bearing for the behavioral claim.
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

from ange.e3_data import gen_split, build_prompt
from ange.engine import parse_answer, score
from ange.probes import HEADER, FOOT_LIST

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"

# single-token entity alphabet: common CJK block, verified single-token
NODE_CH = [chr(0x4E00 + i) for i in range(200)]   # n0..n199
REL_CH = [chr(0x3400 + i) for i in range(16)]     # r1..r16


def single_map(text: str) -> str:
    import re
    def node_sub(m):
        i = int(m.group(1))
        return NODE_CH[i] if i < len(NODE_CH) else m.group(0)
    def rel_sub(m):
        j = int(m.group(1))
        return REL_CH[j - 1] if 1 <= j <= len(REL_CH) else m.group(0)
    # substitute relations first (inside -r-> and r tokens), then nodes
    text = re.sub(r'\br(\d+)\b', lambda m: rel_sub(m), text)
    text = re.sub(r'\bn(\d+)\b', lambda m: node_sub(m), text)
    return text


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    test = [s for s in gen_split(300, 315, 30, n=20) if s["task"] == "nbr"]
    print(f"[data] {len(test)} nbr probes (e3 held-out seeds)", flush=True)

    out = open(f"{R}/single_token_ablation.jsonl", "w")
    for enc in ("edge_list", "pointer2"):
        for mode in ("fragmented", "single"):
            prompts = []
            for s in test:
                p = build_prompt(s["graph"], s["q"], s["task"], enc, s["meta"])
                if mode == "single":
                    p = single_map(p)
                prompts.append(p)
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
            for s, pr, o in zip(test, prompts, outs):
                txt = o.outputs[0].text
                gold = s["a"].split("ANSWER: ")[-1]
                if mode == "single":
                    gold = single_map(gold)
                pred = parse_answer(txt, "list")
                out.write(json.dumps({"enc": enc, "mode": mode, "task": "nbr",
                                      "gold": gold, "pred": pred,
                                      "score": score(pred, gold, "list")}) + "\n")
            out.flush()
            print(f"[{enc}/{mode}] done", flush=True)
    out.close()

    from collections import defaultdict
    acc = defaultdict(list)
    for line in open(f"{R}/single_token_ablation.jsonl"):
        r = json.loads(line)
        acc[(r["enc"], r["mode"])].append(r["score"])
    print("\n== forced single-token ablation (frozen 7B, nbr F1) ==")
    for k in sorted(acc):
        v = acc[k]
        print(f"  {k}: {sum(v)/len(v):.3f} (n={len(v)})")
    print("DONE")


if __name__ == "__main__":
    main()
