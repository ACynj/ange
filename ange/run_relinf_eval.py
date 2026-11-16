#!/usr/bin/env python
"""TRIX rel_inf independent evaluation: infer the HIDDEN relation between
two entities from their surrounding connections. 200 probes, modes T1
(readable names) and T3 (fully permuted IDs), arms frozen / LoRA(pointer)
/ LoRA(triples). Chance = 1/len(subgraph rels)."""
from __future__ import annotations
import json, random, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

from ange.trix_data import load_codex_s, build_subgraph
from ange.trix_domain import render_probe
from ange.run_trix import parse_answer
from collections import defaultdict

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"


def gen_relinf(n=200, seed=42):
    triples, ents, rels = load_codex_s()
    rng = random.Random(seed)
    sub = build_subgraph(triples, 4200)
    edges_by_head = defaultdict(list)
    for h, r, t in triples:
        edges_by_head[h].append((h, r, t))
    tests = [e for e in sub if e in triples]
    rng.shuffle(tests)
    probes = []
    for h, r, t in tests:
        if len(probes) >= n:
            break
        # surrounding edges: neighbors of h and t (excluding the query edge)
        ctx = set()
        for e in edges_by_head.get(h, []) + edges_by_head.get(t, []):
            ctx.add(e)
        ctx.discard((h, r, t))
        ctx = [e for e in ctx if e[2] in ents and e[0] in ents][:40]
        if len(ctx) < 4:
            continue
        probes.append({"edges": ctx, "q": (h, r, t), "kind": "rel_inf"})
    return probes, ents, rels


def main():
    probes, ents, rels = gen_relinf(200)
    print(f"[data] {len(probes)} rel_inf probes (hidden relation)", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    tok = AutoTokenizer.from_pretrained(B7)

    def fit(p):
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) <= 8192 - 40:
            return p
        return (tok.decode(ids[:5500]) + "\n... (truncated) ...\n"
                + tok.decode(ids[-2400:]))

    arms = [("frozen", "triples", None),
            ("lora_pointer", "pointer", "/mnt/ssd/ange/runs/trix_dom_ptr/adapter"),
            ("lora_triples", "triples", "/mnt/ssd/ange/runs/trix_dom_tri/adapter")]
    out = open(f"{R}/relinf_eval.jsonl", "w")
    for mode in ("T1", "T3"):
        for tag, enc, adp in arms:
            req = LoRARequest(tag, 1, adp) if adp else None
            rng = random.Random(42)
            hits = 0
            for d in probes:
                pr, cot = render_probe([tuple(e) for e in d["edges"]],
                                       d["q"], d["kind"], enc, mode,
                                       ents, rels, rng)
                pr = fit(pr)
                gold_num = parse_answer(cot)
                o = llm.chat([[{"role": "user", "content": pr}]], sp,
                             lora_request=req)[0]
                pred = parse_answer(o.outputs[0].text)
                sc = 1.0 if (pred is not None and pred == gold_num) else 0.0
                hits += sc
                out.write(json.dumps({"mode": mode, "arm": tag,
                                      "gold": gold_num, "pred": pred,
                                      "score": sc}) + "\n")
            out.flush()
            print(f"[{mode} {tag}] rel_inf acc = {hits/len(probes):.3f}",
                  flush=True)
    out.close()
    print("RELINF DONE")


if __name__ == "__main__":
    main()
