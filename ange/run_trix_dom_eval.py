#!/usr/bin/env python
"""Evaluate the TRIX-domain arms on the held-out entity-MC test (P3 protocol,
CoDex-S, 200 probes, 10-way, chance 0.10). Arms: frozen + LoRA(triples) +
LoRA(pointer), mode T1 (readable labels)."""
from __future__ import annotations
import json, random, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

from ange.trix_data import load_codex_s
from ange.trix_domain import render_probe
from ange.run_trix import parse_answer

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"


def main():
    probes = json.load(open(f"{R}/trix_ent_test.json"))
    triples, ents, rels = load_codex_s()
    rng = random.Random(41)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.42, enable_lora=True, max_lora_rank=16)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B7)

    def fit8k(p):
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) <= 8192 - 40:
            return p
        # keep head (record block start) and tail (question + options)
        head = tok.decode(ids[:5500])
        tail = tok.decode(ids[-2400:])
        return head + "\n... (records truncated) ...\n" + tail
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    out = open(f"{R}/trix_domain_eval.jsonl", "w")
    arms = [("frozen_triples", "triples", None),
            ("lora_triples", "triples", "/mnt/ssd/ange/runs/trix_dom_tri/adapter"),
            ("lora_pointer", "pointer", "/mnt/ssd/ange/runs/trix_dom_ptr/adapter")]
    for tag, enc, adp in arms:
        req = LoRARequest(tag, 1, adp) if adp else None
        prompts, golds = [], []
        for d in probes:
            q = tuple(d["q"])
            if d["kind"] == "ent":
                q = (q[0], q[1], q[2], list(q[3]))
            pr, ans = render_probe([tuple(e) for e in d["edges"]], q,
                                   d["kind"], enc, "T1", ents, rels, rng)
            prompts.append(fit8k(pr))
            golds.append(ans)
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                        sp, lora_request=req)
        hits = 0
        for p, g, o in zip(prompts, golds, outs):
            txt = o.outputs[0].text
            pred = parse_answer(txt)
            gold_num = parse_answer(g)
            sc = 1.0 if (pred is not None and pred == gold_num) else 0.0
            hits += sc
            out.write(json.dumps({"arm": tag, "enc": enc, "gold": gold_num,
                                  "pred": pred, "score": sc,
                                  "raw": txt[:150]}) + "\n")
        out.flush()
        print(f"[{tag}] entity-MC acc = {hits/len(probes):.3f} ({len(probes)})",
              flush=True)
    out.close()
    print("TRIX DOM EVAL DONE")


if __name__ == "__main__":
    main()
