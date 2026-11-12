#!/usr/bin/env python
"""Reviewer fix #3: does TRAINING localize the content-addressed lookup?

The paper reports that a last-layer single-head attention probe failed to
localize the pointer advantage in the FROZEN 7B reader (matching-record
share 0.009 vs chance 0.015). Here we repeat the probe on the 1.5B
pointer-v2 LoRA arm and its frozen baseline:

  probe  : nbr queries over pointer-v2 prompts (n=20, held-out seeds)
  metric : at the final prompt position (the one predicting the answer),
           per-head attention mass on tokens of MATCHING records
           (records with the queried (src, rel)) divided by the mass on
           the whole record block; chance = matching tokens / block tokens.

If the LoRA arm's share >> frozen share ~= chance, training installs the
content-addressed mechanism the theory describes.
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ange.e3_data import gen_split, build_prompt
from ange.encoders import enc_pointer2

BASE = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"
ADAPTER = "/mnt/ssd/ange/runs/e3_pointer2/adapter"
R = "/mnt/ssd/ange/results"


def record_spans(prompt_text):
    """Char spans of each v2 record line and of the whole record block."""
    lines = prompt_text.split("\n")
    spans, block = [], None
    pos = 0
    for ln in lines:
        if ln.startswith("n") and " -" in ln and "->" in ln:
            if block is None:
                block = [pos, pos + len(ln)]
            else:
                block[1] = pos + len(ln)
            spans.append((pos, pos + len(ln)))
        pos += len(ln) + 1
    return spans, block


@torch.no_grad()
def probe(model, tok, test, n_probes=40):
    shares, chances, maxheads = [], [], []
    for s in test[:n_probes]:
        if s["task"] != "nbr":
            continue
        prompt = build_prompt(s["graph"], s["q"], s["task"], "pointer2", s["meta"])
        msgs = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(msgs, add_special_tokens=False, return_offsets_mapping=True,
                  return_tensors="pt").to(model.device)
        ids, offs = enc["input_ids"], enc["offset_mapping"][0].tolist()
        out = model(ids, output_attentions=True)
        last = out.attentions[-1][0]            # [heads, L, L]
        row = last[:, -1, :]                    # attention from final position
        # token index ranges for each record line
        spans, block = record_spans(prompt)
        if block is None:
            continue
        def tok_idx(span):
            return [i for i, (a, b) in enumerate(offs)
                    if a >= span[0] and b <= span[1]]
        block_ids = tok_idx(block)
        qsrc, qrel = s["meta"][0], s["meta"][1]
        match_ids = []
        for (a, b) in spans:
            line = prompt[a:b]
            if line.startswith(f"{qsrc} -{qrel}->"):
                match_ids += tok_idx((a, b))
        if not match_ids or not block_ids:
            continue
        block_mass = row[:, block_ids].sum(-1)          # [heads]
        match_mass = row[:, match_ids].sum(-1)
        share = (match_mass / block_mass).mean().item()
        shares.append(share)
        chances.append(len(match_ids) / len(block_ids))
        maxheads.append((match_mass / block_mass).max().item())
    n = len(shares)
    return {"mean_share": sum(shares) / n, "max_head_share": max(maxheads),
            "chance": sum(chances) / n, "n_probes": n}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    test = gen_split(300, 315, 30, n=20)

    res = {}
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map="cuda",
                                                 attn_implementation="eager").eval()
    res["frozen"] = probe(model, tok, test)
    del model; torch.cuda.empty_cache()

    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map="cuda",
                                                 attn_implementation="eager").eval()
    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    res["lora_pointer2"] = probe(model, tok, test)

    json.dump(res, open(f"{R}/attn_probe_trained.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k:14s} mean_share={v['mean_share']:.4f} "
              f"max_head={v['max_head_share']:.4f} chance={v['chance']:.4f} "
              f"(n={v['n_probes']})")
    print("DONE")


if __name__ == "__main__":
    main()
