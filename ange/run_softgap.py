#!/usr/bin/env python
"""E-D: quantify the soft-attention approximation gap (limitation ②).

Theorem alignment (paper 1, Thm 4) is proved for hard exact attention;
real models are soft. We measure, for frozen Qwen-7B reading pointer-v2
prompts at graph sizes n = 20/40/60/80/100:
  - behavioral: nbr F1 (the realized soft-attention computation)
  - mechanistic: per-layer max-head matching-record attention share
    (how close any single head comes to the ideal hard match), and the
    layer at which the best head occurs.
Gap trend vs n tells whether softness degrades locality with scale.
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

import torch
from ange.e3_data import gen_split, build_prompt
from ange.engine import parse_answer, score

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"


def record_spans(prompt_text):
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
def probe_sample(model, tok, s):
    prompt = build_prompt(s["graph"], s["q"], s["task"], "pointer2", s["meta"])
    msgs = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(msgs, add_special_tokens=False, return_offsets_mapping=True,
              return_tensors="pt").to(model.device)
    ids, offs = enc["input_ids"], enc["offset_mapping"][0].tolist()
    out = model(ids, output_attentions=True)
    # behavioral answer from the same forward pass
    nxt = out.logits[0, -1].argmax().item()
    gen = model.generate(ids, max_new_tokens=80, do_sample=False,
                         attention_mask=enc["attention_mask"])
    txt = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    gold = s["a"].split("ANSWER: ")[-1]
    pred = parse_answer(txt, "list")
    beh = score(pred, gold, "list")
    # mechanistic: per-layer best-head matching-record share
    spans, block = record_spans(prompt)
    if block is None:
        return None
    def tok_idx(span):
        return [i for i, (a, b) in enumerate(offs) if a >= span[0] and b <= span[1]]
    block_ids = tok_idx(block)
    qsrc, qrel = s["meta"][0], s["meta"][1]
    match_ids = []
    for (a, b) in spans:
        if prompt[a:b].startswith(f"{qsrc} -{qrel}->"):
            match_ids += tok_idx((a, b))
    if not match_ids or not block_ids:
        return None
    per_layer = []
    for att in out.attentions:
        row = att[0, :, -1, :]                     # heads, L
        bm = row[:, block_ids].sum(-1)
        share = (bm / (bm + (row.sum(-1) - bm) + 1e-9))
        mm = row[:, match_ids].sum(-1)
        per_layer.append(((mm / (bm + 1e-9))).max().item())
    return {"behav": beh, "best_layer_share": max(per_layer),
            "best_layer": int(max(range(len(per_layer)),
                                  key=lambda i: per_layer[i]))}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B7)
    model = AutoModelForCausalLM.from_pretrained(
        B7, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    res = {}
    for n in (20, 40, 60, 80, 100):
        test = [s for s in gen_split(700 + n, 708 + n, 25, n=n)
                if s["task"] == "nbr"][:20]
        rows = [r for r in (probe_sample(model, tok, s) for s in test) if r]
        beh = sum(r["behav"] for r in rows) / len(rows)
        mech = sum(r["best_layer_share"] for r in rows) / len(rows)
        layer = sum(r["best_layer"] for r in rows) / len(rows)
        res[n] = {"nbr_f1": round(beh, 3),
                  "best_head_share": round(mech, 4),
                  "argmax_layer": round(layer, 1), "n_probes": len(rows)}
        print(f"n={n}: F1={beh:.3f} best-head share={mech:.4f} "
              f"at layer~{layer:.0f} ({len(rows)} probes)", flush=True)
    json.dump(res, open(f"{R}/softgap.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
