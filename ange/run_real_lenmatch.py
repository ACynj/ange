#!/usr/bin/env python
"""Train + evaluate the real-KG len-matched arm (1.5B): pointer-v2 records
padded per-record with neutral 'z' filler to the v1 record's exact token
count. Fills the last two dashes in Table IV."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.kgdata import KG, gen_lp_mc
from ange.encoders import enc_pointer
from ange.run_e2 import parse_letter
from ange.e3_real import gen_train, prompt_train, gold_cot

B15 = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"


def make_padded_prompt_builder():
    """Per-record padded v2: each 'u -r-> v' record gets floor tokens so its
    token count equals the v1 record for the same edge."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B15)
    n = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])

    from ange.kgdata import _MiniGraph
    from ange.encoders import _node_line, _rel_line

    def enc_ptr2_pad_real(g: "_MiniGraph", query=None) -> str:
        lines = [_node_line(g), _rel_line(g), "Edges (u -rel-> v):"]
        for s, r, t in g.edges:
            v2 = f"{s} -{r}-> {t}"
            v1 = f"EDGE rel={r} src={s} tgt={t}"
            delta = max(0, n(v1) - n(v2))
            lines.append(v2 + (" " + " ".join(["z"] * delta) if delta else ""))
        if query is not None:
            qsrc, qrel = query
            lines.append(f"QUERY: {qsrc} -{qrel}-> ?")
        return "\n".join(lines)

    def prompt_train_pad(kg, p, encoding):
        from ange.kgdata import DEMO_TRIPLES, DEMO_ALIASES, _MiniGraph as MG
        from ange.e3_real import prompt_train as _orig
        # reuse the original builder but swap the graph encoder
        triples, alias_names = __import__('ange.kgdata', fromlist=['to_mrtext']).to_mrtext(kg, p["edges"])
        g = MG(triples, alias_names)
        inv_alias = {name: a for a, name in alias_names.items()}
        head_alias = inv_alias.get(kg.n(p["head"]))
        rel_txt = kg.rel(p["rel"])
        gt = enc_ptr2_pad_real(g, query=(head_alias, rel_txt))
        alias_tbl = "\nAliases:\n" + "\n".join(f"{a} = {n2}" for a, n2 in alias_names.items())
        dg = MG(DEMO_TRIPLES, DEMO_ALIASES)
        dt = enc_ptr2_pad_real(dg, query=("e0", "works_with"))
        dtbl = "\nAliases:\n" + "\n".join(f"{a} = {n2}" for a, n2 in DEMO_ALIASES.items())
        demo = ("Example:\n" + dt + dtbl + "\n\nQuestion: Which entity is the correct "
                "target of the relation 'works_with' starting from entity 'alice'?\n"
                "Choose exactly one letter among A-C.\nOptions:\n  A. paris\n  B. bob\n  C. france\n"
                "The 'works_with' edges from e0 (alice) point to e2 = bob.\nANSWER: B\n\n"
                "Now solve:\n\n")
        cand_lines = "\n".join(f"  {l}. {kg.n(c)}" for l, c in zip("ABCDEFGHIJ", p["cands"]))
        return (demo + gt + alias_tbl + "\n\nQuestion: Which entity is the correct "
                "target of the relation '" + rel_txt + "' starting from entity '"
                + kg.n(p["head"]) + "'?\nChoose exactly one letter among A-J.\nOptions:\n"
                + cand_lines)

    return prompt_train_pad


def train(out):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model
    prompt_pad = make_padded_prompt_builder()
    from ange.kgdata import KG, gen_lp_mc

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(B15)
    model = AutoModelForCausalLM.from_pretrained(B15, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    kg = KG("NELL-betae")
    test_heads = {p["head"] for p in gen_lp_mc(kg, 100, seed=5)} | \
                 {p["head"] for p in gen_lp_mc(kg, 100, seed=6)}
    samples = gen_train(kg, 3000, avoid_heads=test_heads)
    feats = []
    for s in samples:
        pr = prompt_pad(kg, s, "ptr2pad")
        full = tok.apply_chat_template(
            [{"role": "user", "content": pr},
             {"role": "assistant", "content": gold_cot(kg, s)}],
            tokenize=False, add_generation_prompt=False)
        prefix = tok.apply_chat_template([{"role": "user", "content": pr}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(full, add_special_tokens=False)["input_ids"][:5376]
        pre = tok(prefix, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(pre) + ids[len(pre):]
        feats.append({"input_ids": ids[:5376], "labels": labels[:5376]})
    from ange.train_lora import Collator
    targs = TrainingArguments(output_dir=out, num_train_epochs=2.0,
                              per_device_train_batch_size=2, learning_rate=1e-4,
                              warmup_steps=30, lr_scheduler_type="cosine",
                              logging_steps=20, save_strategy="no", bf16=True,
                              report_to=[], remove_unused_columns=False,
                              gradient_checkpointing=True)
    Trainer(model=model, args=targs, train_dataset=feats,
            data_collator=Collator(tok)).train()
    model.save_pretrained(os.path.join(out, "adapter"))
    tok.save_pretrained(os.path.join(out, "adapter"))
    print("saved", out, flush=True)


def evaluate(adapter):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    prompt_pad = make_padded_prompt_builder()
    from ange.kgdata import KG, gen_lp_mc
    llm = LLM(model=B15, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    for ds in ("NELL", "FB15k-237"):
        kg = KG(ds + "-betae")
        probes = gen_lp_mc(kg, 100, seed=5) + gen_lp_mc(kg, 100, seed=6)
        prompts = [prompt_pad(kg, p, "ptr2pad") for p in probes]
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp,
                        lora_request=LoRARequest("lenmatch", 1, adapter))
        with open(f"/mnt/ssd/ange/results/e3_real_fill_{ds.replace('-','')}.jsonl", "a") as f:
            for i, (p, o) in enumerate(zip(probes, outs)):
                txt = o.outputs[0].text
                pred = parse_letter(txt)
                f.write(json.dumps({"arm": "lora_15b_lenmatch", "dataset": ds, "pid": i,
                                    "task": "lp_mc", "gold": p["gold"], "pred": pred,
                                    "score": 1.0 if pred == p["gold"] else 0.0,
                                    "raw": txt[:200]}, ensure_ascii=False) + "\n")
        print(f"[lenmatch @ {ds}] done", flush=True)
    print("LENMATCH DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--out", default="/mnt/ssd/ange/runs/e3_real15_lenmatch")
    a = ap.parse_args()
    if a.phase in ("train", "both"):
        train(a.out)
    if a.phase in ("eval", "both"):
        evaluate(os.path.join(a.out, "adapter"))
