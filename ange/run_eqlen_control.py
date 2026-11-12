#!/usr/bin/env python
"""Reviewer fix #5: equal-length control isolating length from structure.

v1 vs v2 differ in BOTH per-record length (2.4x vs 1.3x) and record form
(key-tokens vs directional symbol). To attribute the v1 extrapolation
collapse to LENGTH (the addressing-economy hypothesis) rather than form,
we pad every v2 record with neutral filler tokens to v1's per-record
token count and retrain the same 1.5B LoRA arm:

  pointer2_pad: 'n3 -r1-> n17 z z z z z'   (filler 'z' carries no info)

If padded-v2 stays extrapolation-safe at n=30/40 (like plain v2), the
collapse of v1 is driven by length, not by key-token structure.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ange.encoders import ENCODINGS, enc_pointer2, _node_line, _rel_line
from ange.e3_data import gen_split, build_prompt
from ange.engine import parse_answer, score

BASE = "/mnt/ssd/ange/models/Qwen2.5-1.5B-Instruct"
R = "/mnt/ssd/ange/results"


def register_pad_encoder():
    """Register pointer2_pad: pad each v2 record to the v1 per-record
    token count with neutral 'z' filler (same convention as the earlier
    edge-list length control)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    n = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    # per-record token delta between v1 and v2 records (representative ids)
    v1_rec = "EDGE rel=r1 src=n3 tgt=n17"
    v2_rec = "n3 -r1-> n17"
    delta = n(v1_rec) - n(v2_rec)
    assert delta > 0, f"unexpected delta {delta}"
    json.dump({"v1_rec_tokens": n(v1_rec), "v2_rec_tokens": n(v2_rec),
               "pad_tokens_per_record": delta},
              open(f"{R}/eqlen_audit.json", "w"), indent=1)
    print(f"[audit] v1 record {n(v1_rec)} tok, v2 record {n(v2_rec)} tok "
          f"-> pad {delta} 'z' per record", flush=True)

    def enc_pointer2_pad(g, query=None):
        lines = [_node_line(g), _rel_line(g), "Edges (u -rel-> v):"]
        lines += [f"{s} -{r}-> {t} " + " ".join(["z"] * delta) for s, r, t in g.edges]
        if query is not None:
            qsrc, qrel = query
            lines.append(f"QUERY: {qsrc} -{qrel}-> ?")
        return "\n".join(lines)

    ENCODINGS["pointer2_pad"] = enc_pointer2_pad


def train(out):
    """Same protocol as train_lora.py (r=16, 3 epochs, data_seed 100)."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from ange.train_lora import build_dataset, Collator

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    split = gen_split(100, 225, 40)                     # same seeds as E3 arms
    from ange.e3_data import make_chat
    chats = make_chat(split, "pointer2_pad")
    feats = build_dataset(tok, chats)
    targs = TrainingArguments(
        output_dir=out, num_train_epochs=3.0, per_device_train_batch_size=16,
        learning_rate=1e-4, warmup_steps=30, lr_scheduler_type="cosine",
        logging_steps=20, save_strategy="no", bf16=True, report_to=[],
        remove_unused_columns=False, gradient_checkpointing=True)
    Trainer(model=model, args=targs, train_dataset=feats,
            data_collator=Collator(tok)).train()
    model.save_pretrained(os.path.join(out, "adapter"))
    tok.save_pretrained(os.path.join(out, "adapter"))
    json.dump({"encoding": "pointer2_pad", "protocol": "identical to e3 arms"},
              open(os.path.join(out, "meta.json"), "w"))
    print("saved", out, flush=True)


def evaluate(adapter, out):
    """Same protocol as eval_p2 size extrapolation: nbr F1 @ n=20/30/40."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=BASE, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    sizes = {"n20": (gen_split(300, 315, 30, n=20), 20),
             "n30": (gen_split(600, 610, 25, n=30), 30),
             "n40": (gen_split(620, 628, 25, n=40), 40)}   # same seeds as eval_p2
    with open(out, "w") as f:
        for tag, (test, nn) in sizes.items():
            prompts = [build_prompt(s["graph"], s["q"], s["task"], "pointer2_pad", s["meta"])
                       for s in test]
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp,
                            lora_request=LoRARequest("pad", 1, adapter))
            for s, o in zip(test, outs):
                txt = o.outputs[0].text
                pred = parse_answer(txt, "list" if not s["task"].startswith("reach") else "bool")
                gold = s["a"].split("ANSWER: ")[-1]
                f.write(json.dumps({"arm": "pointer2_pad", "size": tag,
                                    "task": s["task"], "gold": gold,
                                    "pred": pred, "score": score(pred, gold, "list")}) + "\n")
            f.flush()
            print(f"[{tag}] done", flush=True)
    # summary
    from collections import defaultdict
    acc = defaultdict(list)
    for line in open(out):
        r = json.loads(line)
        acc[(r["size"], r["task"])].append(r["score"])
    print("\n== equal-length control (pointer2_pad) ==")
    for size in ("n20", "n30", "n40"):
        vals = [v for (s, t), vv in acc.items() if s == size for v in vv]
        print(f"{size}: mean over all tasks = {sum(vals)/len(vals):.3f}")
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["train", "eval", "both"])
    ap.add_argument("--out", default="/mnt/ssd/ange/runs/e3_pointer2_pad")
    ap.add_argument("--results", default=f"{R}/eqlen_control.jsonl")
    a = ap.parse_args()
    register_pad_encoder()
    if a.phase in ("train", "both"):
        train(a.out)
    if a.phase in ("eval", "both"):
        evaluate(os.path.join(a.out, "adapter"), a.results)
