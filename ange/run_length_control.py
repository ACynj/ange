#!/usr/bin/env python
"""M5 fix: length-matched control — edge list padded with neutral filler to
JSON token length; 7B frozen; if scores match unpadded edge list, length is
not the mediator. Also tokenizes all encodings with the Qwen tokenizer."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ange.graph import gen_graph
from ange.encoders import ENCODINGS, enc_pointer2
from ange.probes import HEADER, FOOT_BOOL, FOOT_LIST
from ange.engine import parse_answer, score

R = "/mnt/ssd/ange/results"
B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"


def build(g, q, task, filler):
    base = ENCODINGS["edge_list"](g)
    gt = base + ("\n" + "z " * filler if filler else "")
    foot = FOOT_BOOL if task.startswith("reach") else FOOT_LIST
    return HEADER + gt + "\n\nQuestion: " + q + foot


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B7)
    # 1) token-count audit table
    g = gen_graph(11, 20, 4, 1.2)
    audit = {}
    for name, fn in list(ENCODINGS.items()) + [("pointer2", enc_pointer2)]:
        t = fn(g)
        audit[name] = len(tok(t, add_special_tokens=False)["input_ids"])
    el = audit["edge_list"]; js = audit["json"]
    pad_to = js - el
    json.dump({k: v for k, v in audit.items()},
              open(f"{R}/token_audit.json", "w"), indent=1)
    print("token audit:", audit, "| filler tokens:", pad_to)

    # 2) padded control probes (same seeds as E1 protocol)
    from ange.probes import gen_reach, gen_nbr
    probes = gen_reach(21, 1, n=20, n_probes=40) + gen_reach(22, 3, n=20, n_probes=40) \
        + gen_nbr(23, n=20, n_probes=40)
    rng = random.Random(5)
    from vllm import LLM, SamplingParams
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    out = open(f"{R}/length_control.jsonl", "w")
    for tag, filler in [("edge_plain", 0), ("edge_padded", pad_to)]:
        prompts = [build(p.meta["graph"], p.meta["qtext"], p.task, filler) for p in probes]
        outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp)
        for p, o in zip(probes, outs):
            parse = "bool" if p.parse == "bool" else "list"
            pred = parse_answer(o.outputs[0].text, parse)
            out.write(json.dumps({"arm": tag, "task": p.task, "gold": p.gold,
                                  "pred": pred, "score": score(pred, p.gold, parse)}) + "\n")
        out.flush(); print(f"[{tag}] done", flush=True)
    out.close(); print("LENGTH CONTROL DONE")


if __name__ == "__main__":
    main()
