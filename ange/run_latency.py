#!/usr/bin/env python
"""Reviewer fix #6: inference-cost comparison across encodings.

For one 7B reader (vLLM, greedy, batch=1 stream), per encoding:
  prompt tokens, prefill+decode wall time for a fixed 1-hop nbr query,
  at graph sizes n=20 and n=40. Also records CoT (R1-style) token cost
  from the paper's measured 4-6k for context.
"""
import json, sys, time
sys.path.insert(0, '/mnt/ssd/ange/code')
from ange.graph import gen_graph
from ange.encoders import ENCODINGS, enc_pointer2
from ange.e3_data import build_prompt
from ange.probes import HEADER, FOOT_LIST

B7 = "/mnt/ssd/ange/models/Qwen2.5-7B-Instruct"
R = "/mnt/ssd/ange/results"

ENCS = [("edge_list", "edge_list"), ("adjacency_list", "adjacency_list"),
        ("nl", "nl"), ("json", "json"), ("dfs_serial", "dfs_serial"),
        ("sog", "sog"), ("pointer_v1", "pointer"), ("pointer_v2", "pointer2")]


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(B7)
    llm = LLM(model=B7, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.7)
    sp = SamplingParams(temperature=0.0, max_tokens=64)

    res = []
    for n in (20, 40):
        g = gen_graph(11 + n, n, 4, 1.6)
        src, rel = g.nodes()[0], g.rels[0]
        q = f"List ALL nodes that are targets of {rel} edges starting from node {src}. "
        for tag, enc in ENCS:
            if enc in ("pointer", "pointer2"):
                prompt = build_prompt(g, q, "nbr", enc, (src, rel, None))
            else:
                gt = ENCODINGS[enc](g)
                prompt = HEADER + gt + "\n\nQuestion: " + q + FOOT_LIST
            ntok = len(tok(prompt, add_special_tokens=False)["input_ids"])
            # warm + timed runs (batch=1)
            msgs = [[{"role": "user", "content": prompt}]]
            llm.chat(msgs, sp)
            t0 = time.time()
            outs = llm.chat(msgs, sp)
            dt = time.time() - t0
            otok = len(outs[0].outputs[0].token_ids)
            res.append({"enc": tag, "n": n, "prompt_tokens": ntok,
                        "latency_s": round(dt, 3), "out_tokens": otok})
            print(res[-1], flush=True)
    json.dump(res, open(f"{R}/latency_bench.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
