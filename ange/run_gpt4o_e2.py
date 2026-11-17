#!/usr/bin/env python
"""B experiment: strongest closed readers + pointer grammar on real KGs.
GPT-4o (and GLM-5.3) frozen, NELL-995 + FB15k-237 tail prediction,
200 probes each, encodings edge_list vs pointer2. Question: does the
strongest reader + attention-native encoding break the prompt-level
ceiling (0.34) that no frozen family crossed?
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, '/mnt/ssd/ange/code')

from openai import OpenAI
from ange.kgdata import KG, gen_lp_mc
from ange.run_e2 import parse_letter
from ange.e3_real import prompt_train

R = "/mnt/ssd/ange/results"
CFG = json.load(open('/mnt/ssd/ange/api_keys.json'))

MODELS = {
    "gpt4o": (CFG["dmxapi"]["key"], CFG["dmxapi"]["base_url"], "gpt-4o", 200),
    "glm53": (CFG["glm"]["key"], CFG["glm"]["base_url"], "glm-5.3", 4096),
}


def run_model(tag, key, base, model, ds, enc, probes, kg, max_tok=200):
    client = OpenAI(api_key=key, base_url=base, timeout=120, max_retries=2)
    out = open(f"{R}/gpt4o_e2_{ds.replace('-','')}.jsonl", "a")
    hits = 0
    for i, p in enumerate(probes):
        pr = prompt_train(kg, p, enc)
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=max_tok,
                messages=[{"role": "user", "content": pr}])
            txt = resp.choices[0].message.content or ""
        except Exception as e:
            txt = f"[ERROR {type(e).__name__}]"
        pred = parse_letter(txt)
        sc = 1.0 if pred == p["gold"] else 0.0
        hits += sc
        out.write(json.dumps({"model": tag, "ds": ds, "enc": enc,
                              "gold": p["gold"], "pred": pred, "score": sc,
                              "raw": txt[:150]}) + "\n")
        if (i + 1) % 25 == 0:
            out.flush()
            print(f"[{tag} {ds} {enc}] {i+1}/{len(probes)} acc={hits/(i+1):.3f}",
                  flush=True)
    out.close()
    print(f"== {tag} {ds} {enc}: {hits/len(probes):.3f}", flush=True)
    return hits / len(probes)


def main():
    results = {}
    done = set()
    for f in ("NELL", "FB15k237"):
        try:
            for line in open(f"{R}/gpt4o_e2_{f}.jsonl"):
                r = json.loads(line)
                done.add((r["model"], r["ds"], r["enc"]))
        except FileNotFoundError:
            pass
    print("done cells:", len(done), flush=True)
    for ds in ("NELL", "FB15k-237"):
        kg = KG(ds + "-betae")
        probes = gen_lp_mc(kg, 100, seed=5) + gen_lp_mc(kg, 100, seed=6)
        for enc in ("edge_list", "pointer2"):
            for tag, (key, base, model, mt) in MODELS.items():
                if (tag, ds, enc) in done:
                    print(f"[skip {tag} {ds} {enc}]", flush=True)
                    continue
                acc = run_model(tag, key, base, model, ds, enc, probes, kg, mt)
                results[f"{tag}_{ds}_{enc}"] = acc
    json.dump(results, open(f"{R}/gpt4o_e2_summary.json", "w"), indent=1)
    print("\n== SUMMARY ==")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
    print("GPT4O E2 DONE")


if __name__ == "__main__":
    main()
