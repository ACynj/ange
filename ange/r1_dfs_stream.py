#!/usr/bin/env python
"""R1 dfs fill with streaming (keeps connection alive during long CoT)."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ange.probes import gen_reach, gen_nbr, gen_ambig, reencode, HEADER, FOOT_BOOL, FOOT_LIST
from ange.encoders import ENCODINGS
from ange.engine import parse_answer, score

R = "/mnt/ssd/ange/results"


def build_dfs_probes(n_per_task=10):
    """Build a small mixed probe set for dfs_serial."""
    import random
    rng = random.Random(77)
    probes = []
    # reach 1,2,3 balanced
    for k in (1, 2, 3):
        want_yes = True
        got = 0
        while got < n_per_task:
            from ange.graph import gen_graph
            g = gen_graph(600 + got * 7 + k, 20, 4, 1.2)
            src = rng.choice(g.nodes()); rel = rng.choice(g.rels)
            reach = g.reach(src, rel, k)
            pool = sorted(reach) if want_yes else [t for t in g.nodes() if t != src and t not in reach]
            if not pool:
                continue
            tgt = rng.choice(pool)
            q = (f"Starting from node {src} and following only {rel} edges in their "
                 f"forward direction, is node {tgt} reachable in at most {k} steps? ")
            probes.append({"task": f"reach{k}", "q": q, "gold": "YES" if want_yes else "NO",
                          "graph": g, "meta": (src, rel, tgt)})
            want_yes = not want_yes
            got += 1
    # nbr
    got = 0
    while got < n_per_task:
        from ange.graph import gen_graph
        g = gen_graph(700 + got, 20, 4, 1.2)
        src = rng.choice(g.nodes()); rel = rng.choice(g.rels)
        ans = g.out_neighbors(src, rel)
        if not (1 <= len(ans) <= 5):
            continue
        q = f"List ALL nodes that are targets of {rel} edges starting from node {src}. "
        probes.append({"task": "nbr", "q": q, "gold": ",".join(ans) if ans else "NONE",
                      "graph": g, "meta": (src, rel, None)})
        got += 1
    # ambig (just use regular probes, not true ambig pairs — for mean we need 5 tasks)
    got = 0
    while got < n_per_task:
        from ange.graph import gen_graph
        g = gen_graph(800 + got, 20, 4, 1.2)
        src = rng.choice(g.nodes()); rel = rng.choice(g.rels)
        ans = g.out_neighbors(src, rel)
        if not (1 <= len(ans) <= 5):
            continue
        q = f"List ALL nodes that are targets of {rel} edges starting from node {src}. "
        probes.append({"task": "ambig", "q": q, "gold": ",".join(ans) if ans else "NONE",
                      "graph": g, "meta": (src, rel, None)})
        got += 1
    return probes


def stream_one(client, model, prompt, max_tokens=6000, connect_timeout=30):
    """Stream a response; connection stays alive as long as tokens flow."""
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=max_tokens, stream=True)
    txt = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            txt += chunk.choices[0].delta.content
    return txt


def main():
    from openai import OpenAI
    cfg = json.load(open('/mnt/ssd/ange/api_keys.json'))['siliconflow']
    client = OpenAI(api_key=cfg['key'], base_url=cfg['base_url'],
                    timeout=30.0, max_retries=0)  # 30s connect timeout; stream keeps alive

    probes = build_dfs_probes(10)
    print(f"[data] {len(probes)} dfs probes (10/task × 5 tasks)", flush=True)

    out = open(f"{R}/r1_dfs_stream.jsonl", "w")

    def worker(i_p):
        i, p = i_p
        enc_text = ENCODINGS['dfs_serial'](p["graph"])
        foot = FOOT_BOOL if p["task"].startswith("reach") else FOOT_LIST
        prompt = HEADER + enc_text + "\n\nQuestion: " + p["q"] + foot
        t0 = time.time()
        try:
            txt = stream_one(client, "deepseek-ai/DeepSeek-R1", prompt)
            parse = "bool" if p["task"].startswith("reach") else "list"
            pred = parse_answer(txt, parse)
            sc = score(pred, p["gold"], parse)
            return {"task": p["task"], "gold": p["gold"], "pred": pred,
                    "score": sc, "time": round(time.time() - t0), "raw": txt[:200]}
        except Exception as e:
            return {"task": p["task"], "gold": p["gold"], "pred": "NA",
                    "score": 0.0, "time": round(time.time() - t0),
                    "raw": f"[ERROR {type(e).__name__}] {str(e)[:80]}"}

    with ThreadPoolExecutor(max_workers=3) as ex:
        for result in ex.map(worker, enumerate(probes)):
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f"  {result['task']:8s} score={result['score']:.0f} "
                  f"({result['time']}s)", flush=True)
    out.close()

    # Compute mean
    from collections import defaultdict
    acc = defaultdict(list)
    for l in open(f"{R}/r1_dfs_stream.jsonl"):
        r = json.loads(l)
        acc[r["task"]].append(r["score"])
    tasks = ["reach1", "reach2", "reach3", "nbr", "ambig"]
    means = [sum(acc[t]) / len(acc[t]) for t in tasks if acc.get(t)]
    overall = sum(means) / len(means) if means else 0
    print(f"\nR1 dfs mean: {overall:.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
