"""老师轨迹：DeepSeek V4 Pro 在新生成的世界里跑完整的 8 次见面，作为 SFT 的原始数据。

    python -m roommate.train.teacher --seeds 101,102,103,104

- 世界由 roommate.world.generate_world 按新种子生成，和评测用的世界（种子 1、2、3 与短程评测的种子）不重合。
- agent 与 M1 第二版的 DeepSeek V4 Pro 基线完全相同（LLMAgent、同一段系统提示、同一组工具、思考开、
  工具调用循环里把 reasoning_content 原样带回），只多记格式问题和 token 用量（track_format）。
- 请求按内容哈希缓存在 OUT/llm_cache/，重跑时已完成的请求不再花钱；账本 OUT/spend_ledger.json 按价目快照记人民币，
  超过 --cny-limit 就停。
- 只在中转的运行器上跑（DEEPSEEK_API_KEY 只放在 Actions secrets 里）。
"""

import argparse
import gzip
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from .. import llm
from ..agent_llm import LLMAgent
from ..env import Env
from ..grading import summarize
from ..llm import ApiError, BudgetExceeded, Ledger, LLMClient
from ..world import generate_world, world_hash, world_stats

OUT = "runs/m2pilot/teacher"
MODEL = "deepseek-v4-pro"
PRICING = "runs/m1v2/pricing"


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_prices():
    for fn in sorted(os.listdir(PRICING)):
        if fn.startswith("deepseek_"):
            with open(os.path.join(PRICING, fn), encoding="utf-8") as f:
                d = json.load(f)
            llm.DEEPSEEK_PRICES.update({m: {"usd": v["usd"], "cny": v["cny"]} for m, v in d["prices"].items()})


def world_for(seed, out=OUT):
    p = os.path.join(out, f"seed{seed}", "world.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    w = generate_world(seed)
    dump(p, w)
    dump(os.path.join(out, f"seed{seed}", "world_stats.json"), dict(world_stats(w), world_hash=world_hash(w)))
    return w


def save_records(d, recs):
    plain, trans = [], []
    for r in recs:
        r = dict(r)
        t = r.pop("transcript", None)
        plain.append(r)
        if t is not None:
            trans.append({"uid": r["uid"], "idx": r["idx"], "messages": t})
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "sessions.jsonl"), "w", encoding="utf-8") as f:
        for r in plain:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(os.path.join(d, "transcripts.jsonl.gz"), "wt", encoding="utf-8") as f:
        for t in trans:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    s = summarize(plain)
    dump(os.path.join(d, "summary.json"), s)
    return s


def compact_cache(out=OUT):
    cd = os.path.join(out, "llm_cache")
    for fn in os.listdir(cd) if os.path.isdir(cd) else []:
        if fn.endswith(".jsonl"):
            src, dst = os.path.join(cd, fn), os.path.join(cd, fn + ".gz")
            with open(src, "rb") as f, gzip.open(dst, "ab") as g:
                shutil.copyfileobj(f, g)
            os.remove(src)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,102,103,104")
    ap.add_argument("--cny-limit", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    load_prices()
    path = os.path.join(args.out, "spend_ledger.json")
    ledger = Ledger(path, limit_usd=1e9)
    if not os.path.exists(path):
        ledger.data["deepseek_cny_limit"] = args.cny_limit
        dump(path, ledger.data)
    client = LLMClient(os.path.join(args.out, "llm_cache"), ledger, offline=args.offline,
                       blocked_providers={"xai", "siliconflow", "anthropic"}, max_inflight=96)
    stopped = []

    def one(seed):
        world = world_for(seed, args.out)
        agent = LLMAgent(client, MODEL, seed=seed, echo_reasoning=True, track_format=True,
                         truncate_on_cache_miss=args.offline)
        env = Env(world)
        try:
            env.run(agent, workers=args.workers)
        except (BudgetExceeded, ApiError) as e:
            stopped.append(f"seed{seed}: {type(e).__name__}: {e}"[:300])
            print("停止：", stopped[-1], flush=True)
            return
        s = save_records(os.path.join(args.out, f"seed{seed}", "policies", f"llm_{MODEL}"), env.records())
        print(f"seed{seed}: 平均分 {s['mean_score']:.3f} 选对 {s['success_rate']:.3f} 提问 {s['avg_questions']:.2f}",
              flush=True)

    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        list(pool.map(one, seeds))
    dump(os.path.join(args.out, "costs.json"), {"cumulative": client.ledger._load(), "this_run": client.ledger.session,
                                                "retries": client.retries, "stopped": stopped or None})
    compact_cache(args.out)
    if stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
