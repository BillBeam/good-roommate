"""M1 审阅用的复核统计：按用户重抽样的置信区间、剩余空间比例、失分拆解。

只读 runs/m1/policies/ 下的逐次会话记录，不调用任何模型。
运行：python reports/m1_review_stats.py  → 写出 runs/m1/review_stats.json
"""

import json
import os
import random
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m1", "policies")
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m1", "review_stats.json")
TRIVIAL = ["ask_all", "conservative", "remember_no_update", "majority", "random", "nothing"]
MODELS = ["llm_grok-4.7", "llm_grok-4.3", "llm_grok-4.20-0309-non-reasoning"]
N_BOOT = 5000
QUESTION_COST = 0.15


def load(name):
    with open(os.path.join(ROOT, name, "sessions.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def per_user_totals(records):
    tot = {}
    for r in records:
        tot[r["uid"]] = tot.get(r["uid"], 0.0) + r["grade"]["score"]
    return tot


def main():
    recs = {n: load(n) for n in TRIVIAL + MODELS + ["oracle"]}
    tot = {n: per_user_totals(rs) for n, rs in recs.items()}
    uids = sorted(tot["oracle"])
    mean = {n: sum(t.values()) / sum(tot["oracle"].values()) for n, t in tot.items()}
    best_trivial = max(TRIVIAL, key=lambda n: mean[n])

    rng = random.Random(0)
    boot = {m: {"ratio": [], "headroom": []} for m in MODELS}
    for _ in range(N_BOOT):
        sample = [rng.choice(uids) for _ in uids]
        o = sum(tot["oracle"][u] for u in sample)
        b = sum(tot[best_trivial][u] for u in sample)
        for m in MODELS:
            s = sum(tot[m][u] for u in sample)
            boot[m]["ratio"].append(s / o)
            boot[m]["headroom"].append((s - b) / (o - b))

    def ci(xs):
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]

    out = {
        "note": "比例都以标准策略（oracle）的总分为分母；剩余空间比例 = (模型分 - 最简单策略中最高分) / (oracle 分 - 最简单策略中最高分)",
        "n_users": len(uids),
        "n_bootstrap": N_BOOT,
        "ratio_to_oracle": {n: round(v, 4) for n, v in mean.items()},
        "best_trivial_policy": best_trivial,
        "models": {},
    }
    for m in MODELS:
        rs = recs[m]
        n = len(rs)
        fails = [r for r in rs if not r["grade"]["success"]]
        out["models"][m] = {
            "ratio_to_oracle": round(mean[m], 4),
            "ratio_to_oracle_ci95": ci(boot[m]["ratio"]),
            "p_ratio_below_0.80": round(sum(x < 0.8 for x in boot[m]["ratio"]) / N_BOOT, 4),
            "headroom_share": round((mean[m] - mean[best_trivial]) / (1 - mean[best_trivial]), 4),
            "headroom_share_ci95": ci(boot[m]["headroom"]),
            "loss_from_questions_per_session": round(sum(r["grade"]["n_questions"] for r in rs) * QUESTION_COST / n, 4),
            "loss_from_failures_per_session": round(len(fails) / n, 4),
            "failures_by_type_domain_role": [[f"{t}|{d}|{role}", c] for (t, d, role), c in
                                             Counter((r["type"], r["domain"], r["chosen_role"]) for r in fails).most_common()],
        }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
