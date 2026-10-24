"""M1 审阅用的复核统计：按用户重抽样的置信区间、剩余空间比例、失分拆解。

只读 runs/m1/policies/ 下的逐次会话记录，不调用任何模型。
运行：python reports/m1_review_stats.py  → 写出 runs/m1/review_stats.json
"""

import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


TASTE_WORDS = ["辣", "清淡", "口味", "菜系", "火锅"]
TOO_MANY = "一下问太多了"
# 人工逐条核对后认定的"答非所问"：问题只问辣不辣／清淡（属于"口味"话题），
# 却被归类成"饮食"（过敏、饮食限制）来回答。(用户, 会话序号, 第几个问题)
MISROUTED = [("u07", 3, 0), ("u11", 5, 0), ("u19", 6, 1), ("u20", 4, 1)]


def misroutes(rs):
    """模拟用户把关于口味的问题答成了别的话题。"""
    by = {(r["uid"], r["idx"]): r for r in rs}
    qs = [(r, q) for r in rs for q in r["questions"]]
    screen = [(r, q) for r, q in qs if any(w in q["text"] for w in TASTE_WORDS) and "口味" not in q["topics"]]
    cases = []
    for uid, idx, qi in MISROUTED:
        r = by[(uid, idx)]
        q = r["questions"][qi]
        cases.append({"uid": uid, "idx": idx, "question": q["text"], "answered_topics": q["topics"],
                      "answer": q["answer"], "session_success": r["grade"]["success"]})
    # u07：第 4 次会话（idx 3）那条误导回答写进记忆后，之后再也没问，后面的会话全部选错
    later = [by[("u07", i)] for i in range(4, 8)]
    return {
        "note": "keyword_screen 是关键词粗筛：问题提到口味相关词、但实际回答的话题里没有'口味'；"
                "其中大多是一次问了两个以上方面、用户已明说'一下问太多了'，属于设计内的限制。"
                "cases 是人工核对后确认的答非所问。",
        "total_questions": len(qs),
        "keyword_screen": len(screen),
        "keyword_screen_with_too_many_notice": sum(TOO_MANY in q["answer"] for _, q in screen),
        "cases": cases,
        "direct_failures": sum(not c["session_success"] for c in cases),
        "u07_later_sessions": [{"idx": r["idx"], "n_questions": r["grade"]["n_questions"],
                                "chosen_role": r["chosen_role"], "success": r["grade"]["success"]} for r in later],
        "grok47_total_failures": sum(not r["grade"]["success"] for r in rs),
    }


BLANKET_DIET = "饮食上没什么特别限制"
HEALTH_EVENTS = {"怀孕"}


def contradictions(world, recs):
    """用户因健康原因不能吃辣（怀孕事件之后，或本次临时情况是嗓子发炎这类），
    被问到时却说"饮食上没什么特别限制"、且同一句回答里没提不吃辣：用户自相矛盾。
    单纯不爱吃辣的用户说"没什么特别限制"不算矛盾，不计入。"""
    from roommate.usersim import SimUser
    health = {}
    for u in world["users"]:
        sim, preg = SimUser(u), False
        for s in u["sessions"]:
            sim.begin_session(s)
            if s.get("event") and s["event"]["cn"] in HEALTH_EVENTS:
                preg = True
            t = s.get("temp")
            health[(u["uid"], s["idx"])] = sim.effective()["spicy"] == "不吃辣" and (
                preg or bool(t and t["attr"] == "spicy"))
            sim.end_session()
    out = {}
    for m in MODELS:
        hits = [{"uid": r["uid"], "idx": r["idx"], "question": q["text"], "answer": q["answer"],
                 "session_success": r["grade"]["success"]}
                for r in recs[m] for q in r["questions"]
                if health[(r["uid"], r["idx"])] and BLANKET_DIET in q["answer"] and "不吃辣" not in q["answer"]]
        out[m] = {"n": len(hits), "cases": hits}
    return {"note": "用户因健康原因不能吃辣，回答里却说'" + BLANKET_DIET + "'且没提不吃辣", "by_model": out}


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
    out["simulator_misroutes"] = misroutes(recs["llm_grok-4.7"])
    with open(os.path.join(ROOT, "..", "world.json"), encoding="utf-8") as f:
        out["simulator_contradictions"] = contradictions(json.load(f), recs)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
