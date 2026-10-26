"""试跑报告里的全部数字：只读仓库里的日志（runs/m2pilot/ 与 runs/m1v2/、runs/m2/），算好写进
runs/m2pilot/report/numbers.json，图画进 reports/m2pilot_figs/。报告里引用的每个数都出自这里。

    python -m roommate.train.report_numbers
"""

import glob
import gzip
import json
import os
from collections import defaultdict

from .metrics import replay_told, session_row, summarize_rows
from .shortrange import needs_first_rate, short_world

ROOT = "runs/m2pilot"
M = os.path.join(ROOT, "machine")
FIG = "reports/m2pilot_figs"
N_BOOT = 2000
BASELINES = ["llm_deepseek-v4-pro", "llm_deepseek-flash", "llm_claude-haiku-5-5", "llm_claude-sonnet-5-5",
             "llm_claude-opus-5-5", "llm_claude-fable-5-1", "ideal", "remember_no_update", "ask_all"]
TAGS = ["pre", "sft", "grpo1", "grpo2"]


def jl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def world(seed):
    p = f"runs/m1v2/seed{seed}/world.json"
    if os.path.exists(p):
        return jload(p)
    from ..world import generate_world
    return generate_world(seed)


# ---------- 联合基线表（全程，种子 1、2、3） ----------

def baseline_rows(policy, seeds=(1, 2, 3)):
    rows = []
    for s in seeds:
        w = world(s)
        plans = {u["uid"]: u for u in w["users"]}
        p = f"runs/m1v2/seed{s}/policies/{policy}/sessions.jsonl"
        if not os.path.exists(p):
            return None
        by = defaultdict(list)
        for r in jl(p):
            by[r["uid"]].append(r)
        for uid, rs in by.items():
            info = replay_told(plans[uid], rs, w["config"]["repeat_mode"])
            sess = {x["idx"]: x for x in plans[uid]["sessions"]}
            rows += [(s, uid, session_row(r, sess[r["idx"]], info[r["idx"]])) for r in rs if r["idx"] in info]
    return rows


def eval_rows(tag, kind, seeds):
    rows = []
    for s in seeds:
        p = os.path.join(M, "eval", tag, kind, f"seed{s}", "sessions.jsonl")
        if not os.path.exists(p):
            return None
        w = world(s) if kind == "full" else short_world(world(s), [0, 2, 4, 6])
        sess = {(u["uid"], x["idx"]): x for u in w["users"] for x in u["sessions"]}
        orig = {u["uid"]: u.get("orig_uid", u["uid"]) for u in w["users"]}
        rows += [(s, orig[r["uid"]], session_row(r, sess[(r["uid"], r["idx"])])) for r in jl(p)]
    return rows


def tijian_subset():
    """体检（硅基流动，思考开）中期结论用到的那 78 次会话：只用缓存重建，得到（种子, 用户, 第几次）。"""
    from .. import m2
    from ..llm import Ledger, LLMClient
    m2.load_prices()
    c = LLMClient(os.path.join(m2.OUT, "llm_cache"), Ledger(None, 0), offline=True)
    keys, recs = set(), []
    for s in (1, 2, 3):
        for r in m2.run_job(c, ("base", "think", s, 0), cache_only=True):
            if r["domain"] != "util":
                keys.add((s, r["uid"], r["idx"]))
                recs.append((s, r))
    return keys, recs


def teacher_numbers(teacher_dir=os.path.join(ROOT, "teacher"), seeds=(101, 102, 103, 104)):
    """老师轨迹：每条规则（1–4、6）筛掉多少，三个阶段的捷径①、②比例；第 5 条（模板）的数取自 SFT 的 data_stats.json。"""
    from .sft import rule_drop, shortcut_rates, teacher_rows
    if not all(os.path.exists(os.path.join(teacher_dir, f"seed{s}", "world.json")) for s in seeds):
        return None
    dropped, stage = defaultdict(int), {"all": [], "after_rule4": [], "after_rule6": []}
    examples = []
    for s in seeds:
        for r, sess, info in teacher_rows(teacher_dir, s):
            row = session_row(r, sess, info)
            stage["all"].append((s, r["uid"], row))
            why = rule_drop(r, row)
            if why in (None, "6_捷径①"):
                stage["after_rule4"].append((s, r["uid"], row))
            if why is None:
                stage["after_rule6"].append((s, r["uid"], row))
            else:
                dropped[why] += 1
                if why == "6_捷径①":
                    examples.append([s, r["uid"], r["idx"], r["week"]])
    out = {"sessions": len(stage["all"]), "dropped_rules_1_4_6": dict(dropped), "rule6_sessions": examples,
           "shortcuts": {k: shortcut_rates(v) for k, v in stage.items()}}
    ds = os.path.join(M, "sft", "data_stats.json")
    if os.path.exists(ds):
        out["sft_data_stats"] = jload(ds)
    return out


def main():
    out = {}
    out["teacher"] = teacher_numbers()
    # 联合基线表
    joint = {}
    for pol in BASELINES:
        rows = baseline_rows(pol)
        if rows:
            joint[pol] = summarize_rows(rows, n_boot=N_BOOT)
    for tag in TAGS:
        rows = eval_rows(tag, "full", (1, 2, 3))
        if rows:
            joint[f"qwen3.5-4b_{tag}"] = summarize_rows(rows, n_boot=N_BOOT)
    out["joint_full"] = joint
    # 短程评测
    short = {}
    for tag in TAGS:
        rows = eval_rows(tag, "short", (201, 202))
        if rows:
            short[tag] = summarize_rows(rows, n_boot=N_BOOT)
    out["short"] = short
    out["needs_first"] = {str(s): needs_first_rate(short_world(world(s), [0, 2, 4, 6])) for s in (201, 202)}
    out["needs_first_train_pool"] = {str(s): needs_first_rate(short_world(world(s))) for s in (301, 302)}
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("written", os.path.join(ROOT, "report", "numbers.json"))


if __name__ == "__main__":
    main()
