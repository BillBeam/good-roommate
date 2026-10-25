"""M2 起点测量的统计与报告：只读 runs/m2/ 与 runs/m1v2/ 下的逐次会话记录，不调用任何模型。

- 分数只算和偏好有关的会话；置信区间按用户重抽样（每个世界内分别重抽，5000 次），与 M1 第二版同一套代码（stats.py）。
- 剩余空间比例 = (得分 − 最好的简单策略) / (理想记忆策略 − 最好的简单策略)；最好的简单策略沿用 M1 第二版，
  是"记住但不更新"（remember_no_update）。
- 小模型特有的失败（每次与偏好有关的会话算一次）：工具调用格式错误（见 agent_llm.FORMAT_KINDS）、没调 choose_option、
  用满 14 轮还没选。
- 失分拆解同 M1：选错的会话，对照组（知道真实状态）选对了 → 没记住或没察觉变化；对照组也错 → 挑店或格式问题，
  后者再按"没选出任何选项"和"选了错的"分开。
- 组内差异：种子 VAR_SEED 上每个用户的 8 次会话按训练温度重复跑 K 遍。一个用户的 K 遍是一组（GRPO 的一组），
  看每组 K 个总分的差异；另看每个（用户、第几次会话）在 K 遍里分数是否完全相同。
"""

import gzip
import json
import os
import statistics

from . import stats
from .agent_llm import FORMAT_KINDS, MAX_TURNS

REF = [("llm_deepseek-v4-pro", "DeepSeek V4 Pro（温度 0，思考开）"),
       ("llm_deepseek-flash", "DeepSeek Flash（温度 0，思考开）"),
       ("ideal", "理想记忆策略"),
       ("remember_no_update", "记住但不更新")]
BEST_TRIVIAL = "remember_no_update"


def _jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _pref(recs):
    return [r for r in recs if r["domain"] != "util"]


def failure_counts(recs):
    pref = _pref(recs)
    kinds = {k: 0 for k in FORMAT_KINDS}
    sess_kinds = {k: 0 for k in FORMAT_KINDS}
    any_fmt = 0
    for r in pref:
        iss = r.get("format_issues") or []
        hard = [i for i in iss if i["kind"] != "no_tool_call"]
        any_fmt += bool(hard)
        for i in iss:
            kinds[i["kind"]] += 1
        for k in {i["kind"] for i in iss}:
            sess_kinds[k] += 1
    return {
        "n_sessions": len(pref),
        "sessions_with_format_error": any_fmt,          # 不含 no_tool_call（只说话没调工具，会被提醒）
        "format_issue_turns": kinds,
        "sessions_by_format_kind": sess_kinds,
        "no_choose_option": sum(r["chosen"] is None for r in pref),
        "hit_max_turns": sum(bool(r.get("hit_max_turns")) for r in pref),
        "errors": sum(bool(r.get("error")) for r in pref),
        "avg_turns": sum(r.get("turns", 0) for r in pref) / len(pref),
        "max_turns": MAX_TURNS,
    }


def usage_summary(recs):
    """每次会话（全部会话，含无关会话）的 token 用量；只有 track_format 的运行才有。"""
    us = [r["usage"] for r in recs if r.get("usage")]
    if not us:
        return None
    n = len(us)
    return {"n_sessions": n,
            "calls_per_session": sum(u["calls"] for u in us) / n,
            "prompt_tokens_per_session": sum(u["prompt_tokens"] for u in us) / n,
            "completion_tokens_per_session": sum(u["completion_tokens"] for u in us) / n,
            "reasoning_tokens_per_session": sum(u["reasoning_tokens"] for u in us) / n,
            "max_prompt_tokens_per_call_est": None}


def control_breakdown(normal, control):
    ctl = {(r["_seed"], r["uid"], r["idx"]): r for r in control}
    pref = [r for r in normal if r["domain"] != "util" and (r["_seed"], r["uid"], r["idx"]) in ctl]
    b = stats.loss_breakdown(pref, control)
    fails = [r for r in pref if not r["grade"]["success"]]
    other = [r for r in fails if not ctl[(r["_seed"], r["uid"], r["idx"])]["grade"]["success"]]
    b["n_fail_other_no_choice"] = sum(r["chosen"] is None for r in other)
    b["n_fail_other_wrong_choice"] = sum(r["chosen"] is not None for r in other)
    b["n_fail_memory_no_choice"] = sum(r["chosen"] is None for r in fails if r not in other)
    cp = _pref(control)
    b["control_success_rate"] = sum(r["grade"]["success"] for r in cp) / len(cp)
    b["control_mean_score"] = sum(r["grade"]["score"] for r in cp) / len(cp)
    b["control_avg_questions"] = sum(len(r["questions"]) for r in cp) / len(cp)
    b["control_no_choice"] = sum(r["chosen"] is None for r in cp)
    b["control_n_sessions"] = len(cp)
    return b


def variance_stats(out, mode, seed):
    base = os.path.join(out, "variance", f"seed{seed}", f"{mode}_t1")
    if not os.path.isdir(base):
        return None
    reps = sorted(d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, "sessions.jsonl")))
    if len(reps) < 2:
        return None
    runs = {d: _jsonl(os.path.join(base, d, "sessions.jsonl")) for d in reps}
    users = sorted({r["uid"] for r in next(iter(runs.values()))})
    totals = {u: [] for u in users}
    per_sess = {}
    for d, recs in runs.items():
        pu = stats.per_user(recs)
        for u in users:
            totals[u].append(round(pu[u][0], 6))
        for r in _pref(recs):
            per_sess.setdefault((r["uid"], r["idx"]), []).append(round(r["grade"]["score"], 6))
    k = len(reps)
    sds = {u: statistics.pstdev(v) for u, v in totals.items()}
    rngs = {u: max(v) - min(v) for u, v in totals.items()}
    same_user = [u for u, v in totals.items() if len(set(v)) == 1]
    sess_same = [key for key, v in per_sess.items() if len(set(v)) == 1]
    sess_same_all_ok = [key for key in sess_same if per_sess[key][0] > 0.5 - 1e-9 and
                        all(x == per_sess[key][0] for x in per_sess[key])]
    first = {key: v for key, v in per_sess.items() if key[1] == 0}
    succ = {key: [x > 0.5 - 1e-9 for x in v] for key, v in per_sess.items()}
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    sdl = list(sds.values())
    return {
        "mode": mode, "seed": seed, "k": k, "reps": reps, "n_users": len(users),
        "user_totals": totals,
        "user_total_sd": {"mean": statistics.mean(sdl), "median": statistics.median(sdl),
                          "p10": q(sdl, 0.1), "p90": q(sdl, 0.9), "min": min(sdl), "max": max(sdl)},
        "user_total_range": {"mean": statistics.mean(rngs.values()), "max": max(rngs.values())},
        "users_all_identical": len(same_user), "users_all_identical_share": len(same_user) / len(users),
        "n_session_groups": len(per_sess),
        "session_groups_all_identical": len(sess_same),
        "session_groups_all_identical_share": len(sess_same) / len(per_sess),
        "session_groups_identical_all_correct": len(sess_same_all_ok),
        "session_groups_identical_all_wrong_or_costly": len(sess_same) - len(sess_same_all_ok),
        "session_groups_success_mixed": sum(1 for v in succ.values() if 0 < sum(v) < len(v)),
        "first_session_groups": len(first),
        "first_session_groups_all_identical": sum(1 for v in first.values() if len(set(v)) == 1),
        "mean_score_by_rep": {d: stats.pooled_mean({seed: stats.per_user(runs[d])}) for d in reps},
        "mean_score_all_reps": statistics.mean(
            stats.pooled_mean({seed: stats.per_user(runs[d])}) for d in reps),
        "failures_all_reps": failure_counts([r for d in reps for r in runs[d]]),
        "usage_all_reps": usage_summary([r for d in reps for r in runs[d]]),
    }


def write_stats(out, m1, seeds):
    from .m2 import MODES, SHORT, VAR_MODES, VAR_SEED
    pols = {}
    for s in seeds:
        p1 = stats.load_policies(m1, s)
        p2 = stats.load_policies(out, s)
        pols[s] = {k: p1[k] for k, _ in REF if k in p1}
        pols[s].update(p2)
        for recs in pols[s].values():
            for r in recs:
                r["_seed"] = s
    n_pref = {s: len(_pref(pols[s]["ideal"])) for s in seeds}
    keys = [k for k in sorted(set().union(*[set(p) for p in pols.values()]))
            if all(k in pols[s] and len(_pref(pols[s][k])) == n_pref[s] for s in seeds)]
    missing = sorted(k for k in set().union(*[set(p) for p in pols.values()]) if k not in keys)
    per = {k: {s: stats.per_user(pols[s][k]) for s in seeds} for k in keys}
    means = {k: stats.pooled_mean(per[k]) for k in keys}
    boot = stats.bootstrap(per, seeds, lambda m: m)
    bt, idl = BEST_TRIVIAL, "ideal"
    share = lambda m, k: (m[k] - m[bt]) / (m[idl] - m[bt])
    res = {"seeds": seeds, "n_bootstrap": stats.N_BOOT, "best_trivial": bt, "rows": {}, "incomplete": missing}
    for k in keys:
        allr = [r for s in seeds for r in pols[s][k]]
        pref = _pref(allr)
        row = {"mean_score": means[k], "ci95": stats._ci([b[k] for b in boot]),
               "success_rate": sum(r["grade"]["success"] for r in pref) / len(pref),
               "avg_questions": sum(len(r["questions"]) for r in pref) / len(pref),
               "headroom_share": share(means, k), "headroom_share_ci95": stats._ci([share(b, k) for b in boot]),
               "by_seed": {s: stats.pooled_mean({s: per[k][s]}) for s in seeds},
               "n_sessions": len(pref), "source": "m1v2" if k in dict(REF) else "m2",
               "memory_write_session_rate": sum(any(op[0] == "write" for op in r["memory_ops"]) for r in pref) / len(pref),
               "told_before_questions": sum(q["repeated"] for r in pref for q in r["questions"])}
        if k.startswith(f"llm_{SHORT}"):
            row["failures"] = failure_counts(allr)
            row["usage"] = usage_summary(allr)
            row["util_success_rate"] = (lambda u: sum(r["grade"]["success"] for r in u) / len(u))(
                [r for r in allr if r["domain"] == "util"])
        res["rows"][k] = row
    # 和 V4 Pro 的差，按同一批重抽样
    for k in keys:
        if k.startswith(f"llm_{SHORT}") and "llm_deepseek-v4-pro" in keys:
            d = [b[k] - b["llm_deepseek-v4-pro"] for b in boot]
            res["rows"][k]["minus_v4pro"] = means[k] - means["llm_deepseek-v4-pro"]
            res["rows"][k]["minus_v4pro_ci95"] = stats._ci(d)
    res["control"] = {}
    for mode in MODES:
        nk, ck = f"llm_{SHORT}_{mode}", f"llm_{SHORT}_{mode}_true_memory"
        normal = [r for s in seeds if nk in pols[s] for r in pols[s][nk]]
        control = [r for s in seeds if ck in pols[s] for r in pols[s][ck]]
        if normal and control:
            res["control"][mode] = control_breakdown(normal, control)
            res["control"][mode]["seeds"] = sorted({r["_seed"] for r in control})
    for k in ("llm_deepseek-v4-pro", "llm_deepseek-flash"):
        g = _load(os.path.join(m1, "gates.json"))["loss_breakdown"].get(k)
        if g:
            res["control"][k] = g
    res["variance"] = {m: variance_stats(out, m, VAR_SEED) for m in VAR_MODES}
    led_p = os.path.join(out, "spend_ledger.json")
    res["spend"] = _load(led_p) if os.path.exists(led_p) else None
    calls_p = os.path.join(out, "spend_calls.jsonl")
    if os.path.exists(calls_p):
        calls = _jsonl(calls_p)
        lat = sorted(c["latency_s"] for c in calls if "latency_s" in c)
        res["api_calls"] = {"n": len(calls),
                            "prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
                            "completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls),
                            "latency_median_s": lat[len(lat) // 2] if lat else None,
                            "latency_p90_s": lat[int(0.9 * len(lat))] if lat else None,
                            "first_call": calls[0]["t"] if calls else None, "last_call": calls[-1]["t"] if calls else None}
    # 估算：老师（V4 Pro）每次会话的花费 = M1 第二版账本里 V4 Pro 普通运行的人民币 / 它跑过的会话数
    led1 = _load(os.path.join(m1, "spend_ledger.json"))["by_model_role"].get("deepseek-v4-pro|agent")
    n_v4 = sum(len(pols[s]["llm_deepseek-v4-pro"]) for s in seeds if "llm_deepseek-v4-pro" in pols[s])
    nk = f"llm_{SHORT}_nothink"
    if led1 and n_v4 and nk in res["rows"] and res["rows"][nk].get("usage"):
        res["teacher_cost"] = {"cny": led1["cost_cny"], "sessions": n_v4, "cny_per_session": led1["cost_cny"] / n_v4,
                               "source": os.path.join(m1, "spend_ledger.json")}
        res["estimate"] = estimate(res["rows"][nk]["usage"], led1["cost_cny"] / n_v4)
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def write_report(out, path):
    """报告正文由 reports/m2-start.md 手写，数字全部来自 stats.json；这里只检查 stats.json 存在。"""
    if not os.path.exists(os.path.join(out, "stats.json")):
        raise RuntimeError("先跑 stats 阶段")


# ---------- 试跑与正式训练的时长、费用估算 ----------
# 实测量：本轮 Qwen3.5-4B 每次会话的调用数和 token 数（usage_summary），M1 第二版 DeepSeek V4 Pro 每次会话的花费。
# 假设量（没有实测，写进 stats.json，报告里单独标出）：吞吐、步数、组大小、单价。单价来自选型报告。
ASSUME = {
    "teacher_worlds_pilot": 4, "teacher_worlds_full": 20, "sessions_per_world": 192,
    "sft_epochs": 2, "sft_tokens_per_session": 3000,
    "grpo_steps_pilot": 50, "grpo_steps_full": 300,
    "grpo_users_per_step_pilot": 4, "grpo_users_per_step_full": 8, "grpo_group": 8,
    # 单卡吞吐（tokens/秒），低—高两档。4B、LoRA、bf16。Qwen3.5 的线性注意力层训练偏慢（ms-swift 已知问题），取偏低值。
    "gpu": {
        "rtx4090_24g": {"train_tps": [1500, 3000], "decode_tps": [1500, 3000], "prefill_tps": [10000, 20000]},
        "a800_80g": {"train_tps": [4000, 8000], "decode_tps": [3000, 6000], "prefill_tps": [25000, 50000]},
    },
    "uncached_prefill_share": 0.35,     # 有前缀缓存时，每次调用需要重新算的 prompt 比例
    "setup_hours": [3, 6],              # 装环境、调通的上机时长（试跑一次性）
    "price_cny_per_hour": {             # 选型报告里的官方价（优云智算定价页，2026-10-25）
        "rtx4090_24g": {"on_demand": 1.92, "spot": 1.31},
        "a800_80g": {"on_demand": 7.36, "spot": 3.02},
    },
}


def estimate(usage, teacher_cny_per_session, a=ASSUME):
    """usage：Qwen 每次会话的 token 用量（usage_summary 的结果，非思考模式）。返回试跑、正式训练的时长与费用区间。"""
    calls, ptok, ctok = (usage["calls_per_session"], usage["prompt_tokens_per_session"],
                         usage["completion_tokens_per_session"])

    def grpo_hours(gpu, steps, users):
        g = a["gpu"][gpu]
        sess = users * a["grpo_group"] * 8
        out = []
        for i in (1, 0):   # 先算快的一档（低时长），再算慢的
            roll = sess * (ctok / g["decode_tps"][i] + ptok * a["uncached_prefill_share"] / g["prefill_tps"][i])
            train = sess * a["sft_tokens_per_session"] * 3 / g["train_tps"][i]   # 前向+反向，另算一次旧策略的 logprob
            out.append(steps * (roll + train) / 3600)
        return out

    def sft_hours(gpu, worlds):
        g = a["gpu"][gpu]
        tok = worlds * a["sessions_per_world"] * a["sft_tokens_per_session"] * a["sft_epochs"]
        return [tok / g["train_tps"][1] / 3600, tok / g["train_tps"][0] / 3600]

    res = {"assumptions": a, "measured": {"qwen_calls_per_session": calls, "qwen_prompt_tokens_per_session": ptok,
                                           "qwen_completion_tokens_per_session": ctok,
                                           "teacher_cny_per_session": teacher_cny_per_session}}
    for name, gpu, worlds, steps, users, setup in (
            ("pilot", "rtx4090_24g", a["teacher_worlds_pilot"], a["grpo_steps_pilot"], a["grpo_users_per_step_pilot"], a["setup_hours"]),
            ("pilot_a800", "a800_80g", a["teacher_worlds_pilot"], a["grpo_steps_pilot"], a["grpo_users_per_step_pilot"], a["setup_hours"]),
            ("full", "a800_80g", a["teacher_worlds_full"], a["grpo_steps_full"], a["grpo_users_per_step_full"], [0, 0])):
        sft = sft_hours(gpu, worlds)
        grpo = grpo_hours(gpu, steps, users)
        hours = [setup[i] + sft[i] + grpo[i] for i in (0, 1)]
        p = a["price_cny_per_hour"][gpu]
        teacher = worlds * a["sessions_per_world"] * teacher_cny_per_session
        res[name] = {"gpu": gpu, "teacher_sessions": worlds * a["sessions_per_world"], "teacher_cny": teacher,
                     "setup_hours": setup, "sft_hours": sft, "grpo_hours": grpo, "gpu_hours": hours,
                     "gpu_cny_on_demand": [h * p["on_demand"] for h in hours],
                     "gpu_cny_spot": [h * p["spot"] for h in hours],
                     "total_cny_on_demand": [teacher + h * p["on_demand"] for h in hours]}
    return res
