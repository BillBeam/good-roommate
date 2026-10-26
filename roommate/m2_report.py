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

# DeepSeek 两行的实际口径：思考开时 temperature 不生效（请求里带的 0 没用上），按接口默认采样；
# top_p 按官方文档，V4 Pro 带工具时固定 0.95，Flash 没传、取默认值 1。出处见 runs/m1v2/docs/deepseek_sampling_2026-10-26.json
REF = [("llm_deepseek-v4-pro", "DeepSeek V4 Pro（思考开，默认采样，top_p 0.95）"),
       ("llm_deepseek-flash", "DeepSeek Flash（思考开，默认采样，top_p 1.0）"),
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


def load_transcripts(d):
    p = os.path.join(d, "transcripts.jsonl.gz")
    if not os.path.exists(p):
        return {}
    return {(t["uid"], t["idx"]): t["messages"] for t in _jsonl(p)}


def loop_stats(recs, trans):
    """"兜圈子"：同一问题在一次会话里重复问、跑到 14 轮上限、空白填充（被长度上限截断、正文几乎全是空白的一轮）。
    只算与偏好有关的会话。另记"同一个工具调用原样重复 3 次以上"，不并入合计。"""
    pref = _pref(recs)
    rep_q = cap = pad = anyl = same_call = 0
    for r in pref:
        qs = r["questions"]
        keys = [tuple(sorted(q.get("declared") or q.get("attributes") or [])) for q in qs]
        texts_ = [q["text"].strip() for q in qs]
        rq = len(set(keys)) < len(keys) or len(set(texts_)) < len(texts_)
        hc = bool(r.get("hit_max_turns"))
        msgs = trans.get((r["uid"], r["idx"]), [])
        asst = [m for m in msgs if m.get("role") == "assistant"]
        pd = False
        for i in r.get("format_issues") or []:
            if i["kind"] == "truncated" and 0 < i["turn"] <= len(asst):
                c = asst[i["turn"] - 1].get("content") or ""
                if len(c) > 200 and len(c.strip()) < 0.2 * len(c):
                    pd = True
        calls = [json.dumps(tc["function"], sort_keys=True, ensure_ascii=False) for m in asst for tc in m.get("tool_calls") or []]
        sc = any(calls.count(c) >= 3 for c in set(calls))
        rep_q += rq
        cap += hc
        pad += pd
        anyl += rq or hc or pd
        same_call += sc
    n = len(pref)
    return {"n_sessions": n, "repeat_question": rep_q, "hit_max_turns": cap, "whitespace_padding": pad,
            "any_loop": anyl, "any_loop_share": anyl / n if n else None,
            "same_tool_call_3plus": same_call, "transcripts_found": sum((r["uid"], r["idx"]) in trans for r in pref)}


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


def variance_stats(out, mode, seed, sub="variance"):
    base = os.path.join(out, sub, f"seed{seed}", f"{mode}_t1")
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
        "loops_all_reps": loop_stats([dict(r, uid=(d, r["uid"])) for d in reps for r in runs[d]],
                                     {((d, kk[0]), kk[1]): v for d in reps
                                      for kk, v in load_transcripts(os.path.join(base, d)).items()}),
        "usage_all_reps": usage_summary([r for d in reps for r in runs[d]]),
    }


def write_stats(out, m1, seeds):
    from .m2 import MODES, SHORT, VAR_MODES, VAR_SEED  # noqa: F401
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
            tr = {}
            for s in seeds:
                tr.update({(s,) + kk: v for kk, v in load_transcripts(os.path.join(out, f"seed{s}", "policies", k)).items()})
            row["loops"] = loop_stats([dict(r, uid=(r["_seed"], r["uid"])) for r in allr],
                                      {((kk[0], kk[1]), kk[2]): v for kk, v in tr.items()})
            row["failures"] = failure_counts(allr)
            row["usage"] = usage_summary(allr)
            row["util_success_rate"] = (lambda u: sum(r["grade"]["success"] for r in u) / len(u))(
                [r for r in allr if r["domain"] == "util"])
        res["rows"][k] = row
    # 只跑了部分种子的 Qwen 组（"知道真实状态"对照组只跑种子 1）：不进主表，单独给失败、兜圈子和用量
    res["partial_rows"] = {}
    for k in missing:
        if not k.startswith(f"llm_{SHORT}"):
            continue
        ss = [s for s in seeds if k in pols[s]]
        allr = [r for s in ss for r in pols[s][k]]
        pref = _pref(allr)
        tr = {}
        for s in ss:
            tr.update({((s, kk[0]), kk[1]): v for kk, v in load_transcripts(os.path.join(out, f"seed{s}", "policies", k)).items()})
        res["partial_rows"][k] = {
            "seeds": ss, "n_sessions": len(pref),
            "mean_score": sum(r["grade"]["score"] for r in pref) / len(pref),
            "success_rate": sum(r["grade"]["success"] for r in pref) / len(pref),
            "avg_questions": sum(len(r["questions"]) for r in pref) / len(pref),
            "failures": failure_counts(allr), "usage": usage_summary(allr),
            "loops": loop_stats([dict(r, uid=(r["_seed"], r["uid"])) for r in allr], tr)}
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
        per_min = {}
        for c in calls:
            per_min[c["t"][:16]] = per_min.get(c["t"][:16], 0) + c["usage"].get("prompt_tokens", 0) + c["usage"].get("completion_tokens", 0)
        busy = sorted(v for v in per_min.values() if v > 0)
        res["api_calls"] = {"n": len(calls),
                            "prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
                            "completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls),
                            "latency_median_s": lat[len(lat) // 2] if lat else None,
                            "latency_p90_s": lat[int(0.9 * len(lat))] if lat else None,
                            "first_call": calls[0]["t"] if calls else None, "last_call": calls[-1]["t"] if calls else None,
                            "tokens_per_minute": {"n_minutes": len(busy), "median": busy[len(busy) // 2] if busy else None,
                                                  "p90": busy[int(0.9 * len(busy))] if busy else None,
                                                  "max": busy[-1] if busy else None}}
    # 估算：老师（V4 Pro）每次会话的花费 = M1 第二版账本里 V4 Pro 普通运行的人民币 / 它跑过的会话数
    led1 = _load(os.path.join(m1, "spend_ledger.json"))["by_model_role"].get("deepseek-v4-pro|agent")
    n_v4 = sum(len(pols[s]["llm_deepseek-v4-pro"]) for s in seeds if "llm_deepseek-v4-pro" in pols[s])
    # 训练用哪种思考模式待定：两种模式各估一遍，都用种子 1–3、温度 0 的实测量（口径相同，不把任何一种当默认）
    res.pop("estimate", None)
    res["estimate_by_mode"] = {}
    if led1 and n_v4:
        tc = led1["cost_cny"] / n_v4
        res["teacher_cost"] = {"cny": led1["cost_cny"], "sessions": n_v4, "cny_per_session": tc,
                               "source": os.path.join(m1, "spend_ledger.json")}
        ip = os.path.join(out, "interim", "interim_stats.json")
        interim = _load(ip).get("base", {}) if os.path.exists(ip) else {}
        for mode in MODES:
            u = (res["rows"].get(f"llm_{SHORT}_{mode}") or {}).get("usage")
            src = "种子 1–3 全量、温度 0"
            if not u and (interim.get(mode) or {}).get("usage"):
                u, src = interim[mode]["usage"], "中期子集、温度 0（每个用户的前几次会话）"
            if u:
                res["estimate_by_mode"][mode] = dict(estimate(u, tc), usage_source=src)
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res




# ---------- 试跑与正式训练的时长、费用估算 ----------
# 实测量：本轮 Qwen3.5-4B 每次会话的调用数和 token 数（usage_summary），M1 第二版 DeepSeek V4 Pro 每次会话的花费。
# 假设量（没有实测，写进 stats.json，报告里单独标出）：吞吐、步数、组大小、单价。单价来自选型报告。
ASSUME = {
    "teacher_worlds_pilot": 4, "teacher_worlds_full": 20, "sessions_per_world": 192,
    "sft_epochs": 2, "sft_tokens_per_session": 3000,
    "grpo_steps_pilot": 30, "grpo_steps_full": 300,
    "grpo_users_per_step_pilot": 4, "grpo_users_per_step_full": 8, "grpo_group": 8,
    # 单卡吞吐（tokens/秒），低—高两档。4B、LoRA、bf16。Qwen3.5 的线性注意力层训练偏慢（ms-swift 已知问题），取偏低值。
    "gpu": {
        "rtx4090_24g": {"train_tps": [1500, 3000], "decode_tps": [1500, 3000], "prefill_tps": [10000, 20000]},
        "a800_80g": {"train_tps": [3000, 6000], "decode_tps": [3000, 6000], "prefill_tps": [25000, 50000]},
    },
    "uncached_prefill_share": 0.35,
    "first_prompt_tokens": 1300,        # 第一次调用的 prompt（系统提示 + 工具 + 任务），按本轮请求日志的量级     # 有前缀缓存时，每次调用需要重新算的 prompt 比例
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
    # 训练时每次会话是一条完整序列（最后一次调用的 prompt + 它的输出）。上下文按调用线性增长，
    # 第一次调用的 prompt 约 a["first_prompt_tokens"]，所以最后一次 ≈ 2 × 平均 prompt − 第一次。
    per_call = ptok / calls
    train_seq = max(per_call, 2 * per_call - a["first_prompt_tokens"]) + ctok / calls

    def grpo_hours(gpu, steps, users):
        g = a["gpu"][gpu]
        sess = users * a["grpo_group"] * 8
        out = []
        for i in (1, 0):   # 先算快的一档（低时长），再算慢的
            roll = sess * (ctok / g["decode_tps"][i] + ptok * a["uncached_prefill_share"] / g["prefill_tps"][i])
            train = sess * train_seq * (4 / 3) / g["train_tps"][i]   # 前向+反向，另加一次前向算旧策略的 logprob
            out.append(steps * (roll + train) / 3600)
        return out

    def sft_hours(gpu, worlds):
        g = a["gpu"][gpu]
        tok = worlds * a["sessions_per_world"] * a["sft_tokens_per_session"] * a["sft_epochs"]
        return [tok / g["train_tps"][1] / 3600, tok / g["train_tps"][0] / 3600]

    res = {"assumptions": a, "measured": {"qwen_calls_per_session": calls, "qwen_prompt_tokens_per_session": ptok,
                                           "qwen_completion_tokens_per_session": ctok,
                                           "teacher_cny_per_session": teacher_cny_per_session,
                                           "grpo_train_tokens_per_session_est": train_seq}}
    for name, gpu, worlds, steps, users, setup in (
            ("pilot_2x4090", "rtx4090_24g", a["teacher_worlds_pilot"], a["grpo_steps_pilot"], a["grpo_users_per_step_pilot"], a["setup_hours"]),
            ("pilot", "rtx4090_24g", a["teacher_worlds_pilot"], a["grpo_steps_pilot"], a["grpo_users_per_step_pilot"], a["setup_hours"]),
            ("pilot_a800", "a800_80g", a["teacher_worlds_pilot"], a["grpo_steps_pilot"], a["grpo_users_per_step_pilot"], a["setup_hours"]),
            ("full", "a800_80g", a["teacher_worlds_full"], a["grpo_steps_full"], a["grpo_users_per_step_full"], [0, 0])):
        sft = sft_hours(gpu, worlds)
        grpo = grpo_hours(gpu, steps, users)
        hours = [setup[i] + sft[i] + grpo[i] for i in (0, 1)]
        p = a["price_cny_per_hour"][gpu]
        if name == "pilot_2x4090":   # Twinkle 官方示例训练、采样各一张卡；时长按单卡保守估，费用翻倍
            p = {kk: 2 * v for kk, v in p.items()}
        teacher = worlds * a["sessions_per_world"] * teacher_cny_per_session
        label = {"rtx4090_24g": "1 × RTX 4090 24G", "a800_80g": "1 × A800 80G"}[gpu]
        res[name] = {"gpu": "2 × RTX 4090 24G" if name == "pilot_2x4090" else label, "teacher_sessions": worlds * a["sessions_per_world"], "teacher_cny": teacher,
                     "setup_hours": setup, "sft_hours": sft, "grpo_hours": grpo, "gpu_hours": hours,
                     "gpu_cny_on_demand": [h * p["on_demand"] for h in hours],
                     "gpu_cny_spot": [h * p["spot"] for h in hours],
                     "total_cny_on_demand": [teacher + h * p["on_demand"] for h in hours]}
    return res


# ---------- 报告 ----------

KIND_CN = {"bad_json": "参数不是合法 JSON", "unknown_tool": "调了不存在的工具", "truncated": "输出被长度上限截断",
           "think_runaway": "思考用满预算被截断（思考停不下来）", "tool_text": "工具调用写在正文里、没被解析", "no_tool_call": "只说话、没调工具（会被提醒）"}
MODE_CN = {"nothink": "思考关", "think": "思考开"}
THRESHOLD = 0.75     # 卡片约定：起点得分 ≥ 0.75 时不进入训练花费，等决定


def write_report(out, path):
    from .m2 import MODES, SHORT, TRAIN_SAMPLING, VAR_SEED
    from .report import Linker
    L = Linker(path)
    P = lambda *a: os.path.join(out, *a)
    sp = P("stats.json")
    st = _load(sp) if os.path.exists(sp) else {"rows": {}, "seeds": [1, 2, 3]}
    cfg = _load(P("config.json"))
    rows = st["rows"]
    lines = []
    w = lines.append
    q = lambda m, ctrl=False: f"llm_{SHORT}_{m}" + ("_true_memory" if ctrl else "")
    have = [m for m in MODES if q(m) in rows]
    best = max(have, key=lambda m: rows[q(m)]["mean_score"]) if have else None

    w("# M2 起点：未训练的 Qwen3.5-4B 在第二版考场里的得分，以及训练温度下的组内差异")
    w("")
    w("起点低于 0.75，小模型还有东西可学。")
    w("")
    w(f"日志目录：[runs/m2/]({L.rel(out)})。数字由 `python -m roommate.m2 --stages stats,report` 从日志生成，"
      "链接指向生成该数字的文件。考场（世界、规则、判分）是 M1 第二版原样，种子 1、2、3；"
      "分数只算和偏好有关的会话；区间是按用户重抽样（每个世界内分别重抽，5000 次）的 95% 区间。"
      "Qwen3.5-4B 经硅基流动调用（免费），DeepSeek 两行和两个脚本策略取自 M1 第二版的日志，没有重跑；"
      "DeepSeek 两行开着思考，实际是默认采样、不是温度 0（第 10 节）。")
    w("")

    _interim(w, L, P)
    w("---")
    w("")
    w("# 附录：全量运行（后台补跑中，跑完一组更新一组）")
    w("")
    w("以下各节由全量运行的结果生成。中期结论不依赖这里的数字；某一组还没跑完时，相应的行或表是空的。")
    w("")
    # ---------- 结论 ----------
    w("## 全量结论")
    w("")
    if best:
        b = rows[q(best)]
        w(f"- 未训练的 Qwen3.5-4B 最好的一组是{MODE_CN[best]}：偏好会话得分 {L.n(b['mean_score'], sp)}"
          f"（区间 {L.ci(b['ci95'], sp)}），剩余空间比例 {L.n(b['headroom_share'], sp, 'pct')}"
          f"（区间 {L.ci(b['headroom_share_ci95'], sp, 'pct')}）；同一考场上 DeepSeek V4 Pro 是 "
          f"{L.n(rows['llm_deepseek-v4-pro']['mean_score'], sp)}，理想记忆策略 {L.n(rows['ideal']['mean_score'], sp)}。")
        if b["mean_score"] >= THRESHOLD:
            w(f"- **起点得分 ≥ {THRESHOLD}，按约定不进入训练花费，等决定。**")
        else:
            gap = rows["ideal"]["mean_score"] - b["mean_score"]
            w(f"- 起点得分低于 {THRESHOLD}，小模型还有东西可学：离理想记忆策略差 {L.n(gap, sp)} 分，"
              f"离 V4 Pro 差 {L.n(rows['llm_deepseek-v4-pro']['mean_score'] - b['mean_score'], sp)} 分。")
    w("- 训练用哪种思考模式待定，这一步不下结论：两种模式的实测对照见第 4 节，第二步在自部署的 BF16 服务上确认模板如何处理思考、两种模式各测一遍后再定；费用估算两种模式各占一行（第 7 节）。")
    w("")

    # ---------- 1. 得分表 ----------
    w("## 1. 得分")
    w("")
    w("| 策略 | 偏好会话得分 | 95% 区间 | 种子 1 / 2 / 3 | 选对率 | 每次会话提问数 | 剩余空间比例 | 剩余空间 95% 区间 | 日志 |")
    w("|---|---|---|---|---|---|---|---|---|")
    order = [(q(m), f"Qwen3.5-4B（{MODE_CN[m]}，温度 0）") for m in MODES] + REF
    for k, name in order:
        if k not in rows:
            continue
        r = rows[k]
        base = M1_DIR if r["source"] == "m1v2" else out
        logs = " ".join(f"[{s}]({L.rel(os.path.join(base, f'seed{s}', 'policies', k, 'sessions.jsonl'))})" for s in st["seeds"])
        hs = ("—（参照）" if k in ("ideal", BEST_TRIVIAL) else L.n(r["headroom_share"], sp, "pct"))
        hci = ("—" if k in ("ideal", BEST_TRIVIAL) else L.ci(r["headroom_share_ci95"], sp, "pct"))
        w(f"| {name} | {L.n(r['mean_score'], sp)} | {L.ci(r['ci95'], sp)} | "
          f"{' / '.join(L.n(r['by_seed'][str(s)] if str(s) in r['by_seed'] else r['by_seed'][s], sp) for s in st['seeds'])} | "
          f"{L.n(r['success_rate'], sp, 'pct')} | {L.n(r['avg_questions'], sp, 'f2')} | {hs} | {hci} | {logs} |")
    w("")
    w(f"剩余空间比例 =（得分 − 记住但不更新）/（理想记忆策略 − 记住但不更新），和 M1 第二版闸门④同一口径。"
      f"温度与 M1 一致（0）；思考开时 `max_tokens` 8192、关时 2048，硅基流动的 `max_tokens` 不限制思考长度"
      f"（[config.json]({L.rel(P('config.json'))})）。")
    w("")
    for m in have:
        r = rows[q(m)]
        if "minus_v4pro" in r:
            w(f"- {MODE_CN[m]}比 V4 Pro 低 {L.n(-r['minus_v4pro'], sp)} 分（差值区间 {L.ci([-r['minus_v4pro_ci95'][1], -r['minus_v4pro_ci95'][0]], sp)}）；"
              f"写记忆的会话占 {L.n(r['memory_write_session_rate'], sp, 'pct')}，重问已经说过的事 {L.n(r['told_before_questions'], sp, 'int')} 次；"
              f"无关会话（算账、换算）选对 {L.n(r.get('util_success_rate'), sp, 'pct')}。")
    w("")

    # ---------- 2. 小模型特有的失败与兜圈子 ----------
    w("## 2. 小模型特有的失败，以及兜圈子")
    w("")
    w("每次与偏好有关的会话算一次（主表三个种子各 168 次，共 504 次；\"知道真实状态\"对照组只跑了种子 1，168 次）。"
      "格式错误不含\"只说话、没调工具\"：那种情况环境会提醒，模型可以接着调工具。"
      "\"兜圈子\"指一次会话里出现下面任意一种：同一问题重复问（声明的属性相同或问题原文相同）、跑到 14 轮上限、"
      "空白填充（被长度上限截断、正文八成以上是空白的一轮）。")
    w("")
    PR = st.get("partial_rows", {})
    R = lambda k: rows.get(k) or PR.get(k)
    cols = [(m, False) for m in MODES if R(q(m))] + [(m, True) for m in MODES if R(q(m, True))]
    if cols:
        lab = lambda m, c: f"{MODE_CN[m]}{'·知道真实状态（种子 1）' if c else '（种子 1–3）'}"
        w("| | " + " | ".join(lab(m, c) for m, c in cols) + " |")
        w("|---|" + "---|" * len(cols))
        F = lambda m, c: R(q(m, c))["failures"]
        Lp = lambda m, c: R(q(m, c))["loops"]
        n = lambda v: L.n(v, sp, "int")
        w("| 与偏好有关的会话 | " + " | ".join(n(F(m, c)["n_sessions"]) for m, c in cols) + " |")
        w("| 选对率 | " + " | ".join(L.n(R(q(m, c))["success_rate"], sp, "pct") for m, c in cols) + " |")
        w("| **有工具调用格式错误的会话** | " + " | ".join(n(F(m, c)["sessions_with_format_error"]) for m, c in cols) + " |")
        for kk in FORMAT_KINDS:
            w(f"| └ {KIND_CN[kk]}（会话数 / 轮数） | " + " | ".join(
                f"{n(F(m, c)['sessions_by_format_kind'][kk])} / {n(F(m, c)['format_issue_turns'][kk])}" for m, c in cols) + " |")
        w("| **没调 choose_option（没选出任何选项）** | " + " | ".join(n(F(m, c)["no_choose_option"]) for m, c in cols) + " |")
        w("| **超过 14 轮（用满 14 轮还没选）** | " + " | ".join(n(F(m, c)["hit_max_turns"]) for m, c in cols) + " |")
        w("| 接口错误、记成未作答 | " + " | ".join(n(F(m, c)["errors"]) for m, c in cols) + " |")
        w("| 平均每次会话轮数 | " + " | ".join(L.n(F(m, c)["avg_turns"], sp, "f2") for m, c in cols) + " |")
        w("| **兜圈子的会话（合计）** | " + " | ".join(
            f"{n(Lp(m, c)['any_loop'])}（{L.n(Lp(m, c)['any_loop_share'], sp, 'pct')}）" for m, c in cols) + " |")
        w("| └ 同一问题重复问 | " + " | ".join(n(Lp(m, c)["repeat_question"]) for m, c in cols) + " |")
        w("| └ 跑到 14 轮上限 | " + " | ".join(n(Lp(m, c)["hit_max_turns"]) for m, c in cols) + " |")
        w("| └ 空白填充 | " + " | ".join(n(Lp(m, c)["whitespace_padding"]) for m, c in cols) + " |")
        w("| 同一个工具调用原样重复 3 次以上（不并入合计） | " + " | ".join(n(Lp(m, c)["same_tool_call_3plus"]) for m, c in cols) + " |")
        w("")

    # ---------- 3. 知道真实状态的对照 ----------
    w("## 3. 丢分是记忆问题，还是挑店或格式问题")
    w("")
    w("\"知道真实状态\"对照组只跑了种子 1（限速下按优先级缩减），下表 Qwen 两行的失分拆解只算种子 1 的 168 次会话。"
      "对照组：每次会话开始时，把用户当前的真实情况（含本次的临时情况）直接写进记忆，其余不变。"
      "一次选错的会话，对照组选对了 → 记作\"没记住或没察觉变化\"；对照组也错 → 记作\"挑店或格式问题\"，再分成\"没选出任何选项\"和\"选了错的\"。")
    w("")
    ctl = st.get("control", {})
    w("| 策略 | 每次会话失分 | 其中提问 | 没记住或没察觉变化 | 挑店或格式问题 | └ 没选出选项 / 选错 | 对照组选对率 | 对照组得分 | 对照组每次提问 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, name in [(m, f"Qwen3.5-4B（{MODE_CN[m]}）") for m in MODES] + [
            ("llm_deepseek-v4-pro", "DeepSeek V4 Pro（M1，对照只跑了种子 1）"), ("llm_deepseek-flash", "DeepSeek Flash（M1）")]:
        c = ctl.get(k)
        if not c:
            continue
        src = sp if k in MODES else os.path.join(M1_DIR, "gates.json")
        split = (f"{L.n(c['n_fail_other_no_choice'], src, 'int')} / {L.n(c['n_fail_other_wrong_choice'], src, 'int')}"
                 if "n_fail_other_no_choice" in c else "—")
        w(f"| {name} | {L.n(c['total_loss'], src)} | {L.n(c['question_loss'], src)} | {L.n(c['memory_or_change_loss'], src)} | "
          f"{L.n(c['alignment_loss'], src)} | {split} | {L.n(c['control_success_rate'], src, 'pct')} | "
          f"{L.n(c.get('control_mean_score'), src)} | {L.n(c['control_avg_questions'], src, 'f2')} |")
    w("")

    # ---------- 4. 组内差异 ----------
    w("## 4. 两种思考模式的对照，以及训练温度下的组内差异")
    w("")
    w("**训练用哪种思考模式：待定，这一步不下结论。** 下表是两种模式在种子 1–3、温度 0 上的实测对照。"
      "训练用哪种模式，第二步在自部署的 BF16 服务上确认模板如何处理思考（官方模板会去掉早于最后一条用户消息的思考内容，"
      "影响多轮前缀是否一致）、两种模式各测一遍后再定。")
    w("")
    PRr = st.get("partial_rows", {})
    MM = [m for m in MODES if rows.get(q(m))]
    if MM:
        w("| 温度 0，种子 1–3 | " + " | ".join(MODE_CN[m] for m in MM) + " |")
        w("|---|" + "---|" * len(MM))
        cmp_ = lambda label, f: w(f"| {label} | " + " | ".join(f(rows[q(m)]) for m in MM) + " |")
        cmp_("偏好会话得分（95% 区间）", lambda r: f"{L.n(r['mean_score'], sp)}（{L.ci(r['ci95'], sp)}）")
        cmp_("选对率", lambda r: L.n(r["success_rate"], sp, "pct"))
        cmp_("每次会话调用次数", lambda r: L.n(r["usage"]["calls_per_session"], sp, "f2"))
        cmp_("每次会话输入 tokens", lambda r: L.n(r["usage"]["prompt_tokens_per_session"], sp, "int"))
        cmp_("每次会话输出 tokens（其中思考）", lambda r: f"{L.n(r['usage']['completion_tokens_per_session'], sp, 'int')}（{L.n(r['usage']['reasoning_tokens_per_session'], sp, 'int')}）")
        cmp_("空白填充的会话占比", lambda r: L.n(r["loops"]["whitespace_padding"] / r["loops"]["n_sessions"], sp, "pct"))
        cmp_("思考失控（用满思考预算）的会话占比", lambda r: L.n(r["failures"]["sessions_by_format_kind"]["think_runaway"] / r["failures"]["n_sessions"], sp, "pct"))
        cmp_("兜圈子的会话占比（合计）", lambda r: L.n(r["loops"]["any_loop_share"], sp, "pct"))
        cmp_("用满 14 轮还没选的会话", lambda r: L.n(r["failures"]["hit_max_turns"], sp, "int"))
        w("")
    w("组内差异只跑了思考关：这是上一张卡按\"将来训练用的模式\"做的取舍（当时按思考关），在限速下排在基线和对照组之后。"
      "现在训练模式改为待定，思考开在训练温度下的组内差异还没有测，留到第二步和 BF16 重测一起做。")
    w("")
    w(f"采样参数 {json.dumps(TRAIN_SAMPLING)}（GRPO 采样不截断分布）。种子 {VAR_SEED} 的 24 个用户，每人 8 次会话整段重复跑 K 遍；"
      "每一遍的请求 seed 不同（硅基流动的 seed 本来也不保证复现）。一个用户的 K 遍就是 GRPO 的一组。")
    w("")
    V = {m: v for m, v in st.get("variance", {}).items() if v}
    if V:
        w("| | " + " | ".join(MODE_CN[m] for m in V) + " |")
        w("|---|" + "---|" * len(V))
        vrow = lambda label, f: w(f"| {label} | " + " | ".join(f(v) for v in V.values()) + " |")
        vrow("重复遍数 K", lambda v: L.n(v["k"], sp, "int"))
        vrow("K 遍的平均得分（每遍的得分范围）", lambda v: f"{L.n(v['mean_score_all_reps'], sp)}（{L.n(min(v['mean_score_by_rep'].values()), sp)}–{L.n(max(v['mean_score_by_rep'].values()), sp)}）")
        vrow("每个用户 7 次偏好会话总分的组内标准差：中位数", lambda v: L.n(v["user_total_sd"]["median"], sp))
        vrow("　　　　　10% / 90% 分位", lambda v: f"{L.n(v['user_total_sd']['p10'], sp)} / {L.n(v['user_total_sd']['p90'], sp)}")
        vrow("　　　　　最小 / 最大", lambda v: f"{L.n(v['user_total_sd']['min'], sp)} / {L.n(v['user_total_sd']['max'], sp)}")
        vrow("组内总分极差的平均", lambda v: L.n(v["user_total_range"]["mean"], sp))
        vrow("K 遍总分完全相同的用户", lambda v: f"{L.n(v['users_all_identical'], sp, 'int')} / {L.n(v['n_users'], sp, 'int')}")
        vrow("K 遍分数完全相同的会话（同一用户、同一次会话）", lambda v: f"{L.n(v['session_groups_all_identical'], sp, 'int')} / {L.n(v['n_session_groups'], sp, 'int')}（{L.n(v['session_groups_all_identical_share'], sp, 'pct')}）")
        vrow("└ 其中每遍都选对且提问数相同", lambda v: L.n(v["session_groups_identical_all_correct"], sp, "int"))
        vrow("└ 其中每遍都选错（或都没选）且提问数相同", lambda v: L.n(v["session_groups_identical_all_wrong_or_costly"], sp, "int"))
        vrow("选对与否在 K 遍里有对有错的会话", lambda v: L.n(v["session_groups_success_mixed"], sp, "int"))
        vrow("第一次会话（K 遍的 prompt 完全相同）分数完全相同", lambda v: f"{L.n(v['first_session_groups_all_identical'], sp, 'int')} / {L.n(v['first_session_groups'], sp, 'int')}")
        vrow("格式错误会话 / 没选 / 用满 14 轮（K 遍合计）", lambda v: f"{L.n(v['failures_all_reps']['sessions_with_format_error'], sp, 'int')} / {L.n(v['failures_all_reps']['no_choose_option'], sp, 'int')} / {L.n(v['failures_all_reps']['hit_max_turns'], sp, 'int')}")
        w("")
        w("每个用户 K 遍的总分（7 次偏好会话之和）在 stats.json 的 `variance.<模式>.user_totals` 里；逐遍的记录在 "
          + "、".join(f"[{MODE_CN[m]}]({L.rel(P('variance', f'seed{VAR_SEED}', f'{m}_t1'))})" for m in V) + "。")
    else:
        w("（还没有结果）")
    w("")
    _tail(w, L, st, sp, P, cfg, rows, q, have)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


M1_DIR = "runs/m1v2"


def _tail(w, L, st, sp, P, cfg, rows, q, have):
    # ---------- 5. token 用量与吞吐 ----------
    w("## 5. token 用量与接口吞吐")
    w("")
    w("| 组 | 每次会话调用数 | 每次会话 prompt tokens | 每次会话输出 tokens | 其中思考 tokens |")
    w("|---|---|---|---|---|")
    from .m2 import MODES
    for m in MODES:
        for c in (False, True):
            r = rows.get(q(m, c)) or st.get("partial_rows", {}).get(q(m, c))
            if not r or not r.get("usage"):
                continue
            u = r["usage"]
            w(f"| {MODE_CN[m]}{'·知道真实状态（种子 1）' if c else '（种子 1–3，温度 0）'} | {L.n(u['calls_per_session'], sp, 'f2')} | "
              f"{L.n(u['prompt_tokens_per_session'], sp, 'int')} | {L.n(u['completion_tokens_per_session'], sp, 'int')} | "
              f"{L.n(u['reasoning_tokens_per_session'], sp, 'int')} |")
    for m, v in (st.get("variance") or {}).items():
        if v and v.get("usage_all_reps"):
            u = v["usage_all_reps"]
            w(f"| {MODE_CN[m]}·温度 1 | {L.n(u['calls_per_session'], sp, 'f2')} | {L.n(u['prompt_tokens_per_session'], sp, 'int')} | "
              f"{L.n(u['completion_tokens_per_session'], sp, 'int')} | {L.n(u['reasoning_tokens_per_session'], sp, 'int')} |")
    w("")
    a = st.get("api_calls")
    if a:
        cp = P("spend_calls.jsonl")
        w(f"本轮实际发出 {L.n(a['n'], cp, 'int')} 次请求，prompt {L.n(a['prompt_tokens'], cp, 'int')} tokens，"
          f"输出 {L.n(a['completion_tokens'], cp, 'int')} tokens；单次请求耗时中位数 {L.n(a['latency_median_s'], cp, 'f2')} 秒、"
          f"90% 分位 {L.n(a['latency_p90_s'], cp, 'f2')} 秒（{a['first_call']} 到 {a['last_call']}）。")
        t = a["tokens_per_minute"]
        w("")
        w(f"按分钟统计，有请求的 {L.n(t['n_minutes'], cp, 'int')} 分钟里，每分钟 tokens 的中位数 {L.n(t['median'], cp, 'int')}、"
          f"90% 分位 {L.n(t['p90'], cp, 'int')}、最大 {L.n(t['max'], cp, 'int')}。最大值来自刚开始的一小段突发；"
          "之后稳定在定价页上最低一级的 TPM 80000 附近（[价目快照](" + L.rel(P("pricing")) + ")），"
          "所以本账号实际受最低一级的限流。同时发出的请求多到上百个时，多出来的只在对方排队、单次耗时升到几分钟，"
          f"所以正式运行把同时发出的请求限制在 {cfg.get('max_inflight', 64)} 个以内（[运行日志]({L.rel(P('logs'))})）。")
    w("")

    # ---------- 6. 硅基流动与 HF ----------
    w("## 6. 硅基流动上的模型和 Hugging Face 公开权重是不是同一份")
    w("")
    w("详见 [硅基流动与 HF 权重对照](m2-siliconflow-vs-hf.md)。要点：")
    w("")
    w("- **能间接确认的**：分词器和聊天模板是官方的。13 次请求的 prompt token 数，与用 HF 官方模板（版本 `851bf6e`）在本地渲染后逐个相等；"
      "工具调用是官方的 XML 格式（`qwen3_coder`），由服务端解析成 JSON；默认开思考；多轮里的思考内容按官方规则保留。"
      "模型名、上下文长度（262144）、上线日期都和 HF 仓库一致。")
    w("- **确认不了的**：权重是否逐位相同、数值精度。国内站没标精度，国际站把同系列 9B 标为 FP8，4B 很可能也是 FP8。"
      "接口不返回 logprobs，seed 也不保证复现，没法直接比较输出分布。")
    w("- **对分数的影响（判断，不是测量）**：只算量化，估计偏 0–2 分，多半偏低；调用方式不一致（思考开关、"
      "`max_tokens` 是否包含思考、`thinking_budget`、思考内容回传、重试）每一项都可能带来 5–20 分以上，本轮都显式设定了。"
      "工具解析的差异估计 0–3 分，方向不定（本地 vLLM 不一定更好）。要测出来，需要在本地用 BF16 权重和同一个考场重跑，"
      "或者用强制解码打分逐 token 比较，都要一张 GPU。")
    w("- **部署精度与将来训练的权重**：硅基流动的部署精度未知（很可能 FP8），将来训练用的是 HF 上的 BF16 权重。"
      "两者之间的分数差估计在 0–2 分（判断，依据是 FP8 在 8B 以上模型上接近无损的文献，4B 和多步任务可能更敏感），"
      "比本表的区间宽度小，但不能忽略。**正式训练前，会在训练平台上用 BF16 权重、本地 vLLM 和同一个考场重测一遍起点**"
      "（思考开关、温度、`max_tokens` 口径按 [对照报告](m2-siliconflow-vs-hf.md) 的设置对齐），以那次的分数作为训练的起点和对照，本表只作参考。")
    w("")

    # ---------- 7. 选型与估算 ----------
    w("## 7. 训练平台与框架选型，以及试跑和正式训练的估算")
    w("")
    w("选型详见 [平台与框架选型](m2-platform-survey.md)。推荐：")
    w("")
    w("- **平台**：优云智算 Compshare。支付宝、微信支付在官方充值文档里写明；定价页当天看到的价格是 4090 24G 按量 1.92 元/时（抢占 1.31），"
      "A800 80G 按量 7.36（抢占 3.02）；有 OpenAPI 和命令行工具能开机、关机。备选 AutoDL（支付宝、微信已确认；"
      "搜索摘录的单价更低，但普通实例没有开关机接口）。")
    w("- **三条路线**：比较了三条路线，见选型报告 Part C。")
    w("  - **租卡 + 自部署兼容 Tinker 接口的服务**（魔搭 Twinkle、SkyRL 的 Tinker 后端）：训练循环由我们的代码驱动。"
      "8 次会话由我们照常按顺序跑完，记忆由考场自己保存，每次会话切成一条训练样本，优势按会话写 return-to-go。")
    w("  - **租卡 + 开源训练框架**（TRL、verl）：循环由框架驱动，\"一条 prompt 对应一条序列\"。"
      "只能按会话做样本、数据里放旧策略的记忆快照，跨会话回报要近似或靠实验性 API。")
    w("  - **阿里云 PAI**：Model Gallery 的 GRPO 只写了 Qwen3；百炼托管 RL 最小 9B、要商务开通；自建的单价是租卡平台的 2–5 倍。")
    w("- **推荐**：走路线一。服务端先试 Twinkle（官方示例默认就是 Qwen3.5-4B，需 2 卡），备选 SkyRL（单卡 A800，要求驱动 ≥ r580）；"
      "路线二的 TRL 作保底，不用 PAI。理由是这个结构在路线一里是普通的 Python，现有考场代码只需换一个记录 token 的 client。")
    w("- **把本考场接进去的主要风险**："
      "两家 Tinker 服务端都没有\"Qwen3.5-4B + 多轮工具\"的公开验证；它们把 LoRA 动态加载进 vLLM，线性注意力层上可能出错；"
      "SDK 版本锁定落后于最新 cookbook；老师轨迹、SFT、rollout、评测四处的模板和工具解析器必须一致；"
      "线性注意力层在训练端和 vLLM 端的 logprob 偏差；环境版本要求新。")
    w("")
    EM = st.get("estimate_by_mode") or {}
    if EM:
        tc = st["teacher_cost"]
        a = next(iter(EM.values()))["assumptions"]
        w("**估算**。训练用哪种思考模式待定（第 4 节），两种模式各占一行，实测量的来源见下表第二列（全量没跑完时取中期子集），已经包含兜圈子的开销。"
          "SFT 之后兜圈子会变少，rollout 会更便宜，所以 GRPO 一栏偏保守。"
          f"老师 DeepSeek V4 Pro 每次会话 {L.n(tc['cny_per_session'], tc['source'], 'f3')} 元，来自 M1 第二版账本。"
          "吞吐、步数、组大小是假设，写在 stats.json 的 `estimate_by_mode.<模式>.assumptions` 里。")
        w("")
        w("| 模式 | 实测量来自 | 每次会话调用 / 输入 / 输出 tokens（实测） | 训练序列（估） |")
        w("|---|---|---|---|")
        for mode, e in EM.items():
            m = e["measured"]
            w(f"| {MODE_CN[mode]} | {e.get('usage_source', '')} | {L.n(m['qwen_calls_per_session'], sp, 'f2')} / {L.n(m['qwen_prompt_tokens_per_session'], sp, 'int')} / "
              f"{L.n(m['qwen_completion_tokens_per_session'], sp, 'int')} | {L.n(m['grpo_train_tokens_per_session_est'], sp, 'int')} |")
        w("")
        w("| 方案 | 模式 | 卡 | 老师轨迹 | 装环境 | SFT | GRPO | 上机合计 | 显卡费（按量） | 显卡费（抢占） | 合计（老师 + 按量显卡） |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        hr = lambda x: f"{x[0]:.1f}–{x[1]:.1f} 小时"
        cy = lambda x: f"{x[0]:.0f}–{x[1]:.0f} 元"
        plans = (("pilot", f"试跑：{a['teacher_worlds_pilot']} 个新世界的老师轨迹 + SFT + {a['grpo_steps_pilot']} 步 GRPO，单卡"),
                 ("pilot_2x4090", "试跑，2 卡 4090（Twinkle 官方示例的配置，时长按单卡保守估）"),
                 ("pilot_a800", "试跑，换 A800"),
                 ("full", f"正式：{a['teacher_worlds_full']} 个新世界 + SFT + {a['grpo_steps_full']} 步 GRPO"))
        for key, name in plans:
            for mode, e in EM.items():
                x = e[key]
                w(f"| {name} | {MODE_CN[mode]} | {x['gpu']} | {x['teacher_sessions']} 次会话，{L.n(x['teacher_cny'], sp, 'f2')} 元 | "
                  f"{hr(x['setup_hours'])} | {hr(x['sft_hours'])} | {hr(x['grpo_hours'])} | {hr(x['gpu_hours'])} | "
                  f"{cy(x['gpu_cny_on_demand'])} | {cy(x['gpu_cny_spot'])} | {cy(x['total_cny_on_demand'])} |")
        w("")
        w(f"GRPO 每步 = {a['grpo_users_per_step_pilot']}（试跑）或 {a['grpo_users_per_step_full']}（正式）个用户 × 每组 {a['grpo_group']} 遍 × 8 次会话；"
          "rollout 时长按实测的输出 tokens 和未命中前缀缓存的输入 tokens 除以假设吞吐计算，训练按估计的序列长度、"
          "前向加反向、另加一次前向算旧策略 logprob 计算；SFT 按每次会话 3000 tokens、2 遍。这些吞吐没有实测，误差可能到 2 倍；"
          "试跑的头 10 步就是用来把它们换成实测值的。思考开的输出里思考 tokens 占大头，解码吞吐的假设对它影响更大。")
    w("")

    # ---------- 8. 花费 ----------
    w("## 8. 花费与 token 用量")
    w("")
    led = st.get("spend")
    lp = P("spend_ledger.json")
    if led:
        w(f"- 硅基流动（Qwen3.5-4B，免费）：累计 {L.n(led.get('siliconflow_cny_total', 0), lp, 'f2')} 元，上限 {L.n(led.get('siliconflow_cny_limit'), lp, 'f2')} 元。"
          f"逐次调用明细在 [spend_calls.jsonl]({L.rel(P('spend_calls.jsonl'))})。")
        tok = {k: v for k, v in led["by_model_role"].items()}
        tp = sum(v["prompt_tokens"] for v in tok.values())
        tc_ = sum(v["completion_tokens"] for v in tok.values())
        calls = sum(v["api_calls"] for v in tok.values())
        w(f"- 账本记下的调用 {L.n(calls, lp, 'int')} 次，prompt {L.n(tp, lp, 'int')} tokens，输出 {L.n(tc_, lp, 'int')} tokens（按用途的明细见账本）。")
    w("- 另有：硅基流动对照调研时手动发了 17 次请求（免费，记录在 [siliconflow_probe/](../runs/m2/siliconflow_probe)，不在账本里，均为 0 元）；"
      "选型只做网页调研，没有注册、充值、租卡。")
    tot = (led or {}).get("siliconflow_cny_total", 0.0)
    w(f"- 本卡总花费：{L.n(tot, lp, 'f2')} 元（上限 50 元）。")
    w("")

    # ---------- 9. 复现 ----------
    w("## 9. 复现")
    w("")
    rp = P("repro_check.json")
    if os.path.exists(rp):
        r = _load(rp)
        w(f"只用缓存重跑每一组（`python -m roommate.m2 --stages repro`），和保存的逐次记录逐字比较："
          f"{L.n(sum(r['by_run'].values()), rp, 'int')} / {L.n(len(r['by_run']), rp, 'int')} 组一致。"
          "请求缓存在 runs/m2/llm_cache/（gzip）。")
    tp = P("tests.txt")
    if os.path.exists(tp):
        w(f"单元测试：[{open(tp, encoding='utf-8').read().strip().splitlines()[-1]}]({L.rel(tp)})。")
    w("")
    w("## 10. 没解决的问题")
    w("")
    w("- 硅基流动上的权重和精度只能间接确认；本地 BF16 与硅基流动之间的分数差没有实测。")
    w("- 评测用温度 0，请求参数和 M1 相同；但 M1 的 DeepSeek 开着思考，DeepSeek 官方文档写明思考模式下 temperature 不生效、设了也不报错，"
      "所以 M1 的 DeepSeek 实际是默认采样（V4 Pro 带工具时 top_p 固定 0.95，Flash 没传 top_p、取默认值 1；"
      f"[文档原文与核对]({L.rel(os.path.join('runs', 'm1v2', 'docs', 'deepseek_sampling_2026-10-26.json'))})）。"
      "和\"同一批会话上的 V4 Pro\"比时，两边的解码方式并不相同：Qwen 按温度 0 请求，V4 Pro 是默认采样。"
      "另外官方不建议 Qwen 的思考模式用贪心解码；思考开这一组可能因此被低估。")
    w("- 估算里的显卡吞吐是假设，没有实测。")
    w("")


def _interim(w, L, P):
    """报告开头的中期结论：只读 runs/m2/interim/interim_stats.json。"""
    ip = P("interim", "interim_stats.json")
    if not os.path.exists(ip):
        return
    I = _load(ip)
    B, C, V = I.get("base", {}), I.get("ctrl", {}), I.get("variance")
    n = lambda v, k="f3": L.n(v, ip, k)
    w("## 中期结论（只用已跑完的子集）")
    w("")
    w("**已完成的范围**（下面每个数字都来自这个子集，不是全量）：")
    w("")
    for m, e in B.items():
        spu = "、".join(f"种子 {s} 每人约 {v:.1f} 次" for s, v in e["sessions_per_user"].items())
        w(f"- {MODE_CN[m]}、温度 0：种子 {'、'.join(str(s) for s in e['seeds'])}，{n(e['n_users'], 'int')} 个用户，"
          f"每个用户的前几次会话（{spu}），共 {n(e['n_sessions'], 'int')} 次与偏好有关的会话。")
    for m, c in C.items():
        w(f"- \"知道真实状态\"对照组，{MODE_CN[m]}：种子 1，{n(c['n_sessions'], 'int')} 次与偏好有关的会话。")
    VL = I.get("variance_loose")
    if VL:
        cb = "、".join(f"第 {r} 遍 {c}" for r, c in VL["completed_by_rep"].items())
        w(f"- 组内差异：思考关、温度 1，种子 1。计划是 24 个用户 × 前 3 次会话 × 4 遍；运行被打断时只跑完了第一次会话（{cb} 个用户）。"
          f"至少 2 遍跑完的用户 {n(VL['n_users_ge2_reps'], 'int')} 个、共 {n(VL['n_completed_first_sessions'], 'int')} 次第一次会话；"
          f"4 遍都跑完的只有 {n(V['n_users'] if V else 0, 'int')} 个用户。")
    w("")
    w("子集是\"每个用户的前 k 次会话\"（全量运行按会话序号一轮一轮推进，从缓存重建时每个用户在第一次缺缓存的会话处截断）。"
      "前几次会话要先问，分数天然偏低，所以每个比较都在同一批会话上做：DeepSeek V4 Pro、理想记忆策略、记住但不更新也只取这些会话（M1 第二版的完整记录）。"
      f"区间按用户重抽样。数据：[interim_stats.json]({L.rel(ip)})，生成脚本 `python -m roommate.m2_interim`。")
    w("")
    w("**问题一：起点分数是否低于 0.75？**")
    w("")
    w("| 温度 0 | 子集会话数 | Qwen3.5-4B 偏好会话得分 | 95% 区间 | 同一批会话上 V4 Pro | 理想记忆策略 | 记住但不更新 | 剩余空间比例 | 低于 0.75 的重抽样比例 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m, e in B.items():
        r = e["refs_same_sessions"]
        w(f"| {MODE_CN[m]} | {n(e['n_sessions'], 'int')} | {n(e['mean_score'])} | {L.ci(e['ci95'], ip)} | "
          f"{n(r['llm_deepseek-v4-pro'])} | {n(r['ideal'])} | {n(r['remember_no_update'])} | "
          f"{n(e['headroom_share'], 'pct')}（{L.ci(e['headroom_share_ci95'], ip, 'pct')}） | {n(e['p_below_075'], 'pct')} |")
    w("")
    w("丢分去向（同一子集）：")
    w("")
    w("| 温度 0 | 选错的会话 | └ 没选出任何选项 | └ 其中用满 14 轮 | └ 选了错的 | 每次会话提问扣分 | 兜圈子的会话占比 | 每次会话调用 / 输入 / 输出 tokens |")
    w("|---|---|---|---|---|---|---|---|")
    for m, e in B.items():
        fb, lp, u = e["fail_breakdown"], e["loops"], e["usage"]
        w(f"| {MODE_CN[m]} | {n(fb['n_fail'], 'int')} / {n(e['n_sessions'], 'int')} | {n(fb['no_choice'], 'int')} | "
          f"{n(fb['no_choice_hit_max_turns'], 'int')} | {n(fb['wrong_choice'], 'int')} | {n(fb['question_loss_per_session'])} | "
          f"{n(lp['any_loop_share'], 'pct')}（重复问 {n(lp['repeat_question'], 'int')}、14 轮 {n(lp['hit_max_turns'], 'int')}、空白填充 {n(lp['whitespace_padding'], 'int')}） | "
          f"{n(u['calls_per_session'], 'f2')} / {n(u['prompt_tokens_per_session'], 'int')} / {n(u['completion_tokens_per_session'], 'int')} |")
    for m, c in C.items():
        w(f"| 对照组·{MODE_CN[m]}（种子 1） | 选对率 {n(c['success_rate'], 'pct')} | {n(c['no_choice'], 'int')} | {n(c['hit_max_turns'], 'int')} | — | — | — | — |")
    w("")
    w("**问题二：温度大于 0 时，同一用户几遍之间的分数有没有差异？**")
    w("")
    VL = I.get("variance_loose")
    if VL:
        w(f"- **有差异的用户占比**：第一次会话（各遍 prompt 完全相同）在至少 2 遍里跑完的用户 {n(VL['n_users_ge2_reps'], 'int')} 个，"
          f"其中几遍分数不全相同的 {n(VL['n_users_differ'], 'int')} 个（{n(VL['share_users_differ'], 'pct')}），"
          f"选对与否有对有错的 {n(VL['n_users_success_mixed'], 'int')} 个；跑完的 {n(VL['n_completed_first_sessions'], 'int')} 次第一次会话里，"
          f"用满 14 轮的 {n(VL['n_first_sessions_hit_max_turns'], 'int')} 次。")
        w("- 注意：组内差异子集只到第一次会话，而且没跑完就被打断（见下面\"运行中断\"）；跑得慢的会话（兜圈子的）更可能没跑完，"
          "所以子集偏向短会话，用满 14 轮的比例会被低估。第一次会话各遍的 prompt 完全相同，比较是干净的。")
    if V:
        w(f"- 严格口径（4 遍都跑完，只有 {n(V['n_users'], 'int')} 个用户，样本太小，只作参考）：几遍分数不全相同的 "
          f"{n(V['users_with_difference'], 'int')} / {n(V['n_users'], 'int')}。")
        w(f"- 用户总分的组内标准差：中位数 {n(V['user_total_sd']['median'])}，10% / 90% 分位 {n(V['user_total_sd']['p10'])} / {n(V['user_total_sd']['p90'])}。")
        w(f"- 按（用户、会话）看：{n(V['session_groups_all_identical'], 'int')} / {n(V['n_session_groups'], 'int')} 组 {n(V['k'], 'int')} 遍分数完全相同"
          f"（{n(V['session_groups_all_identical_share'], 'pct')}；其中每遍都选对且提问数相同 {n(V['session_groups_identical_all_correct'], 'int')} 组），"
          f"选对与否有对有错的 {n(V['session_groups_success_mixed'], 'int')} 组；第一次会话（{n(V['k'], 'int')} 遍 prompt 完全相同）分数完全相同的 "
          f"{n(V['first_session_groups_all_identical'], 'int')} / {n(V['first_session_groups'], 'int')}。")
        w(f"- 这 {n(V['k'], 'int')} 遍的平均得分 {n(V['mean_score_all_reps'])}；用满 14 轮没选 {n(V['failures_all_reps']['hit_max_turns'], 'int')} 次。")
    w("")
    w("")
    w("**结论属于哪一种**：")
    w("")
    for m, e in B.items():
        fb = e["fail_breakdown"]
        nc_share = fb["no_choice"] / fb["n_fail"] if fb["n_fail"] else 0
        hi = e["ci95"][1]
        if hi < 0.75 and nc_share >= 0.5:
            cat = "**起点低，且主要丢在格式和兜圈子**"
            why = (f"得分 {n(e['mean_score'])}（区间上界 {n(hi)}，低于 0.75）；选错的 {n(fb['n_fail'], 'int')} 次里 "
                   f"{n(fb['no_choice'], 'int')} 次没选出任何选项（{n(nc_share, 'pct')}），全部是用满 14 轮；"
                   f"另外每次会话被重复提问扣 {n(fb['question_loss_per_session'])} 分。")
        elif hi < 0.75:
            cat = ("**起点中等，且组内有差异**" if (V or VL) and m == "nothink" else
                   "**起点中等**（这一模式在训练温度下的组内差异还没测；思考关上测到有差异）")
            if e["max_idx"] is not None and e["max_idx"] <= 1:
                cat += "（证据只到前两次会话，跨会话记忆还没被检验）"
            why = (f"得分 {n(e['mean_score'])}（区间上界 {n(hi)}，低于 0.75），同一批会话上 V4 Pro {n(e['refs_same_sessions']['llm_deepseek-v4-pro'])}；"
                   f"丢分主要在提问（每次会话 {n(fb['question_loss_per_session'])}）和选错（{n(fb['wrong_choice'], 'int')} 次），"
                   f"没选出选项只有 {n(fb['no_choice'], 'int')} 次。")
        else:
            cat = "**起点 ≥ 0.75**"
            why = f"得分 {n(e['mean_score'])}，区间 {L.ci(e['ci95'], ip)}。"
        w(f"- {MODE_CN[m]}：{cat}。{why}")
    if VL:
        w(f"- 两种模式都不属于\"起点 ≥ 0.75，或组内几乎没有差异\"：温度 1 时同一用户几遍之间的分数有差异的占 {n(VL['share_users_differ'], 'pct')}"
          "（测的是思考关；思考开在训练温度下的组内差异还没测）。所以小模型还有东西可学，GRPO 在组内有信号可用；"
          "思考关要先解决兜圈子（SFT 的任务），思考开的起点更高、丢分在提问和挑选。训练用哪种模式仍待定（附录第 4 节）。")
    w("")
    rp = P("logs", "restarts.txt")
    w(f"**运行中断**：2026-10-25 约 14:01（UTC）起，本环境的出口代理按网络策略拒绝连接 `api.siliconflow.cn`（CONNECT 返回 403），"
      f"之后没有新的请求成功；按规则没有重试、没有换渠道。已完成的请求全部在缓存里，网络恢复后用 "
      f"`python -m roommate.m2 --stages runs` 补跑，已跑过的部分从缓存逐字重放，结果只更新附录。经过见 [restarts.txt]({L.rel(rp)})。")
    w("")
