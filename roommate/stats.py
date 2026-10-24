"""统计与闸门：只读 runs/m1v2/ 下的逐次会话记录，不调用任何模型。

- 所有分数只算和偏好有关的会话；无关会话（算账、换算）单独报。
- 置信区间按用户重抽样（在每个世界内分别重抽用户，5000 次），取 2.5% 与 97.5% 分位。
- 闸门定义见 README 路线：
  ① 判分自检通过、每个长期变化都能从台词推出来、模拟用户不自相矛盾；
  ② 每次都问、不写记忆的策略 ≤ 理想记忆策略的 50%；
  ③ 不看用户的规则或分类器选对率 ≤ 1/6 + 5 个百分点；
  ④ 最强现成模型在"最简单策略之上、理想记忆策略之下"的空间里拿到的比例，95% 区间上界 < 80%（只报告，不挡路）。
- 失分拆解：每次会话的失分 = 提问花掉的分 + 选错的 1 分；选错再按同一会话在"知道真实状态"对照组里
  选没选对分成两块：对照组选对了 → 没记住或没察觉变化；对照组也选错 → 选项没对齐。
"""

import json
import os
import random

from .policies import TRIVIAL

N_BOOT = 5000
GATE4_MODELS = ["deepseek-v4-pro"]   # grok 停用后，闸门④用 V4 Pro
LIMIT3 = 1 / 6 + 0.05


def _jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _ci(xs):
    xs = sorted(xs)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


def load_policies(out, seed, sub="policies"):
    base = os.path.join(out, f"seed{seed}", sub)
    res = {}
    if not os.path.isdir(base):
        return res
    for key in sorted(os.listdir(base)):
        p = os.path.join(base, key, "sessions.jsonl")
        if os.path.exists(p):
            res[key] = _jsonl(p)
    return res


def per_user(recs):
    """(总分, 会话数) 按用户，只算与偏好有关的会话。"""
    t = {}
    for r in recs:
        if r["domain"] == "util":
            continue
        a = t.setdefault(r["uid"], [0.0, 0])
        a[0] += r["grade"]["score"]
        a[1] += 1
    return t


def pooled_mean(by_seed_user, keys=None):
    tot = n = 0
    for seed, users in by_seed_user.items():
        for u in (keys[seed] if keys else users):
            tot += users[u][0]
            n += users[u][1]
    return tot / n


def bootstrap(per, seeds, fn, n=N_BOOT, seed=0):
    """per：{策略: {种子: {用户: [总分, 会话数]}}}；fn(means) -> 数值。在每个世界内重抽用户。"""
    rng = random.Random(seed)
    uids = {s: sorted(next(iter(per.values()))[s]) for s in seeds}
    out = []
    for _ in range(n):
        sample = {s: [rng.choice(uids[s]) for _ in uids[s]] for s in seeds}
        means = {k: pooled_mean({s: v[s] for s in seeds}, sample) for k, v in per.items()}
        out.append(fn(means))
    return out


def loss_breakdown(normal, control):
    """normal、control：同一批世界的逐次会话记录。返回每次与偏好有关的会话的平均失分拆解。"""
    ctl = {(r["_seed"], r["uid"], r["idx"]): r for r in control} if control else {}
    pref = [r for r in normal if r["domain"] != "util"]
    n = len(pref)
    q = sum(r["grade"]["success"] - r["grade"]["score"] for r in pref) / n
    fails = [r for r in pref if not r["grade"]["success"]]
    if ctl:
        mem = sum(1 for r in fails if ctl[(r["_seed"], r["uid"], r["idx"])]["grade"]["success"])
        align = len(fails) - mem
    else:
        mem = align = None
    return {"n_sessions": n, "total_loss": q + len(fails) / n, "question_loss": q,
            "fail_rate": len(fails) / n,
            "memory_or_change_loss": None if mem is None else mem / n,
            "alignment_loss": None if align is None else align / n,
            "n_fail": len(fails), "n_fail_memory_or_change": mem, "n_fail_alignment": align}


def model_detail(recs):
    pref = [r for r in recs if r["domain"] != "util"]
    qs = [q for r in pref for q in r["questions"]]
    by_kind = {}
    for r in pref:
        k = r["grade"]["near_miss"]
        e = by_kind.setdefault(k, {"n": 0, "success": 0, "chose_near_miss": 0})
        e["n"] += 1
        e["success"] += r["grade"]["success"]
        e["chose_near_miss"] += r["grade"]["near_miss_error"]
    writes = sum(any(op[0] == "write" for op in r["memory_ops"]) for r in pref)
    return {
        "n": len(pref), "success_rate": sum(r["grade"]["success"] for r in pref) / len(pref),
        "avg_questions": len(qs) / len(pref),
        "told_before_rate": (sum(q["repeated"] for q in qs) / len(qs)) if qs else None,
        "too_many_rate": (sum(q.get("too_many", False) for q in qs) / len(qs)) if qs else None,
        "memory_write_session_rate": writes / len(pref),
        "near_miss_error_rate": sum(r["grade"]["near_miss_error"] for r in pref) / len(pref),
        "by_near_miss_kind": by_kind,
        "no_choice": sum(r["chosen"] is None for r in pref),
        "errors": sum(bool(r.get("error")) for r in pref),
    }


def write(out, gate_seeds, compare_seed):
    pols = {s: load_policies(out, s) for s in gate_seeds}
    v1m = {s: load_policies(out, s, "policies_v1mode") for s in gate_seeds}
    for s in gate_seeds:
        for recs in pols[s].values():
            for r in recs:
                r["_seed"] = s
    n_pref = {s: len([r for r in pols[s]["oracle"] if r["domain"] != "util"]) for s in gate_seeds}
    # 完整的：三个闸门世界都有、且每个世界的偏好会话一个不缺；其余（只跑了部分世界、或缓存截断的）单独报
    allkeys = sorted(set().union(*[set(p) for p in pols.values()]))
    keys = [k for k in allkeys if all(k in pols[s] and
                                      sum(r["domain"] != "util" for r in pols[s][k]) == n_pref[s] for s in gate_seeds)]
    partial = [k for k in allkeys if k not in keys]
    per = {k: {s: per_user(pols[s][k]) for s in gate_seeds} for k in keys}
    means = {k: pooled_mean(per[k]) for k in keys}
    boot = bootstrap(per, gate_seeds, lambda m: m)
    ci = {k: _ci([b[k] for b in boot]) for k in keys}
    util = {}
    for k in keys:
        us = [r for s in gate_seeds for r in pols[s][k] if r["domain"] == "util"]
        util[k] = {"n": len(us), "success_rate": sum(r["grade"]["success"] for r in us) / len(us),
                   "mean_score": sum(r["grade"]["score"] for r in us) / len(us)}
    res = {"gate_seeds": gate_seeds, "n_bootstrap": N_BOOT, "policies": {}, "util_sessions": util}
    for k in keys:
        allr = [r for s in gate_seeds for r in pols[s][k]]
        res["policies"][k] = dict(mean_score=means[k], ci95=ci[k], **model_detail(allr),
                                  by_seed={s: pooled_mean({s: per[k][s]}) for s in gate_seeds})
    res["partial_runs"] = {}
    for k in partial:
        rs = [r for s in gate_seeds if k in pols[s] for r in pols[s][k] if r["domain"] != "util"]
        res["partial_runs"][k] = {
            "mean_score": sum(r["grade"]["score"] for r in rs) / len(rs), "n_sessions": len(rs),
            "by_seed": {s: {"n_sessions": sum(r["domain"] != "util" for r in pols[s][k]),
                            "n_full": n_pref[s],
                            "mean_score": (lambda x: sum(r["grade"]["score"] for r in x) / len(x) if x else None)(
                                [r for r in pols[s][k] if r["domain"] != "util"])}
                        for s in gate_seeds if k in pols[s]},
            **model_detail(rs)}
    # v1 行为（总会重新回答）下的对照
    res["v1_mode"] = {}
    for k in sorted(set.intersection(*[set(p) for p in v1m.values()])) if all(v1m.values()) else []:
        res["v1_mode"][k] = pooled_mean({s: per_user(v1m[s][k]) for s in gate_seeds})

    # ① 环境自检
    sc = _load(os.path.join(out, "selfcheck.json"))
    cons = _load(os.path.join(out, "consistency.json"))
    cue_p = os.path.join(out, "cue_validation.json")
    cue = _load(cue_p) if os.path.exists(cue_p) else None
    g1 = {"selfcheck_pass": sc["all_pass"], "consistency_pass": cons["all_pass"],
          "consistency_n_answers": cons["n_answers_checked"], "consistency_n_problems": cons["n_problems"],
          "cue_validation_done": cue is not None,
          "cue_judges": cue["models"] if cue else None,
          "cue_items": cue["n_items"] if cue else None,
          "cue_items_pass": cue["n_items_pass_all_models"] if cue else None,
          "cue_items_single_judge": len(cue.get("items_judged_by_fewer_models", [])) if cue else None,
          "cue_pass": bool(cue and cue["all_pass"])}
    g1["pass"] = g1["selfcheck_pass"] and g1["consistency_pass"] and g1["cue_pass"]

    # ② 记忆是必需的
    ratio = means["ask_all"] / means["ideal"]
    rb = [b["ask_all"] / b["ideal"] for b in boot]
    g2 = {"ask_all": means["ask_all"], "ideal": means["ideal"], "ratio": ratio, "ratio_ci95": _ci(rb),
          "threshold": 0.5, "pass": ratio <= 0.5}

    # ③ 选项不泄题
    lp = _load(os.path.join(out, "leak_probes.json"))
    g3 = {"limit": LIMIT3, "modal": lp["modal"], "conservative": lp["conservative"],
          "position_best": lp["position_best"], "classifier": lp["classifier"]["test_acc"],
          "classifier_large_test": lp["classifier_large_test"]["test_acc"],
          "random_policy": means.get("random"), "chi2_p": lp["chi2"]["p_value"], "pass": lp["all_pass"]}

    # ④ 环境没被现成模型做穿（只报告）
    trivial = [k for k in TRIVIAL if k in means]
    best_trivial = max(trivial, key=lambda k: means[k])
    llm_keys = {m: f"llm_{m}" for m in GATE4_MODELS if f"llm_{m}" in means}
    g4 = {"best_trivial": best_trivial, "best_trivial_score": means[best_trivial], "ideal": means["ideal"]}
    if llm_keys:
        best_m = max(llm_keys, key=lambda m: means[llm_keys[m]])
        k = llm_keys[best_m]
        share = (means[k] - means[best_trivial]) / (means["ideal"] - means[best_trivial])
        sb = [(b[k] - b[best_trivial]) / (b["ideal"] - b[best_trivial]) for b in boot]
        g4.update(model=best_m, model_score=means[k], headroom_share=share, headroom_share_ci95=_ci(sb),
                  threshold=0.8, pass_not_too_easy=_ci(sb)[1] < 0.8,
                  all_models={m: {"score": means[kk],
                                  "headroom_share": (means[kk] - means[best_trivial]) / (means["ideal"] - means[best_trivial]),
                                  "headroom_share_ci95": _ci([(b[kk] - b[best_trivial]) / (b["ideal"] - b[best_trivial])
                                                              for b in boot])}
                              for m, kk in {k2[4:]: k2 for k2 in keys if k2.startswith("llm_")
                                            and not k2.endswith("_true_memory")}.items()})
    res["gates"] = {"1": g1, "2": g2, "3": g3, "4": g4}

    # 失分拆解与"知道真实状态"对照：只算有对照组记录的那些会话
    res["loss_breakdown"] = {}
    llm_all = [k for k in allkeys if k.startswith("llm_") and not k.endswith("_true_memory")]
    for k in llm_all:
        ck = k + "_true_memory"
        control = [r for s in gate_seeds if ck in pols[s] for r in pols[s][ck]]
        normal = [r for s in gate_seeds if k in pols[s] for r in pols[s][k]]
        if control:
            have = {(r["_seed"], r["uid"], r["idx"]) for r in control}
            normal = [r for r in normal if (r["_seed"], r["uid"], r["idx"]) in have]
        res["loss_breakdown"][k] = loss_breakdown(normal, control or None)
        res["loss_breakdown"][k]["seeds"] = sorted({r["_seed"] for r in (control or normal)})
        if control:
            cp = [r for r in control if r["domain"] != "util"]
            res["loss_breakdown"][k]["control_success_rate"] = sum(r["grade"]["success"] for r in cp) / len(cp)
            res["loss_breakdown"][k]["control_avg_questions"] = sum(len(r["questions"]) for r in cp) / len(cp)
            res["loss_breakdown"][k]["control_n_sessions"] = len(cp)
    for k in ("ideal", "ask_all", "remember_all_reask", "remember_no_update"):
        normal = [r for s in gate_seeds for r in pols[s][k]]
        oracle = [r for s in gate_seeds for r in pols[s]["oracle"]]
        res["loss_breakdown"][k] = loss_breakdown(normal, oracle)

    # 种子 0：第二版规则下的前后对照（不参与闸门）
    p0 = load_policies(out, compare_seed)
    res["compare_seed0"] = {k: {"pref_mean_score": pooled_mean({0: per_user(v)}), **model_detail(v)}
                            for k, v in p0.items()}
    v1p = os.path.join("runs", "m1", "review_stats.json")
    if os.path.exists(v1p):
        v1 = _load(v1p)["preference_sessions_only"]
        res["compare_seed0_v1"] = {"source": v1p, "policies": v1["policies"], "ideal_ceiling_estimate": v1["ideal_ceiling"]}
    with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res
