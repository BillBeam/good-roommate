"""M2 起点测量的中期结论：只用已经跑完的部分，不等全量。

    python -m roommate.m2_interim

- 基线与对照组：从请求缓存的快照（runs/m2/interim/cache_snapshot_*）只用缓存重建每一组，
  每个用户从第一次缺缓存的会话起截断，所以子集是"每个用户的前 k 次会话"。
- 比较都在同一批会话上做：Qwen 子集里有的（种子、用户、第几次会话），DeepSeek V4 Pro、理想记忆策略、
  记住但不更新也只取这些会话（来自 M1 第二版的完整记录）。区间按用户重抽样（每个世界内分别重抽）。
- 组内差异：runs/m2/variance_interim/ 下的子集（种子 1、24 个用户、前 3 次会话、思考关、温度 1、4 遍）。
- 结果写到 runs/m2/interim/interim_stats.json，报告由 m2_report 读它生成开头的"中期结论"。
"""

import json
import os
import random
import shutil

from . import m2, stats
from .llm import Ledger, LLMClient
from .m2_report import _pref, failure_counts, loop_stats, usage_summary

OUT = m2.OUT
INT = os.path.join(OUT, "interim")
REFS = ["llm_deepseek-v4-pro", "ideal", "remember_no_update"]
N_BOOT = 5000


def snapshot_cache():
    """复制一份缓存，去掉正在写的半行，避免读到另一个进程写了一半的行。"""
    src = os.path.join(OUT, "llm_cache")
    dst = os.path.join(INT, "cache_snapshot")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for fn in os.listdir(src):
        if fn.endswith(".jsonl"):
            good = []
            with open(os.path.join(src, fn), encoding="utf-8") as f:
                for line in f:
                    try:
                        json.loads(line)
                        good.append(line)
                    except ValueError:
                        pass
            with open(os.path.join(dst, fn), "w", encoding="utf-8") as f:
                f.writelines(good)
        else:
            shutil.copy(os.path.join(src, fn), dst)
    return dst


def rebuild(cache, jobs):
    c = LLMClient(cache, Ledger(None, 0), offline=True)
    out = {}
    for job in jobs:
        recs = [dict(r, _seed=job[2]) for r in m2.run_job(c, job, cache_only=True)]
        trans = {((job[2], r["uid"]), r["idx"]): r.get("transcript", []) for r in recs}
        out[m2.job_name(job)] = (recs, trans)
    return out


def rebuild_variance(cache, until=3):
    """组内差异子集：只用缓存重建思考关、温度 1 的 K 遍（前 until 次会话），只保留 K 遍都跑完的（用户、会话）。
    写到 runs/m2/variance_interim/，再由 m2_report.variance_stats 统计。"""
    c = LLMClient(cache, Ledger(None, 0), offline=True)
    reps = {}
    for r in range(1, m2.VAR_REPS + 1):
        job = ("var", "nothink", m2.VAR_SEED, r)
        reps[r] = m2.run_job(c, job, cache_only=True, until=until)
    common = set.intersection(*[{(x["uid"], x["idx"]) for x in v} for v in reps.values()])
    # 每个用户只保留"前几次会话在 K 遍里都完成"的连续前缀，保证同一用户在各遍里比较的是同一批会话
    keep = set()
    for uid in {u for u, _ in common}:
        i = 0
        while (uid, i) in common:
            keep.add((uid, i))
            i += 1
    base = os.path.join(OUT, "variance_interim", f"seed{m2.VAR_SEED}", "nothink_t1")
    if os.path.exists(base):
        shutil.rmtree(base)
    for r, recs in reps.items():
        rows = [{k: v for k, v in x.items() if k != "transcript"} for x in recs if (x["uid"], x["idx"]) in keep]
        d = os.path.join(base, f"rep{r}")
        os.makedirs(d)
        with open(os.path.join(d, "sessions.jsonl"), "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    # 宽口径：每个用户第一次会话（各遍 prompt 完全相同）在至少 2 遍里跑完，就比较这几遍的分数
    first = {}
    for r, recs in reps.items():
        for x in recs:
            if x["idx"] == 0 and x["domain"] != "util":
                first.setdefault(x["uid"], []).append({"rep": r, "score": round(x["grade"]["score"], 6),
                                                      "success": x["grade"]["success"], "chosen": x["chosen"],
                                                      "n_questions": len(x["questions"]),
                                                      "hit_max_turns": bool(x.get("hit_max_turns"))})
    multi = {u: v for u, v in first.items() if len(v) >= 2}
    loose = {"n_users_ge2_reps": len(multi),
             "n_users_differ": sum(len({e["score"] for e in v}) > 1 for v in multi.values()),
             "n_users_success_mixed": sum(0 < sum(e["success"] for e in v) < len(v) for v in multi.values()),
             "reps_per_user": {u: len(v) for u, v in sorted(multi.items())},
             "scores": {u: [e["score"] for e in v] for u, v in sorted(multi.items())},
             "n_completed_first_sessions": sum(len(v) for v in first.values()),
             "n_first_sessions_hit_max_turns": sum(e["hit_max_turns"] for v in first.values() for e in v),
             "n_first_sessions_no_choice": sum(e["chosen"] is None for v in first.values() for e in v),
             "completed_by_rep": {r: sum(1 for x in recs if x["idx"] == 0) for r, recs in reps.items()}}
    loose["share_users_differ"] = loose["n_users_differ"] / loose["n_users_ge2_reps"] if multi else None
    return {"loose_first_session": loose, "n_pairs_kept": len(keep), "n_users_kept": len({u for u, _ in keep}),
            "sessions_per_user": {u: sum(1 for uu, _ in keep if uu == u) for u in sorted({u for u, _ in keep})}}


def subset_compare(qrecs, seeds):
    """qrecs：某一模式在若干种子上的子集记录。返回在同一批会话上 Qwen 与参照策略的得分和区间。"""
    have = {(r["_seed"], r["uid"], r["idx"]) for r in qrecs}
    pols = {"qwen": qrecs}
    for k in REFS:
        rs = []
        for s in seeds:
            for r in stats.load_policies(m2.M1, s).get(k, []):
                if (s, r["uid"], r["idx"]) in have:
                    rs.append(dict(r, _seed=s))
        pols[k] = rs
    per = {k: {s: stats.per_user([r for r in v if r["_seed"] == s]) for s in seeds} for k, v in pols.items()}
    # 只保留每个世界里在子集中有偏好会话的用户
    for s in seeds:
        users = set(per["qwen"][s])
        for k in per:
            per[k][s] = {u: per[k][s][u] for u in users if u in per[k][s]}
    means = {k: stats.pooled_mean(per[k]) for k in per}
    rng = random.Random(0)
    uids = {s: sorted(per["qwen"][s]) for s in seeds}
    boot = []
    for _ in range(N_BOOT):
        sample = {s: [rng.choice(uids[s]) for _ in uids[s]] for s in seeds if uids[s]}
        boot.append({k: stats.pooled_mean({s: per[k][s] for s in sample}, sample) for k in per})
    share = lambda m: (m["qwen"] - m["remember_no_update"]) / (m["ideal"] - m["remember_no_update"])
    pref = _pref(qrecs)
    return {
        "n_sessions": len(pref), "n_users": sum(len(v) for v in uids.values()),
        "sessions_per_user": {s: (len(_pref([r for r in qrecs if r["_seed"] == s])) / len(uids[s]) if uids[s] else 0)
                              for s in seeds},
        "max_idx": max(r["idx"] for r in qrecs) if qrecs else None,
        "mean_score": means["qwen"], "ci95": stats._ci([b["qwen"] for b in boot]),
        "p_below_075": sum(b["qwen"] < 0.75 for b in boot) / len(boot),
        "refs_same_sessions": {k: means[k] for k in REFS},
        "minus_v4pro_ci95": stats._ci([b["qwen"] - b["llm_deepseek-v4-pro"] for b in boot]),
        "headroom_share": share(means), "headroom_share_ci95": stats._ci([share(b) for b in boot]),
        "success_rate": sum(r["grade"]["success"] for r in pref) / len(pref),
        "avg_questions": sum(len(r["questions"]) for r in pref) / len(pref),
    }


def main():
    os.makedirs(INT, exist_ok=True)
    m2.load_prices()
    cache = snapshot_cache()
    jobs = [j for j in m2.plan() if j[0] in ("base", "ctrl")]
    built = rebuild(cache, jobs)
    res = {"note": "中期结论：只用已跑完的部分（每个用户的前 k 次会话），数字都来自子集", "base": {}, "ctrl": {}}
    for mode in m2.MODES:
        qrecs = [r for s in m2.SEEDS for r in built[f"{mode}@{s}"][0]]
        trans = {}
        for s in m2.SEEDS:
            trans.update(built[f"{mode}@{s}"][1])
        if not _pref(qrecs):
            continue
        seeds = sorted({r["_seed"] for r in qrecs})
        e = subset_compare(qrecs, seeds)
        e["seeds"] = seeds
        e["failures"] = failure_counts(qrecs)
        e["loops"] = loop_stats([dict(r, uid=(r["_seed"], r["uid"])) for r in qrecs], trans)
        e["usage"] = usage_summary(qrecs)
        # 丢分去向：选错的会话里，没选出任何选项（格式、兜圈子、用满轮数）占多少
        pref = _pref(qrecs)
        fails = [r for r in pref if not r["grade"]["success"]]
        e["fail_breakdown"] = {
            "n_fail": len(fails),
            "no_choice": sum(r["chosen"] is None for r in fails),
            "no_choice_hit_max_turns": sum(r["chosen"] is None and bool(r.get("hit_max_turns")) for r in fails),
            "wrong_choice": sum(r["chosen"] is not None for r in fails),
            "question_loss_per_session": sum(r["grade"]["success"] - r["grade"]["score"] for r in pref) / len(pref),
            "fail_loss_per_session": len(fails) / len(pref),
            "loss_from_no_choice_per_session": sum(r["chosen"] is None for r in fails) / len(pref),
        }
        res["base"][mode] = e
    for mode in m2.MODES:
        recs = built.get(f"{mode}@1@ctrl", ([], {}))[0]
        if _pref(recs):
            pref = _pref(recs)
            res["ctrl"][mode] = {"n_sessions": len(pref), "success_rate": sum(r["grade"]["success"] for r in pref) / len(pref),
                                 "mean_score": sum(r["grade"]["score"] for r in pref) / len(pref),
                                 "no_choice": sum(r["chosen"] is None for r in pref),
                                 "hit_max_turns": sum(bool(r.get("hit_max_turns")) for r in pref)}
    # 组内差异子集（只用缓存重建）
    vinfo = rebuild_variance(cache)
    vb = os.path.join(OUT, "variance_interim", f"seed{m2.VAR_SEED}", "nothink_t1")
    res["variance_loose"] = vinfo["loose_first_session"]
    if os.path.isdir(vb) and vinfo["n_pairs_kept"]:
        from .m2_report import variance_stats
        res["variance"] = variance_stats(OUT, "nothink", m2.VAR_SEED, sub="variance_interim")
        if res["variance"]:
            res["variance"]["rebuild"] = vinfo
            res["variance"]["users_with_difference"] = res["variance"]["n_users"] - res["variance"]["users_all_identical"]
            res["variance"]["users_with_difference_share"] = res["variance"]["users_with_difference"] / res["variance"]["n_users"]
    with open(os.path.join(INT, "interim_stats.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    shutil.rmtree(cache)
    return res


if __name__ == "__main__":
    r = main()
    for m, e in r["base"].items():
        print(m, "sessions", e["n_sessions"], "users", e["n_users"], "max_idx", e["max_idx"], "score", round(e["mean_score"], 3),
              e["ci95"], "v4pro same", round(e["refs_same_sessions"]["llm_deepseek-v4-pro"], 3), "fails", e["fail_breakdown"])
    print("ctrl", r["ctrl"])
    v = r.get("variance")
    if v:
        print("variance k", v["k"], "users differ", v["users_with_difference"], "/", v["n_users"])
