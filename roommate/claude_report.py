"""Claude 基线的统计与报告：只读 runs/m1v2/ 下的记录与账本，不调用任何模型。

    python -m roommate.claude_run --stages stats,report

口径与 M1 第二版的 DeepSeek 基线相同（roommate/stats.py）：分数只算和偏好有关的会话；区间按用户重抽样（每个世界内分别重抽，
5000 次）取 95%；剩余空间比例 = (模型分 − 最简单策略里的最高分) / (理想记忆策略 − 最简单策略里的最高分)，
三个量都在同一批世界上算；失分拆解按同一会话在"知道真实状态"对照组里选没选对来分。
报告里引用代码位置时用钉住提交、带行号的 GitHub 链接（code_link）。
"""

import json
import os
import subprocess

from . import stats
from .policies import TRIVIAL
from .report import MODEL_CN, NM_CN, REF_CN, Linker

OUT = "runs/m1v2"
CDIR = os.path.join(OUT, "claude")
SEEDS = [1, 2, 3]
DEEPSEEK = ["deepseek-flash", "deepseek-v4-pro"]
CLAUDE = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]
LLMS = DEEPSEEK + CLAUDE
REFS = ["ideal", "remember_no_update", "ask_all"]
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5",
         "claude-fable-5-1": "Fable 5.1", "deepseek-v4-pro": "V4 Pro", "deepseek-flash": "Flash"}
ROLE_CN = {"correct": "选对", "near_miss": "选到近似项（用户自己的状态差一点）", "decoy": "选到别的用户的选项",
           None: "没选（没作答）"}
STATS = os.path.join(CDIR, "claude_stats.json")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _ci(xs):
    return stats._ci(xs)


# ---------- 统计 ----------

def _block(pols, keys, seeds, users=None):
    """keys 在 seeds 上的均分、区间、剩余空间比例。users：{种子: 只算这些用户}（Fable 只跑了部分用户时用）。"""
    per = {}
    for k in keys:
        per[k] = {}
        for s in seeds:
            pu = stats.per_user(pols[s][k])
            if users:
                pu = {u: v for u, v in pu.items() if u in users[s]}
            per[k][s] = pu
    means = {k: stats.pooled_mean(per[k]) for k in keys}
    boot = stats.bootstrap(per, seeds, lambda m: m)
    trivial = [k for k in TRIVIAL if k in means]
    bt = max(trivial, key=lambda k: means[k])
    share = lambda m, k: (m[k] - m[bt]) / (m["ideal"] - m[bt])
    out = {"seeds": seeds, "best_trivial": bt, "n_users": {s: len(per["ideal"][s]) for s in seeds},
           "n_sessions": sum(v[1] for s in seeds for v in per["ideal"][s].values()), "rows": {}}
    for k in keys:
        pref = [r for s in seeds for r in pols[s][k] if r["domain"] != "util" and (not users or r["uid"] in users[s])]
        out["rows"][k] = {"mean_score": means[k], "ci95": _ci([b[k] for b in boot]),
                          "success_rate": sum(r["grade"]["success"] for r in pref) / len(pref),
                          "avg_questions": sum(len(r["questions"]) for r in pref) / len(pref),
                          "headroom_share": share(means, k), "headroom_share_ci95": _ci([share(b, k) for b in boot]),
                          "gap_to_ideal": means["ideal"] - means[k],
                          "gap_to_ideal_ci95": _ci([b["ideal"] - b[k] for b in boot])}
    return out


def _complete(pols, k, s, n_pref):
    return k in pols[s] and sum(r["domain"] != "util" for r in pols[s][k]) == n_pref[s]


def _choice_split(recs):
    """不用对照组的粗分：选错时选到的是哪类选项。近似项 = 用户自己的状态差一点（旧状态、临时情况等），多半是记忆或变化的问题。"""
    pref = [r for r in recs if r["domain"] != "util"]
    n = len(pref)
    cnt = {}
    for r in pref:
        cnt[r["chosen_role"]] = cnt.get(r["chosen_role"], 0) + 1
    return {"n": n, "share": {str(k): v / n for k, v in cnt.items()}, "count": {str(k): v for k, v in cnt.items()},
            "question_loss": sum(r["grade"]["success"] - r["grade"]["score"] for r in pref) / n}


def _behavior(recs):
    pref = [r for r in recs if r["domain"] != "util"]
    allr = recs
    issues = [i for r in allr for i in (r.get("format_issues") or [])]
    kinds = {}
    for i in issues:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    return {"n_sessions": len(allr), "n_pref": len(pref),
            "no_choice_pref": sum(r["chosen"] is None for r in pref),
            "hit_max_turns": sum(bool(r.get("hit_max_turns")) for r in allr),
            "refused_sessions": sum(bool(r.get("refused")) for r in allr),
            "errors": sum(bool(r.get("error")) for r in allr),
            "avg_turns": sum(r.get("turns", 0) for r in allr) / len(allr) if allr else None,
            "issue_counts": kinds}


def _usage(recs):
    """逐次会话的 token 与花费（记录里的 usage 是这次会话原响应的 usage；缓存命中时也照原样算，等于"这次会话按价目要花多少"）。"""
    rs = [r for r in recs if r.get("usage")]
    if not rs:
        return None
    n = len(rs)
    tot = {k: sum(r["usage"][k] for r in rs) for k in ("calls", "input_tokens", "cache_write_tokens",
                                                       "cache_read_tokens", "output_tokens", "cost_usd")}
    return {"n_sessions": n, "total": tot, "per_session": {k: v / n for k, v in tot.items()}}


def write_stats():
    g = _load(os.path.join(OUT, "gates.json"))
    pols = {s: stats.load_policies(OUT, s) for s in SEEDS}
    for s in SEEDS:
        for recs in pols[s].values():
            for r in recs:
                r["_seed"] = s
    n_pref = {s: sum(r["domain"] != "util" for r in pols[s]["oracle"]) for s in SEEDS}
    lk = {m: f"llm_{m}" for m in LLMS}
    res = {"gate_seeds": SEEDS, "n_bootstrap": stats.N_BOOT}
    # ① 三个世界都完整的模型：和 gates.json 同一套数（同一个重抽样种子，区间逐位相同）
    full3 = [m for m in LLMS if all(_complete(pols, lk[m], s, n_pref) for s in SEEDS)]
    res["three_seed"] = dict(_block(pols, [lk[m] for m in full3] + REFS + [k for k in TRIVIAL if k not in REFS], SEEDS),
                             models=full3)
    # ② 种子 1：所有在种子 1 上完整的模型（Fable 至少跑了种子 1）
    full1 = [m for m in LLMS if _complete(pols, lk[m], 1, n_pref)]
    res["seed1"] = dict(_block(pols, [lk[m] for m in full1] + REFS + [k for k in TRIVIAL if k not in REFS], [1]),
                        models=full1)
    # ③ Fable 有部分世界或部分用户：按它实际跑到的（种子, 用户）和所有模型同批比较
    fk = lk["claude-fable-5-1"]
    partial = {}
    for s in SEEDS:
        if fk in pols[s]:
            us = sorted({r["uid"] for r in pols[s][fk]})
            if us:
                partial[s] = us
    res["fable_coverage"] = {s: len(v) for s, v in partial.items()}
    fable_full = [s for s in SEEDS if _complete(pols, fk, s, n_pref)]
    res["fable_full_seeds"] = fable_full
    if partial and any(not _complete(pols, fk, s, n_pref) for s in partial):
        seeds_p = sorted(partial)
        keys = [lk[m] for m in LLMS if all(lk[m] in pols[s] and
                                           {r["uid"] for r in pols[s][lk[m]]} >= set(partial[s]) for s in seeds_p)]
        res["fable_subset"] = dict(_block(pols, keys + REFS + [k for k in TRIVIAL if k not in REFS], seeds_p,
                                          users={s: set(v) for s, v in partial.items()}),
                                   models=[k[4:] for k in keys], users=partial)
    # 细节：每个模型在它跑完的世界上（三个世界都完整就用三个世界）
    res["detail"], res["behavior"], res["choice_split"], res["usage"], res["seeds_used"] = {}, {}, {}, {}, {}
    for m in LLMS:
        seeds_m = SEEDS if m in full3 else [s for s in SEEDS if lk[m] in pols[s]]
        recs = [r for s in seeds_m for r in pols[s].get(lk[m], [])]
        if not recs:
            continue
        res["seeds_used"][m] = seeds_m
        res["detail"][m] = stats.model_detail(recs)
        res["behavior"][m] = _behavior(recs)
        res["choice_split"][m] = _choice_split(recs)
        res["usage"][m] = _usage(recs)
        ck = lk[m] + "_true_memory"
        crecs = [r for s in SEEDS for r in pols[s].get(ck, [])]
        if crecs and m.startswith("claude-"):
            res["usage"][m + ":control"] = _usage(crecs)
            res["behavior"][m + ":control"] = _behavior(crecs)
    # ④ 失分拆解（有对照组的）：直接取 gates.json 的（同口径）；另给 V4 Pro 与 Opus 5.5 在种子 1 上的同批比较
    res["loss_breakdown"] = {m: g["loss_breakdown"][lk[m]] for m in LLMS if lk[m] in g["loss_breakdown"]}
    res["loss_breakdown_refs"] = {k: g["loss_breakdown"][k] for k in ("ideal", "remember_no_update")}
    # 对照组也选错的会话里，选到的是哪类选项、有没有作答（"挑店或格式问题"再细分）
    res["alignment_detail"] = {}
    for m in LLMS:
        ck = lk[m] + "_true_memory"
        if not any(ck in pols[s] for s in SEEDS):
            continue
        ctl = {(r["_seed"], r["uid"], r["idx"]): r for s in SEEDS for r in pols[s].get(ck, [])}
        both = [(r, ctl[(r["_seed"], r["uid"], r["idx"])]) for s in SEEDS for r in pols[s].get(lk[m], [])
                if (r["_seed"], r["uid"], r["idx"]) in ctl and r["domain"] != "util" and not r["grade"]["success"]]
        al = [(r, c) for r, c in both if not c["grade"]["success"]]
        res["alignment_detail"][m] = {
            "n_fail": len(both), "n_alignment": len(al),
            "alignment_no_choice": sum(r["chosen"] is None for r, _ in al),
            "alignment_same_wrong_as_control": sum(r["chosen"] == c["chosen"] for r, c in al),
            "memory_fail_roles": _count([r["chosen_role"] for r, c in both if c["grade"]["success"]])}
    # ⑤ 花费：Claude 账本（实时累计的那本）与 DeepSeek 账本
    cl = os.path.join(CDIR, "spend_ledger.json")
    res["claude_ledger"] = _load(cl) if os.path.exists(cl) else None
    dl = _load(os.path.join(OUT, "spend_ledger.json"))
    res["deepseek_ledger_rows"] = {k: dict(v, n_sessions=sum(len(pols[s].get(lk[k.split("|")[0]] + (
        "_true_memory" if k.endswith("|control") else ""), [])) for s in SEEDS))
                                   for k, v in dl["by_model_role"].items() if k.split("|")[0] in DEEPSEEK
                                   and k.split("|")[1] in ("agent", "control")}
    for name in ("pilot.json", "estimate.json", "repro_check.json", "full_log.json", "config.json"):
        p = os.path.join(CDIR, name)
        res[name.split(".")[0]] = _load(p) if os.path.exists(p) else None
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def _count(xs):
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


# ---------- 代码位置：钉住提交、带行号的 GitHub 链接 ----------

def _git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def _repo_url():
    u = _git("remote", "get-url", "origin")
    u = u.replace("git@github.com:", "https://github.com/")
    if "github.com/" in u:
        u = "https://github.com/" + u.split("github.com/", 1)[1]
    return u[:-4] if u.endswith(".git") else u


def code_link(path, anchor, span=1, label=None):
    """path 在最后一次改它的提交里、第一处包含 anchor 的那一行（往下 span 行）。提交必须已经推到 origin。"""
    sha = _git("log", "-1", "--format=%H", "--", path)
    if not _git("branch", "-r", "--contains", sha):
        raise RuntimeError(f"{path} 的提交 {sha[:7]} 还没推到 origin，链接会失效")
    text = _git("show", f"{sha}:{path}").split("\n")
    ln = next(i + 1 for i, line in enumerate(text) if anchor in line)
    rng = f"L{ln}" + (f"-L{ln + span - 1}" if span > 1 else "")
    return f"[{label or f'{path}#{rng}'}]({_repo_url()}/blob/{sha}/{path}#{rng})"


# ---------- 报告 ----------

def _name(m):
    if m in MODEL_CN:
        return MODEL_CN[m]
    if m.startswith("llm_"):
        return MODEL_CN.get(m[4:], m)
    return REF_CN.get(m, m)


def write_report(path):
    S = _load(STATS)
    L = Linker(path)
    sp = STATS
    n = lambda v, kind="f3": L.n(v, sp, kind)
    ci = lambda v, kind="f3": L.ci(v, sp, kind)
    lines = []
    w = lines.append
    T3, T1 = S["three_seed"], S["seed1"]
    claude3 = [m for m in T3["models"] if m.startswith("claude-")]
    led = S["claude_ledger"] or {"total_usd": 0.0, "by_model_role": {}}
    lp = os.path.join(CDIR, "spend_ledger.json")

    # 最强的模型：三个世界都完整的模型里分最高的；Fable 只在种子 1 上比
    best = max(T3["models"], key=lambda m: T3["rows"][f"llm_{m}"]["mean_score"])
    b3 = T3["rows"][f"llm_{best}"]
    first = _headline(S, best)
    w("# M1 第二版：Claude 系列基线（Haiku 5.5、Sonnet 5.5、Opus 5.5、Fable 5.1），与 DeepSeek 同考场并排")
    w("")
    w(first)
    w("")
    w("日志目录：[runs/m1v2/claude/](" + L.rel(CDIR) + ")（试跑、估算、计划、账本、逐次调用、复现检查）与 "
      "`runs/m1v2/seed{1,2,3}/policies/llm_claude-*`（逐次会话记录与对话）。数字由 `python -m roommate.claude_run --stages stats,report` "
      f"从日志生成，链接指向 [claude_stats.json]({L.rel(sp)})，它再由逐次会话记录算出。"
      "分数只算和偏好有关的会话；区间都是按用户重抽样（每个世界内分别重抽，5000 次）的 95% 区间。"
      "Claude 的输出只用于评测和分析，不进任何训练数据（[DATA_POLICY.json](" + L.rel(os.path.join(CDIR, "DATA_POLICY.json")) + ")）。")
    w("")

    # ---------- 三个问题 ----------
    w("## 1. 三个问题")
    w("")
    w(f"**最强的模型离理想记忆策略还差多少。** 三个闸门世界上分最高的是 {_name(best)}：每次会话 {n(b3['mean_score'])} 分"
      f"（区间 {ci(b3['ci95'])}），理想记忆策略 {n(T3['rows']['ideal']['mean_score'])}，差 {n(b3['gap_to_ideal'])}"
      f"（区间 {ci(b3['gap_to_ideal_ci95'])}）；剩余空间比例 {n(b3['headroom_share'], 'pct')}（区间 {ci(b3['headroom_share_ci95'], 'pct')}）。"
      + _fable_sentence(S, n, ci))
    w("")
    w("**丢分落在哪几类。** " + _loss_sentence(S, n))
    w("")
    w("**差距随模型变大怎么变。** " + _scaling_sentence(S, n, ci))
    w("")

    # ---------- 设置与对不齐 ----------
    w("## 2. 设置：和 DeepSeek 基线一致的地方与对不齐的地方")
    w("")
    w("一致：同一个考场（`runs/m1v2/seed{1,2,3}/world.json`，种子 1、2、3 各 24 个用户 × 8 次会话，没有重新生成）、"
      f"同一段系统提示和同一组工具（{code_link('roommate/agent_claude.py', 'def claude_tools', 3, 'agent_claude.claude_tools')} "
      f"把 {code_link('roommate/agent_llm.py', 'def tools(question_cost)', 1, 'agent_llm.tools')} 的名字、说明、参数原样搬过来，"
      f"系统提示直接用 {code_link('roommate/agent_llm.py', 'def system_prompt', 1, 'agent_llm.system_prompt')}）、"
      "同样的 14 轮上限和\"只输出文字时提醒调用工具\"、同一套判分（roommate/grading.py，没有改）、"
      "每次会话从零开始、跨会话只靠记忆工具。\"知道真实状态\"对照组同样是把真实状态写进记忆，只给最强档（Opus 5.5）跑种子 1，"
      "和 DeepSeek 只给 V4 Pro 跑种子 1 的口径一致。")
    w("")
    w("对不齐的设置（逐条）：")
    w("")
    w("| 设置 | DeepSeek 基线 | Claude 基线 | 可能的影响 |")
    w("|---|---|---|---|")
    for row in _mismatch_rows(S):
        w("| " + " | ".join(row) + " |")
    w("")

    # ---------- 试跑与估算 ----------
    w("## 3. 全量之前：试跑实测与估算")
    w("")
    pil, est = S.get("pilot"), S.get("estimate")
    if pil and est:
        pp, ep = os.path.join(CDIR, "pilot.json"), os.path.join(CDIR, "estimate.json")
        w(f"每个模型先只跑种子 {pil['seed']} 的前 {pil['n_users']} 个用户的全部会话（每人 8 次），Opus 5.5 的对照组同样试跑。"
          "试跑的请求照常进缓存，全量时这两个用户直接命中缓存，不重复花钱。")
        w("")
        w("| 试跑 | 会话数 | 每次会话花费（美元） | 两个用户各自的每次会话花费 | 每次会话调用数 | 每次会话 tokens：未缓存输入 / 缓存写入 / 缓存命中 / 输出 | 试跑得分 |")
        w("|---|---|---|---|---|---|---|")
        for k, v in pil["runs"].items():
            t = v["tokens_per_session"]
            by = " / ".join(f"{c:.4f}" for c in v["cost_per_session_by_user"].values())
            w(f"| {_name(k.split(':')[1])}{'（知道真实状态）' if k.endswith('true_memory') else ''} | {v['n_sessions']} | "
              f"{L.n(v['cost_per_session'], pp, 'p')} | {by} | {t['calls']:.1f} | "
              f"{t['input']:.0f} / {t['cache_write']:.0f} / {t['cache_read']:.0f} / {t['output']:.0f} | {L.n(v['mean_score_pref'], pp)} |")
        w("")
        w(f"按实测重新估算的全卡总价（四个模型各三个世界 + Opus 对照组）：{L.n(est['full_card_usd'], ep, 'usd')}"
          f"（按两个用户里较贵的那个算 {L.n(est['full_card_usd_high'], ep, 'usd')}），上限 {est['budget_usd']:.0f} 美元。"
          + ("估算超过上限，按优先级裁剪：" if est["over_budget"] else "估算没超过上限，不用裁剪。"))
        w("")
        w("| 组 | 会话数（其中要新跑的） | 估算（美元） | 计划 |")
        w("|---|---|---|---|")
        planned = {(p["model"], p["control"], p["seed"]): p for p in est["plan"]}
        skipped = {(p["model"], p["control"], p["seed"]): p for p in est["skipped"]}
        for r in est["rows"]:
            k = (r["model"], r["control"], r["seed"])
            p, sk = planned.get(k), skipped.get(k)
            plan = ("跑" if p and not p.get("users") else f"只跑前 {p['users']} 个用户" if p else "不跑") + \
                   (f"（{sk['reason']}）" if sk else "")
            w(f"| {_name(r['model'])}{'（知道真实状态）' if r['control'] else ''} · 种子 {r['seed']} | {r['sessions']}（{r['sessions_not_cached']}） | "
              f"{L.n(r['est_usd'], ep, 'f2')} | {plan} |")
        w("")
        w(f"计划花费（含试跑）：{L.n(est['planned_usd'], ep, 'usd')}。账本另留 {est['reserve_usd']:.0f} 美元给并发中还没记账的请求："
          f"累计加余量到 {est['budget_usd']:.0f} 美元就不再发新请求（{code_link('roommate/llm.py', 'self.data[\"total_usd\"] + self.reserve >= self.limit', 1, 'llm.Ledger.check')}）。"
          f"全量时每组开跑前再按已花的钱核一遍（{code_link('roommate/claude_run.py', 'def stage_full', 1, 'claude_run.stage_full')}）。")
        fl = S.get("full_log") or []
        if fl:
            w("")
            w("全量运行记录（[full_log.json](" + L.rel(os.path.join(CDIR, "full_log.json")) + ")）：")
            w("")
            for batch in fl:
                for r in batch["runs"]:
                    w(f"- {r['run']}：{ {'done': '跑完', 'skipped': '跳过', 'stopped': '停止'}[r['status']] }"
                      + (f"，{r['reason']}" if r.get("reason") else "")
                      + (f"，累计 {r['cum_usd']:.2f} 美元" if r.get("cum_usd") is not None else ""))
    else:
        w("还没有试跑记录。")
    w("")

    # ---------- 结果 ----------
    w("## 4. 结果：DeepSeek 与 Claude 同表")
    w("")
    w(f"三个闸门世界（每个模型 {T3['n_sessions']} 次偏好会话）：")
    w("")
    _main_table(w, S, T3, n, ci)
    w(f"剩余空间比例 = (模型分 − 最简单策略里的最高分 {n(T3['rows'][T3['best_trivial']]['mean_score'])}，即\"{_name(T3['best_trivial'])}\") / "
      f"(理想记忆策略 {n(T3['rows']['ideal']['mean_score'])} − 最简单策略里的最高分)。"
      "M1 第二版报告第 7 节的基线表已把 Claude 一并列进去（reports/m1v2-baseline.md）；闸门④的定义没有改，仍按约定只看 DeepSeek V4 Pro。")
    w("")
    w(f"种子 1（Fable 5.1 至少跑了这个世界；每个模型 {T1['n_sessions']} 次偏好会话，区间只在种子 1 内重抽用户）：")
    w("")
    _main_table(w, S, T1, n, ci, detail=False)
    if S.get("fable_subset"):
        F = S["fable_subset"]
        cov = "；".join(f"种子 {s}：{len(u)} 个用户" for s, u in F["users"].items())
        w(f"Fable 5.1 只跑了部分世界或部分用户（{cov}）。在同一批（种子, 用户）上并排：")
        w("")
        _main_table(w, S, F, n, ci, detail=False)

    # ---------- 失分 ----------
    w("## 5. 丢分落在哪几类")
    w("")
    w("**失分拆解**（每次偏好会话的平均失分；口径和 DeepSeek 相同：选错的会话按同一会话在\"知道真实状态\"对照组里选没选对来分，"
      "对照组选对了算\"没记住或没察觉变化\"，对照组也选错算\"挑店或格式问题\"；所以只算有对照组记录的会话）：")
    w("")
    w("| 策略 | 算了哪些会话 | 总失分 | 提问 | 没记住或没察觉变化 | 挑店或格式问题 | 对照组成功率 | 对照组平均提问 |")
    w("|---|---|---|---|---|---|---|---|")
    for m, v in S["loss_breakdown"].items():
        if v.get("memory_or_change_loss") is None:
            continue
        w(f"| {_name(m)} | 种子 {'、'.join(map(str, v['seeds']))}，{v['n_sessions']} 次 | {n(v['total_loss'])} | {n(v['question_loss'])} | "
          f"{n(v['memory_or_change_loss'])} | {n(v['alignment_loss'])} | {n(v.get('control_success_rate'), 'pct')} | {n(v.get('control_avg_questions'), 'f2')} |")
    for k, v in S["loss_breakdown_refs"].items():
        w(f"| {_name(k)} | 三个闸门世界 | {n(v['total_loss'])} | {n(v['question_loss'])} | {n(v['memory_or_change_loss'])} | "
          f"{n(v['alignment_loss'])} | （对照用标准策略） | — |")
    w("")
    ad = S.get("alignment_detail", {})
    if ad:
        w("\"挑店或格式问题\"再细分（对照组也选错的会话里）：")
        w("")
        for m, v in ad.items():
            w(f"- {_name(m)}：选错 {n(v['n_fail'], 'int')} 次，其中对照组也选错 {n(v['n_alignment'], 'int')} 次；"
              f"这些里没作答（格式或轮数用满）{n(v['alignment_no_choice'], 'int')} 次，和对照组选了同一个错项 {n(v['alignment_same_wrong_as_control'], 'int')} 次。"
              f"对照组选对、普通运行选错的那些会话选到的选项：" +
              "，".join(f"{ROLE_CN.get(None if r == 'None' else r, r)} {c} 次" for r, c in v["memory_fail_roles"].items()))
        w("")
    w("**不用对照组的粗分**（每个模型都有）：提问花掉的分，加上选错时选到的是哪类选项。近似项是用户自己的状态差一点"
      "（事件之前的旧状态、忽略或误用临时情况、多一个没发生的变化），选到它多半是没记住或没察觉变化；"
      "选到别的用户的选项、或没作答，多半是挑店或格式问题。")
    w("")
    w("| 模型 | 世界 | 提问花掉的分 | 选对 | 选到近似项 | 选到别的用户的选项 | 没作答 |")
    w("|---|---|---|---|---|---|---|")
    for m in LLMS:
        c = S["choice_split"].get(m)
        if not c:
            continue
        sh = c["share"]
        w(f"| {_name(m)} | {'、'.join(map(str, S['seeds_used'][m]))} | {n(c['question_loss'])} | {n(sh.get('correct', 0), 'pct')} | "
          f"{n(sh.get('near_miss', 0), 'pct')} | {n(sh.get('decoy', 0), 'pct')} | {n(sh.get('None', 0), 'pct')} |")
    w("")
    w("按近似项种类看选对率（每个模型在它跑完的世界上合计；括号里是会话数）：")
    w("")
    kinds = list(NM_CN)
    w("| 模型 | " + " | ".join(NM_CN[k] for k in kinds) + " |")
    w("|---|" + "---|" * len(kinds))
    for m in LLMS:
        d = S["detail"].get(m)
        if not d:
            continue
        cells = []
        for k in kinds:
            e = d["by_near_miss_kind"].get(k)
            cells.append(f"{n(e['success'] / e['n'], 'pct')}（{e['n']}）" if e else "—")
        w(f"| {_name(m)} | " + " | ".join(cells) + " |")
    w("")

    # ---------- 规模 ----------
    w("## 6. 随模型变大")
    w("")
    w("Claude 四档从小到大（Haiku 5.5 → Sonnet 5.5 → Opus 5.5 → Fable 5.1）。三个世界都跑完的用三个世界的数，"
      "只跑了种子 1 的用种子 1 的数（标在\"世界\"一列）；不同世界的数不能直接比，同一列里要比就看第 4 节种子 1 的表。")
    w("")
    w("| 模型 | 世界 | 得分 | 剩余空间比例 | 离理想记忆还差 | 每次提问 | 提问里被答\"说过了\" | 写记忆的会话 | 提问花掉的分 | 选到近似项 | 没作答 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in CLAUDE:
        if m not in S["detail"]:
            continue
        T = T3 if m in T3["models"] else T1 if m in T1["models"] else None
        if T is None:
            continue
        r, d, c = T["rows"][f"llm_{m}"], S["detail"][m], S["choice_split"][m]
        w(f"| {_name(m)} | {'、'.join(map(str, T['seeds']))} | {n(r['mean_score'])} | {n(r['headroom_share'], 'pct')} | {n(r['gap_to_ideal'])} | "
          f"{n(d['avg_questions'], 'f2')} | {n(d['told_before_rate'], 'pct')} | {n(d['memory_write_session_rate'], 'pct')} | "
          f"{n(c['question_loss'])} | {n(c['share'].get('near_miss', 0), 'pct')} | {n(c['share'].get('None', 0), 'pct')} |")
    w("")

    # ---------- 行为与格式 ----------
    w("## 7. 工具调用与格式")
    w("")
    w("| 记录 | 会话数 | 平均轮数 | 没作答的偏好会话 | 用满 14 轮 | 拒答（安全分类器） | 只输出文字、没调工具的轮 | 输出被截断的轮 | 接口出错 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m, b in S["behavior"].items():
        if not m.startswith("claude-"):
            continue
        ic = b["issue_counts"]
        w(f"| {_name(m.split(':')[0])}{'（知道真实状态）' if m.endswith(':control') else ''} | {b['n_sessions']} | {n(b['avg_turns'], 'f2')} | "
          f"{n(b.get('no_choice_pref'), 'int')} | {n(b['hit_max_turns'], 'int')} | {n(b['refused_sessions'], 'int')} | "
          f"{n(ic.get('no_tool_call', 0) + ic.get('tool_text', 0), 'int')} | {n(ic.get('truncated', 0), 'int')} | {n(b['errors'], 'int')} |")
    w("")

    # ---------- token 与花费 ----------
    w("## 8. token 用量与实际花费")
    w("")
    w(f"Claude 累计实际花费 {L.n(led['total_usd'], lp, 'usd')}（上限 150 美元），由账本按官方价目快照"
      "（[anthropic_pricing_2026-10-26.json](" + L.rel(os.path.join(OUT, "pricing", "anthropic_pricing_2026-10-26.json")) + ")）"
      "和每次响应里的 usage 实时累计，逐次调用见 [spend_calls.jsonl](" + L.rel(os.path.join(CDIR, "spend_calls.jsonl")) + ")。"
      "缓存命中的请求不再花钱，也不计入\"实际请求\"。")
    w("")
    w("| 模型 | 用途 | 实际请求 | 输入 tokens（其中缓存命中 / 缓存写入） | 输出 tokens（含思考） | 实际花费（美元） |")
    w("|---|---|---|---|---|---|")
    tot = 0.0
    for k, v in sorted(led["by_model_role"].items(), key=lambda kv: (CLAUDE.index(kv[0].split("|")[0])
                                                                       if kv[0].split("|")[0] in CLAUDE else 9, kv[0])):
        m, role = k.split("|")
        tot += v["cost_usd"]
        w(f"| {_name(m)} | {'对照组（知道真实状态）' if role == 'control' else '普通运行（含试跑）'} | {v['api_calls']} | "
          f"{v['prompt_tokens']}（{v['cached_prompt_tokens']} / {v.get('cache_write_tokens', 0)}） | {v['completion_tokens']} | {L.n(v['cost_usd'], lp, 'usd')} |")
    w(f"| **合计** | | | | | {L.n(tot, lp, 'usd')} |")
    w("")
    w("每次会话的平均用量（按记录里每次会话原响应的 usage 算，缓存命中的会话也照原样计入，所以等于\"按价目每次会话要花多少\"）：")
    w("")
    w("| 记录 | 会话数 | 调用数 | 未缓存输入 | 缓存写入 | 缓存命中 | 输出（含思考） | 花费（美元） |")
    w("|---|---|---|---|---|---|---|---|")
    for m, u in S["usage"].items():
        if not u:
            continue
        p = u["per_session"]
        w(f"| {_name(m.split(':')[0])}{'（知道真实状态）' if m.endswith(':control') else ''} | {u['n_sessions']} | {p['calls']:.2f} | "
          f"{p['input_tokens']:.0f} | {p['cache_write_tokens']:.0f} | {p['cache_read_tokens']:.0f} | {p['output_tokens']:.0f} | "
          f"{n(p['cost_usd'], 'p')} |")
    w("")
    dlp = os.path.join(OUT, "spend_ledger.json")
    w("对照：DeepSeek 基线的花费（[M1 第二版账本](" + L.rel(dlp) + ")，闸门世界三个种子，含重跑；tokenizer 不同，token 数不能直接和 Claude 比）：")
    w("")
    w("| 模型 | 用途 | 实际请求 | 输入 tokens（其中缓存命中） | 输出 tokens（含思考） | 花费（美元） | 每次会话（美元） |")
    w("|---|---|---|---|---|---|---|")
    for k, v in S["deepseek_ledger_rows"].items():
        m, role = k.split("|")
        w(f"| {_name(m)} | {'对照组' if role == 'control' else '普通运行'} | {v['api_calls']} | {v['prompt_tokens']}（{v['cached_prompt_tokens']}） | "
          f"{v['completion_tokens']} | {L.n(v['cost_usd'], dlp, 'usd')} | {v['cost_usd'] / v['n_sessions']:.4f}（{v['n_sessions']} 次） |")
    w("")

    # ---------- 复现、局限 ----------
    w("## 9. 复现、数据用途与局限")
    w("")
    rp = S.get("repro_check")
    if rp:
        w(f"- 只用缓存重跑全部 Claude 记录：{L.n(sum(rp['runs'].values()), os.path.join(CDIR, 'repro_check.json'), 'int')} / {len(rp['runs'])} 组与保存的记录逐字一致。"
          "Claude 不能设温度 0，重新向接口发同样的请求不会得到同样的回答；复现只靠缓存。")
    w(f"- Claude 的输出只用于评测和分析：存放它们的目录都有 DATA_POLICY.json（for_training = false），"
      f"以后读训练数据的代码要先过 {code_link('roommate/data_policy.py', 'def assert_trainable', 1, 'data_policy.assert_trainable')}。")
    w("- 密钥只从环境变量 CLAUDE_EVAL_API_KEY 读取（" + code_link("roommate/llm.py", "key = os.environ.get(ANTHROPIC_KEY_ENV)", 1, "llm.LLMClient._post_anthropic") +
      "），不写进文件、日志和提交；请求显式发到 https://api.anthropic.com，不用环境里给别的工具配的 ANTHROPIC_BASE_URL。")
    for x in _limitations(S):
        w(f"- {x}")
    w("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _main_table(w, S, T, n, ci, detail=True):
    w("| 模型 / 策略 | 每次会话得分 | 区间 | 选对率 | 每次会话提问数 | 剩余空间比例 | 区间 | 离理想记忆还差 |")
    w("|---|---|---|---|---|---|---|---|")
    for m in T["models"]:
        r = T["rows"][f"llm_{m}"]
        w(f"| {_name(m)} | {n(r['mean_score'])} | {ci(r['ci95'])} | {n(r['success_rate'], 'pct')} | {n(r['avg_questions'], 'f2')} | {n(r['headroom_share'], 'pct')} | "
          f"{ci(r['headroom_share_ci95'], 'pct')} | {n(r['gap_to_ideal'])} |")
    for k in REFS:
        r = T["rows"][k]
        w(f"| {_name(k)} | {n(r['mean_score'])} | {ci(r['ci95'])} | {n(r['success_rate'], 'pct')} | {n(r['avg_questions'], 'f2')} | "
          f"{n(r['headroom_share'], 'pct')} | — | {n(r['gap_to_ideal'])} |")
    w("")


def _headline(S, best):
    """报告第一句：不超过 20 个字的结论。"""
    T3 = S["three_seed"]
    r = T3["rows"][f"llm_{best}"]
    nm = SHORT.get(best, best)
    pct = round(100 * r["headroom_share"])
    if r["headroom_share_ci95"][1] < 1.0:
        cands = [f"{nm} 最强，仍未到理想记忆。", f"{nm} 最强，剩余空间 {pct}%。", f"{nm} 最强，仍有差距。"]
    else:
        cands = [f"{nm} 最强，已接近理想记忆。", f"{nm} 最强，剩余空间 {pct}%。", f"{nm} 已接近理想。"]
    return next(c for c in cands if len(c) <= 20)


def _fable_sentence(S, n, ci):
    T1 = S["seed1"]
    if "claude-fable-5-1" not in T1["models"]:
        return ""
    f = T1["rows"]["llm_claude-fable-5-1"]
    others = sorted((m for m in T1["models"] if m != "claude-fable-5-1"),
                    key=lambda m: -T1["rows"][f"llm_{m}"]["mean_score"])
    o = others[0]
    ro = T1["rows"][f"llm_{o}"]
    return (f"Fable 5.1 在种子 1 上得 {n(f['mean_score'])}（区间 {ci(f['ci95'])}，剩余空间比例 {n(f['headroom_share'], 'pct')}），"
            f"同一个世界上 {_name(o)} 得 {n(ro['mean_score'])}（剩余空间比例 {n(ro['headroom_share'], 'pct')}）。")


def _loss_sentence(S, n):
    lb = S["loss_breakdown"]
    parts = []
    for m in ("claude-opus-5-5", "deepseek-v4-pro"):
        v = lb.get(m)
        if v and v.get("memory_or_change_loss") is not None:
            parts.append(f"{_name(m)}（种子 1，有对照组）每次会话失分 {n(v['total_loss'])}：提问 {n(v['question_loss'])}、"
                         f"没记住或没察觉变化 {n(v['memory_or_change_loss'])}、挑店或格式问题 {n(v['alignment_loss'])}")
    ref = S["loss_breakdown_refs"]["ideal"]
    return "；".join(parts) + f"。理想记忆策略只在提问上失 {n(ref['question_loss'])}。其余模型没有对照组，第 5 节给出按选错选项种类的粗分。"


def _scaling_sentence(S, n, ci):
    T3 = S["three_seed"]
    ms = [m for m in CLAUDE if m in T3["models"]]
    if not ms:
        return "Claude 还没有三个世界都跑完的结果。"
    cells = "，".join(f"{_name(m)} {n(T3['rows'][f'llm_{m}']['mean_score'])}（剩余空间 {n(T3['rows'][f'llm_{m}']['headroom_share'], 'pct')}）"
                      for m in ms)
    q = "，".join(f"{SHORT[m]} {n(S['choice_split'][m]['question_loss'])}" for m in ms)
    nm = "，".join(f"{SHORT[m]} {n(S['choice_split'][m]['share'].get('near_miss', 0), 'pct')}" for m in ms)
    return (f"三个世界上从小到大：{cells}。提问花掉的分：{q}；选到近似项（多半是没记住或没察觉变化）的比例：{nm}。"
            "具体看第 6 节。")


def _mismatch_rows(S):
    cfg = S.get("config") or {}
    eff = cfg.get("effort", "high")
    return [
        ["温度与 seed", "请求里带 temperature = 0（DeepSeek 开思考时这个参数不生效，接口不报错）；DeepSeek 不收 seed",
         "不能设温度 0：Opus 5.5、Fable 5.1 已去掉 temperature / top_p / top_k，Sonnet 5.5、Haiku 5.5 只收默认值，"
         "而且 2026-10-21 起所有模型带这几个参数都返回 400；只能用接口默认采样，也没有 seed。"
         f"缓存键另带\"种子|用户|会话\"（{code_link('roommate/llm.py', 'cache_tag（如', 1, 'llm.LLMClient.chat')}），不同用户碰巧相同的请求各自独立采样",
         "Claude 每次会话是一次随机采样，同一个用户重跑会有出入，区间里含这部分波动；DeepSeek 开思考时实际上也不是温度 0，两边都不是确定性解码。复现都只靠缓存"],
        ["思考", "V4 默认开思考，工具调用循环里把 reasoning_content 原样带回",
         f"四个模型都开自适应思考（{code_link('roommate/agent_claude.py', '\"thinking\": {\"type\": \"adaptive\"', 1, 'agent_claude')}），"
         f"努力程度统一显式设成 {eff}（Opus 5.5、Haiku 5.5 默认是 medium，Sonnet 5.5、Fable 5.1 默认是 high）；Opus 5.5、Fable 5.1 本来就关不掉思考。思考块（含签名）原样带回",
         "两边都是\"开思考\"，但思考深度的旋钮不同，不能说是同等的思考量。四个 Claude 用同一个努力程度，档位之间可比；换成默认努力程度，Opus 5.5 和 Haiku 5.5 会想得更少、更便宜"],
        ["思考内容可见", "reasoning_content 是原文", "只返回摘要（display = summarized），原始思考不对外",
         "不影响模型行为和花费（思考照常计费）；对话记录里能看到的推理只有摘要，分析失分时证据少一些"],
        ["请求格式", "OpenAI 兼容的 chat/completions：tool 角色的消息、function 工具",
         "Messages API：工具结果放在同一条 user 消息的 tool_result 块里，工具定义是 input_schema；内容逐字相同",
         "模型看到的信息相同；两家各自在服务端把工具定义转成自己的提示，这部分没法对齐"],
        ["最大输出", "没设 max_tokens（用接口默认）", f"max_tokens = {cfg.get('max_tokens', 32000)}，流式取最终消息",
         "截断的轮数见第 7 节；没有截断就没有影响"],
        ["提示缓存", "DeepSeek 自动缓存", "开顶层自动缓存（cache_control，5 分钟）", "只影响花费，不影响模型看到的内容"],
        ["拒答", "没有这一类", "安全分类器可能拒答（stop_reason = refusal）；不开换模型重试（会把别的模型的回答混进来），拒答这次会话按没作答计",
         "拒答次数见第 7 节"],
        ["对照组", "V4 Pro 种子 1", "Opus 5.5 种子 1", "失分拆解只能在种子 1 的 168 次偏好会话上比，两边同口径"],
        ["Fable 5.1 的覆盖", "—", "按预算优先级至少跑种子 1（见第 3 节）", "只跑了种子 1 时，和别的模型比要看第 4 节种子 1 的表，不能和三个世界的数直接比"],
        ["tokenizer", "DeepSeek 自己的", "Claude 自己的", "token 数不能跨厂商比，花费可以比"],
    ]


def _limitations(S):
    out = ["每个模型只在同一个考场上跑一遍；Claude 不能设温度 0，同一个用户重跑的波动没有单独量（M2 起点体检里的重复采样方法可以照搬，这次没做）。",
           "失分拆解里的\"没记住或没察觉变化 / 挑店或格式问题\"只有带对照组的模型有（"
           + "、".join(_name(m) for m, v in S["loss_breakdown"].items() if v.get("memory_or_change_loss") is not None)
           + "），其他模型只有按选错选项种类的粗分。",
           "试跑只用了 2 个用户，估算有偏差的余地；花费以账本里的实际累计为准。"]
    fc = S.get("fable_full_seeds") or []
    if fc != SEEDS:
        out.append(f"Fable 5.1 跑完的世界：{('种子 ' + '、'.join(map(str, fc))) if fc else '没有完整的世界'}，没有三个世界的区间。")
    return out
