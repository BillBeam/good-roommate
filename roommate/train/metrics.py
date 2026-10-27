"""评测与训练曲线共用的指标，以及按用户重抽样的置信区间。

每次会话的记录来自 roommate.env（判分、提问）加本地 agent 的 notes（思考长度、格式问题等）。
分数、选对率、提问数只算和偏好有关的会话（与 M1、M2 的口径相同）。

两个捷径指标：
① 信息不够就选（没问够，会话级）：会话开始时用户已经提过的偏好（回答过的，或事件台词里明说、可推断的）加上本次台词里的临时情况，
   还不足以从 6 个选项里定出正确的那个——也就是这次的选择要用到用户从没提过的偏好；在这样的会话里，
   agent 选择时手里的信息（已提过的 + 本次问到的）仍然定不出唯一的选项，算一次"信息不够就选"
   （问了几次都可能算：问到的不够也算；一次没问是它的特例）。
   比例 = 信息不够就选的会话 / 要用到没提过的偏好的会话（只算做了选择的会话）。
② 又去问（属性级）：会话开始时用户已经告诉过、而且长期值没变的属性（属于本次任务的领域、不是本次的临时情况），
   agent 这次又问了它。比例 = 又问了的（会话, 属性）对 / 这样的对。
"会话开始时用户告诉过什么"取自模拟用户（本地 agent 记在 notes 里；API 基线的记录用 replay_told 按提问记录重放得到）。
"""

import random

from ..grading import violations
from ..schema import DOMAINS
from ..usersim import SimUser

FORMAT_KINDS = ["bad_json", "unknown_tool", "truncated", "think_runaway", "tool_text", "no_tool_call",
                "context_overflow"]


def replay_told(plan, records, repeat_mode="refuse_told"):
    """按记录里的提问重放模拟用户，得到每次会话开始时的 told 和长期状态（API 基线用）。"""
    u = SimUser(plan, repeat_mode)
    out = {}
    recs = {r["idx"]: r for r in records}
    for sess in plan["sessions"]:
        r = recs.get(sess["idx"])
        if r is None:
            break
        u.begin_session(sess)
        out[sess["idx"]] = {"told_at_start": dict(u.told), "state_long": dict(u.state),
                            "temp_attr": sess["temp"]["attr"] if sess.get("temp") else None}
        for q in r["questions"]:
            if q.get("declared"):
                ans = u.answer(q["declared"])
                assert ans["text"] == q["answer"], (plan["uid"], sess["idx"], q)
        u.end_session()
    return out


def consistent_options(sess, known):
    """只看 known 这些属性（按用户当时的真实取值），和用户一致的选项。"""
    k = set(known) & set(DOMAINS[sess["domain"]])
    return [o["id"] for o in sess["options"] if not set(violations(o, sess["eff_state"], sess["domain"])) & k]


def session_row(rec, sess, info=None):
    """一次会话的各项计数（分子、分母分开，便于按用户合并）。"""
    info = info or rec
    pers = rec["domain"] != "util"
    asked = {a for q in rec["questions"] for a in (q.get("attributes") or [])}
    told = info.get("told_at_start")
    row = {"pers": int(pers), "score": rec["grade"]["score"] if pers else 0.0,
           "success": int(rec["grade"]["success"]) if pers else 0, "questions": len(rec["questions"]) if pers else 0,
           "no_choice": int(pers and rec["chosen"] is None)}
    ts = rec.get("turn_stats") or []
    row["turns"] = len(ts)
    row["think_tokens"] = sum(t["think_tokens"] for t in ts)
    row["think_hit"] = sum(int(t["think_hit"]) for t in ts)
    issues = rec.get("format_issues") or []
    row["turns_with_issue"] = len({i["turn"] for i in issues if i["kind"] in FORMAT_KINDS})
    for k in FORMAT_KINDS:
        row[f"fmt_{k}"] = sum(1 for i in issues if i["kind"] == k)
    row["s1_num"] = row["s1_den"] = row["s2_num"] = row["s2_den"] = 0
    if pers and told is not None:
        temp = info.get("temp_attr")
        if rec["chosen"] is not None:
            k0 = set(told) | ({temp} if temp else set())
            if len(consistent_options(sess, k0)) > 1:          # 要用到没提过的偏好
                row["s1_den"] = 1
                row["s1_num"] = int(len(consistent_options(sess, k0 | asked)) > 1)
        K = {a for a in DOMAINS[rec["domain"]] if a in told and told[a] == info["state_long"][a] and a != temp}
        row["s2_den"] = len(K)
        row["s2_num"] = len(K & asked)
    return row


RATIOS = {   # 指标名 -> (分子, 分母)
    "mean_score": ("score", "pers"), "success_rate": ("success", "pers"), "avg_questions": ("questions", "pers"),
    "no_choice_rate": ("no_choice", "pers"), "think_tokens_per_turn": ("think_tokens", "turns"),
    "think_over_limit_rate": ("think_hit", "turns"), "format_error_rate": ("turns_with_issue", "turns"),
    "shortcut1_guess_unmentioned": ("s1_num", "s1_den"), "shortcut2_reask_told": ("s2_num", "s2_den"),
}


def per_user(rows):
    """rows：[(组, 用户, 行)] -> {组: {用户: 合计}}；组是世界（种子），用户在组内重抽样。"""
    out = {}
    for g, u, r in rows:
        acc = out.setdefault(g, {}).setdefault(u, {})
        for k, v in r.items():
            acc[k] = acc.get(k, 0) + v
    return out


def ratios(tot):
    return {m: (tot.get(a, 0) / tot[b] if tot.get(b) else None) for m, (a, b) in RATIOS.items()}


def summarize_rows(rows, n_boot=2000, seed=0):
    pu = per_user(rows)
    total = {}
    for g in pu.values():
        for acc in g.values():
            for k, v in acc.items():
                total[k] = total.get(k, 0) + v
    point = ratios(total)
    rng = random.Random(seed)
    groups = {g: sorted(us) for g, us in pu.items()}
    boots = {m: [] for m in RATIOS}
    for _ in range(n_boot):
        tot = {}
        for g, us in groups.items():
            for _ in us:
                acc = pu[g][rng.choice(us)]
                for k, v in acc.items():
                    tot[k] = tot.get(k, 0) + v
        for m, v in ratios(tot).items():
            if v is not None:
                boots[m].append(v)
    ci = {}
    for m, xs in boots.items():
        xs.sort()
        ci[m] = [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]] if len(xs) > 20 else None
    fmt = {k[4:]: total.get(k, 0) for k in total if k.startswith("fmt_")}
    return {"n_sessions_pref": total.get("pers", 0), "n_turns": total.get("turns", 0),
            "n_users": sum(len(v) for v in pu.values()), "point": point, "ci95": ci, "format_counts": fmt,
            "n_boot": n_boot}
