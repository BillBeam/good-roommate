"""判分：完全由代码对照用户当时的真实状态计算，不用任何评判模型。

判分规则和第一版相同：选对得 1 分，每问一次扣 question_cost 分（默认 0.15，世界配置项）。
"""

from .schema import DIET_TAG, EARLY_CUTOFF, LATE_FROM, PRICE_BANDS, area_attr

QUESTION_COST = 0.15  # 默认值；实际值以世界配置为准


def band_of(price, domain):
    for band, (lo, hi) in PRICE_BANDS[domain].items():
        if lo <= price <= hi:
            return band
    raise ValueError((price, domain))


def violations(opt, state, domain):
    """选项 opt 违反了状态 state 的哪些属性（列表为空表示满足）。"""
    v = []
    if state["allergy"] != "无" and state["allergy"] in opt["allergens"]:
        v.append("allergy")
    if state["diet"] != "普通" and DIET_TAG[state["diet"]] not in opt["diet_tags"]:
        v.append("diet")
    if state["cuisine"] != opt["cuisine"]:
        v.append("cuisine")
    want = "偏辣" if state["spicy"] == "爱吃辣" else "清淡"
    if opt["spice"] != want:
        v.append("spicy")
    if band_of(opt["price"], domain) != state["budget"]:
        v.append("budget")
    aa = area_attr(domain)
    if opt["area"] != state[aa]:
        v.append(aa)
    if domain == "dinner":
        if state["commute"] == "开车" and not opt["parking"]:
            v.append("commute")
        if state["kid"] == "有婴儿" and not opt["kid_ok"]:
            v.append("kid")
        if state["pet"] == "有狗" and not opt["pet_ok"]:
            v.append("pet")
        if state["evening"] == "要早回" and opt["slot"] > EARLY_CUTOFF:
            v.append("evening")
        if state["evening"] == "下班晚" and opt["slot"] < LATE_FROM:
            v.append("evening")
    return v


def satisfies(opt, state, domain):
    return not violations(opt, state, domain)


def grade_session(session, chosen_id, n_questions, question_cost=QUESTION_COST):
    """返回本次会话的判分记录。chosen_id 为 None 表示没有选择。"""
    correct = chosen_id is not None and chosen_id == session["correct"]
    role = session["roles"].get(chosen_id) if chosen_id else None
    return {
        "success": bool(correct),
        "n_questions": n_questions,
        "score": (1.0 if correct else 0.0) - question_cost * n_questions,
        "near_miss": session.get("near_miss"),
        "near_miss_error": role == "near_miss",
    }


def _rate(num, den):
    return None if den == 0 else num / den


def summarize(records):
    """records：每次会话一条。分数只算和偏好有关的会话；无关会话单独报。"""
    pers = [r for r in records if r["domain"] != "util"]
    util = [r for r in records if r["domain"] == "util"]
    follow = [r for r in pers if r["idx"] >= 1]
    qs = [q for r in pers for q in r["questions"]]
    by_type = {}
    for t in ("P", "E", "T"):
        rs = [r for r in pers if r["type"] == t]
        by_type[t] = {"n": len(rs), "success_rate": _rate(sum(r["grade"]["success"] for r in rs), len(rs)),
                      "mean_score": _rate(sum(r["grade"]["score"] for r in rs), len(rs))}
    by_nm = {}
    for r in pers:
        k = r["grade"]["near_miss"] or "none"
        e = by_nm.setdefault(k, {"n": 0, "success": 0, "near_miss_error": 0})
        e["n"] += 1
        e["success"] += r["grade"]["success"]
        e["near_miss_error"] += r["grade"]["near_miss_error"]
    n = len(pers)
    return {
        "n_sessions": n,
        "mean_score": sum(r["grade"]["score"] for r in pers) / n,
        "success_rate": _rate(sum(r["grade"]["success"] for r in pers), n),
        "followup_success_rate": _rate(sum(r["grade"]["success"] for r in follow), len(follow)),
        "n_followup": len(follow),
        "avg_questions": len(qs) / n,
        "question_loss": sum(r["grade"]["success"] - r["grade"]["score"] for r in pers) / n,
        "n_questions": len(qs),
        "told_before_rate": _rate(sum(q["repeated"] for q in qs), len(qs)),
        "near_miss_error_rate": _rate(sum(r["grade"]["near_miss_error"] for r in pers if r["grade"]["near_miss"]),
                                      sum(1 for r in pers if r["grade"]["near_miss"])),
        "by_near_miss": by_nm,
        "no_choice": sum(r["chosen"] is None for r in pers),
        "by_type": by_type,
        "util": {"n": len(util), "success_rate": _rate(sum(r["grade"]["success"] for r in util), len(util)),
                 "mean_score": _rate(sum(r["grade"]["score"] for r in util), len(util)),
                 "avg_questions": _rate(sum(len(r["questions"]) for r in util), len(util))},
    }
