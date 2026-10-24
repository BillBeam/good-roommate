"""判分：完全由代码对照用户当时的真实状态计算，不用任何评判模型。"""

from .schema import DIET_TAG, PRICE_BANDS

QUESTION_COST = 0.15  # 每次提问扣的分


def band_of(price, domain):
    for band, (lo, hi) in PRICE_BANDS[domain].items():
        if lo <= price <= hi:
            return band
    raise ValueError((price, domain))


def violations(opt, state, domain):
    """选项 opt 违反了状态 state 的哪些属性（列表为空表示满足）。"""
    v = []
    if domain in ("dinner", "lunch"):
        if state["allergy"] != "无" and state["allergy"] in opt["allergens"]:
            v.append("allergy")
        if state["cuisine"] != opt["cuisine"]:
            v.append("cuisine")
        want = "偏辣" if state["spicy"] == "爱吃辣" else "清淡"
        if opt["spice"] != want:
            v.append("spicy")
        area_attr = "home" if domain == "dinner" else "office"
        if opt["area"] != state[area_attr]:
            v.append(area_attr)
    if state["diet"] != "普通" and DIET_TAG[state["diet"]] not in opt["diet_tags"]:
        v.append("diet")
    if band_of(opt["price"], domain) != state["budget"]:
        v.append("budget")
    if domain in ("dinner", "hotel"):
        if state["commute"] == "开车" and not opt["parking"]:
            v.append("commute")
        if state["kid"] == "有婴儿" and not opt["kid_ok"]:
            v.append("kid")
    if domain == "dinner" and state["evening"] == "要早回" and opt["slot"] > "18:30":
        v.append("evening")
    if domain == "hotel" and state["pet"] == "有狗" and not opt["pet_ok"]:
        v.append("pet")
    return v


def satisfies(opt, state, domain):
    return not violations(opt, state, domain)


def grade_session(session, chosen_id, n_questions):
    """返回本次会话的判分记录。chosen_id 为 None 表示没有选择。"""
    correct = chosen_id is not None and chosen_id == session["correct"]
    rec = {
        "success": bool(correct),
        "n_questions": n_questions,
        "score": (1.0 if correct else 0.0) - QUESTION_COST * n_questions,
        "stale_trap": session.get("stale_state") is not None,
        "temp_trap": session.get("temp_trap_state") is not None,
        "stale_error": False,
        "temp_error": False,
    }
    if session["domain"] != "util" and chosen_id is not None and not correct:
        opt = next(o for o in session["options"] if o["id"] == chosen_id)
        if rec["stale_trap"]:
            rec["stale_error"] = satisfies(opt, session["stale_state"], session["domain"])
        if rec["temp_trap"]:
            rec["temp_error"] = satisfies(opt, session["temp_trap_state"], session["domain"])
    return rec


def _rate(num, den):
    return None if den == 0 else num / den


def summarize(records):
    """records：每次会话一条（含 grade 字段和提问明细）。"""
    n = len(records)
    pers = [r for r in records if r["domain"] != "util"]
    follow = [r for r in pers if r["idx"] >= 1]
    util = [r for r in records if r["domain"] == "util"]
    qs = [q for r in records for q in r["questions"]]
    stale = [r for r in records if r["grade"]["stale_trap"]]
    temp = [r for r in records if r["grade"]["temp_trap"]]
    by_type = {}
    for t in ("P", "E", "T", "U"):
        rs = [r for r in records if r["type"] == t]
        by_type[t] = {"n": len(rs), "success_rate": _rate(sum(r["grade"]["success"] for r in rs), len(rs))}
    return {
        "n_sessions": n,
        "mean_score": sum(r["grade"]["score"] for r in records) / n,
        "success_rate": _rate(sum(r["grade"]["success"] for r in records), n),
        "followup_success_rate": _rate(sum(r["grade"]["success"] for r in follow), len(follow)),
        "n_followup": len(follow),
        "avg_questions": len(qs) / n,
        "avg_questions_util": _rate(sum(len(r["questions"]) for r in util), len(util)),
        "repeat_question_rate": _rate(sum(q["repeated"] for q in qs), len(qs)),
        "n_questions": len(qs),
        "stale_error_rate": _rate(sum(r["grade"]["stale_error"] for r in stale), len(stale)),
        "n_stale_trap": len(stale),
        "temp_error_rate": _rate(sum(r["grade"]["temp_error"] for r in temp), len(temp)),
        "n_temp_trap": len(temp),
        "no_choice": sum(r["chosen"] is None for r in records),
        "by_type": by_type,
    }
