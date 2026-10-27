"""短程考场：每条轨迹只含同一个用户连续的 2 次见面（第 k、k+1 次会话）。

由新生成的完整世界（8 次见面）切出来，不改原有的世界和考场规则：
- 起点状态 = 这个用户在第 k 次会话之前的长期真实状态（把前面各次的事件依次作用上去）；
- agent 和用户都是"第一次打交道"：agent 记忆为空，用户也不记得告诉过 agent 什么（told 为空），
  所以第 2 次会话里能用到的、不用再问的信息，只能来自第 1 次会话里 agent 自己记下的东西；
- 两次会话的任务、选项、事件、临时情况都原样沿用，判分规则不变。
派生出来的世界照常交给 roommate.env.Env 跑。
"""

import copy

from ..grading import violations
from ..schema import DOMAINS


def state_before(plan, k):
    s = dict(plan["init_state"])
    spicy = plan.get("init_spicy_reason", "taste")
    from .. import texts
    for sess in plan["sessions"][:k]:
        ev = sess.get("event")
        if ev:
            for a, (old, new) in ev["changes"].items():
                assert s[a] == old
                s[a] = new
            if ev["type"] in texts.HEALTH_SPICY_EVENTS:
                spicy = ev["type"]
    return s, spicy


def window(plan, k):
    s, spicy = state_before(plan, k)
    sessions = []
    for j, sess in enumerate(plan["sessions"][k:k + 2]):
        x = copy.deepcopy(sess)
        x["orig_idx"] = sess["idx"]
        x["idx"] = j
        sessions.append(x)
    return {"uid": f"{plan['uid']}w{k}", "orig_uid": plan["uid"], "start": k, "init_state": s,
            "init_spicy_reason": spicy, "sessions": sessions}


def short_world(world, starts=None):
    """starts：每个用户取哪些起点 k（默认 0..n-2 全部）。"""
    starts = list(range(world["n_sessions"] - 1)) if starts is None else starts
    users = [window(u, k) for u in world["users"] for k in starts]
    return {"version": world["version"], "seed": world["seed"], "n_users": len(users), "n_sessions": 2,
            "events": world["events"], "config": world["config"], "users": users, "derived_from": "short-range",
            "starts": starts}


def decision_attrs(sess):
    """这次任务里真正起作用的属性：有选项因为它而不满足用户当时的真实状态。"""
    if sess["domain"] == "util":
        return set()
    out = set()
    for o in sess["options"]:
        out |= set(violations(o, sess["eff_state"], sess["domain"]))
    return out & set(DOMAINS[sess["domain"]])


def needs_first(w):
    """第 2 次会话要用到第 1 次会话能得到的信息吗？
    条件：第 2 次是和偏好有关的任务；有某项属性 a 在第 2 次里起作用，a 在第 1 次里能得到
    （第 1 次是偏好任务且 a 属于它的领域、可以问；或第 1 次的事件台词提到了 a），
    两次之间 a 的长期值没变，而且第 2 次自己的台词（事件、临时情况）没有提到 a。"""
    s1, s2 = w["sessions"]
    if s2["domain"] == "util":
        return False, []
    avail = set(DOMAINS[s1["domain"]]) if s1["domain"] != "util" else set()
    if s1.get("event"):
        avail |= set(s1["event"]["cues"])
    stated2 = set(s2["event"]["cues"]) if s2.get("event") else set()
    if s2.get("temp"):
        stated2.add(s2["temp"]["attr"])
    changed = set(s2["event"]["changes"]) if s2.get("event") else set()
    attrs = sorted((decision_attrs(s2) & avail) - stated2 - changed)
    return bool(attrs), attrs


def needs_first_rate(sw):
    flags = [needs_first(u)[0] for u in sw["users"]]
    return {"n": len(flags), "needs_first": sum(flags), "rate": sum(flags) / len(flags) if flags else None}
