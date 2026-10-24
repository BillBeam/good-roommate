"""按随机种子生成模拟用户的"一生"（第二版）。

会话类型（比例与第一版相同）：
- P：普通的个性化任务
- E：用户在提出任务时顺口提到一个生活事件（改动若干偏好，本次任务就按新状态判）
- T：用户提到一个临时情况（只影响本次任务，长期偏好不变）
- U：与偏好无关的任务（换算、算账等）

生成分两遍：先模拟全部用户的状态变化，再为每次会话造候选选项。

选项的造法（第二版）：6 个选项分成 3 对，每对是两个只在少数属性上不同的"近似状态"各造出的一个选项。
- 用户这一对：一个满足用户当时的真实状态（正确答案），另一个是"差一点"的近似项：
  事件之前的旧状态（stale）、忽略本次临时情况（temp_ignored）、把上次的临时情况当成长期（temp_trap）、
  或者一个没发生的假想变化（hyp_event / hyp_temp）。
- 另外两对来自同一世界里其他用户当时的状态，各自再加一个事件或临时情况式的变化。
每个选项的特征都只由它背后的那个状态按同一套随机规则生成，所以 6 个选项在不看用户时彼此对称：
不看用户的规则选不出正确答案，选项也不会透露用户的状态。正确答案在对里是"变化之后"的那一个，
还是"变化之前"的那一个，在全世界范围内各占一半（生成时配平），因此"挑看起来刚变过的那个"也没用。
"""

import datetime as dt
import hashlib
import json
import random

from . import texts
from .grading import band_of, satisfies
from .texts import other
from .schema import (ALLERGENS, AREAS, ATTRS, CUISINES, DIET_TAG, DIET_TAGS, DOMAINS, EARLY_SLOTS, LATE_SLOTS,
                     PRICE_BANDS, TWO_SIDED, area_attr)

VERSION = 2
N_OPTIONS = 6
BASE_DATE = dt.date(2026, 1, 5)  # 第 1 周周一
WEEKDAYS = "一二三四五六日"
LETTERS = "ABCDEF"
ALL_EVENTS = list(texts.EVENTS)
DEFAULT_CONFIG = {"repeat_mode": "refuse_told", "question_cost": 0.15}
POST_KINDS = {"stale", "temp_ignored", "hyp_prior_event", "hyp_prior_temp"}  # 正确答案是"变化之后"的那个
PRE_KINDS = {"temp_trap", "hyp_event", "hyp_temp"}                          # 正确答案是"变化之前"的那个
EVENT_ATTRS = {"fitness": ("diet", "cuisine"), "move": ("home", "commute"), "baby": ("kid", "evening"),
               "dog": ("pet", "evening"), "promotion": ("budget", "office"), "newjob": ("budget", "office"),
               "pregnancy": ("spicy", "cuisine"), "checkup": ("diet", "spicy")}


def sample_initial_state(rng):
    """随机数的调用顺序与第一版相同（同一种子下初始状态的大部分属性与第一版一致）。"""
    s = {}
    s["allergy"] = rng.choices(ATTRS["allergy"], [0.4, 0.2, 0.2, 0.2])[0]
    s["diet"] = rng.choices(ATTRS["diet"], [0.65, 0.15, 0.1, 0.1])[0]
    s["cuisine"] = rng.choices(CUISINES, [1, 1, 1, 1, 1, 0.3, 1])[0]
    hot = s["cuisine"] in ("川菜", "火锅")
    s["spicy"] = "爱吃辣" if rng.random() < (1.0 if hot else 0.45) else "不吃辣"
    s["budget"] = rng.choice(ATTRS["budget"])
    s["home"] = rng.choice(AREAS)
    s["office"] = rng.choice(AREAS)
    s["commute"] = rng.choices(ATTRS["commute"], [0.45, 0.25, 0.3])[0]
    s["kid"] = "有婴儿" if rng.random() < 0.1 else "无"
    s["pet"] = "有狗" if rng.random() < 0.1 else "无"
    early = s["kid"] != "无" or s["pet"] != "无" or rng.random() < 0.5
    s["evening"] = "要早回" if early else "下班晚"
    return s


# ---------- 变化（事件、临时情况）作用在状态上 ----------

def _two_sided_in(changes, domain):
    return any(a in TWO_SIDED for a in changes if a in DOMAINS[domain])


# ---------- 由状态生成选项特征 ----------

def _price_in(rng, domain, band):
    lo, hi = PRICE_BANDS[domain][band]
    step = 1 if domain == "lunch" else 5
    return lo + step * rng.randint(0, (hi - lo) // step)


def option_from_state(rng, L, domain, k_allergens, m_tags):
    """按状态 L 生成一个满足 L 的选项。除了 L 要求的特征，其余特征随机；规则对所有选项相同。"""
    o = {"cuisine": L["cuisine"], "area": L[area_attr(domain)],
         "spice": "偏辣" if L["spicy"] == "爱吃辣" else "清淡",
         "price": _price_in(rng, domain, L["budget"])}
    pool = [a for a in ALLERGENS if a != L["allergy"]]
    o["allergens"] = sorted(rng.sample(pool, min(k_allergens, len(pool))))
    need = DIET_TAG.get(L["diet"])
    rest = [t for t in DIET_TAGS if t != need]
    tags = ([need] if need else []) + rng.sample(rest, m_tags - (1 if need else 0))
    o["diet_tags"] = sorted(tags)
    if domain == "dinner":
        o["slot"] = rng.choice(EARLY_SLOTS if L["evening"] == "要早回" else LATE_SLOTS)
        o["parking"] = L["commute"] == "开车" or rng.random() < 0.5
        o["kid_ok"] = L["kid"] == "有婴儿" or rng.random() < 0.5
        o["pet_ok"] = L["pet"] == "有狗" or rng.random() < 0.5
    assert satisfies(o, L, domain)
    return o


def _signature(o, domain):
    d = {k: v for k, v in o.items() if k not in ("id", "name", "price")}
    return json.dumps(d, sort_keys=True, ensure_ascii=False) + band_of(o["price"], domain)


def _names(rng, opts):
    used, names = set(), []
    for o in opts:
        pool = [n for n in texts.DINING_NAMES[o["cuisine"]] if n not in used]
        n = rng.choice(pool)
        used.add(n)
        names.append(n)
    return names


def build_options(rng, domain, cur, near, decoy_pairs):
    """cur：用户当时的真实状态；near：近似项的状态；decoy_pairs(rng)：造两对干扰状态。
    返回 (选项列表, 正确选项字母, 每个选项的角色, 每个选项背后的状态)。"""
    k_all = rng.choice([1, 2])
    m_tags = rng.choice([1, 2])
    for _ in range(200):
        latents = [("correct", cur), ("near_miss", near)]
        for pair in decoy_pairs(rng):
            rng.shuffle(pair)
            latents += [("decoy", pair[0]), ("decoy", pair[1])]
        opts = [(role, option_from_state(rng, L, domain, k_all, m_tags), L) for role, L in latents]
        sat = [satisfies(o, cur, domain) for _, o, _ in opts]
        sigs = {_signature(o, domain) for _, o, _ in opts}
        if sat == [True] + [False] * (N_OPTIONS - 1) and len(sigs) == N_OPTIONS:
            break
    else:
        raise RuntimeError("生成选项失败")
    rng.shuffle(opts)
    names = _names(rng, [o for _, o, _ in opts])
    out, roles, states = [], {}, {}
    for i, (role, o, L) in enumerate(opts):
        o = dict(o, id=LETTERS[i], name=names[i])
        out.append(o)
        roles[o["id"]] = role
        states[o["id"]] = {a: L[a] for a in DOMAINS[domain]}
    correct = [o["id"] for o in out if satisfies(o, cur, domain)]
    assert len(correct) == 1, correct
    return out, correct[0], roles, states


# ---------- 与偏好无关的任务 ----------

def build_util_task(rng, date):
    kind = rng.choice(["aa", "km", "date", "end"])
    if kind == "aa":
        n = rng.randint(3, 6)
        per = rng.randint(45, 160)
        q = f"我们 {n} 个人吃饭一共花了 {n * per} 元，AA 的话每人多少钱？"
        ans = str(per)
        cands = {str(per + d) for d in (-12, -7, -3, 4, 9, 15)}
        fmt = lambda x: f"{x} 元"
    elif kind == "km":
        x = rng.choice([3, 5, 8, 10, 12, 21, 42])
        q = f"跑 {x} 公里大概是多少英里？"
        ans = f"{x * 0.6214:.1f}"
        cands = {f"{x * f:.1f}" for f in (0.5, 0.55, 0.7, 0.8, 1.6, 0.45)}
        fmt = lambda x: f"{x} 英里"
    elif kind == "date":
        n = rng.choice([10, 17, 25, 40, 45, 60])
        d = date + dt.timedelta(days=n)
        q = f"从今天算起，{n} 天后是几月几号？"
        ans = f"{d.month} 月 {d.day} 日"
        cands = {f"{(d + dt.timedelta(days=k)).month} 月 {(d + dt.timedelta(days=k)).day} 日" for k in (-3, -2, -1, 1, 2, 3)}
        fmt = lambda x: x
    else:
        h, m = rng.randint(9, 16), rng.choice([0, 10, 15, 20, 30, 40, 45])
        dur = rng.choice([45, 50, 75, 90, 105, 135])
        end = dt.datetime(2026, 1, 1, h, m) + dt.timedelta(minutes=dur)
        q = f"会议 {h}:{m:02d} 开始，开 {dur} 分钟，几点结束？"
        ans = end.strftime("%H:%M")
        cands = {(end + dt.timedelta(minutes=k)).strftime("%H:%M") for k in (-30, -15, -10, 10, 15, 30)}
        fmt = lambda x: x
    cands.discard(ans)
    vals = [ans] + rng.sample(sorted(cands), N_OPTIONS - 1)
    rng.shuffle(vals)
    options = [{"id": LETTERS[i], "name": fmt(v)} for i, v in enumerate(vals)]
    correct = LETTERS[vals.index(ans)]
    return q, options, correct


# ---------- 第一遍：用户的一生 ----------

def _pick_event(rng, state, exclude, events):
    cands = [e for e in events if e not in exclude and texts.EVENTS[e]["pre"](state)]
    return rng.choice(cands) if cands else None


def _pick_domain(rng, pending, avoid=None):
    """优先选能检验待测属性的领域。"""
    doms = [d for d in DOMAINS if d != avoid] or list(DOMAINS)
    if pending:
        score = {d: len(set(DOMAINS[d]) & pending) for d in doms}
        best = max(score.values())
        if best > 0:
            doms = [d for d in doms if score[d] == best]
    return rng.choice(doms)


def near_candidates(sess, domain):
    """某次会话在给定领域里可用的"有依据的"近似项：{种类: (变化类型, 要改回或改成的属性取值)}。
    sess 需要已有 type、temp、eff_state、hist。"""
    eff = sess["eff_state"]
    cand = {}
    ev, lt = sess["hist"]["last_event"], sess["hist"]["last_temp"]
    if sess["type"] == "T":
        t = sess["temp"]
        if t["attr"] in DOMAINS[domain]:
            cand["temp_ignored"] = (t["type"], {t["attr"]: t["long_value"]})
        return cand
    if lt and lt["attr"] in DOMAINS[domain] and eff[lt["attr"]] != lt["value"]:
        cand["temp_trap"] = (lt["type"], {lt["attr"]: lt["value"]})
    if ev and _two_sided_in(ev["changes"], domain) and all(eff[a] == new for a, (old, new) in ev["changes"].items()):
        cand["stale"] = (ev["type"], {a: old for a, (old, new) in ev["changes"].items()})
    return cand


def prior_state(rng, state, domain, kind, change):
    """假想的"变化之前"：找一个状态 P，使 state 看起来像是 P 经历了这个事件或临时情况之后的样子。
    找不到返回 None。"""
    if kind == "hyp_prior_temp":
        tv = texts.TEMPS[change]
        a = tv["attr"]
        if domain not in tv["domains"]:
            return None
        if change == "elsewhere":
            return dict(state, home=other(rng, AREAS, state["home"]))
        if state[a] != {"throat": "不吃辣", "expense": "高", "frugal": "低"}[change]:
            return None
        vals = [v for v in ATTRS[a] if tv["pre"](dict(state, **{a: v})) and v != state[a]]
        return dict(state, **{a: rng.choice(vals)}) if vals else None
    S = state
    if change == "fitness" and S["diet"] == "控碳水" and S["cuisine"] == "轻食":
        P = dict(S, diet="普通", cuisine=other(rng, CUISINES, "轻食"))
    elif change == "pregnancy" and S["spicy"] == "不吃辣" and S["cuisine"] == "粤菜":
        P = dict(S, spicy="爱吃辣", cuisine=other(rng, CUISINES, "粤菜"))
    elif change == "checkup" and S["diet"] == "低糖" and S["spicy"] == "不吃辣":
        P = dict(S, diet="普通", spicy="爱吃辣")
    elif change == "baby" and S["kid"] == "有婴儿" and S["evening"] == "要早回":
        P = dict(S, kid="无", evening="下班晚")
    elif change == "dog" and S["pet"] == "有狗" and S["evening"] == "要早回":
        P = dict(S, pet="无", evening="下班晚")
    elif change == "promotion" and S["budget"] == "高":
        P = dict(S, budget=rng.choice(["低", "中"]), office=other(rng, AREAS, S["office"]))
    elif change == "newjob" and S["budget"] == "低":
        P = dict(S, budget=rng.choice(["中", "高"]), office=other(rng, AREAS, S["office"]))
    elif change == "move" and S["commute"] == ("骑车" if S["home"] == S["office"] else
                                               "开车" if S["home"] == "余杭" else "地铁"):
        P = dict(S, home=other(rng, AREAS, S["home"]), commute=rng.choice(ATTRS["commute"]))
    else:
        return None
    if not _two_sided_in({a: 1 for a in EVENT_ATTRS[change] if P[a] != S[a]}, domain):
        return None
    return P


def hyp_options(state, domain, events):
    """假想变化的候选：[(种类, 变化类型)]。假想事件要求在本领域里改到至少一项两向属性。
    hyp_event / hyp_temp：近似项是"多了一个没发生的变化"（正确答案在变化之前）；
    hyp_prior_event / hyp_prior_temp：近似项是"一个说得通的更早状态"（正确答案看起来在变化之后）。"""
    out = []
    for e in events:
        if prior_state(random.Random(0), state, domain, "hyp_prior_event", e) is not None:
            out.append(("hyp_prior_event", e))
    for t in texts.TEMPS:
        if prior_state(random.Random(0), state, domain, "hyp_prior_temp", t) is not None:
            out.append(("hyp_prior_temp", t))
    for e in events:
        if texts.EVENTS[e]["pre"](state):
            probe = texts.event_effects(e, state, random.Random(0))
            if _two_sided_in(probe, domain):
                out.append(("hyp_event", e))
    for t, tv in texts.TEMPS.items():
        if tv["pre"](state) and domain in tv["domains"]:
            out.append(("hyp_temp", t))
    return out


def generate_life(rng, uid, n_sessions, events):
    init = sample_initial_state(rng)
    state = dict(init)
    types = ["P"] * n_sessions
    e1 = rng.choice([2, 3])
    types[e1] = "E"
    t1 = rng.choice([i for i in (4, 5) if i < n_sessions - 1 and types[i] == "P"])
    types[t1] = "T"
    if n_sessions >= 8 and rng.random() < 0.5 and types[n_sessions - 2] == "P":
        types[n_sessions - 2] = "E"
    free = [i for i in range(1, n_sessions) if types[i] == "P" and types[i - 1] not in ("E", "T")]
    types[rng.choice(free)] = "U"

    sessions, week, used_events, prev_domain = [], 1, set(), None
    last_event = None   # {"changes": {a: [old, new]}}：最近一次事件
    last_temp = None    # {"attr", "value"}：最近一次临时情况，下一次相关会话里检验
    pending = set()
    for idx, typ in enumerate(types):
        if idx > 0:
            week += rng.choice([1, 2, 2, 3])
        date = BASE_DATE + dt.timedelta(days=(week - 1) * 7 + rng.randint(0, 6))
        sess = {"idx": idx, "week": week, "date": date.isoformat(), "weekday": WEEKDAYS[date.weekday()],
                "type": typ, "event": None, "temp": None}
        mention = ""
        if typ == "E":
            etype = _pick_event(rng, state, used_events, events)
            if etype is None:
                typ = sess["type"] = "P"
            else:
                used_events.add(etype)
                changes = texts.event_effects(etype, state, rng)
                after = dict(state, **changes)
                mention, cues = texts.event_mention(etype, after, rng)
                assert set(changes) <= set(cues), (etype, changes, cues)
                sess["event"] = {"type": etype, "cn": texts.EVENTS[etype]["cn"], "mention": mention,
                                 "cues": cues, "changes": {a: [state[a], v] for a, v in changes.items()}}
                last_event = sess["event"]
                state = after
                pending |= set(changes)
        if typ == "U":
            q, options, correct = build_util_task(rng, date)
            sess.update(domain="util", request=q, user_text=q, options=options, correct=correct,
                        roles={o["id"]: ("correct" if o["id"] == correct else "distractor") for o in options},
                        long_state=dict(state), eff_state=None, near_miss=None)
            sessions.append(sess)
            continue
        eff = dict(state)
        if typ == "T":
            cands = [t for t, tv in texts.TEMPS.items() if tv["pre"](state)]
            ttype = rng.choice(cands)
            tv = texts.TEMPS[ttype]
            val = texts.temp_value(ttype, state, rng)
            domain = rng.choice(tv["domains"])
            eff[tv["attr"]] = val
            sess["temp"] = {"type": ttype, "cn": tv["cn"], "reason": tv["reason"], "attr": tv["attr"],
                            "value": val, "long_value": state[tv["attr"]]}
            mention = rng.choice(tv["mentions"]).format(temp_value=val)
        elif typ == "E":
            covering = [d for d in DOMAINS if _two_sided_in(sess["event"]["changes"], d)]
            domain = rng.choice(covering) if covering else _pick_domain(rng, set(sess["event"]["changes"]))
        else:
            want = set(pending) | ({last_temp["attr"]} if last_temp else set())
            domain = _pick_domain(rng, want, avoid=prev_domain if idx <= 1 else None)
            if idx == 0:
                domain = rng.choice(["dinner", "lunch"])
        # 记下本次会话之前的历史，第二遍用来为任意领域构造近似项
        sess["hist"] = {"last_event": last_event if typ != "E" else sess["event"],
                        "last_temp": last_temp if typ != "T" else None}
        sess["eff_state"] = eff
        cand = near_candidates(sess, domain)
        if typ == "T":
            last_temp = {"type": sess["temp"]["type"], "attr": sess["temp"]["attr"], "value": sess["temp"]["value"]}
        elif "temp_trap" in cand or (last_temp and last_temp["attr"] in DOMAINS[domain]):
            last_temp = None
        pending -= set(DOMAINS[domain])
        if domain == "dinner" and sess["temp"] and sess["temp"]["type"] == "elsewhere":
            req = rng.choice(texts.REQUESTS["dinner_elsewhere"])
        else:
            req = rng.choice(texts.REQUESTS[domain])
        sess.update(domain=domain, request=req, user_text=(mention + req) if mention else req,
                    long_state=dict(state), eff_state=eff, near_miss_candidates=cand)
        sessions.append(sess)
        prev_domain = domain
    return {"uid": uid, "init_state": init, "sessions": sessions}


# ---------- 第二遍：配平近似项并造选项 ----------

def _assign_near_misses(rng, users, events):
    """为每个任务选定近似项（种类与变化类型），使在每个（领域, 变化类型）里，
    "正确答案是变化之后的那个"和"是变化之前的那个"的次数尽量相等。"""
    cnt = {}  # (领域, 变化类型) -> [变化之后, 变化之前]

    def bump(s, kind, change):
        s["near_miss"], s["near_change"] = kind, change
        cnt.setdefault((s["domain"], change), [0, 0])[0 if kind in POST_KINDS else 1] += 1

    flexible = []
    for u in users:
        for s in u["sessions"]:
            if s["domain"] == "util":
                continue
            c = s["near_miss_candidates"]
            if s["type"] == "E" and "stale" in c:
                bump(s, "stale", c["stale"][0])
            elif s["type"] == "T":
                bump(s, "temp_ignored", c["temp_ignored"][0])
            elif "temp_trap" in c:
                bump(s, "temp_trap", c["temp_trap"][0])
            else:
                flexible.append(s)
    rng.shuffle(flexible)
    for s in flexible:
        c = s["near_miss_candidates"]
        opts = ([("stale", c["stale"][0])] if "stale" in c else []) + hyp_options(s["eff_state"], s["domain"], events)

        def deficit(o):
            post, pre = cnt.get((s["domain"], o[1]), [0, 0])
            return (pre - post) if o[0] in POST_KINDS else (post - pre)
        best = max(deficit(o) for o in opts)
        top = [o for o in opts if deficit(o) == best]
        if best > 0 and "stale" in c and deficit(("stale", c["stale"][0])) > 0:
            top = [("stale", c["stale"][0])]  # 需要"变化之后"时优先用真实发生过的变化（检验记忆）
        bump(s, *rng.choice(top))
    return {f"{d}|{c}": {"post": v[0], "pre": v[1]} for (d, c), v in sorted(cnt.items())}


def make_near(rng, sess, domain, kind, change):
    """在给定领域里为某次会话构造一种近似项的状态。"""
    cur = sess["eff_state"]
    if kind in ("stale", "temp_ignored", "temp_trap"):
        return dict(cur, **near_candidates(sess, domain)[kind][1])
    if kind == "hyp_event":
        return dict(cur, **texts.event_effects(change, cur, rng))
    if kind == "hyp_temp":
        return dict(cur, **{texts.TEMPS[change]["attr"]: texts.temp_value(change, cur, rng)})
    P = prior_state(rng, cur, domain, kind, change)
    return P if P is not None else prior_state(random.Random(0), cur, domain, kind, change)


def _decoy_pairs(rng, domain, same, pool, events, n=2):
    """n 个干扰对，各来自同一世界里不同的另一个用户的同一次会话：
    直接借用同领域的那一对（真实状态, 近似项），这样干扰对和用户这一对由完全相同的机制产生；
    同领域的不够时，按同样的规则为别的用户现造一对。"""
    out = []
    same = rng.sample(same, min(n, len(same)))
    for sj in same:
        out.append([dict(sj["eff_state"]), dict(sj["_near"])])
    rest = [x for x in pool if x not in same]
    need = n - len(out)
    # 用户很少时（只在测试里出现）允许重复借用同一个用户
    picks = rng.sample(rest, need) if len(rest) >= need else [rng.choice(pool) for _ in range(need)]
    for sj in picks:
        opts = [(k, v[0]) for k, v in near_candidates(sj, domain).items()] + hyp_options(sj["eff_state"], domain, events)
        kind, change = rng.choice(opts)
        out.append([dict(sj["eff_state"]), make_near(rng, sj, domain, kind, change)])
    return out


def generate_world(seed, n_users=24, n_sessions=8, events=None, config=None):
    """events：允许出现的事件类型（用于按事件类型划出留出集，见 event_split）；默认全部。"""
    assert n_sessions >= 6
    events = list(ALL_EVENTS if events is None else events)
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    rng = random.Random(seed)
    user_rngs = [random.Random(rng.random()) for _ in range(n_users)]
    users = [generate_life(user_rngs[i], f"u{i:02d}", n_sessions, events) for i in range(n_users)]
    balance = _assign_near_misses(random.Random(f"{seed}|balance"), users, events)
    # 先为每次任务定下近似项
    for u in users:
        for s in u["sessions"]:
            if s["domain"] == "util":
                continue
            nrng = random.Random(f"{seed}|near|{u['uid']}|{s['idx']}")
            near = make_near(nrng, s, s["domain"], s["near_miss"], s["near_change"])
            s["_near"] = near
            s["near_miss_state"] = {a: near[a] for a in DOMAINS[s["domain"]]}
            s["correct_is_post"] = s["near_miss"] in POST_KINDS
    # 再造选项
    for i, u in enumerate(users):
        for s in u["sessions"]:
            if s["domain"] == "util":
                continue
            orng = random.Random(f"{seed}|opt|{u['uid']}|{s['idx']}")
            others = [v["sessions"][s["idx"]] for j, v in enumerate(users) if j != i]
            pool = [x for x in others if x["domain"] != "util"]
            same = [x for x in pool if x["domain"] == s["domain"]]
            opts, correct, roles, states = build_options(
                orng, s["domain"], s["eff_state"], s["_near"],
                lambda r: _decoy_pairs(r, s["domain"], same, pool, events))
            s.update(options=opts, correct=correct, roles=roles, option_states=states)
    for u in users:
        for s in u["sessions"]:
            s.pop("_near", None)
            s.pop("near_miss_candidates", None)
    return {"version": VERSION, "seed": seed, "n_users": n_users, "n_sessions": n_sessions,
            "events": events, "config": cfg, "near_miss_balance": balance, "users": users}


def event_split(holdout):
    """按事件类型划分：返回 (训练用的事件类型, 留出的事件类型)。"""
    holdout = list(holdout)
    assert set(holdout) <= set(ALL_EVENTS), holdout
    return [e for e in ALL_EVENTS if e not in holdout], holdout


def world_hash(world):
    return hashlib.sha256(json.dumps(world, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def world_stats(world):
    ev_multi = ev_single = temps = util = 0
    by_event, by_temp, by_nm, by_dom = {}, {}, {}, {}
    for u in world["users"]:
        for s in u["sessions"]:
            if s["event"]:
                k = len(s["event"]["changes"])
                ev_multi += k >= 2
                ev_single += k == 1
                by_event[s["event"]["cn"]] = by_event.get(s["event"]["cn"], 0) + 1
            if s["temp"]:
                temps += 1
                by_temp[s["temp"]["cn"]] = by_temp.get(s["temp"]["cn"], 0) + 1
            util += s["domain"] == "util"
            by_dom[s["domain"]] = by_dom.get(s["domain"], 0) + 1
            if s["domain"] != "util":
                by_nm[s["near_miss"]] = by_nm.get(s["near_miss"], 0) + 1
    n_pref = sum(by_nm.values())
    return {
        "version": world.get("version"), "seed": world["seed"],
        "n_users": world["n_users"], "n_sessions_per_user": world["n_sessions"],
        "n_sessions_total": sum(len(u["sessions"]) for u in world["users"]),
        "n_preference_sessions": n_pref,
        "events_multi_attr": ev_multi, "events_single_attr": ev_single,
        "temp_situations": temps, "unrelated_sessions": util,
        "events_by_type": by_event, "temps_by_type": by_temp, "sessions_by_domain": by_dom,
        "near_miss_by_kind": by_nm,
        "correct_is_post_share": sum(v for k, v in by_nm.items() if k in POST_KINDS) / n_pref,
        "events_allowed": world["events"], "config": world["config"],
    }
