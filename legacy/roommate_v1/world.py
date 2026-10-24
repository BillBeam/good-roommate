"""按随机种子生成模拟用户的"一生"。

每个用户有初始真实状态，以及一份会话计划。会话类型：
- P：普通的个性化任务
- E：用户在提出任务时顺口提到一个生活事件（隐式改动若干偏好，本次任务就按新状态判）
- T：用户提到一个临时情况（只影响本次任务，长期偏好不变）
- U：与已存偏好无关的任务（换算、算账等，不需要个性化）

生成器会顺着时间模拟用户的状态，为每个任务造出恰好一个正确选项；
若之前有偏好变化或临时情况，再放入"过时记忆陷阱"或"临时情况陷阱"选项。
"""

import copy
import datetime as dt
import hashlib
import json
import random

from . import texts
from .grading import band_of, satisfies, violations
from .schema import ALLERGENS, AREAS, ATTRS, CUISINES, DIET_TAG, DOMAINS, EARLY_SLOTS, LATE_SLOTS, PRICE_BANDS

N_OPTIONS = 6
P_TRUE = 0.45  # 干扰项在每个属性上取"正确值"的概率；让正确值约占一半，选项本身不泄露答案
BASE_DATE = dt.date(2026, 1, 5)  # 第 1 周周一
WEEKDAYS = "一二三四五六日"
LETTERS = "ABCDEF"


def sample_initial_state(rng):
    s = {}
    s["allergy"] = rng.choices(ATTRS["allergy"], [0.4, 0.2, 0.2, 0.2])[0]
    s["diet"] = rng.choices(ATTRS["diet"], [0.65, 0.15, 0.1, 0.1])[0]
    s["cuisine"] = rng.choices(CUISINES, [1, 1, 1, 1, 1, 0.3, 1])[0]
    hot = s["cuisine"] in ("川菜", "火锅")
    s["spicy"] = "爱吃辣" if rng.random() < (0.85 if hot else 0.45) else "不吃辣"
    s["budget"] = rng.choice(ATTRS["budget"])
    s["home"] = rng.choice(AREAS)
    s["office"] = rng.choice(AREAS)
    s["commute"] = rng.choices(ATTRS["commute"], [0.45, 0.25, 0.3])[0]
    s["kid"] = "有婴儿" if rng.random() < 0.1 else "无"
    s["pet"] = "有狗" if rng.random() < 0.1 else "无"
    early = s["kid"] != "无" or s["pet"] != "无" or rng.random() < 0.1
    s["evening"] = "要早回" if early else "可以晚归"
    return s


# ---------- 事件与临时情况 ----------

def _other(rng, values, cur):
    return rng.choice([v for v in values if v != cur])


def event_effects(etype, s, rng):
    eff = texts.EVENTS[etype]["effects"]
    if etype == "move":
        new_home = _other(rng, AREAS, s["home"])
        if new_home == s["office"]:
            mode = "骑车"
        elif new_home == "余杭":
            mode = "开车"
        else:
            mode = "地铁"
        if mode == s["commute"]:
            mode = _other(rng, ATTRS["commute"], s["commute"])
        return {"home": new_home, "commute": mode}
    if etype == "promotion":
        return {"budget": "高", "office": _other(rng, AREAS, s["office"])}
    if etype == "newjob":
        return {"budget": "低", "office": _other(rng, AREAS, s["office"])}
    return eff(s, rng)


def event_mention(etype, changes, rng):
    ev = texts.EVENTS[etype]
    if etype == "move":
        tpl = rng.choice(ev["mentions"][changes["commute"]])
    else:
        tpl = rng.choice(ev["mentions"])
    return tpl.format(new_home=changes.get("home", ""), new_office=changes.get("office", ""))


def temp_value(ttype, s, rng):
    t = texts.TEMPS[ttype]
    if t["value"] == "_other_area":
        return _other(rng, AREAS, s["home"])
    return t["value"](s, rng)


# ---------- 选项生成 ----------

def _random_option(rng, domain):
    o = {"price": None, "diet_tags": [], "parking": False, "kid_ok": False}
    if domain in ("dinner", "lunch"):
        o["cuisine"] = rng.choice(CUISINES)
        o["area"] = rng.choice(AREAS)
        o["spice"] = rng.choice(["偏辣", "清淡"])
        o["allergens"] = sorted(a for a in ALLERGENS if rng.random() < 0.35)
    o["diet_tags"] = sorted(t for t in DIET_TAG.values() if rng.random() < 0.4)
    band = rng.choice(list(PRICE_BANDS[domain]))
    o["price"] = _price_in(rng, domain, band)
    if domain in ("dinner", "hotel"):
        o["parking"] = rng.random() < 0.5
        o["kid_ok"] = rng.random() < 0.5
    if domain == "dinner":
        o["slot"] = rng.choice(EARLY_SLOTS + LATE_SLOTS)
    if domain == "hotel":
        o["pet_ok"] = rng.random() < 0.5
    return o


def _price_in(rng, domain, band):
    lo, hi = PRICE_BANDS[domain][band]
    step = 1 if domain == "lunch" else 5 if domain == "dinner" else 10
    return rng.randrange(lo, hi + 1, step) if step == 1 else lo + step * rng.randint(0, (hi - lo) // step)


def _set_attr(o, attr, state, domain, rng, satisfy):
    """把选项 o 在属性 attr 上设成满足（satisfy=True）或违反 state 的样子。"""
    val = state[attr]
    if attr == "allergy":
        if val == "无":
            return satisfy
        s = set(o["allergens"])
        s.discard(val) if satisfy else s.add(val)
        o["allergens"] = sorted(s)
    elif attr == "diet":
        if val == "普通":
            return satisfy
        s = set(o["diet_tags"])
        s.add(DIET_TAG[val]) if satisfy else s.discard(DIET_TAG[val])
        o["diet_tags"] = sorted(s)
    elif attr == "cuisine":
        o["cuisine"] = val if satisfy else _other(rng, CUISINES, val)
    elif attr == "spicy":
        want = "偏辣" if val == "爱吃辣" else "清淡"
        o["spice"] = want if satisfy else ("清淡" if want == "偏辣" else "偏辣")
    elif attr == "budget":
        band = val if satisfy else _other(rng, list(PRICE_BANDS[domain]), val)
        o["price"] = _price_in(rng, domain, band)
    elif attr in ("home", "office"):
        o["area"] = val if satisfy else _other(rng, AREAS, val)
    elif attr == "commute":
        if val != "开车":
            return satisfy
        o["parking"] = satisfy
    elif attr == "kid":
        if val == "无":
            return satisfy
        o["kid_ok"] = satisfy
    elif attr == "evening":
        if val == "可以晚归":
            return satisfy
        o["slot"] = rng.choice(EARLY_SLOTS if satisfy else LATE_SLOTS)
    elif attr == "pet":
        if val == "无":
            return satisfy
        o["pet_ok"] = satisfy
    return True


def _option_satisfying(rng, domain, state):
    o = _random_option(rng, domain)
    for a in DOMAINS[domain]:
        _set_attr(o, a, state, domain, rng, True)
    assert satisfies(o, state, domain)
    return o


def _trap(rng, domain, cur, other, changed_attrs):
    """造一个满足 other（过时或被临时情况污染的状态）、但违反当前状态 cur 的选项。"""
    o = _option_satisfying(rng, domain, other)
    for a in changed_attrs:
        # 在 other 下仍满足、在 cur 下违反
        if a in ("cuisine", "spicy", "budget", "home", "office"):
            continue  # 两向属性：满足 other 就已经违反 cur
        trial = copy.deepcopy(o)
        _set_attr(trial, a, cur, domain, rng, False)
        if satisfies(trial, other, domain):
            o = trial
    if satisfies(o, cur, domain) or not satisfies(o, other, domain):
        return None
    return o


def _decoy_values(rng, attr, cur, extra_states):
    """两向属性的干扰取值：优先用过时值 / 临时值，再补一个随机值。"""
    vals = []
    for st in extra_states:
        if st is not None and st[attr] != cur[attr] and st[attr] not in vals:
            vals.append(st[attr])
    if not vals:
        pool = ATTRS[attr] if attr not in ("home", "office") else AREAS
        vals.append(_other(rng, pool, cur[attr]))
    return vals


def _distractor(rng, domain, cur, extras):
    o = _random_option(rng, domain)
    for a in DOMAINS[domain]:
        if a in ("cuisine", "budget", "home", "office"):
            if rng.random() < P_TRUE:
                _set_attr(o, a, cur, domain, rng, True)
            else:
                v = rng.choice(_decoy_values(rng, a, cur, extras))
                _set_attr(o, a, dict(cur, **{a: v}), domain, rng, True)
        elif a == "spicy":
            _set_attr(o, a, cur, domain, rng, rng.random() < P_TRUE)
    return o


def _signature(o):
    return json.dumps({k: v for k, v in o.items() if k not in ("id", "name", "price")}, sort_keys=True, ensure_ascii=False) + \
        str(band_of(o["price"], o["_domain"]))


def build_options(rng, domain, cur, stale, temp_trap_state):
    opts = [("correct", _option_satisfying(rng, domain, cur))]
    if stale is not None:
        changed = [a for a in DOMAINS[domain] if stale[a] != cur[a]]
        t = _trap(rng, domain, cur, stale, changed)
        if t is None:
            stale = None
        else:
            opts.append(("stale_trap", t))
    if temp_trap_state is not None:
        changed = [a for a in DOMAINS[domain] if temp_trap_state[a] != cur[a]]
        t = _trap(rng, domain, cur, temp_trap_state, changed)
        if t is None:
            temp_trap_state = None
        else:
            opts.append(("temp_trap", t))
    seen = set()
    for _, o in opts:
        o["_domain"] = domain
        seen.add(_signature(o))
    tries = 0
    while len(opts) < N_OPTIONS:
        tries += 1
        assert tries < 5000, "生成干扰项失败"
        o = _distractor(rng, domain, cur, [stale, temp_trap_state])
        o["_domain"] = domain
        if satisfies(o, cur, domain) or _signature(o) in seen:
            continue
        seen.add(_signature(o))
        opts.append(("distractor", o))
    rng.shuffle(opts)
    names = _names(rng, domain, [o for _, o in opts])
    out, roles = [], {}
    for i, (role, o) in enumerate(opts):
        o = {k: v for k, v in o.items() if k != "_domain"}
        o["id"] = LETTERS[i]
        o["name"] = names[i]
        out.append(o)
        roles[o["id"]] = role
    correct = [o["id"] for o in out if satisfies(o, cur, domain)]
    assert len(correct) == 1, correct
    return out, correct[0], roles, stale, temp_trap_state


def _names(rng, domain, opts):
    if domain == "hotel":
        return rng.sample(texts.HOTEL_NAMES, len(opts))
    used, names = set(), []
    for o in opts:
        base = texts.DINING_NAMES[o["cuisine"]]
        pool = [n for n in base + [b + "（新店）" for b in base] if n not in used]
        n = rng.choice(pool)
        used.add(n)
        names.append(n)
    return names


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


# ---------- 会话计划 ----------

def _pick_event(rng, state, exclude):
    cands = [e for e, ev in texts.EVENTS.items() if e not in exclude and ev["pre"](state)]
    weights = [0.5 if e in ("checkup", "car") else 1.0 for e in cands]
    return rng.choices(cands, weights)[0] if cands else None


def _pick_domain(rng, pending, avoid=None):
    """优先选能检验待测属性的领域。"""
    doms = [d for d in DOMAINS if d != avoid] or list(DOMAINS)
    if pending:
        score = {d: len(set(DOMAINS[d]) & pending) for d in doms}
        best = max(score.values())
        if best > 0:
            doms = [d for d in doms if score[d] == best]
    return rng.choice(doms)


def generate_user(rng, uid, n_sessions):
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

    sessions = []
    week = 1
    prev_val = {}          # 属性 -> 最近一次事件之前的值
    pending = set()        # 还没被后续任务检验过的变化属性
    last_temp = None       # (属性, 临时值)
    used_events = set()
    prev_domain = None
    for idx, typ in enumerate(types):
        if idx > 0:
            week += rng.choice([1, 2, 2, 3])
        date = BASE_DATE + dt.timedelta(days=(week - 1) * 7 + rng.randint(0, 6))
        sess = {"idx": idx, "week": week, "date": date.isoformat(), "weekday": WEEKDAYS[date.weekday()],
                "type": typ, "event": None, "temp": None}
        mention = ""
        if typ == "E":
            etype = _pick_event(rng, state, used_events)
            if etype is None:
                typ = sess["type"] = "P"
            else:
                used_events.add(etype)
                changes = event_effects(etype, state, rng)
                sess["event"] = {"type": etype, "cn": texts.EVENTS[etype]["cn"],
                                 "changes": {a: [state[a], v] for a, v in changes.items()}}
                for a, v in changes.items():
                    prev_val[a] = state[a]
                    state[a] = v
                mention = event_mention(etype, changes, rng)
                sess["event"]["mention"] = mention
                pending |= set(changes)
        if typ == "U":
            q, options, correct = build_util_task(rng, date)
            sess.update(domain="util", request=q, user_text=q, options=options, correct=correct,
                        roles={o["id"]: ("correct" if o["id"] == correct else "distractor") for o in options},
                        long_state=dict(state), eff_state=None, stale_state=None, temp_trap_state=None)
            sessions.append(sess)
            continue
        eff = dict(state)
        if typ == "T":
            cands = [t for t, tv in texts.TEMPS.items() if tv["pre"](state)]
            ttype = rng.choice(cands)
            tv = texts.TEMPS[ttype]
            val = temp_value(ttype, state, rng)
            domain = rng.choice(tv["domains"])
            eff[tv["attr"]] = val
            sess["temp"] = {"type": ttype, "cn": tv["cn"], "attr": tv["attr"], "value": val,
                            "long_value": state[tv["attr"]]}
            mention = rng.choice(tv["mentions"]).format(temp_value=val)
        elif typ == "E":
            domain = _pick_domain(rng, set(sess["event"]["changes"]))
        else:
            want = set(pending)
            if last_temp:
                want.add(last_temp[0])
            domain = _pick_domain(rng, want, avoid=prev_domain if idx <= 1 else None)
            if idx == 0:
                domain = rng.choice(["dinner", "lunch"])
        stale = None
        skip = sess["temp"]["attr"] if sess["temp"] else None
        if any(a in DOMAINS[domain] and a != skip and prev_val[a] != eff[a] for a in prev_val):
            stale = dict(eff, **{a: v for a, v in prev_val.items() if a in DOMAINS[domain] and a != skip})
            if all(stale[a] == eff[a] for a in DOMAINS[domain]):
                stale = None
        temp_trap = None
        if last_temp and typ != "T" and last_temp[0] in DOMAINS[domain]:
            ta, tval = last_temp
            if eff[ta] != tval:
                temp_trap = dict(eff, **{ta: tval})
        options, correct, roles, stale, temp_trap = build_options(rng, domain, eff, stale, temp_trap)
        if stale is not None and typ != "E":
            pending -= set(DOMAINS[domain])
        if temp_trap is not None:
            last_temp = None
        if typ == "T":
            last_temp = (sess["temp"]["attr"], sess["temp"]["value"])
        req = rng.choice(texts.REQUESTS[domain]).format(city=rng.choice(texts.CITIES))
        sess.update(domain=domain, request=req, user_text=(mention + req) if mention else req,
                    options=options, correct=correct, roles=roles,
                    long_state=dict(state), eff_state=eff, stale_state=stale, temp_trap_state=temp_trap)
        sessions.append(sess)
        prev_domain = domain
    return {"uid": uid, "init_state": init, "sessions": sessions}


def generate_world(seed, n_users=24, n_sessions=8):
    assert n_sessions >= 6
    rng = random.Random(seed)
    users = [generate_user(random.Random(rng.random()), f"u{i:02d}", n_sessions) for i in range(n_users)]
    return {"seed": seed, "n_users": n_users, "n_sessions": n_sessions, "users": users}


def world_hash(world):
    return hashlib.sha256(json.dumps(world, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def world_stats(world):
    ev_multi = ev_single = temps = util = stale = temp_traps = 0
    by_event, by_temp = {}, {}
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
            stale += s["stale_state"] is not None
            temp_traps += s["temp_trap_state"] is not None
    return {
        "n_users": world["n_users"], "n_sessions_per_user": world["n_sessions"],
        "n_sessions_total": sum(len(u["sessions"]) for u in world["users"]),
        "events_multi_attr": ev_multi, "events_single_attr": ev_single,
        "temp_situations": temps, "unrelated_sessions": util,
        "stale_trap_sessions": stale, "temp_trap_sessions": temp_traps,
        "events_by_type": by_event, "temps_by_type": by_temp,
    }
