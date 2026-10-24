"""事件台词的线索校验（闸门①的一部分）：每个事件改动的每一项属性，都要能从台词里读出来。

做法：让评判模型只看一句事件台词和用户原来的状态，推出改了哪几项、改成什么；
和程序里的真实改动逐项比较，完全一致才算推对。每句台词要被每个评判模型都推对才算合格。
这里用到了评判模型（README 规则 1 要求单独标注）：它只用于检验台词，不参与任何判分。

校验的台词：闸门世界与对照世界里实际出现的全部事件台词，外加每个台词模板在随机状态上的若干实例
（保证每个模板都被校验到）。
"""

import json
import random

from . import texts
from .schema import ATTR_CN, ATTR_ORDER, ATTRS, CN_ATTR, PRICE_BANDS
from .world import sample_initial_state

VALUE_NOTES = {
    "diet": {"普通": "普通（不吃素、不控碳水、不控糖）"},
    "budget": {b: f"{b}（晚饭人均 {PRICE_BANDS['dinner'][b][0]}–{PRICE_BANDS['dinner'][b][1]} 元）" for b in "低中高"},
    "evening": {"下班晚": "下班晚（晚饭只能订 19:30 以后）", "要早回": "要早回（晚饭只能订 18:30 以前）"},
    "office": {},
    "cuisine": {},
}
ATTR_NOTES = {"cuisine": "菜系（最爱吃的）", "office": "公司（在哪个区上班）", "home": "住处（住在哪个区）",
              "evening": "晚归（晚上的作息）", "commute": "出行（平时怎么出门）"}

PROMPT = """下面是一位用户原来的情况，以及他今天和私人助理聊天时顺口说的一句话。
请判断：按这句话，用户的哪些情况变了、变成了什么。

各项情况及可选取值：
{domains}

用户原来的情况：{old}

用户说："{line}"

只写这句话让它变了的项，没变的不要写；取值只能从上面的可选取值里选。
只输出 JSON，例如 {{"changes": {{"住处": "滨江", "出行": "地铁"}}}}。"""


def _domains_text():
    rows = []
    for a in ATTR_ORDER:
        vals = [VALUE_NOTES.get(a, {}).get(v, v) for v in ATTRS[a]]
        rows.append(f"- {ATTR_NOTES.get(a, ATTR_CN[a])}：" + " / ".join(vals))
    return "\n".join(rows)


def build_prompt(old, line):
    old_s = "，".join(f"{ATTR_CN[a]}={old[a]}" for a in ATTR_ORDER)
    return PROMPT.format(domains=_domains_text(), old=old_s, line=line)


def items_from_worlds(worlds):
    out = []
    for w in worlds:
        for u in w["users"]:
            state = dict(u["init_state"])
            for s in u["sessions"]:
                ev = s.get("event")
                if ev:
                    old = dict(state)
                    for a, (o, n) in ev["changes"].items():
                        state[a] = n
                    out.append({"source": f"seed{w['seed']}/{u['uid']}/{s['idx']}", "event": ev["type"],
                                "line": ev["mention"], "old": old,
                                "changes": {a: n for a, (o, n) in ev["changes"].items()}})
    return out


def _template_id(etype, line, after):
    ev = texts.EVENTS[etype]
    tpls = [t for v in ev["mentions"].values() for t in v] if isinstance(ev["mentions"], dict) else ev["mentions"]
    for i, (tpl, cues) in enumerate(tpls):
        if tpl.format(new_home=after["home"], new_office=after["office"]) == line:
            return f"{etype}#{i}"
    return f"{etype}#?"


def synthetic_items(per_template=3, seed=0):
    """每个台词模板至少 per_template 个实例（随机的原状态）。"""
    rng = random.Random(seed)
    want = {}
    for e, ev in texts.EVENTS.items():
        tpls = [t for v in ev["mentions"].values() for t in v] if isinstance(ev["mentions"], dict) else ev["mentions"]
        for i in range(len(tpls)):
            want[f"{e}#{i}"] = per_template
    out = []
    for _ in range(20000):
        if not any(v > 0 for v in want.values()):
            break
        s = sample_initial_state(rng)
        e = rng.choice(list(texts.EVENTS))
        if not texts.EVENTS[e]["pre"](s):
            continue
        ch = texts.event_effects(e, s, rng)
        after = dict(s, **ch)
        line, cues = texts.event_mention(e, after, rng)
        tid = _template_id(e, line, after)
        if want.get(tid, 0) <= 0:
            continue
        want[tid] -= 1
        out.append({"source": f"synthetic/{tid}/{per_template - want[tid]}", "event": e, "line": line, "old": s, "changes": ch})
    return out


def judge(client, model, item, extra=None):
    body = {"model": model, "temperature": 0, "seed": 1,
            "messages": [{"role": "user", "content": build_prompt(item["old"], item["line"])}],
            "response_format": {"type": "json_object"}, **(extra or {})}
    resp = client.chat(body, role="cuecheck")
    raw = resp["choices"][0]["message"].get("content") or ""
    try:
        got = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])["changes"]
        got = {CN_ATTR.get(k, k): v for k, v in got.items()}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"raw": raw, "pred": None, "ok": False}
    pred = {a: v for a, v in got.items() if not (a in item["old"] and item["old"][a] == v)}
    return {"raw": raw, "pred": pred, "ok": pred == item["changes"]}


def validate(client, models, items, extras=None, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    extras = extras or {}
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for m in models:
            res = list(pool.map(lambda it: judge(client, m, it, extras.get(m)), items))
            for it, r in zip(items, res):
                rows.append({"model": m, **{k: it[k] for k in ("source", "event", "line", "changes")},
                             "pred": r["pred"], "ok": r["ok"], "raw": r["raw"]})
    by_line = {}
    for r in rows:
        e = by_line.setdefault(r["line"] + "|" + json.dumps(r["changes"], ensure_ascii=False, sort_keys=True),
                               {"line": r["line"], "event": r["event"], "n": 0, "ok": {}})
        e["ok"].setdefault(r["model"], []).append(r["ok"])
    instances_ok = {}
    for r in rows:
        k = (r["source"], r["line"])
        instances_ok.setdefault(k, []).append(r["ok"])
    per_model = {m: sum(r["ok"] for r in rows if r["model"] == m) / max(1, sum(r["model"] == m for r in rows))
                 for m in models}
    n_inst = len(instances_ok)
    n_pass = sum(all(v) and len(v) == len(models) for v in instances_ok.values())
    failures = [r for r in rows if not r["ok"]]
    return {"models": models, "n_items": n_inst, "n_items_pass_all_models": n_pass,
            "all_pass": n_pass == n_inst, "accuracy_by_model": per_model,
            "failures": failures, "rows": rows}
