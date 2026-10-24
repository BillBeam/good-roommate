"""模拟用户的前后一致性检查（闸门①的一部分）。

遍历世界里所有用户、所有会话、所有属性，按几种不同的"提问历史"生成回答，逐条检查：
1. 只答所问：回答里只出现被问到的属性（饮食限制附带的健康原因不吃辣除外，那正是要求说出来的）；
2. 回答能被短句表完整反解（没有模板以外的话）；
3. 每条陈述和用户当时的真实状态一致（长期陈述对长期状态，"这次"的陈述对本次有效状态）；
4. 不和自己说过的话矛盾：独立记一本"用户对 agent 说过什么"的账（回答里的陈述 + 事件台词标注），
   新的长期陈述必须和账上最近一次的说法相同（账只会被回答和事件台词更新）；
5. "说过了"只在账上确有这一项、且账上的值等于当前真实值时出现；
6. 全称否定式短句（texts.BLANKET）只在这一方面确实没有约束时出现；
7. 健康原因造成的不吃辣（怀孕、体检、本次嗓子发炎），被问到饮食限制且实际作答时一定说出来；
8. 每项属性第一次被问到、且之前没在台词里告诉过时，回答就是用户当时的真实值；
9. 事件台词里标"明说"的改动，新值的关键词确实出现在台词里。
"""

import copy
import random

from . import texts
from .schema import ATTR_ORDER, AREAS, CUISINES
from .usersim import SimUser

# "明说"的关键词：属性 -> 取值 -> 台词里应出现的字样
EXPLICIT_WORDS = {
    "home": {a: [a] for a in AREAS},
    "office": {a: [a] for a in AREAS},
    "cuisine": {c: [c] for c in CUISINES},
    "commute": {"开车": ["开车"], "骑车": ["骑车"], "地铁": ["地铁"]},
    "diet": {"控碳水": ["碳水"], "低糖": ["糖"], "素食": ["素"]},
    "spicy": {"不吃辣": ["辣"]},
    "kid": {"有婴儿": ["宝宝", "满月", "闺女"]},
    "pet": {"有狗": ["柯基", "金毛", "狗"]},
    "budget": {"高": ["高档"], "低": ["便宜"]},
    "evening": {"要早回": ["回家"]},
}

HISTORIES = ("never", "every_session", "random")


def _health_spicy(user):
    return user.state["spicy"] == "不吃辣" and user.spicy_reason in texts.SPICY_REASON


def check_world(world, repeat_mode="refuse_told", histories=HISTORIES, seed=0):
    problems, n_answers, n_lines = [], 0, 0
    first_answers = []  # (uid, idx, attr, 回答, 真实值) —— 第 8 条的明细
    for u in world["users"]:
        for hist in histories:
            rng = random.Random(f"{seed}|{u['uid']}|{hist}")
            user = SimUser(u, repeat_mode)
            ledger = {}          # 属性 -> 用户最近一次对 agent 说的长期值（独立于 SimUser.told）
            asked_ever = set()
            for s in u["sessions"]:
                user.begin_session(s)
                if s.get("event"):
                    n_lines += 1
                    ev = s["event"]
                    for a, kind in ev["cues"].items():
                        val = user.state[a]
                        if kind == "明说" and a in EXPLICIT_WORDS and val in EXPLICIT_WORDS[a]:
                            if not any(w in ev["mention"] for w in EXPLICIT_WORDS[a][val]):
                                problems.append((u["uid"], s["idx"], "line", f"明说的 {a}={val} 在台词里找不到"))
                        ledger[a] = val
                    for a in ev["changes"]:
                        if a not in ev["cues"]:
                            problems.append((u["uid"], s["idx"], "line", f"改动 {a} 没有线索"))
                eff = user.effective()
                # 逐属性生成回答（在副本上，不影响这条历史）
                for a in ATTR_ORDER:
                    probe = copy.deepcopy(user)
                    r = probe.answer([a])
                    n_answers += 1
                    _check_answer(problems, u["uid"], s, [a], r, user, eff, ledger)
                    if a not in asked_ever and a not in ledger:
                        first_answers.append((u["uid"], s["idx"], a, r["text"], user.state[a]))
                # 推进这条历史：决定这次会话真的问了哪些属性
                if hist == "every_session":
                    ask = list(ATTR_ORDER)
                elif hist == "random":
                    ask = [a for a in ATTR_ORDER if rng.random() < 0.3]
                else:
                    ask = []
                for i in range(0, len(ask), 2):
                    r = user.answer(ask[i:i + 2])
                    n_answers += 1
                    _check_answer(problems, u["uid"], s, ask[i:i + 2], r, user, eff, ledger)
                    for c in texts.parse_claims(r["text"]):
                        if c["kind"] == "value" and c["scope"] == "long":
                            ledger[c["attr"]] = c["value"]
                    asked_ever |= set(ask[i:i + 2])
                user.end_session()
    return {"n_answers_checked": n_answers, "n_event_lines": n_lines, "n_problems": len(problems),
            "problems": problems[:50], "n_first_answers": len(first_answers),
            "first_answers_truthful": sum(_first_ok(t) for t in first_answers),
            "first_answer_failures": [t for t in first_answers if not _first_ok(t)][:20]}


def _first_ok(t):
    uid, idx, a, text, truth = t
    claims = [c for c in texts.parse_claims(text) if c["kind"] == "value" and c["attr"] == a]
    return any(c["value"] == truth for c in claims) or any(c["scope"] == "temp" for c in claims)


def _check_answer(problems, uid, s, asked, r, user, eff, ledger):
    text = r["text"]
    claims = texts.parse_claims(text)
    where = (uid, s["idx"], "+".join(asked))
    for c in claims:
        if c["kind"] == "unknown":
            problems.append(where + (f"无法反解的片段：{c['text']}",))
            continue
        attr = c["attr"]
        if attr not in asked and not ("diet" in asked and attr == "spicy"):
            problems.append(where + (f"答非所问：问 {asked}，答了 {attr}",))
        if c["kind"] == "told":
            if attr not in ledger:
                problems.append(where + ("说'之前说过了'，但从没告诉过",))
            elif ledger[attr] != user.state[attr]:
                problems.append(where + (f"说'之前说过了'，但上次说的是 {ledger[attr]}，现在是 {user.state[attr]}",))
            continue
        if c["scope"] == "long":
            if c["value"] != user.state[attr]:
                problems.append(where + (f"长期陈述 {attr}={c['value']} 与真实值 {user.state[attr]} 不符",))
            if attr in ledger and ledger[attr] != c["value"]:
                problems.append(where + (f"与自己说过的话矛盾：之前说 {attr}={ledger[attr]}，现在说 {c['value']}",))
        else:
            if c["value"] != eff[attr]:
                problems.append(where + (f"本次陈述 {attr}={c['value']} 与本次真实值 {eff[attr]} 不符",))
    for phrase, ok in texts.BLANKET.items():
        if phrase in text and not ok(eff):
            problems.append(where + (f"全称否定'{phrase}'与真实状态不符",))
    if "diet" in asked and r["status"].get("diet") == "answered":
        temp_spicy = user.temp and user.temp["attr"] == "spicy"
        if (_health_spicy(user) or temp_spicy) and not any(
                c["kind"] == "value" and c["attr"] == "spicy" and c["value"] == "不吃辣" for c in claims):
            problems.append(where + ("健康原因不吃辣，问到饮食限制时没有说出来",))
    if user.temp and user.temp["attr"] in asked:
        ta = user.temp["attr"]
        if not any(c["kind"] == "value" and c["attr"] == ta and c["scope"] == "temp" for c in claims):
            problems.append(where + ("本次临时情况改到的属性，没有照实回答临时值",))
    for a in asked:
        if not any(c["attr"] == a for c in claims if c["kind"] != "unknown"):
            problems.append(where + (f"问了 {a} 却没有回答",))
