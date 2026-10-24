"""脚本策略：判分自检用的参照，以及几种典型捷径。

- oracle：知道用户真实状态的标准策略（特权），不提问，应得满分
- nothing：什么都不做，应得零分
- ask_all：每次把与本任务相关的问题都问一遍（每问两项），完美理解回答
- conservative：不做个性化，选"对随机用户最可能合适"的保守默认
- random：随机选
- remember_no_update：问过就记、从不察觉变化（非特权，用来验证"过时记忆陷阱"确实生效）
- majority：每个属性取选项里出现最多的值，选最"随大流"的选项（检验选项本身不泄露答案）

与偏好无关的任务（换算、算账）有客观答案，除 nothing / random / majority 外的脚本策略都直接答对，不提问。
"""

import json
import random
from collections import Counter

from .grading import satisfies, violations
from .schema import TOPICS, topics_for_domain
from .world import sample_initial_state


def _pairs(xs):
    return [xs[i:i + 2] for i in range(0, len(xs), 2)]


class Oracle:
    name = "oracle"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            ctx.choose(s["correct"])
            return
        st = ctx.oracle_state()
        ctx.choose(next(o["id"] for o in s["options"] if satisfies(o, st, s["domain"])))


class Nothing:
    name = "nothing"
    raise_errors = True

    def run_session(self, ctx):
        pass


class AskAll:
    name = "ask_all"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            ctx.choose(s["correct"])
            return
        known = {}
        for pair in _pairs([t for t in topics_for_domain(s["domain"]) if t != "近况"]):
            known.update(ctx.ask_structured(pair))
        ctx.choose(_best_match(s, known))


def _best_match(s, known):
    """在已知属性上违反最少的选项（未知属性不算）。"""
    def n_viol(o):
        return sum(1 for a in violations(o, _fill(known), s["domain"]) if a in known)
    return min(s["options"], key=lambda o: (n_viol(o), o["id"]))["id"]


def _fill(known):
    # 未知属性填一个占位值，判违反时再按 known 过滤
    base = {"allergy": "无", "diet": "普通", "cuisine": "?", "spicy": "爱吃辣", "budget": "中", "home": "?",
            "office": "?", "commute": "地铁", "kid": "无", "pet": "无", "evening": "可以晚归"}
    return dict(base, **known)


class Conservative:
    name = "conservative"
    raise_errors = True

    def __init__(self, n_prior=2000, seed=12345):
        rng = random.Random(seed)
        self.prior = [sample_initial_state(rng) for _ in range(n_prior)]

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            ctx.choose(s["correct"])
            return
        def rate(o):
            return sum(satisfies(o, st, s["domain"]) for st in self.prior)
        ctx.choose(max(s["options"], key=lambda o: (rate(o), -ord(o["id"])))["id"])


class RandomPolicy:
    name = "random"
    raise_errors = True

    def run_session(self, ctx):
        ctx.choose(ctx.rng("random").choice(ctx.sess["options"])["id"])


class RememberNoUpdate:
    name = "remember_no_update"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            ctx.choose(s["correct"])
            return
        ctx.memory_read()
        known = {k[5:]: json.loads(v["value"]) for k, v in ctx.ep.memory.entries.items() if k.startswith("attr:")}
        missing = [t for t in topics_for_domain(s["domain"])
                   if t != "近况" and any(a not in known for a in TOPICS[t])]
        for pair in _pairs(missing):
            got = ctx.ask_structured(pair)
            for a, v in got.items():
                ctx.memory_write("attr:" + a, json.dumps(v, ensure_ascii=False))
            known.update(got)
        ctx.choose(_best_match(s, known))


class Majority:
    name = "majority"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        opts = s["options"]
        keys = [k for k in opts[0] if k not in ("id", "name", "price")]
        if s["domain"] == "util":
            ctx.choose(ctx.rng("majority").choice(opts)["id"])
            return
        modes = {}
        for k in keys:
            modes[k] = Counter(json.dumps(o[k], ensure_ascii=False) for o in opts).most_common(1)[0][0]
        def agree(o):
            return sum(json.dumps(o[k], ensure_ascii=False) == modes[k] for k in keys)
        ctx.choose(max(opts, key=lambda o: (agree(o), -ord(o["id"])))["id"])


SCRIPTED = {p.name: p for p in (Oracle, Nothing, AskAll, Conservative, RandomPolicy, RememberNoUpdate, Majority)}
