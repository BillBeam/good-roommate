"""脚本策略（第二版）：参照、捷径，以及不看用户的规则。脚本策略提问时同样按属性声明，
并且只从回答的文字里拿信息（texts.parse_claims），和大模型 agent 看到的完全一样。

- oracle：标准策略（特权：直接知道用户当前真实状态），不提问。只用于判分自检，应得满分。
- ideal：理想记忆策略，满分参照。每项属性只在第一次需要时问（需要 = 不问就分不出正确选项），
  按事件台词的标注（texts.EVENTS 的 cues）正确更新，临时情况从台词得知，从不重问。
  它的"读台词"用的是标注（特权），代表一个完全理解台词的 agent。
- ask_all：每次都问、不写记忆。每次把本任务涉及的全部属性问一遍（每次问两项）。
- remember_all_reask：记住所有回答、每次都重问、不读台词。
- remember_no_update：问过就记、不再问，也不读台词（从不察觉变化）。
- conservative：保守默认，不做个性化，选"对随机用户最可能合适"的那家。
- modal：选菜系、区域、价位档次和其他选项撞得最多的那家（审阅脚本里的规则）。
- random：随机选。nothing：什么都不做。

与偏好无关的任务（换算、算账）单独计分：会算的策略直接答对，不看题的策略随机选。
"""

import itertools
import json
import random
from collections import Counter

from . import texts
from .grading import band_of, satisfies, violations
from .schema import ATTRS, ATTR_ORDER, DOMAINS, MAX_ATTRS_PER_QUESTION
from .world import sample_initial_state

MEM_PREFIX = "attr:"


def _pairs(xs):
    return [xs[i:i + MAX_ATTRS_PER_QUESTION] for i in range(0, len(xs), MAX_ATTRS_PER_QUESTION)]


_NEUTRAL = {"allergy": "无", "diet": "普通", "commute": "地铁", "kid": "无", "pet": "无"}


def consistent(o, known, domain):
    """选项 o 是否不违反已知的属性（未知属性不算）。"""
    filled = {a: known.get(a, _NEUTRAL.get(a, "?")) for a in ATTR_ORDER}
    return not [a for a in violations(o, filled, domain) if a in known]


def n_viol(o, known, domain):
    filled = {a: known.get(a, _NEUTRAL.get(a, "?")) for a in ATTR_ORDER}
    return sum(1 for a in violations(o, filled, domain) if a in known)


def best_match(ctx, known):
    """在已知属性上违反最少的选项；平局时随机（按会话固定的随机数）。"""
    s = ctx.sess
    best = min(n_viol(o, known, s["domain"]) for o in s["options"])
    tied = [o["id"] for o in s["options"] if n_viol(o, known, s["domain"]) == best]
    return ctx.rng("tie").choice(tied)


def absorb(claims, long_known, temp_known):
    """把回答里的陈述并入已知：长期值进 long_known，本次临时值进 temp_known。"""
    for c in claims:
        if c["kind"] != "value":
            continue
        (temp_known if c["scope"] == "temp" else long_known)[c["attr"]] = c["value"]


def _solve_util(ctx):
    ctx.choose(ctx.sess["correct"])


def _load_memory(ctx):
    ctx.memory_read()
    return {k[len(MEM_PREFIX):]: json.loads(v["value"]) for k, v in ctx.ep.memory.entries.items()
            if k.startswith(MEM_PREFIX)}


def _save_memory(ctx, before, after):
    for a, v in after.items():
        if before.get(a) != v:
            ctx.memory_write(MEM_PREFIX + a, json.dumps(v, ensure_ascii=False))


class Oracle:
    name = "oracle"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            return _solve_util(ctx)
        st = ctx.oracle_state()
        ctx.choose(next(o["id"] for o in s["options"] if satisfies(o, st, s["domain"])))


class Nothing:
    name = "nothing"
    raise_errors = True

    def run_session(self, ctx):
        pass


class RandomPolicy:
    name = "random"
    raise_errors = True

    def run_session(self, ctx):
        ctx.choose(ctx.rng("random").choice(ctx.sess["options"])["id"])


class Ideal:
    """理想记忆策略。"""
    name = "ideal"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            return _solve_util(ctx)
        d = s["domain"]
        mem = _load_memory(ctx)
        known = dict(mem)
        if s["event"]:  # 按台词标注更新：标注里的每项属性都能从台词读出新值
            for a in s["event"]["cues"]:
                if a in s["event"]["changes"]:
                    known[a] = s["event"]["changes"][a][1]
                # 标注了但没改的属性（如搬家后出行方式没变）：台词里说的就是当前值
                elif a in ctx.ep.user.state:
                    known[a] = ctx.ep.user.state[a]
        temp = {}
        if s["temp"]:
            temp[s["temp"]["attr"]] = s["temp"]["value"]
        while True:
            eff = dict(known, **temp)
            cands = [o for o in s["options"] if consistent(o, eff, d)]
            unknown = [a for a in DOMAINS[d] if a not in eff]
            if len(cands) <= 1 or not unknown:
                break
            ask = self._most_useful(cands, eff, unknown, d)
            claims = ctx.ask_structured(ask)
            absorb(claims, known, temp)
            assert all(c["kind"] != "told" for c in claims), "理想记忆策略不应重问"
        _save_memory(ctx, mem, known)
        ctx.choose(best_match(ctx, dict(known, **temp)))

    @staticmethod
    def _most_useful(cands, eff, unknown, d):
        """选一组（最多两项）未知属性，使问完之后剩下的候选选项最少（各取值等可能）。"""
        best = None
        groups = [list(g) for k in (1, 2) for g in itertools.combinations(unknown, k)]
        for g in groups:
            combos = list(itertools.product(*[ATTRS[a] for a in g]))
            rem = sum(sum(consistent(o, dict(eff, **dict(zip(g, vs))), d) for o in cands) for vs in combos)
            key = (rem / len(combos), -len(g) if rem / len(combos) <= 1 else len(g))
            if best is None or key < best[0]:
                best = (key, g)
        g = best[1]
        if len(g) == 1 and len(unknown) > 1:
            # 问一项就够分清时只问一项；否则顺手把第二项也问了（一次提问能答两项）
            if sum(consistent(o, eff, d) for o in cands) and best[0][0] > 1:
                g = g + [a for a in unknown if a not in g][:1]
        return g


class AskAll:
    """每次都问、不写记忆。"""
    name = "ask_all"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            return _solve_util(ctx)
        long_known, temp = {}, {}
        for pair in _pairs(DOMAINS[s["domain"]]):
            absorb(ctx.ask_structured(pair), long_known, temp)
        ctx.choose(best_match(ctx, dict(long_known, **temp)))


class RememberAllReask:
    """记住所有回答、每次都重问、不读台词。"""
    name = "remember_all_reask"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            return _solve_util(ctx)
        mem = _load_memory(ctx)
        known, temp = dict(mem), {}
        for pair in _pairs(DOMAINS[s["domain"]]):
            absorb(ctx.ask_structured(pair), known, temp)  # "说过了"的属性沿用记忆
        _save_memory(ctx, mem, known)
        ctx.choose(best_match(ctx, dict(known, **temp)))


class RememberNoUpdate:
    """问过就记、不再问，也不读台词。"""
    name = "remember_no_update"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            return _solve_util(ctx)
        mem = _load_memory(ctx)
        known, temp = dict(mem), {}
        missing = [a for a in DOMAINS[s["domain"]] if a not in known]
        for pair in _pairs(missing):
            absorb(ctx.ask_structured(pair), known, temp)
        _save_memory(ctx, mem, known)
        ctx.choose(best_match(ctx, dict(known, **temp)))


class Conservative:
    name = "conservative"
    raise_errors = True

    def __init__(self, n_prior=2000, seed=12345):
        rng = random.Random(seed)
        self.prior = [sample_initial_state(rng) for _ in range(n_prior)]

    def run_session(self, ctx):
        s = ctx.sess
        if s["domain"] == "util":
            return ctx.choose(ctx.rng("random").choice(s["options"])["id"])
        rate = {o["id"]: sum(satisfies(o, st, s["domain"]) for st in self.prior) for o in s["options"]}
        best = max(rate.values())
        ctx.choose(ctx.rng("tie").choice([i for i in rate if rate[i] == best]))


class Modal:
    name = "modal"
    raise_errors = True

    def run_session(self, ctx):
        s = ctx.sess
        opts = s["options"]
        if s["domain"] == "util":
            return ctx.choose(ctx.rng("random").choice(opts)["id"])
        F = {o["id"]: dict(o, band=band_of(o["price"], s["domain"])) for o in opts}
        keys = ["cuisine", "area", "band"]
        mode = {k: Counter(F[i][k] for i in F).most_common(1)[0][0] for k in keys}
        sc = {i: sum(F[i][k] == mode[k] for k in keys) for i in F}
        ctx.choose(ctx.rng("tie").choice([i for i in F if sc[i] == max(sc.values())]))


SCRIPTED = {p.name: p for p in (Oracle, Ideal, Nothing, RandomPolicy, AskAll, RememberAllReask,
                                RememberNoUpdate, Conservative, Modal)}
# 不看用户的规则（闸门③）
BLIND = ["conservative", "modal", "random"]
# 最简单策略（闸门④的下界）：捷径与不看用户的规则
TRIVIAL = ["ask_all", "remember_all_reask", "remember_no_update", "conservative", "modal", "random", "nothing"]
