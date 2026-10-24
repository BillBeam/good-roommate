"""选项泄题检验：完全不看用户、只看候选选项的规则与分类器能选对多少。

- modal：审阅脚本里的规则——挑菜系、区域、价位档次和其他选项撞得最多的那家（平局取期望）
- position：永远选同一个位置（A–F 各算一遍，报最好的）
- conservative：选"对随机用户最可能合适"的那家（按初始状态分布估计）
- classifier：只用选项特征训练的条件 logit 分类器；在另外生成的世界上训练，在闸门世界上测
- 正确答案在 A–F 上的分布，做卡方检验

只用 Python 标准库与 numpy。
"""

import math
import random
from collections import Counter

import numpy as np

from .grading import band_of, satisfies
from .schema import ALLERGENS, AREAS, CUISINES, DIET_TAGS, DOMAINS
from .world import LETTERS, sample_initial_state


def pref_sessions(world):
    return [s for u in world["users"] for s in u["sessions"] if s["domain"] != "util"]


def _feat(o, d):
    return dict(o, band=band_of(o["price"], d))


def modal_rule(world):
    hit, n = 0.0, 0
    for s in pref_sessions(world):
        d = s["domain"]
        F = {o["id"]: _feat(o, d) for o in s["options"]}
        keys = ["cuisine", "area", "band"]
        mode = {k: Counter(F[i][k] for i in F).most_common(1)[0][0] for k in keys}
        sc = {i: sum(F[i][k] == mode[k] for k in keys) for i in F}
        cands = [i for i in F if sc[i] == max(sc.values())]
        hit += 1 / len(cands) if s["correct"] in cands else 0.0
        n += 1
    return hit / n


def position_rule(world):
    ss = pref_sessions(world)
    cnt = Counter(s["correct"] for s in ss)
    return {L: cnt.get(L, 0) / len(ss) for L in LETTERS}, cnt


def conservative_rule(world, n_prior=2000, seed=12345):
    rng = random.Random(seed)
    prior = [sample_initial_state(rng) for _ in range(n_prior)]
    hit, n = 0.0, 0
    for s in pref_sessions(world):
        d = s["domain"]
        rate = {o["id"]: sum(satisfies(o, st, d) for st in prior) for o in s["options"]}
        best = max(rate.values())
        cands = [i for i in rate if rate[i] == best]
        hit += 1 / len(cands) if s["correct"] in cands else 0.0
        n += 1
    return hit / n


def chi2_uniform(counts, k=6):
    n = sum(counts.get(L, 0) for L in LETTERS[:k])
    e = n / k
    x = sum((counts.get(L, 0) - e) ** 2 / e for L in LETTERS[:k])
    return x, chi2_sf(x, k - 1)


def chi2_sf(x, df):
    """卡方分布的上尾概率（df 为整数）。"""
    if df % 2 == 0:
        m = x / 2
        term, s = 1.0, 1.0
        for i in range(1, df // 2):
            term *= m / i
            s += term
        return math.exp(-m) * s
    s = math.erfc(math.sqrt(x / 2))
    term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    acc = 0.0
    for i in range(1, (df + 1) // 2):
        acc += term
        term *= x / (2 * i + 1)
    return s + acc


# ---------- 分类器 ----------

ATTR_FEATS = ["cuisine", "area", "band", "spice", "slot_early", "parking", "kid_ok", "pet_ok"]


def _option_vector(o, opts, d):
    """单个选项的特征：自身取值（独热）+ 与其他选项撞值的个数 + 与最近邻的差异，
    以及"在和最近邻不同的那些属性上自己取什么值"（用来捕捉"看起来刚变过"的方向）。"""
    F = [_feat(x, d) for x in opts]
    me = _feat(o, d)

    def vals(x):
        v = {"cuisine": x["cuisine"], "area": x["area"], "band": x["band"], "spice": x["spice"]}
        if d == "dinner":
            v.update(slot_early=x["slot"] < "18:30", parking=x["parking"], kid_ok=x["kid_ok"], pet_ok=x["pet_ok"])
        for a in ALLERGENS:
            v["alg_" + a] = a in x["allergens"]
        for t in DIET_TAGS:
            v["tag_" + t] = t in x["diet_tags"]
        return v

    mv = vals(me)
    others = [vals(x) for x in F if x["id"] != o["id"]]
    dist = [sum(mv[k] != ov[k] for k in mv) for ov in others]
    nn = others[int(np.argmin(dist))]
    f = {}
    f["dom_" + d] = 1.0
    for k, v in mv.items():
        f[f"{d}|{k}={v}"] = 1.0
        f[f"{d}|collide|{k}"] = sum(ov[k] == v for ov in others) / 5
        if nn[k] != v:
            f[f"{d}|nn_diff|{k}={v}"] = 1.0
    f[f"{d}|nn_dist"] = min(dist) / 10
    f[f"{d}|nn_unique"] = float(sorted(dist)[0] < sorted(dist)[1])
    f[f"{d}|mean_dist"] = sum(dist) / len(dist) / 10
    f[f"{d}|price_rank"] = sorted(x["price"] for x in F).index(o["price"]) / 5
    return f


def _design(worlds):
    rows = []
    for w in worlds:
        for s in pref_sessions(w):
            feats = [_option_vector(o, s["options"], s["domain"]) for o in s["options"]]
            y = [o["id"] for o in s["options"]].index(s["correct"])
            rows.append((feats, y))
    return rows


def _vectorize(rows, vocab=None):
    if vocab is None:
        vocab = {}
        for feats, _ in rows:
            for f in feats:
                for k in f:
                    vocab.setdefault(k, len(vocab))
    X = np.zeros((len(rows), 6, len(vocab)))
    Y = np.zeros(len(rows), dtype=int)
    for i, (feats, y) in enumerate(rows):
        for j, f in enumerate(feats):
            for k, v in f.items():
                if k in vocab:
                    X[i, j, vocab[k]] = v
        Y[i] = y
    return X, Y, vocab


def train_logit(X, Y, l2=1e-2, steps=600, lr=0.5):
    w = np.zeros(X.shape[2])
    for _ in range(steps):
        z = X @ w
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(p)
        onehot[np.arange(len(Y)), Y] = 1
        grad = np.einsum("nj,njk->k", p - onehot, X) / len(Y) + l2 * w
        w -= lr * grad
    return w


def eval_logit(w, X, Y):
    z = X @ w
    pred = z.argmax(axis=1)
    return float((pred == Y).mean())


def classifier_probe(train_worlds, test_worlds, l2=1e-2):
    Xtr, Ytr, vocab = _vectorize(_design(train_worlds))
    w = train_logit(Xtr, Ytr, l2=l2)
    Xte, Yte, _ = _vectorize(_design(test_worlds), vocab)
    return {"train_acc": eval_logit(w, Xtr, Ytr), "test_acc": eval_logit(w, Xte, Yte),
            "n_train_sessions": len(Ytr), "n_test_sessions": len(Yte), "n_features": len(vocab), "l2": l2}
