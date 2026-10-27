"""要不要跑 GRPO 种子 2：种子 1 训完后的全程考试（eval_grpo1）和 SFT 后的全程考试（eval_sft）按用户配对比较。

规则（2026-10-27 04:3x UTC 定下并提交，当时种子 1 的考试还没有任何一组跑完）：
- 两次考试都是全程、种子 1–3，设置相同（温度 0、思考上限 4096、回答上限 8192）。
- 只比两次都有的、与偏好有关的会话，按（种子, 用户）配对：
  差 = Σ_用户 (GRPO 后得分之和 − SFT 后得分之和) / Σ_用户 会话数，即同一批会话上的平均分之差。
- 差 > 0（GRPO 后高于 SFT 后）就照原计划跑种子 2 的训练和考试；差 ≤ 0 就不跑，直接收尾。
- 只看点估计；按用户重抽样的 95% 区间照算照记，不参与判定。
- 这条规则只决定种子 2；其余停止规则、预算和收尾要求不变。

    python -m roommate.train.seed2_rule [--out runs/m2pilot/machine]
"""

import argparse
import json
import os
import random
from collections import defaultdict

from .common import FULL_SEEDS, OUT, dump, world
from .metrics import session_row

SKIP_RC = 3          # grpo.py 按规则不跑种子 2 时的退出码


def _rows(out, tag, seed, sess):
    p = os.path.join(out, "eval", tag, "full", f"seed{seed}", "sessions.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    rows = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[(r["uid"], r["idx"])] = session_row(r, sess[(r["uid"], r["idx"])])
    return rows


def decide(out=OUT, n_boot=2000, write=True):
    a, b, n = defaultdict(float), defaultdict(float), defaultdict(int)
    for s in FULL_SEEDS:
        w = world(s)
        sess = {(u["uid"], x["idx"]): x for u in w["users"] for x in u["sessions"]}
        sft, grpo = _rows(out, "sft", s, sess), _rows(out, "grpo1", s, sess)
        for key in sorted(set(sft) & set(grpo)):
            if not sft[key]["pers"]:
                continue
            u = (s, key[0])
            a[u] += sft[key]["score"]
            b[u] += grpo[key]["score"]
            n[u] += 1
    users = sorted(n)
    diff = lambda us: sum(b[u] - a[u] for u in us) / sum(n[u] for u in us)
    rng = random.Random(0)
    boots = sorted(diff([rng.choice(users) for _ in users]) for _ in range(n_boot))
    d = diff(users)
    out_d = {"rule": "差 = 同一批会话上 GRPO 种子 1 后与 SFT 后的平均分之差（全程，种子 1–3，按用户配对）；差 > 0 跑种子 2，否则不跑",
             "sessions": sum(n.values()), "users": len(users),
             "sft_mean": sum(a.values()) / sum(n.values()), "grpo1_mean": sum(b.values()) / sum(n.values()),
             "diff": d, "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
             "run_seed2": d > 0}
    if write:
        dump(os.path.join(out, "seed2_decision.json"), out_d)
    return out_d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    print(json.dumps(decide(args.out, write=False), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
