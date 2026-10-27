"""GRPO：从 SFT 模型出发，在短程考场（每条轨迹 = 同一用户连续 2 次见面）上训练。

    python -m roommate.train.grpo --seed 1 --adapter /root/work/ckpt/sft/final
    python -m roommate.train.grpo --seed 1 --resume        # 从最近一次检查点接着跑

每步：4 个用户（短程轨迹的起点），每人采样 8 次（温度 1、top_p 1，思考上限 4096），共 32 条轨迹、64 次会话。

奖励与优势（同一用户的 8 次采样是一组）：
- 每次会话的奖励 r_k = 判分（选对 1 分，每问一次扣 0.15；与偏好无关的任务同样按判分）。
- 第 k 次会话里的各段，回报取"从这次往后"的奖励之和：G_1 = r_1 + r_2，G_2 = r_2，
  所以第 1 次会话里记没记下信息，会通过第 2 次的得分反映到第 1 次的样本上。
- 优势 A_k = (G_k − 组内 G_k 的均值) / (组内 G_k 的标准差 + 1e-6)；组内 G_k 全相同时 A_k = 0（这组不提供信号）。
- 一段里每个计入 loss 的 token 都用这段所在会话的 A_k；loss 和截断重要性采样见 trainer.py。

每步记录：分数、选对率、提问数、两个捷径指标、思考长度、采样端与训练端 logprob 之差（均值、P99、最大值）、
对这个差的修正（截断重要性采样，上限 TIS_CAP，被截断的 token 比例），以及时间。
停止规则：分数上涨主要来自捷径①上升时停下（见 shortcut_stop），不调参绕过去。
"""

import argparse
import json
import os
import random
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from ..env import Env
from . import serve
from .agent_local import LocalAgent
from .common import (CKPT, EXPORT, GRPO_SEEDS, MAX_MODEL_LEN, MODEL, OUT, SHM, TRAIN_SETTING, append_jsonl, dump,
                     make_chat, rows_for, status, summary, world)
from .sampler import VLLMClient
from .shortrange import needs_first, short_world

MAX_TRAIN_LEN = 24576


def pool_of(seed):
    """训练用的短程轨迹池：GRPO_SEEDS 的世界里全部 7 个起点，按种子打乱。"""
    items = []
    for s in GRPO_SEEDS:
        sw = short_world(world(s))
        items += [(s, u) for u in sw["users"]]
    random.Random(seed).shuffle(items)
    return items


def advantages(G):
    """G：[样本][会话] 的回报 -> 同形的优势。"""
    A = [[0.0] * len(G[0]) for _ in G]
    for k in range(len(G[0])):
        col = [g[k] for g in G]
        mu = sum(col) / len(col)
        sd = statistics.pstdev(col)
        for i, g in enumerate(G):
            A[i][k] = 0.0 if sd < 1e-9 else (g[k] - mu) / (sd + 1e-6)
    return A


def returns(rewards):
    """reward-to-go：G_k = sum_{j>=k} r_j。"""
    out, acc = [], 0.0
    for r in reversed(rewards):
        acc += r
        out.append(acc)
    return out[::-1]


def rollout_group(agent_for, seed, user, n):
    """同一条短程轨迹采样 n 次，返回 [每次的会话记录列表]。"""
    w = world(seed)
    base = {"version": w["version"], "seed": seed, "n_users": 1, "n_sessions": 2, "events": w["events"],
            "config": w["config"], "users": [user]}

    def one(i):
        env = Env(base)
        env.run(agent_for(i), workers=1)
        return env.records()
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, range(n))), base


def build_items(trajs, A, gi=0, max_len=MAX_TRAIN_LEN):
    """一组轨迹 -> 训练样本：第 i 次采样、第 k 次会话里的每一段，都带优势 A[i][k]。"""
    items, dropped = [], 0
    for i, recs in enumerate(trajs):
        for k, r in enumerate(recs):
            for seg in r.get("segments", []):
                if not sum(seg["mask"]):
                    continue
                if len(seg["ids"]) > max_len:
                    dropped += 1
                    continue
                items.append({"ids": seg["ids"], "mask": seg["mask"], "lp": seg["lp"], "adv": A[i][k],
                              "g": gi, "i": i, "k": k})
    return items, dropped


def shortcut_stop(hist, window=5, min_steps=10):
    """分数上涨主要来自捷径①：最近 window 步对比最初 window 步，分数涨了 0.05 以上，捷径①（信息不够就选）涨了 0.10 以上，
    并且分数的涨幅里，少问省下的分（−0.15 × 提问数的变化）占一半以上。"""
    if len(hist) < min_steps:
        return False, None
    a, b = hist[:window], hist[-window:]
    m = lambda xs, k: sum(x[k] for x in xs if x.get(k) is not None) / max(1, sum(x.get(k) is not None for x in xs))
    d_score = m(b, "score") - m(a, "score")
    d_s1 = m(b, "shortcut1") - m(a, "shortcut1")
    d_q = m(b, "questions") - m(a, "questions")
    saved = -0.15 * d_q
    info = {"d_score": d_score, "d_shortcut1": d_s1, "d_questions": d_q, "question_saving": saved}
    return (d_score > 0.05 and d_s1 > 0.10 and saved >= 0.5 * d_score), info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adapter", help="SFT 的 LoRA（从这里出发）")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, help="演练中断：跑到这一步就退出（不清理检查点）")
    ap.add_argument("--gpu-util", type=float, default=0.40)
    ap.add_argument("--sleep-mode", action="store_true")
    args = ap.parse_args(argv)
    if args.seed == 2:
        # 种子 2 跑不跑，按 seed2_rule 的规则（种子 1 考试出结果之前定的）：GRPO 后不高于 SFT 后就不跑
        from .seed2_rule import SKIP_RC, decide
        d = decide()
        if not d["run_seed2"]:
            status("grpo_s2", stage="skipped_by_rule", diff=round(d["diff"], 4), sessions=d["sessions"])
            raise SystemExit(SKIP_RC)
    from .trainer import Trainer
    tag = f"grpo_s{args.seed}"
    out = os.path.join(OUT, tag)
    ck = os.path.join(CKPT, tag)
    latest = os.path.join(ck, "latest")
    log = os.path.join(out, "steps.jsonl")
    t_start = time.time()
    chat = make_chat()
    if args.resume and os.path.exists(os.path.join(latest, "trainer_state.pt")):
        tr = Trainer(MODEL, lr=args.lr, adapter=latest, seed=args.seed)
        extra = tr.load_state(latest)
        start = extra["step"]
        hist = extra["hist"]
        append_jsonl(os.path.join(out, "events.jsonl"), {"t": time.time(), "event": "resume", "from_step": start})
    else:
        assert args.adapter, "第一次运行要给 SFT 的 LoRA"
        tr = Trainer(MODEL, lr=args.lr, adapter=args.adapter, seed=args.seed)
        start, hist = 0, []
        if os.path.exists(log):
            shutil.move(log, log + f".old{int(time.time())}")
    proc, boot = serve.start(MODEL, os.path.join(OUT, "logs", f"vllm_{tag}.log"), gpu_util=args.gpu_util,
                             sleep_mode=args.sleep_mode)
    sampler = VLLMClient(f"http://127.0.0.1:{serve.PORT}", "policy")
    try:
        t0 = time.time()
        tr.export(SHM)
        sampler.update_weights(SHM)
        sync0 = time.time() - t0
        pool = pool_of(args.seed)
        dump(os.path.join(out, "config.json"), {**vars(args), "pool_size": len(pool), "train_setting": TRAIN_SETTING,
                                                "needs_first_in_pool": sum(needs_first(u)[0] for _, u in pool) / len(pool),
                                                "tis_cap": __import__("roommate.train.trainer", fromlist=["x"]).TIS_CAP,
                                                "vllm_boot_seconds": boot, "first_sync_seconds": sync0})
        for step in range(start, args.steps):
            if args.stop_after is not None and step >= args.stop_after:
                append_jsonl(os.path.join(out, "events.jsonl"), {"t": time.time(), "event": "stop_after", "step": step})
                return
            ts = time.time()
            groups = pool[step * args.groups:(step + 1) * args.groups]
            status(tag, step=step + 1, total=args.steps, stage="rollout")

            def agent_for(gi):
                return lambda i: LocalAgent(sampler, chat, max_model_len=MAX_MODEL_LEN, seed=args.seed,
                                            rep=1 + step * 1000 + gi * 100 + i, label=tag, **TRAIN_SETTING)
            with ThreadPoolExecutor(max_workers=args.groups) as pool_ex:
                results = list(pool_ex.map(lambda g: rollout_group(agent_for(g[0]), g[1][0], g[1][1], args.samples),
                                           enumerate(groups)))
            t_roll = time.time() - ts
            items, rows, groups_log, dropped = [], [], [], 0
            for gi, ((seed, user), (trajs, base)) in enumerate(zip(groups, results)):
                R = [[r["grade"]["score"] for r in recs] for recs in trajs]
                G = [returns(x) for x in R]
                A = advantages(G)
                groups_log.append({"world": seed, "uid": user["uid"], "needs_first": needs_first(user)[0],
                                   "rewards": R, "returns": G, "adv": A,
                                   "chosen": [[r["chosen"] for r in recs] for recs in trajs],
                                   "questions": [[len(r["questions"]) for r in recs] for recs in trajs]})
                for recs in trajs:
                    rows += rows_for(base, recs, group=f"{seed}:{gi}")
                its, dr = build_items(trajs, A, gi)
                items += its
                dropped += dr
            m = summary(rows, n_boot=0)["point"] if rows else {}
            status(tag, step=step + 1, total=args.steps, stage="train")
            t1 = time.time()
            if args.sleep_mode:
                sampler.sleep(1)
            train_items = [it for it in items if it["adv"] != 0.0]
            st = tr.train_step(train_items, "grpo") if train_items else {"tokens": 0}
            t_train = time.time() - t1
            t2 = time.time()
            tr.export(SHM)
            if args.sleep_mode:
                sampler.wake_up()
            sampler.update_weights(SHM)
            t_sync = time.time() - t2
            row = {"step": step + 1, "score": m.get("mean_score"), "success": m.get("success_rate"),
                   "questions": m.get("avg_questions"), "shortcut1": m.get("shortcut1_guess_unmentioned"),
                   "shortcut2": m.get("shortcut2_reask_told"), "think_tokens_per_turn": m.get("think_tokens_per_turn"),
                   "think_over_limit": m.get("think_over_limit_rate"), "format_error": m.get("format_error_rate"),
                   "segments": len(items), "segments_trained": len(train_items), "dropped_long": dropped,
                   "zero_adv_groups": sum(all(a == 0 for row_ in gl["adv"] for a in row_) for gl in groups_log),
                   "mismatch_abs_mean": st.get("mismatch_abs_mean"), "mismatch_abs_p99": st.get("mismatch_abs_p99"),
                   "mismatch_abs_max": st.get("mismatch_abs_max"),
                   "correction": f"TIS(cap={__import__('roommate.train.trainer', fromlist=['x']).TIS_CAP})",
                   "tis_clipped_frac": (st["tis_clipped"] / st["tis_n"]) if st.get("tis_n") else None,
                   "loss": st.get("loss"), "grad_norm": st.get("grad_norm"), "train_tokens": st.get("tokens"),
                   "t_rollout": t_roll, "t_train": t_train, "t_sync": t_sync, "t_step": time.time() - ts,
                   "wall": time.time() - t_start}
            hist.append(row)
            append_jsonl(log, row)
            append_jsonl(os.path.join(out, "groups.jsonl"), {"step": step + 1, "groups": groups_log})
            stop, info = shortcut_stop(hist)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps or stop:
                extra = {"step": step + 1, "hist": hist, "seed": args.seed}
                tr.save(latest, extra)
                exp = os.path.join(EXPORT, f"{tag}_latest")
                tr.save(exp, extra)
                append_jsonl(os.path.join(out, "events.jsonl"), {"t": time.time(), "event": "checkpoint",
                                                                 "step": step + 1})
            if stop:
                dump(os.path.join(out, "stopped_by_rule.json"), {"step": step + 1, **info})
                status(tag, stage="stopped_by_rule", step=step + 1)
                break
        tr.save(os.path.join(ck, "final"), {"step": len(hist), "hist": hist, "seed": args.seed})
        tr.save(os.path.join(EXPORT, f"{tag}_final"), {"step": len(hist), "hist": hist, "seed": args.seed})
        dump(os.path.join(out, "done.json"), {"steps": len(hist), "seconds": round(time.time() - t_start, 1)})
        status(tag, stage="done", steps=len(hist))
    finally:
        serve.stop(proc)


if __name__ == "__main__":
    main()
