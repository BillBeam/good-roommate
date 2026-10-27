"""评测：训练前、SFT 后、GRPO 后三个时点，设置完全相同。

    python -m roommate.train.evaluate --tag pre
    python -m roommate.train.evaluate --tag sft --adapter /root/work/ckpt/sft/final

- 全程：M1 第二版种子 1、2、3 的世界（各 24 个用户 × 8 次见面），结果并进联合基线表。
- 短程：新世界（种子 201、202）切出的 2 次见面轨迹，每个用户取起点 0、2、4、6。
- 口径（与体检一致）：开思考、温度 0、思考上限 4096 token（用满就补上 </think> 再给回答 8192 token）。
- 每次会话的记录、对话、指标（含按用户重抽样的 95% 区间）存到 OUT/eval/<tag>/。
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..env import Env
from . import serve
from .agent_local import LocalAgent
from .common import (EVAL_SETTING, FULL_SEEDS, MAX_MODEL_LEN, MODEL, OUT, SHM, SHORT_EVAL_SEEDS, SHORT_EVAL_STARTS,
                     dump, make_chat, rows_for, save_records, status, summary, world)
from .sampler import VLLMClient
from .shortrange import needs_first_rate, short_world


def merge_adapter_to(path, adapter, device="cpu"):
    from .trainer import Trainer
    t = Trainer(MODEL, adapter=adapter, device=device, grad_ckpt=False)
    n = t.export(path, with_lora=True)
    del t
    return n


def run_sets(sampler, chat, tag, sets, workers=24, limit_users=None, setting=None):
    setting = dict(EVAL_SETTING, **(setting or {}))
    agent = LocalAgent(sampler, chat, max_model_len=MAX_MODEL_LEN, seed=0, label=tag, keep_tokens=False, **setting)
    jobs = []
    if "full" in sets:
        jobs += [("full", s, world(s)) for s in FULL_SEEDS]
    if "short" in sets:
        jobs += [("short", s, short_world(world(s), SHORT_EVAL_STARTS)) for s in SHORT_EVAL_SEEDS]
    if limit_users:
        jobs = [(k, s, dict(w, users=w["users"][:limit_users], n_users=limit_users)) for k, s, w in jobs]
    res, lock = {}, threading.Lock()

    def one(job):
        kind, s, w = job
        t0 = time.time()
        env = Env(w)
        env.run(agent, workers=workers)
        recs = env.records()
        save_records(os.path.join(OUT, "eval", tag, kind, f"seed{s}"), recs)
        with lock:
            res[(kind, s)] = (w, recs, time.time() - t0)
        status(f"eval:{tag}", done=[f"{k}@{x}" for k, x in res])

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(one, jobs))
    out = {"tag": tag, "setting": setting, "sets": {}}
    for kind in ("full", "short"):
        ks = [(k, s) for (k, s) in res if k == kind]
        if not ks:
            continue
        rows = [r for key in ks for r in rows_for(res[key][0], res[key][1])]
        out["sets"][kind] = {"all": summary(rows), "by_seed": {str(s): summary(rows_for(res[(kind, s)][0],
                                                                                        res[(kind, s)][1]), n_boot=500)
                                                               for _, s in ks},
                             "seconds": {str(s): round(res[(kind, s)][2], 1) for _, s in ks}}
        if kind == "short":
            out["sets"][kind]["needs_first"] = {str(s): needs_first_rate(res[(kind, s)][0]) for _, s in ks}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--sets", default="full,short")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit-users", type=int)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    args = ap.parse_args(argv)
    t0 = time.time()
    status(f"eval:{args.tag}", stage="start")
    chat = make_chat()
    proc, boot = serve.start(MODEL, os.path.join(OUT, "logs", f"vllm_eval_{args.tag}.log"), gpu_util=args.gpu_util)
    try:
        sampler = VLLMClient(f"http://127.0.0.1:{serve.PORT}", "policy")
        if args.adapter:
            merge_adapter_to(SHM, args.adapter)
            sampler.update_weights(SHM)
        out = run_sets(sampler, chat, args.tag, args.sets.split(","), args.workers, args.limit_users)
        out.update({"adapter": args.adapter, "vllm_boot_seconds": boot, "seconds": round(time.time() - t0, 1),
                    "sampler_stats": sampler.stats})
        dump(os.path.join(OUT, "eval", args.tag, "metrics.json"), out)
        status(f"eval:{args.tag}", stage="done", seconds=out["seconds"])
    finally:
        serve.stop(proc)


if __name__ == "__main__":
    main()
