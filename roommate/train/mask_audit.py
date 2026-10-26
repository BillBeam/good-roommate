"""自检第 7 项的诊断（不是自检）：用和自检相同的设置多采样一些轨迹，把原规则会标出的计入 loss 的片段原文全部存下来，
并逐 token 核对它们是不是采样端生成的。用来先看清被标出的是什么，再决定修训练数据处理还是修检查。

    python -m roommate.train.mask_audit --reps 4

结果：OUT/mask_audit/flagged.json（每处：原因、匹配到的文字、上下文、是否全是采样端生成的）和 summary.json。
"""

import argparse
import os
import time

from . import serve
from .common import MODEL, OUT, dump, make_chat, world
from .sampler import VLLMClient
from .selfcheck import flagged_spans, loss_not_from_sampler, rollouts
from .shortrange import short_world


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args(argv)
    d = os.path.join(OUT, "mask_audit")
    os.makedirs(d, exist_ok=True)
    chat = make_chat()
    proc, boot = serve.start(MODEL, os.path.join(OUT, "logs", "vllm_mask_audit.log"), gpu_util=0.80)
    try:
        sampler = VLLMClient(f"http://127.0.0.1:{serve.PORT}", "policy")
        sw = short_world(world(301), starts=[0, 3])
        t0 = time.time()
        trajs = rollouts(sampler, chat, sw["users"][:12], sw, n_rep=args.reps, seed=17)   # 自检用 seed=7、1 次
        flags, spans, not_gen, segs = [], 0, 0, 0
        for recs in trajs:
            for r in recs:
                for s in r.get("segments", []):
                    segs += 1
                    not_gen += len(loss_not_from_sampler(s, chat.stop_ids))
                    f, n = flagged_spans(s, r["transcript"], chat)
                    spans += n
                    flags += [dict(x, uid=r["uid"], idx=r["idx"]) for x in f]
        dump(os.path.join(d, "flagged.json"), flags)
        dump(os.path.join(d, "summary.json"), {
            "trajectories": len(trajs), "segments": segs, "loss_spans": spans, "flagged": len(flags),
            "flagged_by_reason": {k: sum(f["why"] == k for f in flags) for k in ("template_marker", "tool_text", "user_text")},
            "flagged_not_all_from_sampler": sum(not f["all_from_sampler"] for f in flags),
            "loss_tokens_not_from_sampler": not_gen, "seconds": round(time.time() - t0, 1), "vllm_boot_s": boot})
        print("flagged", len(flags), "of", spans, "spans; not from sampler:", not_gen)
    finally:
        serve.stop(proc)


if __name__ == "__main__":
    main()
