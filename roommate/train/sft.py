"""SFT：用老师轨迹（DeepSeek V4 Pro 在新世界里的对话）训练 LoRA。

    python -m roommate.train.sft --teacher runs/m2pilot/teacher

筛选规则（逐次会话，全部满足才留下；每条规则筛掉多少写进 OUT/sft/data_stats.json）：
1. 正常结束：没有出错（error 为空），做了选择，没有用满 14 轮；
2. 选对了（判分 success）；
3. 没有格式问题（工具调用都能解析、没有只输出文字的轮次，因而整次会话没有插入提醒，是一整段）；
4. 没有重问：每个问题都没有被用户答"之前说过了"；
5. 套 Qwen3.5 模板后不超过 MAX_LEN 个 token，并且逐轮前缀一致（见 qwen_chat.assistant_spans）。
读每个文件之前都过 data_policy.assert_trainable（Claude 的输出不能进训练数据）。
"""

import argparse
import gzip
import json
import os
import random
import time

from ..agent_llm import system_prompt
from ..data_policy import assert_trainable
from ..world import DEFAULT_CONFIG
from .common import CKPT, EXPORT, MODEL, OUT, TEACHER_SEEDS, append_jsonl, dump, make_chat, status
from .qwen_chat import assistant_spans

MAX_LEN = 16384


def teacher_messages(transcript, config):
    sysmsg = {"role": "system", "content": system_prompt(config["question_cost"], config["repeat_mode"])}
    msgs = [sysmsg]
    for m in transcript:
        m = dict(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"]["arguments"]
                calls.append({"id": c.get("id"), "type": "function",
                              "function": {"name": c["function"]["name"],
                                           "arguments": json.loads(args) if isinstance(args, str) else args}})
            m["tool_calls"] = calls
        msgs.append(m)
    return msgs


def load_teacher(chat, teacher_dir, seeds, config=DEFAULT_CONFIG):
    stats = {"sessions": 0, "kept": 0, "dropped": {}, "tokens": 0, "by_seed": {}}
    data = []

    def drop(reason):
        stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1

    for seed in seeds:
        d = os.path.join(teacher_dir, f"seed{seed}", "policies", "llm_deepseek-v4-pro")
        sp, tp = os.path.join(d, "sessions.jsonl"), os.path.join(d, "transcripts.jsonl.gz")
        assert_trainable(sp)
        assert_trainable(tp)
        recs = {(r["uid"], r["idx"]): r for r in map(json.loads, open(sp, encoding="utf-8"))}
        with gzip.open(tp, "rt", encoding="utf-8") as f:
            trans = {(t["uid"], t["idx"]): t["messages"] for t in map(json.loads, f)}
        kept = 0
        for key, r in sorted(recs.items()):
            stats["sessions"] += 1
            if r.get("error") or r["chosen"] is None or r.get("hit_max_turns"):
                drop("1_未正常结束")
                continue
            if not r["grade"]["success"]:
                drop("2_选错")
                continue
            if r.get("format_issues"):
                drop("3_格式问题")
                continue
            if any(st == "told_before" for q in r["questions"] for st in q["status"].values()):
                drop("4_重问")
                continue
            try:
                ids, mask = assistant_spans(chat, teacher_messages(trans[key], config))
            except Exception:
                drop("5_模板前缀不一致")
                continue
            if len(ids) > MAX_LEN:
                drop("5_超长")
                continue
            data.append({"seed": seed, "uid": r["uid"], "idx": r["idx"], "domain": r["domain"], "ids": ids,
                         "mask": mask})
            stats["tokens"] += len(ids)
            kept += 1
        stats["by_seed"][str(seed)] = kept
    stats["kept"] = len(data)
    stats["loss_tokens"] = sum(sum(x["mask"]) for x in data)
    return data, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/m2pilot/teacher")
    ap.add_argument("--seeds", default=",".join(map(str, TEACHER_SEEDS)))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args(argv)
    from .trainer import Trainer, cosine_lr
    t_start = time.time()
    chat = make_chat()
    data, stats = load_teacher(chat, args.teacher, [int(s) for s in args.seeds.split(",")])
    rng = random.Random(0)
    users = sorted({(x["seed"], x["uid"]) for x in data})
    rng.shuffle(users)
    val_users = set(users[:max(1, int(len(users) * args.val_frac))])       # 按用户留出验证集
    val = [x for x in data if (x["seed"], x["uid"]) in val_users]
    train = [x for x in data if (x["seed"], x["uid"]) not in val_users]
    stats.update({"train_sequences": len(train), "val_sequences": len(val), "val_users": len(val_users)})
    dump(os.path.join(OUT, "sft", "data_stats.json"), stats)
    steps_per_epoch = (len(train) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    ck = os.path.join(CKPT, "sft")
    latest = os.path.join(ck, "latest")
    if args.resume and os.path.exists(os.path.join(latest, "trainer_state.pt")):
        tr = Trainer(MODEL, lr=args.lr, adapter=latest)
        extra = tr.load_state(latest)
        start = extra["step"]
    else:
        tr = Trainer(MODEL, lr=args.lr)
        start = 0
    log = os.path.join(OUT, "sft", "train_log.jsonl")

    def val_loss():
        tr.model.eval()
        tot = n = 0.0
        for x in val:
            lp = tr.logprobs(x["ids"], x["mask"])
            tot += float(-lp.sum())
            n += lp.numel()
        return tot / max(1, n)

    if start == 0:
        append_jsonl(log, {"step": 0, "val_loss": val_loss(), "t": time.time() - t_start})
    for step in range(start, total):
        if args.max_steps and step >= args.max_steps:
            break
        ep, k = divmod(step, steps_per_epoch)
        order = list(range(len(train)))
        random.Random(1000 + ep).shuffle(order)
        batch = [train[i] for i in order[k * args.batch:(k + 1) * args.batch]]
        lr = cosine_lr(step, total, args.lr)
        tr.set_lr(lr)
        st = tr.train_step(batch, "sft")
        row = {"step": step + 1, "epoch": ep + 1, "lr": lr, "loss": st["loss"], "tokens": st["tokens"],
               "grad_norm": st["grad_norm"], "seconds": st["seconds"], "t": time.time() - t_start}
        if (step + 1) % 10 == 0 or step + 1 == total:
            row["val_loss"] = val_loss()
        append_jsonl(log, row)
        status("sft", step=step + 1, total=total, loss=st["loss"])
        if (step + 1) % args.save_every == 0 or step + 1 == total:
            tr.save(latest, {"step": step + 1, "total": total})
            tr.save(os.path.join(EXPORT, "sft_latest"), {"step": step + 1, "total": total})
    tr.save(os.path.join(ck, "final"), {"step": total, "total": total})
    tr.save(os.path.join(EXPORT, "sft_final"), {"step": total, "total": total})
    dump(os.path.join(OUT, "sft", "done.json"), {"steps": total, "seconds": round(time.time() - t_start, 1),
                                                 **{k: stats[k] for k in ("kept", "train_sequences", "val_sequences")}})
    status("sft", stage="done")


if __name__ == "__main__":
    main()
