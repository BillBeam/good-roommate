"""试跑在机器上的公共设置：路径、模型、模板、记录的读写、指标。"""

import gzip
import json
import os
import time

from ..agent_llm import tools
from ..grading import QUESTION_COST
from .metrics import session_row, summarize_rows
from .qwen_chat import QwenChat

WORK = os.environ.get("M2P_WORK", "/root/work")
MODEL = os.environ.get("M2P_MODEL", os.path.join(WORK, "models", "Qwen3.5-4B"))
OUT = os.environ.get("M2P_OUT", os.path.join(WORK, "out"))           # 定期同步到仓库（不含检查点）
CKPT = os.environ.get("M2P_CKPT", os.path.join(WORK, "ckpt"))        # 本机检查点
EXPORT = os.path.join(OUT, "ckpt_export")                            # 最近一次检查点的副本，中转存成 artifact
SHM = "/dev/shm/m2p_weights.safetensors"
FULL_SEEDS = [1, 2, 3]                 # 全程评测：M1 第二版的三个世界
SHORT_EVAL_SEEDS = [201, 202]          # 短程评测
SHORT_EVAL_STARTS = [0, 2, 4, 6]
GRPO_SEEDS = [301, 302]                # GRPO 训练用的短程轨迹
TEACHER_SEEDS = [101, 102, 103, 104]   # 老师轨迹（SFT）
EVAL_SETTING = {"temperature": 0.0, "think_budget": 4096, "answer_cap": 8192}   # 与体检口径一致
TRAIN_SETTING = {"temperature": 1.0, "think_budget": 4096, "answer_cap": 2048}
MAX_MODEL_LEN = 32768


def make_chat(model_path=MODEL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    return QwenChat(tok, tools(QUESTION_COST), enable_thinking=True)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def status(phase, **kw):
    dump(os.path.join(OUT, "status.json"), {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            "phase": phase, **kw})


def world(seed):
    p = f"runs/m1v2/seed{seed}/world.json"
    if os.path.exists(p):
        return load(p)
    from ..world import generate_world
    return generate_world(seed)


def save_records(d, recs):
    """记录里去掉逐 token 的段（太大），对话另存 transcripts.jsonl.gz。"""
    os.makedirs(d, exist_ok=True)
    plain, trans = [], []
    for r in recs:
        r = {k: v for k, v in r.items() if k != "segments"}
        t = r.pop("transcript", None)
        plain.append(r)
        if t is not None:
            trans.append({"uid": r["uid"], "idx": r["idx"], "messages": t})
    with open(os.path.join(d, "sessions.jsonl"), "w", encoding="utf-8") as f:
        for r in plain:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(os.path.join(d, "transcripts.jsonl.gz"), "wt", encoding="utf-8") as f:
        for t in trans:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def rows_for(world_obj, recs, group=None):
    """指标行：(组, 用户, 计数)。短程轨迹按原来的用户分组重抽样（同一个用户的几条窗口不独立）。"""
    sess = {(u["uid"], s["idx"]): s for u in world_obj["users"] for s in u["sessions"]}
    orig = {u["uid"]: u.get("orig_uid", u["uid"]) for u in world_obj["users"]}
    g = world_obj["seed"] if group is None else group
    return [(g, orig[r["uid"]], session_row(r, sess[(r["uid"], r["idx"])])) for r in recs]


def summary(rows, n_boot=2000):
    return summarize_rows(rows, n_boot=n_boot)
