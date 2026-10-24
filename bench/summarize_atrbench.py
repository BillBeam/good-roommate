"""把 ATRBench 的判分结果抽成只含数字和编号的摘要（不含任何基准文本），写入 runs/m1/bench/。"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.environ.get("ATR_OUTPUTS_ROOT", os.path.join(ROOT, "external", "atrbench_outputs"))
PRICES = {  # 美元 / 百万 tokens，来自 runs/m1/pricing 下的 xAI 官方查价快照
    "grok-4.20-0309-non-reasoning": (1.25, 0.20, 2.50),
}


def main(dest=os.path.join(ROOT, "runs", "m1", "bench", "atrbench_summary.json")):
    cells = []
    for ev in sorted(glob.glob(os.path.join(OUT_ROOT, "*", "_oracle", "*", "eval.json"))):
        cell = os.path.dirname(ev)
        e = json.load(open(ev))
        m = json.load(open(os.path.join(cell, "cell_manifest.json")))
        sess = [{"session_id": r["session_id"], "domain": r["domain"], "check_type": r["check_type"],
                 "task_success": r["task_success"]} for r in e["session_results"]]
        cells.append({"persona_id": m["persona_id"], "variant": m["variant"], "model": m["model"], "seed": m["seed"],
                      "n_test_sessions": len(sess), "n_success": sum(s["task_success"] for s in sess),
                      "accuracy": sum(s["task_success"] for s in sess) / len(sess), "sessions": sess})
    tok = {}
    for f in sorted(glob.glob(os.path.join(OUT_ROOT, "_summary", "evaluator_*.json"))):
        tok = json.load(open(f))
    cost = 0.0
    for row in tok.get("rows", []):
        pin, pcache, pout = PRICES[row["model"]]
        u = row["token_usage"]["agent"]
        cost += ((u["prompt_tokens"] - u["cached_tokens"]) * pin + u["cached_tokens"] * pcache
                 + u["completion_tokens"] * pout) / 1e6
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({"benchmark": "ATRBench", "commit": "a2cfefec1832804caf143b230de3a6be86c698b7",
               "license": "代码 Apache-2.0；数据 CC BY 4.0（派生自 NVIDIA Nemotron-Personas-USA）",
               "estimated_cost_usd": cost,
               "cost_note": "按 token 数与 xAI 官方单价估算（ATRBench 的请求不经过本项目的计费账本）",
               "cells": cells, "evaluator_summary_raw": _strip(tok)}, open(dest, "w"), ensure_ascii=False, indent=1)
    print(dest)


def _strip(d):
    # 只保留数字类字段，避免把基准内容带进仓库
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if isinstance(v, (int, float, bool, dict, list)) or k in ("cell_id", "model", "variant", "persona_id")}
    if isinstance(d, list):
        return [_strip(x) for x in d]
    return d


if __name__ == "__main__":
    main(*sys.argv[1:])
