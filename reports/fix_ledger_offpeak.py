"""一次性的账本更正（2026-10-24）：第二版最初的 DeepSeek 计费一律按高峰价、只记美元。
这批调用全部发生在 2026-10-24（UTC 周六，北京时间周六晚到周日凌晨），按官方规则属于空闲时段，
所以按 token 数改用空闲价重算美元与人民币，并在账本里写下 DeepSeek 剩余部分的人民币上限。

运行：python reports/fix_ledger_offpeak.py <剩余部分的上限（元）> <从这些 用途 起算的条目，逗号分隔>
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LEDGER = os.path.join("runs", "m1v2", "spend_ledger.json")


def main():
    budget = float(sys.argv[1])
    remaining_keys = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
    snap = json.load(open(glob.glob(os.path.join("runs", "m1v2", "pricing", "deepseek_*.json"))[0], encoding="utf-8"))
    led = json.load(open(LEDGER, encoding="utf-8"))
    old_ds_usd = 0.0
    tot_cny = 0.0
    before = 0.0
    for k, v in led["by_model_role"].items():
        m = k.split("|")[0]
        if not m.startswith("deepseek"):
            continue
        p = snap["prices"][m]
        hit = v["cached_prompt_tokens"]
        miss = v["prompt_tokens"] - hit
        out = v["completion_tokens"]
        f = lambda q: (hit * q["input_cache_hit"] + miss * q["input_cache_miss"] + out * q["output"]) / 1e6
        old_ds_usd += v["cost_usd"]
        v["cost_usd_peak_recorded"] = v["cost_usd"]
        v["cost_usd"] = f(p["usd"]["off_peak"])
        v["cost_cny"] = f(p["cny"]["off_peak"])
        tot_cny += v["cost_cny"]
        if k not in remaining_keys:
            before += v["cost_cny"]
    new_ds_usd = sum(v["cost_usd"] for k, v in led["by_model_role"].items() if k.startswith("deepseek"))
    led["total_usd"] += new_ds_usd - old_ds_usd
    led["deepseek_cny_total"] = tot_cny
    led["deepseek_cny_budget_from"] = before
    led["deepseek_cny_limit"] = before + budget
    led["note"] = ("DeepSeek 的条目起初一律按高峰美元价记账（cost_usd_peak_recorded）；这些调用全部发生在 2026-10-24 UTC 周六，"
                   "属于空闲时段，已按 token 数改用空闲价重算美元与人民币。此后的调用按调用时刻取价。"
                   f"缩减口径之后的 DeepSeek 上限 {budget} 元，从 {before:.2f} 元算起"
                   f"（{', '.join(sorted(remaining_keys)) or '无'} 整条算作剩余部分）。")
    json.dump(led, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"total_usd": led["total_usd"], "deepseek_cny_total": tot_cny, "budget_from": before,
                      "limit": led["deepseek_cny_limit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
