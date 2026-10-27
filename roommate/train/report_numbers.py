"""试跑报告里的全部数字：只读仓库里的日志（runs/m2pilot/ 与 runs/m1v2/、runs/m2/），算好写进
runs/m2pilot/report/numbers.json，图画进 reports/m2pilot_figs/。报告里引用的每个数都出自这里。

    python -m roommate.train.report_numbers
"""

import glob
import gzip
import json
import os
from collections import defaultdict

from .metrics import replay_told, session_row, summarize_rows
from .shortrange import needs_first_rate, short_world

ROOT = "runs/m2pilot"
M = os.path.join(ROOT, "machine")
FIG = "reports/m2pilot_figs"
N_BOOT = 2000
BASELINES = ["llm_deepseek-v4-pro", "llm_deepseek-flash", "llm_claude-haiku-5-5", "llm_claude-sonnet-5-5",
             "llm_claude-opus-5-5", "llm_claude-fable-5-1", "ideal", "remember_no_update", "ask_all"]
TAGS = ["pre", "sft", "grpo1", "grpo2"]


def jl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def world(seed):
    p = f"runs/m1v2/seed{seed}/world.json"
    if os.path.exists(p):
        return jload(p)
    from ..world import generate_world
    return generate_world(seed)


# ---------- 联合基线表（全程，种子 1、2、3） ----------

def baseline_rows(policy, seeds=(1, 2, 3)):
    rows = []
    for s in seeds:
        w = world(s)
        plans = {u["uid"]: u for u in w["users"]}
        p = f"runs/m1v2/seed{s}/policies/{policy}/sessions.jsonl"
        if not os.path.exists(p):
            return None
        by = defaultdict(list)
        for r in jl(p):
            by[r["uid"]].append(r)
        for uid, rs in by.items():
            info = replay_told(plans[uid], rs, w["config"]["repeat_mode"])
            sess = {x["idx"]: x for x in plans[uid]["sessions"]}
            rows += [(s, uid, session_row(r, sess[r["idx"]], info[r["idx"]])) for r in rs if r["idx"] in info]
    return rows


def eval_rows(tag, kind, seeds):
    rows = []
    for s in seeds:
        p = os.path.join(M, "eval", tag, kind, f"seed{s}", "sessions.jsonl")
        if not os.path.exists(p):
            return None
        w = world(s) if kind == "full" else short_world(world(s), [0, 2, 4, 6])
        sess = {(u["uid"], x["idx"]): x for u in w["users"] for x in u["sessions"]}
        orig = {u["uid"]: u.get("orig_uid", u["uid"]) for u in w["users"]}
        rows += [(s, orig[r["uid"]], session_row(r, sess[(r["uid"], r["idx"])])) for r in jl(p)]
    return rows


def tijian_subset():
    """体检（硅基流动，思考开）中期结论用到的那 78 次会话：只用缓存重建，得到（种子, 用户, 第几次）。"""
    from .. import m2
    from ..llm import Ledger, LLMClient
    m2.load_prices()
    c = LLMClient(os.path.join(m2.OUT, "llm_cache"), Ledger(None, 0), offline=True)
    keys, recs = set(), []
    for s in (1, 2, 3):
        for r in m2.run_job(c, ("base", "think", s, 0), cache_only=True):
            if r["domain"] != "util":
                keys.add((s, r["uid"], r["idx"]))
            recs.append((s, r))
    return keys, recs


def tijian_compare(tag="pre"):
    """起点分数和体检比：只比体检做过的那 78 次会话（种子, 用户, 第几次）。
    体检在硅基流动上跑，本次在本地 vLLM 上跑；两边都是思考开、温度 0、思考上限 4096、回答上限 8192。"""
    keys, recs = tijian_subset()
    hc = []
    for s in (1, 2, 3):
        w = world(s)
        plans = {u["uid"]: u for u in w["users"]}
        by = defaultdict(list)
        for s2, r in recs:
            if s2 == s:
                by[r["uid"]].append(r)
        for uid, rs in by.items():
            info = replay_told(plans[uid], rs, w["config"]["repeat_mode"])
            sess = {x["idx"]: x for x in plans[uid]["sessions"]}
            hc += [(s, uid, session_row(r, sess[r["idx"]], info[r["idx"]])) for r in rs
                   if (s, uid, r["idx"]) in keys]
    ours = []
    for s in (1, 2, 3):
        w = world(s)
        sess = {(u["uid"], x["idx"]): x for u in w["users"] for x in u["sessions"]}
        ours += [(s, r["uid"], session_row(r, sess[(r["uid"], r["idx"])]))
                 for r in jl(os.path.join(M, "eval", tag, "full", f"seed{s}", "sessions.jsonl"))
                 if (s, r["uid"], r["idx"]) in keys]
    # 同一批会话的配对差（本次 − 体检），按用户重抽样
    import random
    a, b = defaultdict(float), defaultdict(float)
    n = defaultdict(int)
    for s, uid, row in hc:
        a[(s, uid)] += row["score"]
        n[(s, uid)] += row["pers"]
    for s, uid, row in ours:
        b[(s, uid)] += row["score"]
    users = sorted(n)
    diff = lambda us: sum(b[u] - a[u] for u in us) / sum(n[u] for u in us)
    rng = random.Random(0)
    boots = sorted(diff([rng.choice(users) for _ in users]) for _ in range(N_BOOT))
    return {"n_keys": len(keys), "tijian_siliconflow": summarize_rows(hc, n_boot=N_BOOT),
            f"{tag}_local_vllm": summarize_rows(ours, n_boot=N_BOOT),
            "paired_diff_mean_score": {"point": diff(users), "ci95": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]],
                                       "n_users": len(users)}}


def teacher_numbers(teacher_dir=os.path.join(ROOT, "teacher"), seeds=(101, 102, 103, 104)):
    """老师轨迹：每条规则（1–4、6）筛掉多少，三个阶段的捷径①、②比例；第 5 条（模板）的数取自 SFT 的 data_stats.json。"""
    from .sft import rule_drop, shortcut_rates, teacher_rows
    if not all(os.path.exists(os.path.join(teacher_dir, f"seed{s}", "world.json")) for s in seeds):
        return None
    dropped, stage = defaultdict(int), {"all": [], "after_rule4": [], "after_rule6": []}
    examples = []
    for s in seeds:
        for r, sess, info in teacher_rows(teacher_dir, s):
            row = session_row(r, sess, info)
            stage["all"].append((s, r["uid"], row))
            why = rule_drop(r, row)
            if why in (None, "6_捷径①"):
                stage["after_rule4"].append((s, r["uid"], row))
            if why is None:
                stage["after_rule6"].append((s, r["uid"], row))
            else:
                dropped[why] += 1
                if why == "6_捷径①":
                    examples.append([s, r["uid"], r["idx"], r["week"]])
    out = {"sessions": len(stage["all"]), "dropped_rules_1_4_6": dict(dropped), "rule6_sessions": examples,
           "shortcuts": {k: shortcut_rates(v) for k, v in stage.items()}}
    ds = os.path.join(M, "sft", "data_stats.json")
    if os.path.exists(ds):
        out["sft_data_stats"] = jload(ds)
    return out


def credit_mix(groups_file):
    """GRPO 里第 1 次见面的优势被第 2 次的得分带偏的程度（讲解 4、报告"后续改进"的对照实验 A 的基线）。
    第 1 次没选对 = 第 1 次得分 ≤ 0（选错或没选；选对至少得 1 − 0.15 × 提问数，问 7 次以上才会 ≤ 0）。
    同一步记了两次（续跑演练）只取最后一次。"""
    if not os.path.exists(groups_file):
        return None
    last = {}
    for r in jl(groups_file):
        last[r["step"]] = r
    c = defaultdict(int)
    for st in sorted(last):
        for g in last[st]["groups"]:
            m1 = sum(x[0] for x in g["rewards"]) / len(g["rewards"])
            for i in range(len(g["rewards"])):
                r1, a1, q1 = g["rewards"][i][0], g["adv"][i][0], g["questions"][i][0]
                if r1 <= 0:
                    c["s1_not_right"] += 1
                    c["s1_not_right_adv_pos"] += a1 > 0
                    if q1 == 0:
                        c["s1_guess_not_right"] += 1
                        c["s1_guess_not_right_adv_pos"] += a1 > 0
                else:
                    c["s1_right"] += 1
                    c["s1_right_adv_neg"] += a1 < 0
                    # 第 1 次得分不低于同组第 1 次的平均，优势却是负的：负号来自第 2 次的得分
                    c["s1_right_adv_neg_from_s2"] += a1 < 0 and r1 >= m1
    c = dict(c)
    c["rate_not_right_adv_pos"] = c["s1_not_right_adv_pos"] / c["s1_not_right"] if c.get("s1_not_right") else None
    c["rate_guess_not_right_adv_pos"] = (c["s1_guess_not_right_adv_pos"] / c["s1_guess_not_right"]
                                         if c.get("s1_guess_not_right") else None)
    c["rate_right_adv_neg"] = c["s1_right_adv_neg"] / c["s1_right"] if c.get("s1_right") else None
    return c


def _ts(x):
    from datetime import datetime
    return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def outage_numbers(relay=os.path.join(ROOT, "relay")):
    """这次断连：几点断的、断了多久、这段时间旧机器的费用、重启前后的检查结果、释放与换机。
    只读中转的记录：relay_runs.jsonl（每轮摘要）、logs/<轮次>.jsonl（逐条日志）、machines_past.jsonl（释放过的机器）。"""
    runs_p, logs_d = os.path.join(relay, "relay_runs.jsonl"), os.path.join(relay, "logs")
    if not (os.path.exists(runs_p) and os.path.isdir(logs_d)):
        return None
    runs = jl(runs_p)
    causes_p = os.path.join(relay, "outage_causes.json")
    causes = {k: v for k, v in jload(causes_p).items() if not k.startswith("_")} if os.path.exists(causes_p) else {}
    events = []
    for f in sorted(glob.glob(os.path.join(logs_d, "*.jsonl"))):
        rid = os.path.basename(f)[:-6]
        events += [dict(e, run=rid) for e in jl(f)]
    events.sort(key=lambda e: e["t"])
    # 只算平台/网络造成的断连；中转自身原因、做镜像等核实过不算的，单列
    def platform(e):
        return causes.get(e["run"], {}).get("counts_as_platform_outage", True)
    ssh_fail = [e for e in events if (e.get("rc") == 255 or e.get("msg") in ("job_status_empty", "unreachable"))
                and platform(e)]
    if not ssh_fail:
        return None
    first_fail = ssh_fail[0]
    # 断之前最后一轮全部动作成功、且结束于第一次失败之前的中转
    ok_runs = [r for r in runs if r.get("actions") and all(a["ok"] for a in r["actions"]) and not r.get("error")
               and _ts(r["finished"]) < _ts(first_fail["t"])]
    last_ok = max(ok_runs, key=lambda r: _ts(r["finished"]))
    # 这一轮里失败的动作是从哪一刻开始做的（失败行之前最近的一条 action 行）
    started = max((e for e in events if e["run"] == first_fail["run"] and e.get("msg") == "action"
                   and e["t"] <= first_fail["t"]), key=lambda e: e["t"])
    reboot = next((e for e in events if e.get("msg") == "reboot"), None)
    after = next((e for e in events if e.get("msg") == "rebooted"), None)
    after_fail = next((e for e in events if reboot and e["t"] > reboot["t"] and e.get("rc") == 255), None)
    past = [p for p in jl(os.path.join(relay, "machines_past.jsonl"))] if os.path.exists(
        os.path.join(relay, "machines_past.jsonl")) else []
    old = past[0] if past else None
    out = {"last_ok_round": {"run": last_ok["run_id"], "finished": last_ok["finished"]},
           "first_failed_action": {"run": first_fail["run"], "action_started": started["t"], "failed_at": first_fail["t"],
                                   "what": started.get("do")},
           "ssh_failures_logged": len(ssh_fail),
           "not_counted": {k: {"cause": v["cause"], "detail": v.get("detail")} for k, v in causes.items()
                           if not v.get("counts_as_platform_outage", True)},
           "reboot": reboot and {"at": reboot["t"], "ssh_after": after and after.get("out", "").split("\n")[0],
                                 "ssh_after_at": after and after["t"],
                                 "next_failure_at": after_fail and after_fail["t"]}}
    if old:
        lo, hi = _ts(last_ok["finished"]), _ts(started["t"])      # 断开发生在这两个时刻之间
        rel = _ts(old["released"])
        out["old_machine"] = {"name": old["name"], "gpu": old["gpu"], "hourly": old["hourly"], "released": old["released"],
                              "outage_hours_max": round((rel - lo) / 3600, 2), "outage_hours_min": round((rel - hi) / 3600, 2),
                              "outage_cost_max": round((rel - lo) / 3600 * old["hourly"], 2),
                              "outage_cost_min": round((rel - hi) / 3600 * old["hourly"], 2),
                              "total_hours": round((rel - _ts(old["created"])) / 3600, 2),
                              "total_cost": round((rel - _ts(old["created"])) / 3600 * old["hourly"], 2)}
    return out


def main():
    out = {}
    out["outage"] = outage_numbers()
    out["teacher"] = teacher_numbers()
    # 联合基线表
    joint = {}
    for pol in BASELINES:
        rows = baseline_rows(pol)
        if rows:
            joint[pol] = summarize_rows(rows, n_boot=N_BOOT)
    for tag in TAGS:
        rows = eval_rows(tag, "full", (1, 2, 3))
        if rows:
            joint[f"qwen3.5-4b_{tag}"] = summarize_rows(rows, n_boot=N_BOOT)
    out["joint_full"] = joint
    # 短程评测
    short = {}
    for tag in TAGS:
        rows = eval_rows(tag, "short", (201, 202))
        if rows:
            short[tag] = summarize_rows(rows, n_boot=N_BOOT)
    out["short"] = short
    out["tijian"] = tijian_compare("pre")
    out["credit_mix"] = {f"grpo_s{k}": credit_mix(os.path.join(M, f"grpo_s{k}", "groups.jsonl")) for k in (1, 2)}
    sd = os.path.join(M, "seed2_decision.json")
    if os.path.exists(os.path.join(M, "eval", "grpo1", "full")):
        from .seed2_rule import decide
        out["seed2_rule"] = decide(M, n_boot=N_BOOT, write=False)
        out["seed2_rule"]["machine_decision_file"] = jload(sd) if os.path.exists(sd) else None
    out["needs_first"] = {str(s): needs_first_rate(short_world(world(s), [0, 2, 4, 6])) for s in (201, 202)}
    out["needs_first_train_pool"] = {str(s): needs_first_rate(short_world(world(s))) for s in (301, 302)}
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("written", os.path.join(ROOT, "report", "numbers.json"))


if __name__ == "__main__":
    main()
