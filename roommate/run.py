"""一条命令跑完 M1：生成世界 → 参照策略 → 判分自检 → 大模型基线 → 复现与恢复检查 → 报告。

    python -m roommate.run                 # 默认：种子 0，24 个用户 × 8 次会话，三个 xAI 模型
    python -m roommate.run --offline       # 只用缓存（不联网、不花钱），结果应与首次运行逐字一致
    python -m roommate.run --skip-llm      # 只跑脚本策略与自检

固定种子时：世界与脚本策略的结果完全由代码决定；大模型的每个请求都按内容哈希缓存在
runs/m1/llm_cache/，重跑时全部命中缓存，因此整份结果可逐字复现。
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import urllib.request

from . import report
from .agent_llm import LLMAgent
from .classifier_gold import GOLD
from .env import Env
from .grading import satisfies, summarize
from .llm import BudgetExceeded, Ledger, XAIClient, make_classifier, USERSIM_MODEL
from .policies import SCRIPTED
from .world import generate_world, world_hash, world_stats

DEFAULT_MODELS = ["grok-4.20-0309-non-reasoning", "grok-4.3", "grok-4.7"]


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def records_hash(recs):
    return hashlib.sha256(json.dumps(recs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def no_classify(q):
    raise AssertionError("脚本策略不应走自由文本提问")


# ---------- 判分自检 ----------

def self_check(world, results):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    n_one, n_total, trap_ok, trap_total = 0, 0, 0, 0
    for u in world["users"]:
        for s in u["sessions"]:
            n_total += 1
            if s["domain"] == "util":
                n_one += sum(o["id"] == s["correct"] for o in s["options"]) == 1
                continue
            ok_ids = [o["id"] for o in s["options"] if satisfies(o, s["eff_state"], s["domain"])]
            n_one += ok_ids == [s["correct"]]
            for role, st in (("stale_trap", s["stale_state"]), ("temp_trap", s["temp_trap_state"])):
                if st is None:
                    continue
                trap_total += 1
                ids = [i for i, r in s["roles"].items() if r == role]
                o = next(o for o in s["options"] if o["id"] == ids[0]) if ids else None
                trap_ok += bool(o) and satisfies(o, st, s["domain"]) and not satisfies(o, s["eff_state"], s["domain"])
    add("每个任务恰好一个正确选项", n_one == n_total, f"{n_one}/{n_total}")
    add("陷阱选项满足过时/被污染的状态、违反当前真实状态", trap_ok == trap_total, f"{trap_ok}/{trap_total}")

    o, z = results["oracle"], results["nothing"]
    add("标准策略（知道真实状态）得满分", o["mean_score"] == 1.0 and o["success_rate"] == 1.0,
        f"平均分 {o['mean_score']}，成功率 {o['success_rate']}，提问 {o['n_questions']} 次")
    add("什么都不做得零分", z["mean_score"] == 0.0 and z["success_rate"] == 0.0,
        f"平均分 {z['mean_score']}，成功率 {z['success_rate']}")
    a = results["ask_all"]
    add("每次都问一遍：只靠回答即可全部做对（回答信息充分）", a["success_rate"] == 1.0,
        f"成功率 {a['success_rate']}，平均每次会话提问 {a['avg_questions']:.3f} 次")
    r = results["remember_no_update"]
    add("只记不更新的策略会掉进过时记忆陷阱（陷阱有效）", (r["stale_error_rate"] or 0) > 0.5,
        f"过时记忆错误率 {r['stale_error_rate']:.3f}")
    rnd, maj = results["random"], results["majority"]
    add("随大流选项不明显优于随机（选项本身不泄露答案）",
        maj["followup_success_rate"] - rnd["followup_success_rate"] < 0.15,
        f"随大流 {maj['followup_success_rate']:.3f}，随机 {rnd['followup_success_rate']:.3f}")
    st = world_stats(world)
    add("规模：至少 20 个用户、每人至少 6 次会话", st["n_users"] >= 20 and st["n_sessions_per_user"] >= 6,
        f"{st['n_users']} 个用户 × {st['n_sessions_per_user']} 次会话")
    add("同时存在多项偏好事件、临时情况、无关会话",
        st["events_multi_attr"] > 0 and st["temp_situations"] > 0 and st["unrelated_sessions"] > 0,
        f"多项偏好事件 {st['events_multi_attr']}，临时情况 {st['temp_situations']}，无关会话 {st['unrelated_sessions']}")
    w2 = generate_world(world["seed"], world["n_users"], world["n_sessions"])
    add("同一种子重新生成的世界完全相同", world_hash(w2) == world_hash(world), world_hash(world))
    return {"all_pass": all(c["pass"] for c in checks), "checks": checks}


# ---------- 归类准确率 ----------

def eval_classifier(client):
    clf = make_classifier(client)
    rows, exact = [], 0
    for q, gold in GOLD:
        pred = clf(q)
        ok = set(pred) == set(gold)
        exact += ok
        rows.append({"question": q, "gold": gold, "pred": pred, "match": ok})
    return {"model": USERSIM_MODEL, "n": len(GOLD), "exact_set_accuracy": exact / len(GOLD), "items": rows}


# ---------- 主流程 ----------

def run_policy(world, policy, classify, workers=1):
    env = Env(world, classify)
    env.run(policy, workers=workers)
    return env.records()


def split_transcripts(recs):
    plain, trans = [], []
    for r in recs:
        r = dict(r)
        t = r.pop("transcript", None)
        plain.append(r)
        if t is not None:
            trans.append({"uid": r["uid"], "idx": r["idx"], "messages": t})
    return plain, trans


def fetch_pricing(out):
    pdir = os.path.join(out, "pricing")
    os.makedirs(pdir, exist_ok=True)
    if any(f.startswith("xai_language_models_") for f in os.listdir(pdir)):
        return  # 已有查价快照，保留首次查价的日期
    path = os.path.join(pdir, f"xai_language_models_{dt.date.today().isoformat()}.json")
    req = urllib.request.Request("https://api.x.ai/v1/language-models",
                                 headers={"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    dump(path, {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "source": "GET https://api.x.ai/v1/language-models（xAI 官方接口）",
                "unit_note": "价格字段单位：1e-4 美元 / 百万 tokens（如 12500 = 1.25 美元/百万 tokens）；"
                             "响应里的 cost_in_usd_ticks 单位为 1e-10 美元",
                "models": data["models"]})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--users", type=int, default=24)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--budget", type=float, default=25.0, help="累计花费上限（美元），到达即停止")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="runs/m1")
    ap.add_argument("--report", default="reports/m1-baseline.md")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--resume-k", type=int, default=4, help="中断—恢复检查在第几个会话边界存档")
    args = ap.parse_args(argv)
    out = args.out
    os.makedirs(out, exist_ok=True)

    world = generate_world(args.seed, args.users, args.sessions)
    dump(os.path.join(out, "world.json"), world)
    dump(os.path.join(out, "world_stats.json"), dict(world_stats(world), world_hash=world_hash(world)))
    print("世界：", json.dumps(world_stats(world), ensure_ascii=False))

    # 参照策略
    results, rec_hash = {}, {}
    for name, P in SCRIPTED.items():
        recs = run_policy(world, P(), no_classify)
        results[name] = summarize(recs)
        rec_hash[name] = records_hash(recs)
        dump_jsonl(os.path.join(out, "policies", name, "sessions.jsonl"), recs)
        dump(os.path.join(out, "policies", name, "summary.json"), results[name])
        print(f"{name:20s} 平均分 {results[name]['mean_score']:.3f}")
    sc = self_check(world, results)
    dump(os.path.join(out, "selfcheck.json"), sc)
    print("判分自检全部通过" if sc["all_pass"] else "判分自检有失败项！", [c["name"] for c in sc["checks"] if not c["pass"]])

    repro = {"world_hash": world_hash(world), "scripted": {}, "llm": {}}
    for name, P in SCRIPTED.items():
        h2 = records_hash(run_policy(generate_world(args.seed, args.users, args.sessions), P(), no_classify))
        repro["scripted"][name] = {"first": rec_hash[name], "rerun": h2, "identical": h2 == rec_hash[name]}

    models = [m for m in args.models.split(",") if m] if not args.skip_llm else []
    resume = {}
    if models:
        if not args.offline:
            fetch_pricing(out)
        ledger = Ledger(os.path.join(out, "spend_ledger.json"), args.budget)
        client = XAIClient(os.path.join(out, "llm_cache"), ledger, offline=args.offline)
        dump(os.path.join(out, "classifier_eval.json"), eval_classifier(client))
        classify = make_classifier(client)
        stopped = None
        for m in models:
            try:
                recs = run_policy(world, LLMAgent(client, m, seed=args.seed), classify, workers=args.workers)
            except BudgetExceeded as e:
                stopped = str(e)
                print("停止：", stopped)
                break
            plain, trans = split_transcripts(recs)
            key = f"llm:{m}"
            results[key] = summarize(plain)
            dump_jsonl(os.path.join(out, "policies", key.replace(":", "_"), "sessions.jsonl"), plain)
            dump_jsonl(os.path.join(out, "policies", key.replace(":", "_"), "transcripts.jsonl"), trans)
            dump(os.path.join(out, "policies", key.replace(":", "_"), "summary.json"), results[key])
            print(f"{key:20s} 平均分 {results[key]['mean_score']:.3f}  累计花费 {ledger.data['total_usd']:.2f} 美元")

            # 复现：只用缓存重跑一遍，逐字比较
            off = XAIClient(os.path.join(out, "llm_cache"), Ledger(None, 0), offline=True)
            recs2 = run_policy(world, LLMAgent(off, m, seed=args.seed), make_classifier(off), workers=args.workers)
            repro["llm"][key] = {"first": records_hash(recs), "rerun_from_cache": records_hash(recs2),
                                 "identical": records_hash(recs) == records_hash(recs2)}

            # 中断—恢复：跑到第 k 个会话边界存档，丢掉内存里的环境，从存档恢复后跑完
            with tempfile.TemporaryDirectory() as td:
                ck = os.path.join(td, "ck.json")
                env = Env(world, make_classifier(off))
                env.run(LLMAgent(off, m, seed=args.seed), until=args.resume_k, workers=args.workers)
                env.save(ck)
                size = os.path.getsize(ck)
                del env
                env2 = Env.load(ck, make_classifier(off))
                env2.run(LLMAgent(off, m, seed=args.seed), workers=args.workers)
                recs3 = env2.records()
            resume[key] = {"k": args.resume_k, "checkpoint_bytes": size, "uninterrupted": records_hash(recs),
                           "resumed": records_hash(recs3), "identical": records_hash(recs) == records_hash(recs3)}
        dump(os.path.join(out, "costs.json"), {"cumulative": ledger._load(), "this_run": ledger.session,
                                               "stopped": stopped})
    dump(os.path.join(out, "repro_check.json"), repro)
    if resume:
        dump(os.path.join(out, "resume_check.json"), resume)
    report.write(out, args.report)
    print("报告：", args.report)


if __name__ == "__main__":
    main()
