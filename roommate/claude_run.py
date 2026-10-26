"""在 M1 第二版考场上补 Claude 系列的基线，和 DeepSeek 基线同一个考场、同一段提示、同一组工具、同一套判分。

    python -m roommate.claude_run                          # 全部阶段：pilot,estimate,full,repro,stats,report
    python -m roommate.claude_run --stages stats,report    # 只用已有记录重算统计、重写报告（不联网）
    python -m roommate.claude_run --offline                # 只用缓存（不花钱），结果应与首次运行逐字一致

阶段：
- pilot：每个模型（和 Opus 5.5 的"知道真实状态"对照组）先只跑种子 1 的前 PILOT_USERS 个用户的全部会话，
  实测每次会话的 token 和花费。请求照常进缓存，正式运行时这几个用户直接命中缓存，不重复花钱。
- estimate：按实测的每次会话花费估算全卡总价，按 BUDGET_USD 和优先级定下要跑哪些（plan.json）。
  优先级：Haiku、Sonnet、Opus 的种子 1、2、3 和 Opus 对照组（种子 1）必跑；Fable 5.1 先跑种子 1，钱够再跑种子 2、3；
  估算连种子 1 都放不下时，Fable 只跑种子 1 里放得下的前若干个用户。
- full：按 plan.json 的顺序跑。每次跑之前按已花的钱和这一组的估算再核一遍，放不下的跳过并记下原因。
- compact：把 Claude 的请求缓存压缩成 .jsonl.gz（仓库里只放压缩后的）。
- repro：只用缓存重跑全部 Claude 记录，检查与保存的记录逐字一致。
- stats / report：统计写进 runs/m1v2/claude/claude_stats.json；M1 第二版的 gates.json 与基线报告一起重算（DeepSeek 与 Claude 同表）；
  再写 reports/m1v2-claude-baseline.md。

花费：单独的账本 runs/m1v2/claude/spend_ledger.json，按官方价目快照实时累计，上限 BUDGET_USD 美元；
累计加上 RESERVE_USD（留给并发中、还没记账的请求）到上限就不再发新请求。
密钥只读环境变量 CLAUDE_EVAL_API_KEY。Claude 的输出只用于评测和分析，不进任何训练数据（见 roommate/data_policy.py）。
"""

import argparse
import datetime as dt
import json
import os

from . import agent_claude, data_policy, llm, run, stats
from .llm import ApiError, BudgetExceeded, Ledger, LLMClient

OUT = run.OUT
CDIR = os.path.join(OUT, "claude")
MODELS = list(run.CLAUDE_MODELS)          # 从小到大：Haiku 5.5、Sonnet 5.5、Opus 5.5、Fable 5.1
CORE = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
CONTROL = ("claude-opus-5-5", 1)
EXTRA = "claude-fable-5-1"
BUDGET_USD = 150.0
RESERVE_USD = 5.0
PILOT_SEED = 1
PILOT_USERS = 2
WORKERS = 12


def dump(path, obj):
    run.dump(path, obj)


def load(path):
    return run.load(path)


def load_prices():
    pp = os.path.join(OUT, "pricing")
    for fn in sorted(os.listdir(pp)):
        if fn.startswith("anthropic_"):
            llm.ANTHROPIC_PRICES.update(load(os.path.join(pp, fn))["prices"])


def make_client(offline):
    ledger = Ledger(os.path.join(CDIR, "spend_ledger.json"), BUDGET_USD, reserve_usd=RESERVE_USD)
    return LLMClient(os.path.join(OUT, "llm_cache"), ledger, offline=offline, blocked_providers=run.BLOCKED_PROVIDERS)


def key_of(m, ctrl):
    return f"llm:{m}" + (":true_memory" if ctrl else "")


def sub_world(seed, n_users=None):
    w = run.get_world(seed)
    if n_users is not None:
        w = dict(w, users=w["users"][:n_users], n_users=n_users)
    return w


def pilot_runs():
    return [(m, False) for m in MODELS] + [(CONTROL[0], True)]


# ---------- pilot ----------

def stage_pilot(client):
    res = {}
    for m, ctrl in pilot_runs():
        k = key_of(m, ctrl)
        try:
            recs = run.run_policy(sub_world(PILOT_SEED, PILOT_USERS), run.agent(client, m, PILOT_SEED, ctrl),
                                  workers=WORKERS)
        except (BudgetExceeded, ApiError) as e:
            print("试跑停止：", f"{type(e).__name__}: {e}"[:300])
            raise
        plain, trans = run.split_transcripts(recs)
        d = os.path.join(CDIR, "pilot", k.replace(":", "_"))
        run.dump_jsonl(os.path.join(d, "sessions.jsonl"), plain)
        run.dump_jsonl(os.path.join(d, "transcripts.jsonl"), trans)
        data_policy.mark_not_for_training(d)
        res[k] = session_costs(plain)
        print(f"试跑 {k:36s} 每次会话 {res[k]['cost_per_session']:.4f} 美元，"
              f"{res[k]['tokens_per_session']['output']:.0f} 输出 tokens，平均分 {res[k]['mean_score_pref']:.3f}")
    dump(os.path.join(CDIR, "pilot.json"), {"seed": PILOT_SEED, "n_users": PILOT_USERS, "runs": res})


def session_costs(recs):
    """逐次会话的实测 token 与花费（来自记录里的 usage，含缓存命中时原响应的 usage）。"""
    n = len(recs)
    by_user = {}
    tok = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "calls": 0}
    for r in recs:
        u = r["usage"]
        by_user[r["uid"]] = by_user.get(r["uid"], 0.0) + u["cost_usd"]
        tok["input"] += u["input_tokens"]
        tok["cache_write"] += u["cache_write_tokens"]
        tok["cache_read"] += u["cache_read_tokens"]
        tok["output"] += u["output_tokens"]
        tok["calls"] += u["calls"]
    pref = [r for r in recs if r["domain"] != "util"]
    per_user_sessions = {u: sum(r["uid"] == u for r in recs) for u in by_user}
    return {"n_sessions": n, "cost_total": sum(by_user.values()), "cost_per_session": sum(by_user.values()) / n,
            "cost_per_session_by_user": {u: c / per_user_sessions[u] for u, c in by_user.items()},
            "tokens_per_session": {k: v / n for k, v in tok.items()},
            "mean_score_pref": sum(r["grade"]["score"] for r in pref) / len(pref) if pref else None}


# ---------- estimate & plan ----------

def stage_estimate():
    pilot = load(os.path.join(CDIR, "pilot.json"))
    per = {k: v["cost_per_session"] for k, v in pilot["runs"].items()}
    per_hi = {k: max(v["cost_per_session_by_user"].values()) for k, v in pilot["runs"].items()}
    n_sess = {s: sum(len(u["sessions"]) for u in run.get_world(s)["users"]) for s in run.GATE_SEEDS}
    pilot_sess = sum(len(u["sessions"]) for u in sub_world(PILOT_SEED, PILOT_USERS)["users"])
    rows = []
    for m in MODELS:
        for s in run.GATE_SEEDS:
            rows.append((m, False, s))
    rows.append((CONTROL[0], True, CONTROL[1]))
    est = []
    for m, ctrl, s in rows:
        k = key_of(m, ctrl)
        n_new = n_sess[s] - (pilot_sess if s == PILOT_SEED else 0)
        est.append({"model": m, "control": ctrl, "seed": s, "sessions": n_sess[s], "sessions_not_cached": n_new,
                    "est_usd": per[k] * n_new, "est_usd_high": per_hi[k] * n_new})
    pilot_spent = sum(v["cost_total"] for v in pilot["runs"].values())
    full_card = pilot_spent + sum(e["est_usd"] for e in est)
    full_card_hi = pilot_spent + sum(e["est_usd_high"] for e in est)
    # 优先级裁剪：必跑的先排；Fable 按种子 1、2、3 依次放，放不下就停在那里（种子 1 放不下时只跑放得下的前若干个用户）
    room = BUDGET_USD - RESERVE_USD - pilot_spent
    plan, skipped = [], []
    core = [e for e in est if e["model"] in CORE]
    core.sort(key=lambda e: (MODELS.index(e["model"]), e["control"], e["seed"]))
    for e in core:
        plan.append(dict(e, users=None))
        room -= e["est_usd"]
    for e in [e for e in est if e["model"] == EXTRA]:
        if e["est_usd"] <= room:
            plan.append(dict(e, users=None))
            room -= e["est_usd"]
        elif e["seed"] == PILOT_SEED and room > 0:
            per_user = per[key_of(EXTRA, False)] * n_sess[e["seed"]] / len(run.get_world(e["seed"])["users"])
            n_u = max(PILOT_USERS, min(len(run.get_world(e["seed"])["users"]), PILOT_USERS + int(room // per_user)))
            plan.append(dict(e, users=n_u, est_usd=per_user * (n_u - PILOT_USERS)))
            skipped += [dict(e, reason=f"估算放不下整个世界，只跑前 {n_u} 个用户")]
            room = 0
        else:
            skipped.append(dict(e, reason="估算超出预算，按优先级不跑"))
    res = {"written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "budget_usd": BUDGET_USD, "reserve_usd": RESERVE_USD, "pilot_spent_usd": pilot_spent,
           "cost_per_session": per, "cost_per_session_high": per_hi, "rows": est,
           "full_card_usd": full_card, "full_card_usd_high": full_card_hi,
           "over_budget": full_card > BUDGET_USD, "plan": plan, "skipped": skipped,
           "planned_usd": pilot_spent + sum(e["est_usd"] for e in plan)}
    dump(os.path.join(CDIR, "estimate.json"), res)
    print(f"全卡估算 {full_card:.2f} 美元（按两个用户里较贵的那个算 {full_card_hi:.2f}）；计划花 {res['planned_usd']:.2f} 美元；"
          f"跳过 {len(skipped)} 组")
    return res


# ---------- full ----------

def stage_full(client):
    est = load(os.path.join(CDIR, "estimate.json"))
    only = os.environ.get("ROOMMATE_CLAUDE_ONLY")   # 分批 / 并行进程：如 "claude-haiku-5-5@1,claude-opus-5-5@1@ctrl"
    log = []
    for e in est["plan"]:
        tag = f"{e['model']}@{e['seed']}" + ("@ctrl" if e["control"] else "")
        if only and tag not in only.split(","):
            continue
        spent = client.ledger._load()["total_usd"]
        if e["model"] == EXTRA and spent + e["est_usd"] > BUDGET_USD - RESERVE_USD:
            log.append({"run": tag, "status": "skipped", "reason": f"已花 {spent:.2f} 美元，这一组估算 {e['est_usd']:.2f} 美元放不下"})
            print("跳过", log[-1])
            continue
        k = key_of(e["model"], e["control"])
        try:
            recs = run.run_policy(sub_world(e["seed"], e.get("users")),
                                  run.agent(client, e["model"], e["seed"], e["control"]), workers=WORKERS)
        except (BudgetExceeded, ApiError) as ex:
            log.append({"run": tag, "status": "stopped", "reason": f"{type(ex).__name__}: {ex}"[:300]})
            print("停止：", log[-1])
            break
        s = run.save_policy(e["seed"], k, recs)
        data_policy.mark_not_for_training(run.pdir(e["seed"], k))
        led = client.ledger._load()
        log.append({"run": tag, "status": "done", "mean_score": s["mean_score"], "cum_usd": led["total_usd"]})
        print(f"种子 {e['seed']} {k:36s} 平均分 {s['mean_score']:.3f}  累计 {led['total_usd']:.2f} 美元")
    p = os.path.join(CDIR, "full_log.json")
    old = load(p) if os.path.exists(p) else []
    dump(p, old + [{"t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "runs": log,
                    "retries": dict(client.retries)}])


# ---------- repro ----------

def stage_repro():
    c = LLMClient(os.path.join(OUT, "llm_cache"), Ledger(None, 0), offline=True)
    out = {}
    for s in run.GATE_SEEDS:
        base = os.path.join(OUT, f"seed{s}", "policies")
        for key in sorted(os.listdir(base)):
            if not key.startswith("llm_claude-"):
                continue
            ctrl = key.endswith("_true_memory")
            m = key[4:].replace("_true_memory", "")
            with open(os.path.join(base, key, "sessions.jsonl"), encoding="utf-8") as f:
                saved = [json.loads(x) for x in f]
            n_users = len({r["uid"] for r in saved})
            again, _ = run.split_transcripts(run.run_policy(sub_world(s, n_users), run.agent(c, m, s, ctrl, cache_only=True),
                                                            workers=WORKERS))
            out[f"{s}/{key}"] = run.records_hash(saved) == run.records_hash(again)
    dump(os.path.join(CDIR, "repro_check.json"), {"all_identical": all(out.values()), "runs": out})
    print("Claude 记录只用缓存重跑：", out)


def stage_compact():
    """Claude 的请求缓存先写未压缩的 .jsonl，这里并入同名的 .jsonl.gz（按键去重、保留先写的），仓库里只放压缩后的。"""
    import gzip
    cd = os.path.join(OUT, "llm_cache")
    for m in MODELS:
        raw, gz = os.path.join(cd, f"{m}.jsonl"), os.path.join(cd, f"{m}.jsonl.gz")
        if not os.path.exists(raw):
            continue
        rows, seen = [], set()
        for p, op in ((gz, gzip.open), (raw, open)):
            if os.path.exists(p):
                with op(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if d["key"] not in seen:
                                seen.add(d["key"])
                                rows.append(line.rstrip("\n"))
        tmp = gz + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for line in rows:
                f.write(line + "\n")
        os.replace(tmp, gz)
        os.remove(raw)
        print(f"缓存 {m}：{len(rows)} 条 → {gz}")


def write_config(args):
    dump(os.path.join(CDIR, "config.json"), {
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "models": MODELS, "core": CORE, "control": CONTROL, "extra": EXTRA,
        "budget_usd": BUDGET_USD, "reserve_usd": RESERVE_USD, "pilot_seed": PILOT_SEED, "pilot_users": PILOT_USERS,
        "workers": WORKERS, "effort": agent_claude.EFFORT, "max_tokens": agent_claude.MAX_TOKENS,
        "thinking": {"type": "adaptive", "display": "summarized"}, "cache_control": {"type": "ephemeral"},
        "tool_choice": {"type": "auto"}, "sampling": "接口默认（这几个模型不接受 temperature / top_p / top_k）",
        "key_env": llm.ANTHROPIC_KEY_ENV, "base_url": llm.ANTHROPIC_BASE_URL,
        "stages": args.stages, "offline": args.offline})


ALL_STAGES = ["pilot", "estimate", "full", "compact", "repro", "stats", "report"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    stages = args.stages.split(",")
    os.makedirs(CDIR, exist_ok=True)
    data_policy.mark_not_for_training(CDIR)
    load_prices()
    client = None
    if any(s in stages for s in ("pilot", "full")):
        write_config(args)
        client = make_client(args.offline)
    if "pilot" in stages:
        stage_pilot(client)
    if "estimate" in stages:
        stage_estimate()
    if "full" in stages:
        stage_full(client)
    if "compact" in stages:
        stage_compact()
    if "repro" in stages:
        stage_repro()
    if "stats" in stages:
        from . import claude_report
        stats.write(OUT, run.GATE_SEEDS, run.COMPARE_SEED)
        claude_report.write_stats()
    if "report" in stages:
        from . import claude_report, report
        cp = os.path.join(OUT, "config.json")       # M1 第二版的运行配置：只补上 Claude 模型的名单，其余不动
        cfg = load(cp)
        if cfg.get("claude_models") != MODELS:
            dump(cp, dict(cfg, claude_models=MODELS))
        report.write(OUT, "reports/m1v2-baseline.md")
        claude_report.write_report("reports/m1v2-claude-baseline.md")
        print("报告：reports/m1v2-claude-baseline.md、reports/m1v2-baseline.md")


if __name__ == "__main__":
    main()
