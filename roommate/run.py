"""一条命令跑完 M1 第二版：生成世界 → 脚本策略 → 判分自检 → 一致性检查 → 选项泄题检验 → 台词线索校验
→ 大模型基线与"知道真实状态"对照 → 复现与恢复检查 → 统计与闸门 → 报告。

    python -m roommate.run                       # 全部阶段
    python -m roommate.run --offline             # 只用缓存（不联网、不花钱），结果应与首次运行逐字一致
    python -m roommate.run --stages scripted,checks,probes,stats,report   # 只跑部分阶段（不需要联网）

闸门用种子 1、2、3 的三个世界；种子 0 只用来和第一版做前后对照，不参与闸门。
固定种子时：世界与脚本策略的结果完全由代码决定；大模型的每个请求都按内容哈希缓存在
runs/m1v2/llm_cache/，重跑时全部命中缓存，因此整份结果可逐字复现。
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import urllib.request
from collections import Counter

from . import checks, cuecheck, llm, probes, report, stats
from .agent_llm import LLMAgent
from .env import Env
from .grading import satisfies, summarize
from .llm import ApiError, BudgetExceeded, Ledger, LLMClient
from .policies import SCRIPTED
from .world import generate_world, world_hash, world_stats

OUT = "runs/m1v2"
GATE_SEEDS = [1, 2, 3]
COMPARE_SEED = 0
CLASSIFIER_TRAIN_SEEDS = list(range(100, 160))
# grok 的余额已耗尽：第二版剩余的调用全部改用 DeepSeek，不再向 xAI 发任何请求（BLOCKED_PROVIDERS）。
# 已经跑完或已缓存的 grok-4.7 结果保留，只从缓存重建、照实报告（CACHED_ONLY）。
BLOCKED_PROVIDERS = {"xai"}
MODELS = ["deepseek-flash", "deepseek-v4-pro"]                 # 闸门世界上的正式基线
CONTROL_RUNS = [("deepseek-v4-pro", 1)]                        # "知道真实状态"对照：只给 V4 Pro 跑种子 1
COMPARE_MODELS = ["deepseek-v4-pro"]                           # 种子 0 对照：大模型只跑 u07 的 8 次会话
# 只从缓存重建的已有结果：(模型, 种子, 是否对照组)。缺缓存的用户从第一次缺的会话起截断（不补跑）。
CACHED_ONLY = ([("grok-4.7", s, False) for s in (1, 2, 3)] + [("grok-4.7", s, True) for s in (1, 2, 3)]
               + [("deepseek-flash", s, True) for s in (1, 2, 3)])
CUE_JUDGES = ["deepseek-v4-pro", "grok-4.7"]                   # grok-4.7 只用缓存里已有的判定
DEEPSEEK_CNY_BUDGET = 40.0                                     # 第二版剩余部分的 DeepSeek 花费上限（元）
MODEL_EXTRA = {}          # 各模型额外的请求参数（如思考模式），在 config.json 里记录
# DeepSeek V4 默认开思考；工具调用循环里要把 reasoning_content 原样带回
ECHO_REASONING = {"deepseek-flash": True, "deepseek-v4-pro": True}


def agent(client, m, seed, ctrl=False, cache_only=False):
    return LLMAgent(client, m, extra=MODEL_EXTRA.get(m), seed=seed, inject_true_memory=ctrl,
                    echo_reasoning=ECHO_REASONING.get(m, False), truncate_on_cache_miss=cache_only)
V1_MODE_POLICIES = ["ask_all", "remember_all_reask", "ideal"]
ALL_STAGES = ["worlds", "scripted", "checks", "probes", "cues", "llm", "repro", "stats", "report"]


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def records_hash(recs):
    return hashlib.sha256(json.dumps(recs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def pdir(seed, key, mode="default"):
    sub = "policies" if mode == "default" else "policies_v1mode"
    top = seed if isinstance(seed, str) else f"seed{seed}"
    return os.path.join(OUT, top, sub, key.replace(":", "_"))


def split_transcripts(recs):
    plain, trans = [], []
    for r in recs:
        r = dict(r)
        t = r.pop("transcript", None)
        plain.append(r)
        if t is not None:
            trans.append({"uid": r["uid"], "idx": r["idx"], "messages": t})
    return plain, trans


def run_policy(world, policy, config=None, workers=1):
    env = Env(world, config)
    env.run(policy, workers=workers)
    return env.records()


def save_policy(seed, key, recs, mode="default"):
    plain, trans = split_transcripts(recs)
    d = pdir(seed, key, mode)
    if not any(r["domain"] != "util" for r in plain):
        raise RuntimeError(f"{seed}/{key} 没有任何与偏好有关的会话记录，不覆盖已有文件")
    dump_jsonl(os.path.join(d, "sessions.jsonl"), plain)
    if trans:
        dump_jsonl(os.path.join(d, "transcripts.jsonl"), trans)
    s = summarize(plain)
    dump(os.path.join(d, "summary.json"), s)
    return s


# ---------- 判分自检 ----------

def self_check(world, res):
    out = []

    def add(name, ok, detail):
        out.append({"name": name, "pass": bool(ok), "detail": detail})

    n_one = n = nm_ok = 0
    for u in world["users"]:
        for s in u["sessions"]:
            n += 1
            if s["domain"] == "util":
                n_one += sum(o["id"] == s["correct"] for o in s["options"]) == 1
                continue
            ok_ids = [o["id"] for o in s["options"] if satisfies(o, s["eff_state"], s["domain"])]
            n_one += ok_ids == [s["correct"]]
            nm = [o for o in s["options"] if s["roles"][o["id"]] == "near_miss"]
            st = dict(s["eff_state"], **s["near_miss_state"])
            nm_ok += len(nm) == 1 and satisfies(nm[0], st, s["domain"]) and not satisfies(nm[0], s["eff_state"], s["domain"])
    n_pref = sum(1 for u in world["users"] for s in u["sessions"] if s["domain"] != "util")
    add("每个任务恰好一个正确选项", n_one == n, f"{n_one}/{n}")
    add("近似项满足它的近似状态、违反用户真实状态", nm_ok == n_pref, f"{nm_ok}/{n_pref}")
    o, z, i = res["oracle"], res["nothing"], res["ideal"]
    add("标准策略（知道真实状态）得满分", o["mean_score"] == 1.0, f"平均分 {o['mean_score']}")
    add("什么都不做得零分", z["mean_score"] == 0.0, f"平均分 {z['mean_score']}")
    add("理想记忆策略全部做对、从不重问", i["success_rate"] == 1.0 and i["told_before_rate"] == 0.0,
        f"成功率 {i['success_rate']}，重问率 {i['told_before_rate']}，平均提问 {i['avg_questions']:.3f}")
    v1 = res["v1mode:ask_all"]
    add("按第一版行为（总会重新回答）时，每次都问一遍能全部做对（回答信息充分）", v1["success_rate"] == 1.0,
        f"成功率 {v1['success_rate']}")
    r = res["remember_no_update"]
    add("不读台词、只记不更新的策略会选到近似项（近似项有效）", (r["near_miss_error_rate"] or 0) > 0.2,
        f"近似项错误率 {r['near_miss_error_rate']:.3f}")
    st = world_stats(world)
    add("规模与会话类型比例同第一版：24 个用户 × 8 次会话，每人 1 次无关会话、1 次临时情况",
        st["n_users"] == 24 and st["n_sessions_per_user"] == 8 and st["unrelated_sessions"] == 24
        and st["temp_situations"] == 24, f"{st['n_users']} × {st['n_sessions_per_user']}，无关 {st['unrelated_sessions']}，临时 {st['temp_situations']}")
    add("正确答案是'变化之后'那个的比例接近一半（选项不暴露方向）", abs(st["correct_is_post_share"] - 0.5) <= 0.03,
        f"{st['correct_is_post_share']:.3f}")
    w2 = generate_world(world["seed"], world["n_users"], world["n_sessions"])
    add("同一种子重新生成的世界完全相同", world_hash(w2) == world_hash(world), world_hash(world))
    return {"seed": world["seed"], "all_pass": all(c["pass"] for c in out), "checks": out}


# ---------- 各阶段 ----------

def stage_worlds(seeds):
    for s in seeds:
        w = generate_world(s)
        dump(os.path.join(OUT, f"seed{s}", "world.json"), w)
        dump(os.path.join(OUT, f"seed{s}", "world_stats.json"), dict(world_stats(w), world_hash=world_hash(w)))
        print(f"世界 {s}：", world_hash(w))
    if os.path.exists(os.path.join("runs", "m1", "world.json")):
        dump(os.path.join(OUT, REPLAY, "world.json"), build_replay())


def get_world(seed):
    return load(os.path.join(OUT, f"seed{seed}", "world.json"))


REPLAY_UID = "u07"
REPLAY = "u07_replay"   # 第一版 u07 的经历（事件、临时情况、会话安排）按第二版规则重放


def build_replay():
    """把第一版种子 0 里 u07 的会话安排固定下来，在第二版的种子 0 世界里重放；只保留 u07 这一个用户。"""
    v1 = load(os.path.join("runs", "m1", "world.json"))
    u1 = next(u for u in v1["users"] if u["uid"] == REPLAY_UID)
    script = [{"type": s["type"], "week": s["week"], "domain": s["domain"],
               "event": s["event"]["type"] if s["event"] else None,
               "temp": [s["temp"]["type"], s["temp"]["value"]] if s["temp"] else None} for s in u1["sessions"]]
    init = next(u for u in generate_world(COMPARE_SEED)["users"] if u["uid"] == REPLAY_UID)["init_state"]
    w = generate_world(COMPARE_SEED, scripts={REPLAY_UID: (init, script)})
    w["users"] = [u for u in w["users"] if u["uid"] == REPLAY_UID]
    w["n_users"] = 1
    w["replay_of"] = {"source": "runs/m1/world.json", "uid": REPLAY_UID, "script": script}
    return w


def stage_scripted(seeds):
    allsc = {}
    for seed in seeds:
        w = get_world(seed)
        res = {}
        for name, P in SCRIPTED.items():
            res[name] = save_policy(seed, name, run_policy(w, P()))
        for name in V1_MODE_POLICIES:
            res["v1mode:" + name] = save_policy(seed, name, run_policy(w, SCRIPTED[name](),
                                                                       {"repeat_mode": "always_answer"}), "v1mode")
        sc = self_check(w, res)
        allsc[seed] = sc
        if seed == COMPARE_SEED and os.path.exists(os.path.join(OUT, REPLAY, "world.json")):
            rw = load(os.path.join(OUT, REPLAY, "world.json"))
            for name in ("ask_all", "ideal", "oracle"):
                save_policy(REPLAY, name, run_policy(rw, SCRIPTED[name]()))
        print(f"种子 {seed}：", {k: round(v["mean_score"], 3) for k, v in res.items()},
              "自检通过" if sc["all_pass"] else "自检失败")
    dump(os.path.join(OUT, "selfcheck.json"), {"all_pass": all(v["all_pass"] for v in allsc.values()),
                                               "by_seed": allsc})


def stage_checks(seeds):
    out = {}
    for seed in seeds:
        w = get_world(seed)
        out[seed] = {m: checks.check_world(w, repeat_mode=m) for m in ("refuse_told", "always_answer")}
        print(f"一致性检查 种子 {seed}：", {m: r["n_problems"] for m, r in out[seed].items()})
    total = sum(r["n_problems"] for v in out.values() for r in v.values())
    n = sum(r["n_answers_checked"] for v in out.values() for r in v.values())
    dump(os.path.join(OUT, "consistency.json"), {"n_answers_checked": n, "n_problems": total,
                                                 "all_pass": total == 0, "by_seed": out})


def stage_probes():
    gate = [get_world(s) for s in GATE_SEEDS]
    limit = 1 / 6 + 0.05
    res = {"limit": limit, "by_seed": {}}
    cnt = Counter()
    for w in gate:
        pos, c = probes.position_rule(w)
        cnt += c
        res["by_seed"][w["seed"]] = {"modal": probes.modal_rule(w), "conservative": probes.conservative_rule(w),
                                     "position": pos}
    n = sum(cnt.values())
    res["modal"] = sum(v["modal"] for v in res["by_seed"].values()) / len(gate)
    res["conservative"] = sum(v["conservative"] for v in res["by_seed"].values()) / len(gate)
    res["position"] = {L: cnt[L] / n for L in "ABCDEF"}
    res["position_best"] = max(res["position"].values())
    x, p = probes.chi2_uniform(cnt)
    res["correct_letter_counts"] = dict(sorted(cnt.items()))
    res["chi2"] = {"statistic": x, "df": 5, "p_value": p}
    train = [generate_world(s) for s in CLASSIFIER_TRAIN_SEEDS]
    res["classifier"] = probes.classifier_probe(train, gate)
    res["classifier"]["train_seeds"] = [CLASSIFIER_TRAIN_SEEDS[0], CLASSIFIER_TRAIN_SEEDS[-1]]
    big = [generate_world(s) for s in range(200, 230)]
    res["classifier_large_test"] = dict(probes.classifier_probe(train, big), test_seeds=[200, 229])
    res["all_pass"] = all(v <= limit for v in (res["modal"], res["conservative"], res["position_best"],
                                               res["classifier"]["test_acc"]))
    dump(os.path.join(OUT, "leak_probes.json"), res)
    print("泄题检验：", {k: res[k] for k in ("modal", "conservative", "position_best")},
          res["classifier"]["test_acc"], res["chi2"])


def make_client(offline, budget, ledger_path=None):
    ledger = Ledger(ledger_path or os.path.join(OUT, "spend_ledger.json"), budget)
    return LLMClient(os.path.join(OUT, "llm_cache"), ledger, offline=offline, blocked_providers=BLOCKED_PROVIDERS)


def load_deepseek_prices():
    pp = os.path.join(OUT, "pricing")
    for fn in sorted(os.listdir(pp)) if os.path.isdir(pp) else []:
        if fn.startswith("deepseek_"):
            d = load(os.path.join(pp, fn))
            llm.DEEPSEEK_PRICES.update({m: {"usd": v["usd"], "cny": v["cny"]} for m, v in d["prices"].items()})


def stage_cues(client, judges):
    worlds = [get_world(s) for s in [COMPARE_SEED] + GATE_SEEDS]
    items = cuecheck.items_from_worlds(worlds) + cuecheck.synthetic_items(3)
    r = cuecheck.validate(client, judges, items, extras={m: MODEL_EXTRA.get(m) for m in judges})
    r["items"] = items
    dump(os.path.join(OUT, "cue_validation.json"), r)
    print("台词校验：", r["n_items_pass_all_models"], "/", r["n_items"], r["accuracy_by_model"])


def llm_plan():
    """(种子, 模型, 是否对照组, 是否只用缓存)。"""
    plan = [(sd, m, False, False) for m in MODELS for sd in GATE_SEEDS]
    plan += [(sd, m, True, False) for m, sd in CONTROL_RUNS]
    plan += [(REPLAY, m, False, False) for m in COMPARE_MODELS]
    plan += [(sd, m, c, True) for m, sd, c in CACHED_ONLY]
    return plan


def rebuild_cached(m, seed, ctrl):
    """只从缓存重建一组已有结果。grok-4.7 的普通运行与对照组在种子 2、3 上曾在两个进程里同时跑，
    两者第一轮请求完全相同，同一个键被写了两个响应。做法：先重建普通运行（同一会话里缺缓存就换另一个响应重试），
    记下每个这种键用了哪个；重建对照组时这些键改用另一个。种子 2 上这样重建的结果与当时保存的记录、对话逐字一致。"""
    c = LLMClient(os.path.join(OUT, "llm_cache"), Ledger(None, 0), offline=True)
    w = get_world(seed)
    normal = run_policy(w, agent(c, m, seed, False, cache_only=True), workers=12)
    if not ctrl:
        return normal
    used = dict(c.chosen)
    c.default_choice = {k: (i + 1) % len(c.alts[k]) for k, i in used.items()}
    return run_policy(w, agent(c, m, seed, True, cache_only=True), workers=12)


def stage_llm(client, budget_stop):
    plan = llm_plan()
    only = os.environ.get("ROOMMATE_LLM_ONLY")  # 分批跑：如 "deepseek-flash@1,deepseek-v4-pro@1@ctrl"
    if only:
        want = set(only.split(","))
        plan = [p for p in plan if f"{p[1]}@{p[0]}" + ("@ctrl" if p[2] else "") in want]
    stopped = None
    for seed, m, ctrl, cache_only in plan:
        key = f"llm:{m}" + (":true_memory" if ctrl else "")
        world = load(os.path.join(OUT, REPLAY, "world.json")) if seed == REPLAY else get_world(seed)
        aseed = COMPARE_SEED if seed == REPLAY else seed
        c = client
        if seed == REPLAY:  # u07 重放单独记账，便于按阶段对账
            c = LLMClient(client.cache_dir, client.ledger, offline=client.offline, role_prefix="u07_v2rules:",
                          blocked_providers=client.blocked)
        try:
            recs = (rebuild_cached(m, seed, ctrl) if cache_only else
                    run_policy(world, agent(c, m, aseed, ctrl), workers=12))
        except (BudgetExceeded, ApiError) as e:
            stopped = f"{type(e).__name__}: {e}"[:300]
            print("停止：", stopped)
            break
        s = save_policy(seed, key, recs)
        n = sum(1 for r in recs if r["domain"] != "util")
        print(f"种子 {seed} {key:36s} 平均分 {s['mean_score']:.3f}（{n} 次偏好会话{'，只用缓存' if cache_only else ''}）"
              f"  累计 {client.ledger.data['total_usd']:.2f} 美元 / DeepSeek {client.ledger.data.get('deepseek_cny_total', 0):.2f} 元")
    dump(os.path.join(OUT, "costs.json"), {"cumulative": client.ledger._load(), "this_run": client.ledger.session,
                                           "stopped": stopped})


def stage_repro(offline_client):
    repro = {"worlds": {}, "scripted": {}, "llm": {}}
    for seed in [COMPARE_SEED] + GATE_SEEDS:
        repro["worlds"][seed] = {"saved": world_hash(get_world(seed)), "regenerated": world_hash(generate_world(seed))}
        w = get_world(seed)
        for name, P in SCRIPTED.items():
            with open(os.path.join(pdir(seed, name), "sessions.jsonl"), encoding="utf-8") as f:
                saved = [json.loads(x) for x in f]
            again = run_policy(generate_world(seed), P())
            repro["scripted"][f"{seed}/{name}"] = records_hash(saved) == records_hash(again)
        for key in sorted(os.listdir(os.path.join(OUT, f"seed{seed}", "policies"))):
            if not key.startswith("llm_"):
                continue
            m = key[4:].replace("_true_memory", "")
            ctrl = key.endswith("_true_memory")
            with open(os.path.join(pdir(seed, key), "sessions.jsonl"), encoding="utf-8") as f:
                saved = [json.loads(x) for x in f]
            again, _ = split_transcripts(rebuild_cached(m, seed, ctrl))
            repro["llm"][f"{seed}/{key}"] = records_hash(saved) == records_hash(again)
    rpath = os.path.join(OUT, REPLAY, "world.json")
    if os.path.exists(rpath):
        rw = load(rpath)
        repro["worlds"][REPLAY] = {"saved": world_hash(rw), "regenerated": world_hash(build_replay())}
        for key in sorted(os.listdir(os.path.join(OUT, REPLAY, "policies"))):
            with open(os.path.join(pdir(REPLAY, key), "sessions.jsonl"), encoding="utf-8") as f:
                saved = [json.loads(x) for x in f]
            if key.startswith("llm_"):
                again, _ = split_transcripts(run_policy(rw, agent(offline_client, key[4:], COMPARE_SEED, cache_only=True)))
            else:
                again = run_policy(build_replay(), SCRIPTED[key]())
            repro["llm" if key.startswith("llm_") else "scripted"][f"{REPLAY}/{key}"] = records_hash(saved) == records_hash(again)
    repro["all_identical"] = all(v["saved"] == v["regenerated"] for v in repro["worlds"].values()) and \
        all(repro["scripted"].values()) and all(repro["llm"].values())
    dump(os.path.join(OUT, "repro_check.json"), repro)
    # 中断—恢复：在第 4 个会话边界存档，丢掉内存里的环境，从存档恢复后跑完
    resume = {}
    seed = GATE_SEEDS[0]
    for key in sorted(os.listdir(os.path.join(OUT, f"seed{seed}", "policies"))):
        if not key.startswith("llm_") or key.endswith("_true_memory"):
            continue
        m = key[4:]
        mk = lambda: agent(offline_client, m, seed)
        full, _ = split_transcripts(run_policy(get_world(seed), mk(), workers=12))
        with tempfile.TemporaryDirectory() as td:
            ck = os.path.join(td, "ck.json")
            env = Env(get_world(seed))
            env.run(mk(), until=4, workers=12)
            env.save(ck)
            size = os.path.getsize(ck)
            del env
            env2 = Env.load(ck)
            env2.run(mk(), workers=12)
            res2, _ = split_transcripts(env2.records())
        resume[f"{seed}/{key}"] = {"k": 4, "checkpoint_bytes": size, "identical": records_hash(full) == records_hash(res2)}
    dump(os.path.join(OUT, "resume_check.json"), resume)
    print("复现：", repro["all_identical"], "恢复：", resume)


def write_config(args):
    from .world import DEFAULT_CONFIG
    dump(os.path.join(OUT, "config.json"), {
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "world_config": DEFAULT_CONFIG, "gate_seeds": GATE_SEEDS, "compare_seed": COMPARE_SEED,
        "classifier_train_seeds": [CLASSIFIER_TRAIN_SEEDS[0], CLASSIFIER_TRAIN_SEEDS[-1]],
        "models": MODELS, "control_runs": CONTROL_RUNS, "compare_models": COMPARE_MODELS,
        "cached_only": CACHED_ONLY, "blocked_providers": sorted(BLOCKED_PROVIDERS),
        "deepseek_cny_budget": DEEPSEEK_CNY_BUDGET,
        "cue_judges": CUE_JUDGES, "model_extra": MODEL_EXTRA, "echo_reasoning": ECHO_REASONING,
        "budget_usd": args.budget,
        "v1_mode_policies": V1_MODE_POLICIES, "stages": args.stages, "offline": args.offline})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--budget", type=float, default=40.0, help="累计花费上限（美元），到达即停止")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--report", default="reports/m1v2-baseline.md")
    args = ap.parse_args(argv)
    stages = args.stages.split(",")
    os.makedirs(OUT, exist_ok=True)
    write_config(args)
    seeds = [COMPARE_SEED] + GATE_SEEDS
    load_deepseek_prices()
    if "worlds" in stages:
        stage_worlds(seeds)
    if "scripted" in stages:
        stage_scripted(seeds)
    if "checks" in stages:
        stage_checks(seeds)
    if "probes" in stages:
        stage_probes()
    client = None
    if any(s in stages for s in ("cues", "llm")):
        client = make_client(args.offline, args.budget)
    if "cues" in stages:
        stage_cues(client, CUE_JUDGES)
    if "llm" in stages:
        stage_llm(client, args.budget)
    if "repro" in stages:
        stage_repro(LLMClient(os.path.join(OUT, "llm_cache"), Ledger(None, 0), offline=True))
    if "stats" in stages:
        stats.write(OUT, GATE_SEEDS, COMPARE_SEED)
    if "report" in stages:
        report.write(OUT, args.report)
        print("报告：", args.report)


if __name__ == "__main__":
    main()
