"""M2 起点测量：未训练的 Qwen3.5-4B（经硅基流动调用）在第二版考场里的得分，以及训练温度下组内分数的差异。

    python -m roommate.m2                        # 全部阶段
    python -m roommate.m2 --offline              # 只用缓存（不联网），结果应与首次运行逐字一致
    python -m roommate.m2 --stages stats,report  # 只重算统计与报告
    ROOMMATE_M2_ONLY=nothink@1,think@1@ctrl,var@nothink@3 python -m roommate.m2 --stages runs   # 分批跑

考场不变：世界、规则、判分都用第二版（runs/m1v2/seed{1,2,3}/world.json），这里只加接入新渠道和新统计的代码。
- 评测（stage runs 里的 base、ctrl）：采样温度与 M1 一致（温度 0），思考模式开、关各一组；
  "知道真实状态"对照组同样开、关各一组。
- 组内差异（var）：按将来训练用的采样（温度 1、top_p 1，见 TRAIN_SAMPLING）把种子 VAR_SEED 上每个用户的
  8 次会话重复跑 VAR_REPS 遍；每一遍的请求 seed 不同（agent 的 rep 参数）。
- 每个请求按内容哈希缓存在 runs/m2/llm_cache/，重跑全部命中缓存；compact 阶段把新写的缓存并入 gzip 文件。
- 花费账本 runs/m2/spend_ledger.json：硅基流动按价目快照（runs/m2/pricing/）记人民币，本卡上限 CNY_BUDGET 元。
"""

import argparse
import datetime as dt
import gzip
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from . import llm
from .agent_llm import LLMAgent
from .env import Env
from .grading import summarize
from .llm import ApiError, BudgetExceeded, Ledger, LLMClient

OUT = "runs/m2"
M1 = "runs/m1v2"
SEEDS = [1, 2, 3]
MODEL = "Qwen/Qwen3.5-4B"
SHORT = "qwen3.5-4b"
# 思考模式开、关。max_tokens 只防止失控的长输出；被截断的轮次记为格式问题 truncated。
MODES = {
    "nothink": {"enable_thinking": False, "max_tokens": 2048},
    "think": {"enable_thinking": True, "max_tokens": 8192},
}
# 将来训练（GRPO 采样）用的采样参数：温度 1、不截断分布，使采样分布就是策略本身的分布。
TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
VAR_SEED = 1
VAR_REPS = 8
VAR_MODES = ["nothink", "think"]
CNY_BUDGET = 50.0
WORKERS = 24          # 每个世界里同时跑的用户数
PARALLEL_JOBS = int(os.environ.get("ROOMMATE_M2_JOBS", "4"))
MAX_INFLIGHT = int(os.environ.get("ROOMMATE_M2_INFLIGHT", "64"))   # 同时发出的请求数上限
ALL_STAGES = ["runs", "compact", "repro", "stats", "report"]


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


def get_world(seed):
    return load(os.path.join(M1, f"seed{seed}", "world.json"))


def key_of(mode, ctrl=False):
    return f"llm_{SHORT}_{mode}" + ("_true_memory" if ctrl else "")


def job_dir(job):
    kind, mode, seed, rep = job
    if kind == "var":
        return os.path.join(OUT, "variance", f"seed{seed}", f"{mode}_t1", f"rep{rep}")
    return os.path.join(OUT, f"seed{seed}", "policies", key_of(mode, kind == "ctrl"))


def job_name(job):
    kind, mode, seed, rep = job
    return {"base": f"{mode}@{seed}", "ctrl": f"{mode}@{seed}@ctrl", "var": f"var@{mode}@{rep}"}[kind]


def plan():
    """(种类, 模式, 种子, 第几遍)；种类：base 评测、ctrl 知道真实状态、var 训练温度下重复。"""
    jobs = [("base", m, s, 0) for m in MODES for s in SEEDS]
    jobs += [("ctrl", m, s, 0) for m in MODES for s in SEEDS]
    jobs += [("var", m, VAR_SEED, r) for m in VAR_MODES for r in range(1, VAR_REPS + 1)]
    return jobs


def make_agent(client, job, cache_only=False):
    kind, mode, seed, rep = job
    extra = dict(MODES[mode], **(TRAIN_SAMPLING if kind == "var" else {}))
    return LLMAgent(client, MODEL, extra=extra, seed=seed, inject_true_memory=kind == "ctrl",
                    echo_reasoning=mode == "think", truncate_on_cache_miss=cache_only,
                    rep=rep, track_format=True)


def run_job(client, job, cache_only=False):
    env = Env(get_world(job[2]))
    env.run(make_agent(client, job, cache_only), workers=WORKERS)
    return env.records()


def save_job(job, recs):
    d = job_dir(job)
    plain, trans = [], []
    for r in recs:
        r = dict(r)
        t = r.pop("transcript", None)
        plain.append(r)
        if t is not None:
            trans.append({"uid": r["uid"], "idx": r["idx"], "messages": t})
    if not any(r["domain"] != "util" for r in plain):
        raise RuntimeError(f"{job_name(job)} 没有任何与偏好有关的会话记录，不覆盖已有文件")
    dump_jsonl(os.path.join(d, "sessions.jsonl"), plain)
    with gzip.open(os.path.join(d, "transcripts.jsonl.gz"), "wt", encoding="utf-8") as f:
        for t in trans:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    s = summarize(plain)
    dump(os.path.join(d, "summary.json"), s)
    return s


def load_prices():
    pp = os.path.join(OUT, "pricing")
    for fn in sorted(os.listdir(pp)):
        if fn.startswith("siliconflow_"):
            d = load(os.path.join(pp, fn))
            llm.SILICONFLOW_PRICES.update({m: v["cny"] for m, v in d["prices"].items()})
            llm.SILICONFLOW_FX["cny_per_usd"] = d["fx_cny_per_usd"]


def make_client(offline):
    path = os.path.join(OUT, "spend_ledger.json")
    ledger = Ledger(path, limit_usd=CNY_BUDGET / llm.SILICONFLOW_FX["cny_per_usd"])
    if not offline and not os.path.exists(path):
        ledger.data["siliconflow_cny_limit"] = CNY_BUDGET
        dump(path, ledger.data)
    return LLMClient(os.path.join(OUT, "llm_cache"), ledger, offline=offline,
                     blocked_providers={"xai", "deepseek"}, max_inflight=MAX_INFLIGHT)


def compact_cache():
    """把本轮新写的 .jsonl 缓存并入同名 .jsonl.gz（仓库里只放压缩后的）。"""
    cd = os.path.join(OUT, "llm_cache")
    for fn in os.listdir(cd):
        if not fn.endswith(".jsonl"):
            continue
        src, dst = os.path.join(cd, fn), os.path.join(cd, fn + ".gz")
        with open(src, "rb") as f, gzip.open(dst, "ab") as g:
            shutil.copyfileobj(f, g)
        os.remove(src)


def stage_runs(client):
    jobs = plan()
    only = os.environ.get("ROOMMATE_M2_ONLY")
    if only:
        want = set(only.split(","))
        jobs = [j for j in jobs if job_name(j) in want]
    stopped = []

    def one(job):
        try:
            recs = run_job(client, job)
        except (BudgetExceeded, ApiError) as e:
            stopped.append(f"{job_name(job)}: {type(e).__name__}: {e}"[:300])
            print("停止：", stopped[-1], flush=True)
            return
        s = save_job(job, recs)
        print(f"{job_name(job):22s} 平均分 {s['mean_score']:.3f}  选对 {s['success_rate']:.3f}  "
              f"提问 {s['avg_questions']:.2f}  未选 {s['no_choice']}  "
              f"累计 {client.ledger._load().get('siliconflow_cny_total', 0):.2f} 元  重试 {client.retries}", flush=True)

    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as pool:
        list(pool.map(one, jobs))
    dump(os.path.join(OUT, "costs.json"), {"cumulative": client.ledger._load(), "this_run": client.ledger.session,
                                           "retries": client.retries, "stopped": stopped or None})


def stage_repro():
    """只用缓存重建每一组结果，和保存的逐次记录逐字比较。"""
    from .run import records_hash
    c = LLMClient(os.path.join(OUT, "llm_cache"), Ledger(None, 0), offline=True)
    out = {}
    for job in plan():
        p = os.path.join(job_dir(job), "sessions.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            saved = [json.loads(x) for x in f]
        again = [{k: v for k, v in r.items() if k != "transcript"} for r in run_job(c, job, cache_only=True)]
        out[job_name(job)] = records_hash(saved) == records_hash(again)
    dump(os.path.join(OUT, "repro_check.json"), {"all_identical": all(out.values()), "by_run": out})
    print("复现：", sum(out.values()), "/", len(out))


def write_config(args):
    dump(os.path.join(OUT, "config.json"), {
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "provider": "siliconflow", "seeds": SEEDS, "worlds_from": M1,
        "modes": MODES, "eval_temperature": 0, "train_sampling": TRAIN_SAMPLING,
        "var_seed": VAR_SEED, "var_reps": VAR_REPS, "var_modes": VAR_MODES,
        "cny_budget": CNY_BUDGET, "workers": WORKERS, "stages": args.stages, "offline": args.offline,
        "only": os.environ.get("ROOMMATE_M2_ONLY")})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--report", default="reports/m2-start.md")
    args = ap.parse_args(argv)
    stages = args.stages.split(",")
    os.makedirs(OUT, exist_ok=True)
    load_prices()
    if "runs" in stages:
        write_config(args)
        stage_runs(make_client(args.offline))
    if "compact" in stages:   # 只在没有别的进程还在写缓存时跑
        compact_cache()
    if "repro" in stages:
        stage_repro()
    if "stats" in stages or "report" in stages:
        from . import m2_report
        if "stats" in stages:
            m2_report.write_stats(OUT, M1, SEEDS)
        if "report" in stages:
            m2_report.write_report(OUT, args.report)
            print("报告：", args.report)


if __name__ == "__main__":
    main()
