"""在训练机器上按顺序跑完试跑的各阶段；中断后再启动，跳过已完成的阶段、GRPO 从最近一次检查点接着跑。

    python -m roommate.train.pipeline                  # 全部
    python -m roommate.train.pipeline --until eval_pre

阶段：selfcheck → eval_pre → sft → eval_sft → grpo1 → eval_grpo1 → grpo2 → eval_grpo2。
- 自检没有全部通过就停下（训练前自检全部通过才开始训练）。
- 没有老师轨迹就停在 sft 之前，状态写 waiting_teacher。
- grpo1 里演练一次"中断后接着跑"：过了第一个检查点之后、在下一步中途把训练进程强行杀掉（SIGKILL），
  再用 --resume 启动，记录在 OUT/grpo_s1/drill.json。
- 显卡花费：按开机时间和单价估算，到 BUDGET_CNY 就不再开始新的阶段。
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

from .common import CKPT, OUT, dump, load, status

STAGES = ["selfcheck", "eval_pre", "sft", "eval_sft", "grpo1", "eval_grpo1", "grpo2", "eval_grpo2"]
PY = [sys.executable, "-m"]


def cost_so_far():
    """显卡费估算：当前机器（开机到现在）+ 释放过的机器（开机到释放），都按开机时的单价。"""
    try:
        import datetime as dt
        ts = lambda x: dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
        m = load("runs/m2pilot/relay/machine.json") if os.path.exists("runs/m2pilot/relay/machine.json") else None
        if not m:
            return None
        cost = (time.time() - ts(m["created"])) / 3600 * float(m["hourly"])
        past = "runs/m2pilot/relay/machines_past.jsonl"
        if os.path.exists(past):
            for line in open(past, encoding="utf-8"):
                p = json.loads(line)
                cost += (ts(p["released"]) - ts(p["created"])) / 3600 * float(p["hourly"])
        return cost
    except Exception:
        return None


def cmd_for(stage):
    if stage == "selfcheck":
        return PY + ["roommate.train.selfcheck"]
    if stage.startswith("eval_"):
        tag = stage[5:]
        ad = {"pre": None, "sft": os.path.join(CKPT, "sft", "final"),
              "grpo1": os.path.join(CKPT, "grpo_s1", "final"), "grpo2": os.path.join(CKPT, "grpo_s2", "final")}[tag]
        return PY + ["roommate.train.evaluate", "--tag", tag] + (["--adapter", ad] if ad else [])
    if stage == "sft":
        return PY + ["roommate.train.sft"] + (["--resume"] if os.path.exists(os.path.join(CKPT, "sft", "latest")) else [])
    seed = int(stage[-1])
    tag = f"grpo_s{seed}"
    if os.path.exists(os.path.join(CKPT, tag, "latest", "trainer_state.pt")):
        return PY + ["roommate.train.grpo", "--seed", str(seed), "--resume"]
    return PY + ["roommate.train.grpo", "--seed", str(seed), "--adapter", os.path.join(CKPT, "sft", "final")]


def run(stage, args):
    log = open(os.path.join(OUT, "logs", f"{stage}.log"), "a")
    c = cmd_for(stage)
    log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(c)}\n")
    log.flush()
    if stage == "grpo1" and args.drill and not os.path.exists(os.path.join(OUT, "grpo_s1", "drill.json")):
        return drill(c, log)
    return subprocess.run(c, stdout=log, stderr=subprocess.STDOUT).returncode


def drill(c, log):
    """跑到第一个检查点之后，在下一步训练中途强行杀掉，再接着跑。"""
    steps = os.path.join(OUT, "grpo_s1", "steps.jsonl")
    p = subprocess.Popen(c, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    killed = None
    while p.poll() is None:
        time.sleep(10)
        n = sum(1 for _ in open(steps)) if os.path.exists(steps) else 0
        st = load(os.path.join(OUT, "status.json")) if os.path.exists(os.path.join(OUT, "status.json")) else {}
        if n >= 6 and st.get("stage") == "train":
            os.killpg(p.pid, signal.SIGKILL)
            killed = {"t": time.time(), "completed_steps_logged": n, "stage_when_killed": st.get("stage"),
                      "step_in_progress": st.get("step")}
            p.wait()
            break
    if killed is None:
        return p.returncode
    time.sleep(5)
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)   # 被杀进程留下的采样端
    time.sleep(10)
    ck = json.load(open(os.path.join(CKPT, "grpo_s1", "latest", "extra.json")))
    t1 = time.time()
    c2 = cmd_for("grpo1")
    log.write(f"\n=== drill resume {' '.join(c2)}\n")
    log.flush()
    rc = subprocess.run(c2, stdout=log, stderr=subprocess.STDOUT).returncode
    dump(os.path.join(OUT, "grpo_s1", "drill.json"), {"killed": killed, "resumed_from_checkpoint_step": ck["step"],
                                                       "resume_command": " ".join(c2[2:]), "resume_rc": rc,
                                                       "resume_started": t1})
    return rc


def main(argv=None):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--until")
    ap.add_argument("--only")
    ap.add_argument("--no-drill", dest="drill", action="store_false")
    ap.add_argument("--budget-cny", type=float, default=235.0)
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    sp = os.path.join(OUT, "pipeline_state.json")
    state = load(sp) if os.path.exists(sp) else {"done": [], "runs": []}
    stages = STAGES[:STAGES.index(args.until) + 1] if args.until else STAGES
    if args.only:
        stages = args.only.split(",")
    for st in stages:
        if st in state["done"]:
            continue
        cost = cost_so_far()
        if cost is not None and cost > args.budget_cny:
            status("pipeline", stage="budget_stop", cost_estimate=round(cost, 1), next=st)
            return
        if st == "sft" and not glob.glob("runs/m2pilot/teacher/seed*/policies/*/transcripts.jsonl.gz"):
            status("pipeline", stage="waiting_teacher")
            return
        subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)   # 上一阶段若异常退出，可能留下采样端
        time.sleep(3)
        t0 = time.time()
        status("pipeline", stage=st)
        rc = run(st, args)
        state["runs"].append({"stage": st, "rc": rc, "seconds": round(time.time() - t0, 1),
                              "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if rc == 0:
            state["done"].append(st)
        dump(sp, state)
        if rc != 0:
            status("pipeline", stage="failed", failed=st, rc=rc)
            return
        if st == "selfcheck" and not load(os.path.join(OUT, "selfcheck", "selfcheck.json"))["all_pass"]:
            state["done"].remove("selfcheck")
            dump(sp, state)
            status("pipeline", stage="selfcheck_failed")
            return
        if st in ("grpo1", "grpo2") and os.path.exists(os.path.join(OUT, f"grpo_s{st[-1]}", "stopped_by_rule.json")):
            # 停止规则（分数上涨主要来自捷径①）：停下 GRPO，不再跑后面的训练；只把停下时的检查点考一次，供报告用
            ev = "eval_" + st
            if ev not in state["done"]:
                status("pipeline", stage=ev)
                t0 = time.time()
                rc = run(ev, args)
                state["runs"].append({"stage": ev, "rc": rc, "seconds": round(time.time() - t0, 1),
                                      "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                if rc == 0:
                    state["done"].append(ev)
                dump(sp, state)
            status("pipeline", stage="stopped_by_shortcut_rule", stopped=st, done=state["done"])
            return
    status("pipeline", stage="all_done", done=state["done"])


if __name__ == "__main__":
    main()
