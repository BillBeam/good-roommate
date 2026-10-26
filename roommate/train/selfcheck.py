"""训练前自检：全部通过才开始训练。结果写到 OUT/selfcheck/selfcheck.json，讲解材料要用的实物写到 material.json。

    python -m roommate.train.selfcheck

检查项（编号与报告一致）：
 1 LoRA 真的生效：带与不带 LoRA 的 logprob 不同（采样端、训练端都看），且采样端与训练端带 LoRA 时接近。
 2 用 2–3k token 的真实会话 prompt 带 LoRA 生成：没有 NaN，没有乱码。
 3 第 0 步时采样端与训练端 logprob 的差（均值、P99、最大值），在真实采样的轨迹上量。
 4 不同微批大小（1、2、4 条序列一起前向）下训练端 logprob 一致；梯度累加的分组方式不影响梯度。
 5 优势符号单步测试：优势 +1 时该段 logprob 上升、−1 时下降；采样端确实逐 token 返回 logprob；温度没被改写。
 6 训练样本的 token 序列与采样时模型看到的逐 token 一致（含模板删除思考的处理）。
 7 只有模型生成的 token 计入 loss：工具返回、模拟用户的话、记忆内容都不计入。
 8 1、8、32 路并发采样时吞吐随并发数上升。
 9 同一用户各段样本的奖励和优势与设计公式一致；记忆保存后再恢复，内容不变。
10 显卡上的 BF16 计算正常。
"""

import copy
import json
import math
import os
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from ..env import Env
from ..agent_llm import NUDGE
from . import serve
from .agent_local import LocalAgent
from .common import MAX_MODEL_LEN, MODEL, OUT, SHM, TRAIN_SETTING, dump, make_chat, world
from .grpo import advantages, build_items, returns
from .sampler import VLLMClient
from .shortrange import short_world

D = os.path.join(OUT, "selfcheck")
RESULTS, MATERIAL = [], {}


def record(no, name, ok, **nums):
    RESULTS.append({"no": no, "name": name, "pass": bool(ok), **nums})
    dump(os.path.join(D, "selfcheck.json"), {"results": RESULTS, "all_pass": all(r["pass"] for r in RESULTS)})
    print(f"[{'PASS' if ok else 'FAIL'}] {no} {name} {json.dumps(nums, ensure_ascii=False)[:400]}", flush=True)


def stats(xs):
    xs = sorted(abs(x) for x in xs)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))], "max": xs[-1]}


def rollouts(sampler, chat, users, w, n_rep=1, setting=None, seed=7):
    setting = dict(TRAIN_SETTING, **(setting or {}))
    base = {k: w[k] for k in ("version", "seed", "events", "config")}

    def one(job):
        u, rep = job
        env = Env(dict(base, users=[u], n_users=1, n_sessions=2))
        env.run(LocalAgent(sampler, chat, max_model_len=MAX_MODEL_LEN, seed=seed, rep=rep, label="selfcheck",
                           **setting), workers=1)
        return env.records()
    jobs = [(u, r) for u in users for r in range(n_rep)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return list(pool.map(one, jobs))


def main():
    import torch
    from .trainer import Trainer
    os.makedirs(D, exist_ok=True)
    t_all = time.time()
    chat = make_chat()
    proc, boot = serve.start(MODEL, os.path.join(OUT, "logs", "vllm_selfcheck.log"), gpu_util=0.40)
    sampler = VLLMClient(f"http://127.0.0.1:{serve.PORT}", "policy")
    try:
        sw = short_world(world(301), starts=[0, 3])
        users = sw["users"][:12]
        # ---------- 采样一批真实轨迹（基础模型，温度 1） ----------
        t0 = time.time()
        trajs = rollouts(sampler, chat, users, sw)
        segs = [s for recs in trajs for r in recs for s in r.get("segments", []) if sum(s["mask"])]
        MATERIAL["rollout"] = {"trajectories": len(trajs), "segments": len(segs), "seconds": time.time() - t0}

        tr = Trainer(MODEL, lr=1e-4, seed=0)
        # ---------- 10 BF16 ----------
        try:
            a = torch.randn(1024, 1024, device="cuda")
            b = torch.randn(1024, 1024, device="cuda")
            ref = a @ b
            got = (a.bfloat16() @ b.bfloat16()).float()
            rel = float((got - ref).norm() / ref.norm())
            record(10, "显卡上的 BF16 计算正常", torch.cuda.is_bf16_supported() and rel < 2e-2 and math.isfinite(rel),
                   gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                   bf16_supported=torch.cuda.is_bf16_supported(), matmul_rel_err=rel)

        except Exception as e:  # 一项出错不影响其余各项
            record(10, "显卡上的 BF16 计算正常", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 3 第 0 步的差（基础模型，LoRA 为 0） ----------
        try:
            diffs, lens = [], []
            for s in segs[:24]:
                lp = tr.logprobs(s["ids"], s["mask"]).tolist()
                sv = [v for v, m in zip(s["lp"], s["mask"]) if m][-len(lp):]
                diffs += [a_ - b_ for a_, b_ in zip(lp, sv) if b_ is not None]
                lens.append(len(s["ids"]))
            st = stats(diffs)
            record(3, "第 0 步采样端与训练端 logprob 之差", st["n"] > 0 and st["mean"] < 0.05, **st,
                   segments=min(24, len(segs)), mean_len=sum(lens) / max(1, len(lens)))
            MATERIAL["step0_diff_hist"] = sorted(abs(x) for x in diffs)[::max(1, len(diffs) // 400)]

        except Exception as e:  # 一项出错不影响其余各项
            record(3, "第 0 步采样端与训练端 logprob 之差", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 4 微批大小 ----------
        try:
            four = sorted(segs, key=lambda s: len(s["ids"]))[:4]
            single = [tr.logprobs(s["ids"], s["mask"]) for s in four]
            res4 = {}
            again = [tr.logprobs(s["ids"], s["mask"]) for s in four]
            rep = sorted(torch.cat([(a - b).abs() for a, b in zip(single, again)]).tolist())
            res4["bs1_repeat_mean_abs_diff"] = sum(rep) / len(rep)
            res4["bs1_repeat_p99_abs_diff"] = rep[int(0.99 * len(rep))]
            res4["bs1_repeat_max_abs_diff"] = rep[-1]
            for bs in (2, 4):
                diffs4 = []
                for i in range(0, 4, bs):
                    outs = tr.logprobs_padded(four[i:i + bs])
                    for j, o in enumerate(outs):
                        diffs4 += (o - single[i + j]).abs().tolist()
                res4[f"bs{bs}_vs_bs1_mean_abs_diff"] = sum(diffs4) / len(diffs4)
                res4[f"bs{bs}_vs_bs1_max_abs_diff"] = max(diffs4)
                res4[f"bs{bs}_vs_bs1_p99_abs_diff"] = sorted(diffs4)[int(0.99 * len(diffs4))]
            # 梯度：同一组样本，两种累加分组（逐条 vs 打乱顺序）得到的梯度
            items = [{"ids": s["ids"], "mask": s["mask"], "lp": s["lp"], "adv": 1.0} for s in four]
            g = []
            for order in ([0, 1, 2, 3], [3, 1, 0, 2]):
                tr.opt.zero_grad(set_to_none=True)
                ntok = sum(sum(it["mask"]) for it in items)
                for k in order:
                    lp = tr.logprobs(items[k]["ids"], items[k]["mask"], grad=True)
                    (-lp.sum() / ntok).backward()
                g.append(torch.cat([p.grad.flatten().float() for p in tr.params if p.grad is not None]).clone())
            cos = float(torch.nn.functional.cosine_similarity(g[0], g[1], dim=0))
            relg = float((g[0] - g[1]).norm() / g[0].norm())
            tr.opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            record(4, "不同微批大小下训练端 logprob 一致",
                   res4["bs2_vs_bs1_mean_abs_diff"] < 0.02 and res4["bs4_vs_bs1_mean_abs_diff"] < 0.02 and cos > 0.999,
                   **res4, grad_cosine_two_orders=cos, grad_rel_diff=relg,
                   criterion="bs2、bs4 与 bs1 的平均差 < 0.02，且两种累加顺序的梯度余弦 > 0.999",
                   criterion_change="原先看最大差（< 0.02）；BF16 下填充成批次后个别 token 的差可到 0.2 以上"
                                    "（第 3 项采样端与训练端的最大差也在 0.5 左右），最大差只反映个别 token 的舍入，"
                                    "改看平均差，最大差、p99 照样记录")

        except Exception as e:  # 一项出错不影响其余各项
            record(4, "不同微批大小下训练端 logprob 一致", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 1 LoRA 生效 ----------
        try:
            probe = max(segs, key=lambda s: len(s["ids"]) if len(s["ids"]) < 4000 else 0)
            pmask = [1 if (t > 0) else 0 for t in range(len(probe["ids"]))]
            gen_pos = [t for t, m in enumerate(probe["mask"]) if m]
            v_base = sampler.prompt_logprobs(probe["ids"])
            with tr.model.disable_adapter():
                t_base = tr.logprobs(probe["ids"], pmask).tolist()
            pos = [t for t in range(1, len(probe["ids"])) if v_base[t] is not None]
            sigma = statistics.mean(abs(t_base[t - 1] - v_base[t]) for t in pos)   # 不带 LoRA 时两端的差：BF16 噪声的大小

            def judge(v_l, t_l):
                """判定规则：(1) 两端的 LoRA 效应都 > 5σ；(2) 两端效应相差 ≤ 15%；(3) 逐 token 的 LoRA 变化两端之差的均值 ≤ 4σ。
                只有 BF16 噪声时 (3) 约 1.4–1.6σ（两份独立噪声相减）；采样端没带上 LoRA、对错层、比例不对时，(1)(2)(3) 至少一条明显不满足。"""
                ps = [t for t in pos if v_l[t] is not None]
                dv = [v_l[t] - v_base[t] for t in ps]
                dt = [t_l[t - 1] - t_base[t - 1] for t in ps]
                ev = statistics.mean(abs(x) for x in dv)
                et = statistics.mean(abs(x) for x in dt)
                dd = stats([a - b for a, b in zip(dv, dt)])
                r = {"effect_sampler": ev, "effect_trainer": et, "effect_rel_diff": abs(ev - et) / max(et, 1e-9),
                     "change_diff": dd, "change_diff_over_sigma": dd["mean"] / sigma,
                     "corr": statistics.correlation(dv, dt) if len(ps) > 2 and ev > 0 else None}
                r["pass"] = ev > 5 * sigma and et > 5 * sigma and r["effect_rel_diff"] <= 0.15 and dd["mean"] <= 4 * sigma
                return r

            def measure(std, seed, faults=()):
                torch.manual_seed(seed)
                for _, mod in tr.lora_modules():                 # 给 LoRA 一个确定的非零初值（B 原本是 0）
                    torch.nn.init.normal_(mod.lora_B["default"].weight, std=std)
                t_l = tr.logprobs(probe["ids"], pmask).tolist()
                tr.export(SHM)
                sampler.update_weights(SHM)
                out = {"correct": judge(sampler.prompt_logprobs(probe["ids"]), t_l)}
                for name, fault in faults:
                    if fault is None:                             # 采样端没带上 LoRA：采样端还是基础模型
                        out[name] = judge(v_base, t_l)
                        continue
                    tr.export(SHM, fault=fault)
                    sampler.update_weights(SHM)
                    out[name] = judge(sampler.prompt_logprobs(probe["ids"]), t_l)
                return out, t_l

            small_std = float(os.environ.get("M2P_LORA_STD", "0.002"))
            big_std = 10 * small_std
            faults = [("fault_sampler_without_lora", None), ("fault_wrong_layer", {"layer_shift": 1}),
                      ("fault_scale_x2", {"scale": 2.0})]
            big, _ = measure(big_std, 2, faults)
            # 恢复成小的 LoRA（同一个随机种子），后面几项（带 LoRA 生成等）和以前一样；小 LoRA 的数照记
            small, t_lora = measure(small_std, 1)
            v_lora = sampler.prompt_logprobs(probe["ids"])
            caught = {k: not v["pass"] for k, v in big.items() if k != "correct"}
            ok = big["correct"]["pass"] and all(caught.values())
            gset = set(gen_pos)
            gp = [t for t in pos if t in gset and v_lora[t] is not None]
            record(1, "LoRA 真的生效", ok, sigma_no_lora=sigma, tokens=len(pos), test_lora_std=big_std,
                   correct=big["correct"], injected_faults={k: v for k, v in big.items() if k != "correct"},
                   faults_caught=caught, small_lora={"std": small_std, **small["correct"]},
                   generated_tokens_only={"tokens": len(gp), "sigma_no_lora": statistics.mean(
                       abs(t_base[t - 1] - v_base[t]) for t in gp) if gp else None},
                   criterion="σ = 不带 LoRA 时采样端与训练端 logprob 之差的均值（BF16 噪声）。测试 LoRA（B 初值 std 0.02）下："
                             "两端效应都 > 5σ，两端效应相差 ≤ 15%，逐 token 的 LoRA 变化两端之差的均值 ≤ 4σ；"
                             "并且同一规则对注入的三种故障（采样端没带 LoRA、LoRA 对错层、比例 ×2）都判不通过",
                   original_rule="（std 0.002 的 LoRA）两端效应都大于带 LoRA 时两端差均值的 5 倍，且该均值 < 0.05",
                   criterion_change="std 0.002 时 LoRA 效应（约 0.11）和 BF16 噪声（σ 约 0.04）同一量级，两端差带不带 LoRA 时"
                                    "的舍入并不相同、相减抵不掉（逐 token 变化之差 0.060 ≈ √(0.045²+0.038²)），按原规则这个大小的"
                                    "LoRA 永远过不了，也分不出对错；改为放大测试 LoRA 并以 σ 为尺度判，同时在自检里注入三种故障，"
                                    "证明规则判得出来")
            MATERIAL["lora_compare"] = [{"token": chat.decode([probe["ids"][t]]), "sampler_without_lora": v_base[t],
                                         "sampler_with_lora": v_lora[t], "trainer_with_lora": t_lora[t - 1],
                                         "trainer_without_lora": t_base[t - 1]} for t in gen_pos[:40]]

        except Exception as e:  # 一项出错不影响其余各项
            record(1, "LoRA 真的生效", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 2 长 prompt 带 LoRA 生成 ----------
        try:
            longs = [s for s in segs if 2000 <= s["prompts"][-1] <= 3500] or sorted(segs, key=lambda s: -s["prompts"][-1])
            lp_ = longs[0]
            prompt = lp_["ids"][:lp_["prompts"][-1]]
            outs = {}
            for T in (1.0, 0.0):
                r = sampler.complete(prompt, 1024, T, seed=5, stop_ids=chat.stop_ids)
                text = chat.decode(r["ids"])
                vals = [v for v in r["lp"] if v is not None]
                printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(1, len(text))
                grams = [text[i:i + 8] for i in range(0, max(0, len(text) - 8))]
                outs[str(T)] = {"tokens": len(r["ids"]), "nan": sum(1 for v in vals if not math.isfinite(v)),
                                "replacement_chars": text.count("�"), "printable_frac": printable,
                                "distinct_8gram_frac": len(set(grams)) / max(1, len(grams)),
                                "has_think_end": "</think>" in text, "excerpt": text[:600]}
            ok2 = all(o["nan"] == 0 and o["replacement_chars"] == 0 and o["printable_frac"] > 0.98 and
                      o["distinct_8gram_frac"] > 0.3 for o in outs.values())
            record(2, "2–3k token 真实 prompt 带 LoRA 生成：无 NaN、无乱码", ok2, prompt_tokens=len(prompt),
                   **{f"T{k}": {kk: vv for kk, vv in v.items() if kk != "excerpt"} for k, v in outs.items()})
            MATERIAL["long_prompt_generation"] = {"prompt_tokens": len(prompt), "outputs": outs}

        except Exception as e:  # 一项出错不影响其余各项
            record(2, "2–3k token 真实 prompt 带 LoRA 生成：无 NaN、无乱码", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 5 优势符号、逐 token logprob、温度 ----------
        try:
            seg = sorted(segs, key=lambda s: sum(s["mask"]))[len(segs) // 2]
            saved = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
            res5 = {}
            for adv in (1.0, -1.0):
                for n, p in tr.model.named_parameters():
                    if n in saved:
                        p.data.copy_(saved[n])
                opt = torch.optim.Adam(tr.params, lr=1e-3)   # 新的 Adam：第一步每个参数大约移动 lr，方向是梯度的符号
                before = float(tr.logprobs(seg["ids"], seg["mask"]).mean())
                tr.opt, keep = opt, tr.opt
                tr.train_step([{"ids": seg["ids"], "mask": seg["mask"], "lp": seg["lp"], "adv": adv}], "grpo",
                              max_grad_norm=1e9)
                tr.opt = keep
                after = float(tr.logprobs(seg["ids"], seg["mask"]).mean())
                res5[f"adv{adv:+.0f}"] = {"before": before, "after": after, "delta": after - before,
                                      "lora_dtype": str(tr.params[0].dtype)}
            for n, p in tr.model.named_parameters():
                if n in saved:
                    p.data.copy_(saved[n])
            gen_tokens = sum(sum(s["mask"]) for s in segs)
            have_lp = sum(1 for s in segs for v, m in zip(s["lp"], s["mask"]) if m and v is not None)
            appended = sum(t["appended_stop"] for recs in trajs for r in recs for t in r.get("turn_stats", []))
            # 温度：温度 1 时"采样到概率最高的 token"的频率应当等于最高概率的平均值；温度被改低时频率会偏高
            temp = {}
            for T in (1.0, 0.6):
                hits, ptop = [], []
                for s in segs[:6]:
                    pr = s["ids"][:s["prompts"][0]]
                    r = sampler.complete(pr, 256, T, seed=11, stop_ids=[], top_logprobs=1)
                    for tok, lpv, top in zip(r["ids"], r["lp"], r["top"]):
                        if not top:
                            continue
                        best = max(top.items(), key=lambda kv: kv[1])
                        hits.append(int(str(best[0]).split(":")[-1]) == tok)
                        ptop.append(math.exp(best[1]))
                n = len(hits)
                var = sum(p * (1 - p) for p in ptop)
                temp[f"T{T}"] = {"n": n, "top1_freq": sum(hits) / n, "mean_p_top1": sum(ptop) / n,
                                 "z": (sum(hits) - sum(ptop)) / math.sqrt(max(var, 1e-9))}
            ok5 = (res5["adv+1"]["delta"] > 0 and res5["adv-1"]["delta"] < 0 and have_lp + appended >= gen_tokens and
                   abs(temp["T1.0"]["z"]) < 4)
            record(5, "优势符号单步测试；逐 token logprob；温度没被改写", ok5, **res5, generated_tokens=gen_tokens,
                   with_sampler_logprob=have_lp, stop_token_appended=appended, temperature_test=temp)

        except Exception as e:  # 一项出错不影响其余各项
            record(5, "优势符号单步测试；逐 token logprob；温度没被改写", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 6 训练样本与采样时看到的逐 token 一致 ----------
        try:
            exact = calls_consistent

            def rescore(s, lp=None):
                """整段重新打分（采样端），和逐轮生成时返回的 logprob 比，查 logprob 有没有对错位置。"""
                lp = s["lp"] if lp is None else lp
                again = sampler.prompt_logprobs(s["ids"])
                return [again[t] - lp[t] for t, m in enumerate(s["mask"])
                        if m and lp[t] is not None and again[t] is not None]

            ex_bad = ex_n = 0
            for s in segs:
                b, n = exact(s)
                ex_bad, ex_n = ex_bad + b, ex_n + n
            # 把 LoRA 清零，使采样端回到基础模型（采样时用的就是基础模型），再整段重新打分
            for _, mod in tr.lora_modules():
                torch.nn.init.zeros_(mod.lora_B["default"].weight)
            tr.export(SHM)
            sampler.update_weights(SHM)
            d6 = []
            for s in segs[:12]:
                d6 += rescore(s)
            st6 = stats(d6)
            rescore_ok = st6["n"] > 0 and st6["mean"] < 0.02 and st6["p99"] < 0.5
            # 故障注入：同一套判定对下面三种"真的不一致"必须判不通过
            s0 = next(s for s in segs if len(s.get("calls", [])) >= 2 and chat.think_end_id in
                      s["ids"][s["calls"][0]["at"]:s["calls"][0]["at"] + s["calls"][0]["recv"]["n"]])
            a0 = s0["calls"][0]["at"]
            k = a0 - 3                                              # 前文里多一个 token（如两个换行并成/拆开）
            ins = {**s0, "ids": s0["ids"][:k] + [s0["ids"][k]] + s0["ids"][k:],
                   "mask": s0["mask"][:k] + [0] + s0["mask"][k:], "lp": s0["lp"][:k] + [None] + s0["lp"][k:]}
            te = s0["ids"].index(chat.think_end_id, a0)              # 第一轮的思考被删掉（模板删思考那类不一致）
            dele = {**s0, "ids": s0["ids"][:a0] + s0["ids"][te:], "mask": s0["mask"][:a0] + s0["mask"][te:],
                    "lp": s0["lp"][:a0] + s0["lp"][te:]}
            shifted_lp = [None] + s0["lp"][:-1]                      # logprob 和 token 错开一位
            f_ins, f_del = exact(ins)[0], exact(dele)[0]
            st_shift = stats(rescore(s0, shifted_lp))
            st_ins = stats(rescore(ins))
            faults = {"inserted_token": {"exact_mismatched_calls": f_ins, "rescore": st_ins,
                                         "caught": f_ins > 0 or not (st_ins["mean"] < 0.02 and st_ins["p99"] < 0.5)},
                      "first_turn_thinking_deleted": {"exact_mismatched_calls": f_del, "caught": f_del > 0},
                      "logprob_shifted_by_one": {"rescore": st_shift,
                                                 "caught": not (st_shift["mean"] < 0.02 and st_shift["p99"] < 0.5)}}
            same_tpl = n_tpl = 0
            for recs in trajs:
                for r in recs:
                    for s in r.get("segments", [])[:1]:
                        msgs = [{"role": "system", "content": __import__("roommate.agent_llm", fromlist=["x"]).system_prompt(
                            0.15, "refuse_told")}] + r["transcript"]
                        nud = [i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"] == NUDGE]
                        tpl = chat.encode(chat.render(msgs[:nud[0]] if nud else msgs, add_generation_prompt=False))
                        n_tpl += 1
                        same_tpl += int(tpl[:len(s["ids"])] == s["ids"][:len(tpl)])
            ok6 = ex_n > 0 and ex_bad == 0 and rescore_ok and all(f["caught"] for f in faults.values())
            record(6, "训练样本的 token 序列与采样时看到的逐 token 一致", ok6,
                   exact_calls_checked=ex_n, exact_calls_mismatched=ex_bad, rescore_vs_rollout=st6,
                   injected_faults=faults, segments_equal_to_template_rendering=same_tpl, segments_checked=n_tpl,
                   criterion="逐次调用核对：训练样本 = 发给采样端的序列 + 采样端返回的序列（指纹比对，零容差）；"
                             "整段重新打分和采样时 logprob 之差：均值 < 0.02，p99 < 0.5（查 logprob 对错位置；最大差照记）；"
                             "并且对注入的三种不一致（前文多一个 token、第一轮思考被删、logprob 错开一位）都判不通过",
                   original_rule="整段重新打分和采样时 logprob 之差：均值 < 0.02，最大差 < 0.5",
                   original_rule_pass=st6["n"] > 0 and st6["max"] < 0.5 and st6["mean"] < 0.02,
                   criterion_change="原规则只靠数值比：单个 token 的 BF16 舍入就能超过最大差门槛（17:44 那次 18246 个 token 里"
                                    "最大 0.54，均值 0.0125），而前文差一个 token 时数值不一定变大；改为指纹逐 token 核对（零容差）"
                                    "判序列一致，数值比只查 logprob 位置，用均值和 p99")
            MATERIAL["template_drop_example"] = template_example(chat, trajs)

        except Exception as e:  # 一项出错不影响其余各项
            record(6, "训练样本的 token 序列与采样时看到的逐 token 一致", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 7 loss 掩码 ----------
        try:
            not_gen, checked, flags = 0, 0, []
            for recs in trajs:
                for r in recs:
                    for s in r.get("segments", []):
                        not_gen += len(loss_not_from_sampler(s, chat.stop_ids))
                        f, n = flagged_spans(s, r["transcript"], chat)
                        checked += n
                        flags += [dict(x, uid=r["uid"], idx=r["idx"]) for x in f]
            # 故障注入：把一段工具返回的 token 错算进 loss，判定必须不通过
            s7 = next(s for s in segs if 0 in s["mask"][s["calls"][0]["at"] + s["calls"][0]["recv"]["n"] + 2:])
            j = next(t for t in range(s7["calls"][0]["at"] + s7["calls"][0]["recv"]["n"] + 2, len(s7["ids"]))
                     if not s7["mask"][t])
            bad7 = {**s7, "mask": s7["mask"][:j] + [1] * 5 + s7["mask"][j + 5:]}
            fault_caught = len(loss_not_from_sampler(bad7, chat.stop_ids)) > 0
            ok7 = checked > 0 and not_gen == 0 and fault_caught
            record(7, "只有模型生成的 token 计入 loss", ok7, spans_checked=checked,
                   loss_tokens_not_from_sampler=not_gen,
                   injected_fault={"tool_response_tokens_marked_as_loss": 5, "caught": fault_caught},
                   old_rule_flags=len(flags), old_rule_flags_not_all_from_sampler=sum(not f["all_from_sampler"] for f in flags),
                   flag_examples=flags[:5],
                   criterion="计入 loss 的每个 token 都在采样端返回的范围里（按每次调用的指纹逐 token 核对；采样端停下但没返回的"
                             "停止 token 除外）；并且把工具返回的 token 错算进 loss 时判不通过",
                   original_rule="计入 loss 的片段里没有模板标记，也不含任何一条完整的工具结果或用户第一句话的前 30 字",
                   original_rule_pass=not flags,
                   criterion_change="原规则按文字找，模型在思考里原样引用工具结果或用户的话也会被标成\"不是生成的\"；"
                                    "改为逐 token 核对是不是采样端生成的（精确）；原规则标出的片段照记，并注明是否全是采样端生成的")
            MATERIAL["mask_example"] = mask_example(chat, trajs)

        except Exception as e:  # 一项出错不影响其余各项
            record(7, "只有模型生成的 token 计入 loss", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 8 吞吐 ----------
        try:
            pr = segs[0]["ids"][:segs[0]["prompts"][0]]
            thr = {}
            for c in (1, 8, 32):
                t0 = time.time()
                with ThreadPoolExecutor(max_workers=c) as pool:
                    rs = list(pool.map(lambda i: sampler.complete(pr, 256, 1.0, seed=100 + i, stop_ids=[], ignore_eos=True),
                                       range(c)))
                sec = time.time() - t0
                thr[str(c)] = {"seconds": sec, "gen_tokens": sum(len(r["ids"]) for r in rs),
                               "tokens_per_s": sum(len(r["ids"]) for r in rs) / sec}
            tp = [thr[k]["tokens_per_s"] for k in ("1", "8", "32")]
            record(8, "1、8、32 路并发采样时吞吐随并发数上升", tp[0] < tp[1] < tp[2], **{f"c{k}": v for k, v in thr.items()})

        except Exception as e:  # 一项出错不影响其余各项
            record(8, "1、8、32 路并发采样时吞吐随并发数上升", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 9 奖励、优势、记忆存取 ----------
        try:
            grp = rollouts(sampler, chat, users[:1], sw, n_rep=8, seed=9)
            R = [[r["grade"]["score"] for r in recs] for recs in grp]
            G = [returns(x) for x in R]
            A = advantages(G)
            ref = []
            for k in range(2):
                col = [sum(R[i][k:]) for i in range(8)]
                mu = sum(col) / 8
                sd = math.sqrt(sum((x - mu) ** 2 for x in col) / 8)
                ref.append([0.0 if sd < 1e-9 else (x - mu) / (sd + 1e-6) for x in col])
            adv_ok = all(abs(A[i][k] - ref[k][i]) < 1e-9 for i in range(8) for k in range(2))
            its, _ = build_items(grp, A, max_len=10 ** 9)
            seg_ok = bool(its) and all(it["adv"] == A[it["i"]][it["k"]] for it in its) and \
                len(its) == sum(1 for recs in grp for r in recs for sg in r["segments"] if sum(sg["mask"]))
            mem_ok, mem_detail = memory_roundtrip(sw, users[1])
            record(9, "奖励和优势与设计公式一致；记忆保存后再恢复内容不变", adv_ok and seg_ok and mem_ok,
                   rewards=R, returns=G, advantages=A, items=len(its), memory=mem_detail)
            MATERIAL["group_example"] = {"uid": users[0]["uid"], "rewards": R, "returns": G, "advantages": A,
                                         "chosen": [[r["chosen"] for r in recs] for recs in grp],
                                         "questions": [[len(r["questions"]) for r in recs] for recs in grp]}
        except Exception as e:  # 一项出错不影响其余各项
            record(9, "奖励和优势与设计公式一致；记忆保存后再恢复内容不变", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        MATERIAL["seconds"] = time.time() - t_all
        MATERIAL["vllm_boot_seconds"] = boot
        dump(os.path.join(D, "material.json"), MATERIAL)
    finally:
        serve.stop(proc)


def memory_roundtrip(sw, user):
    """用脚本策略跑第 1 次会话，存档、恢复，比较记忆；再接着跑第 2 次，和不存档直接跑的结果比较。"""
    from ..policies import RememberNoUpdate
    base = {k: sw[k] for k in ("version", "seed", "events", "config")}
    w = dict(base, users=[user], n_users=1, n_sessions=2)
    e1 = Env(w)
    e1.run(RememberNoUpdate(), until=1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ck.json")
        e1.save(p)
        e2 = Env.load(p)
    same_mem = e1.episodes[0].memory.entries == e2.episodes[0].memory.entries
    e1.run(RememberNoUpdate())
    e2.run(RememberNoUpdate())
    strip = lambda rs: [{k: v for k, v in r.items()} for r in rs]
    same_next = strip(e1.records()) == strip(e2.records())
    return same_mem and same_next, {"memory_entries": len(e1.episodes[0].memory.entries), "same_memory": same_mem,
                                    "same_second_session": same_next,
                                    "memory_after_first": {k: v["value"] for k, v in e1.episodes[0].memory.entries.items()}}


def loss_not_from_sampler(seg, stop_ids):
    """计入 loss（mask = 1）却不在采样端返回范围里的位置。采样端在停止 token 上停下、但输出里没带它时，
    agent 会补上这个停止 token（它是模型自己选的，只是没返回），不算。"""
    got = set()
    for c in seg.get("calls", []):
        if "recv" in c:
            got.update(range(c["at"], c["at"] + c["recv"]["n"]))
    out = []
    for t, m in enumerate(seg["mask"]):
        if m and t not in got:
            appended_stop = seg["ids"][t] in stop_ids and (t - 1) in got and seg["lp"][t] is None
            if not appended_stop:
                out.append(t)
    return out


def flagged_spans(seg, transcript, chat):
    """原来第 7 项按文字找的那几种：计入 loss 的片段里有模板标记、某条完整的工具结果、或用户第一句话的前 30 字。
    每处另记这段是否全是采样端生成的（逐 token 按调用指纹核对）。"""
    got = set()
    for c in seg.get("calls", []):
        if "recv" in c:
            got.update(range(c["at"], c["at"] + c["recv"]["n"]))
    tool_texts = [m["content"] for m in transcript if m["role"] == "tool"]
    user0 = transcript[0]["content"][:30]
    spans, cur = [], []
    for t, m in enumerate(seg["mask"]):
        if m:
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    out = []
    for sp in spans:
        txt = chat.decode([seg["ids"][t] for t in sp])
        why, hit = None, None
        for mk in ("<tool_response>", "<|im_start|>"):
            if mk in txt:
                why, hit = "template_marker", mk
        if why is None:
            hit = next((x for x in tool_texts if len(x) > 8 and x in txt), None)
            why = "tool_text" if hit else None
        if why is None and user0 in txt:
            why, hit = "user_text", user0
        if why:
            i = txt.find(hit)
            out.append({"why": why, "matched": hit[:200], "span_tokens": len(sp),
                        "all_from_sampler": all(t in got for t in sp),
                        "context": txt[max(0, i - 200):i + len(hit) + 120]})
    return out, len(spans)


def calls_consistent(seg):
    """自检第 6 项的逐次调用核对（零容差、不受数值误差影响）：训练样本里每次调用之前的部分 = 当时发给采样端的序列，
    紧跟着的 = 采样端返回的序列（两边都是在 HTTP 这一层记下的指纹）。返回 (对不上的调用数, 调用数)。"""
    from .sampler import ids_sha
    bad = 0
    for c in seg.get("calls", []):
        a = c["at"]
        if "sent" not in c or c["sent"]["n"] != a or ids_sha(seg["ids"][:a]) != c["sent"]["sha"] \
                or ids_sha(seg["ids"][a:a + c["recv"]["n"]]) != c["recv"]["sha"]:
            bad += 1
    return bad, len(seg.get("calls", []))


def template_example(chat, trajs):
    """一处模板删掉思考的前后对比：取一次真实会话的前两轮，在第 1 轮之后插一条真实用户消息（提醒）。"""
    from ..agent_llm import system_prompt
    for recs in trajs:
        for r in recs:
            t = r["transcript"]
            asst = [i for i, m in enumerate(t) if m["role"] == "assistant" and m.get("reasoning_content")]
            if len(asst) >= 1:
                i = asst[0]
                msgs = [{"role": "system", "content": system_prompt(0.15, "refuse_told")}] + t[:i + 1]
                before = chat.render(msgs, add_generation_prompt=True)
                after = chat.render(msgs + [{"role": "user", "content": NUDGE}], add_generation_prompt=True)
                tail = lambda s: s[s.index(t[0]["content"][:20]):] if t[0]["content"][:20] in s else s[-3000:]
                return {"uid": r["uid"], "idx": r["idx"], "before_tail": tail(before)[-2500:],
                        "after_tail": tail(after)[-2500:], "reasoning_chars": len(t[i]["reasoning_content"]),
                        "reasoning_in_before": t[i]["reasoning_content"][:50] in before,
                        "reasoning_in_after": t[i]["reasoning_content"][:50] in after}
    return None


def mask_example(chat, trajs):
    """一条训练样本按"计入 loss / 不计入"切成连续的片段（文字 + 是否计入）。"""
    best = None
    for recs in trajs:
        for r in recs:
            for s in r.get("segments", []):
                n_tool = sum(1 for m in r["transcript"] if m["role"] == "tool")
                if n_tool >= 2 and len(s["ids"]) < 9000:
                    best = s
                    break
            if best:
                break
        if best:
            break
    if best is None:
        return None
    pieces, cur, flag = [], [], None
    for t, m in zip(best["ids"], best["mask"]):
        if flag is None or m == flag:
            cur.append(t)
        else:
            pieces.append({"loss": bool(flag), "text": chat.decode(cur)})
            cur = [t]
        flag = m
    pieces.append({"loss": bool(flag), "text": chat.decode(cur)})
    return {"tokens": len(best["ids"]), "loss_tokens": sum(best["mask"]), "pieces": pieces}


if __name__ == "__main__":
    main()
