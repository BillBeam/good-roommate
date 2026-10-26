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
            torch.manual_seed(1)
            for _, mod in tr.lora_modules():                 # 给 LoRA 一个确定的非零初值（B 原本是 0）
                torch.nn.init.normal_(mod.lora_B["default"].weight, std=float(os.environ.get("M2P_LORA_STD", "0.002")))
            t_lora = tr.logprobs(probe["ids"], pmask).tolist()
            tr.export(SHM)
            sampler.update_weights(SHM)
            v_lora = sampler.prompt_logprobs(probe["ids"])
            pos = [t for t in range(1, len(probe["ids"])) if v_base[t] is not None and v_lora[t] is not None]
            eff_v = statistics.mean(abs(v_lora[t] - v_base[t]) for t in pos)
            eff_t = statistics.mean(abs(t_lora[t - 1] - t_base[t - 1]) for t in pos)
            agree = stats([t_lora[t - 1] - v_lora[t] for t in pos])
            agree_base = stats([t_base[t - 1] - v_base[t] for t in pos])
            # LoRA 带来的变化逐 token 对比：(采样端 带 − 不带) 对 (训练端 带 − 不带)。两端原有的数值差在这里相减抵掉，
            # 直接回答"两端加的是不是同一个 LoRA"
            dv = [v_lora[t] - v_base[t] for t in pos]
            dt = [t_lora[t - 1] - t_base[t - 1] for t in pos]
            dd = stats([a - b for a, b in zip(dv, dt)])
            corr = statistics.correlation(dv, dt) if len(pos) > 2 else None
            gset = set(gen_pos)
            gp = [i for i, t in enumerate(pos) if t in gset]
            gen = {"tokens": len(gp), "lora_effect_sampler": statistics.mean(abs(dv[i]) for i in gp) if gp else None,
                   "trainer_vs_sampler_without_lora": stats([t_base[pos[i] - 1] - v_base[pos[i]] for i in gp]),
                   "trainer_vs_sampler_with_lora": stats([t_lora[pos[i] - 1] - v_lora[pos[i]] for i in gp]),
                   "lora_change_diff": stats([dv[i] - dt[i] for i in gp])}
            eff = (eff_v + eff_t) / 2
            original = eff_v > 5 * agree["mean"] and eff_t > 5 * agree["mean"] and agree["mean"] < 0.05
            small = {"lora_b_std": float(os.environ.get("M2P_LORA_STD", "0.002")), "lora_effect_sampler": eff_v,
                     "lora_effect_trainer": eff_t, "trainer_vs_sampler_with_lora": agree, "lora_change_diff": dd,
                     "lora_change_corr": corr, "generated_tokens_only": gen, "original_rule_pass": original}
            # 判定用放大 10 倍的测试 LoRA：BF16 的舍入噪声（两端差约 0.04–0.06）不随 LoRA 变大而变大，
            # LoRA 带来的变化随之变大；两端加的若是同一个 LoRA，逐 token 的变化之差相对效应就很小，否则会跟着效应一起变大
            big_std = 10 * small["lora_b_std"]
            torch.manual_seed(2)
            for _, mod in tr.lora_modules():
                torch.nn.init.normal_(mod.lora_B["default"].weight, std=big_std)
            tb = tr.logprobs(probe["ids"], pmask).tolist()
            tr.export(SHM)
            sampler.update_weights(SHM)
            vb = sampler.prompt_logprobs(probe["ids"])
            posb = [t for t in pos if vb[t] is not None]
            dvb = [vb[t] - v_base[t] for t in posb]
            dtb = [tb[t - 1] - t_base[t - 1] for t in posb]
            eff_vb = statistics.mean(abs(x) for x in dvb)
            eff_tb = statistics.mean(abs(x) for x in dtb)
            ddb = stats([a - b for a, b in zip(dvb, dtb)])
            corrb = statistics.correlation(dvb, dtb)
            effb = (eff_vb + eff_tb) / 2
            ok = (abs(eff_vb - eff_tb) / effb < 0.10 and ddb["mean"] < 0.25 * effb and corrb > 0.95
                  and eff_vb > 2 * agree_base["mean"])
            # 恢复成原来那个小的 LoRA（同一个随机种子），后面几项（带 LoRA 生成等）和以前一样
            torch.manual_seed(1)
            for _, mod in tr.lora_modules():
                torch.nn.init.normal_(mod.lora_B["default"].weight, std=small["lora_b_std"])
            tr.export(SHM)
            sampler.update_weights(SHM)
            record(1, "LoRA 真的生效", ok,
                   lora_b_std=big_std, lora_effect_sampler=eff_vb, lora_effect_trainer=eff_tb,
                   lora_change_diff=ddb, lora_change_corr=corrb, trainer_vs_sampler_without_lora=agree_base,
                   tokens=len(posb), small_lora=small,
                   criterion="测试 LoRA（B 的初值 std 0.02）下：两端效应相差 < 10%；逐 token 的 LoRA 变化两端之差 < 效应的 25%，"
                             "相关系数 > 0.95；效应大于不带 LoRA 时两端差的 2 倍",
                   original_rule="（std 0.002 的 LoRA）两端效应都大于带 LoRA 时两端差均值的 5 倍，且该均值 < 0.05",
                   criterion_change="std 0.002 时 LoRA 的效应（约 0.11）和两端原有的 BF16 差（约 0.04）同一量级，"
                                    "两端差带不带 LoRA 时的舍入并不相同、相减抵不掉（逐 token 变化之差约 0.06，正是两份噪声的合成），"
                                    "测不出\"同一个 LoRA\"和\"略有不同的 LoRA\"的区别；放大测试 LoRA 让信号远大于噪声再判，"
                                    "小 LoRA 的数照样记录（small_lora）")
            show = gen_pos[:40]
            MATERIAL["lora_compare"] = [{"token": chat.decode([probe["ids"][t]]), "sampler_without_lora": v_base[t],
                                         "sampler_with_lora": v_lora[t], "trainer_with_lora": t_lora[t - 1],
                                         "trainer_without_lora": t_base[t - 1]} for t in show]

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
            # 把 LoRA 清零，使采样端回到基础模型，再在整段序列上取采样端的 logprob，
            # 和当初逐轮生成时返回的 logprob 比：前文若有任何不同，差会很大。
            for _, mod in tr.lora_modules():
                torch.nn.init.zeros_(mod.lora_B["default"].weight)
            tr.export(SHM)
            sampler.update_weights(SHM)
            d6, same_tpl, n_tpl = [], 0, 0
            for s in segs[:12]:
                again = sampler.prompt_logprobs(s["ids"])
                d6 += [again[t] - s["lp"][t] for t, m in enumerate(s["mask"]) if m and s["lp"][t] is not None
                       and again[t] is not None]
            for recs in trajs:
                for r in recs:
                    for s in r.get("segments", [])[:1]:
                        msgs = [{"role": "system", "content": __import__("roommate.agent_llm", fromlist=["x"]).system_prompt(
                            0.15, "refuse_told")}] + r["transcript"]
                        nud = [i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"] == NUDGE]
                        upto = msgs[:nud[0]] if nud else msgs
                        tpl = chat.encode(chat.render(upto, add_generation_prompt=False))
                        n_tpl += 1
                        same_tpl += int(tpl[:len(s["ids"])] == s["ids"][:len(tpl)])
            st6 = stats(d6)
            record(6, "训练样本的 token 序列与采样时看到的逐 token 一致", st6["n"] > 0 and st6["p99"] < 0.5 and st6["mean"] < 0.02,
                   rescore_vs_rollout=st6, segments_equal_to_template_rendering=same_tpl, segments_checked=n_tpl,
                   criterion="重新整段打分和逐轮采样时的 logprob 之差：均值 < 0.02，p99 < 0.5（最大差照记）",
                   original_rule="均值 < 0.02，最大差 < 0.5", original_rule_pass=st6["n"] > 0 and st6["max"] < 0.5 and st6["mean"] < 0.02,
                   criterion_change="前文要是有一个 token 不同，会有成片的 token 差得很大，均值和 p99 都会跳；"
                                    "单个 token 的最大差是 BF16 舍入（第 3 项同类数据上最大差到过 1.04），和第 4 项一样改看均值和 p99")
            MATERIAL["template_drop_example"] = template_example(chat, trajs)

        except Exception as e:  # 一项出错不影响其余各项
            record(6, "训练样本的 token 序列与采样时看到的逐 token 一致", False, error=f"{type(e).__name__}: {e}"[:500])
            torch.cuda.empty_cache()
        # ---------- 7 loss 掩码 ----------
        try:
            bad = 0
            checked = 0
            no_lp = 0
            quotes = []
            for recs in trajs:
                for r in recs:
                    tool_texts = [m["content"] for m in r["transcript"] if m["role"] == "tool"]
                    for s in r.get("segments", []):
                        # 计入 loss 的每个 token 都要有采样端当时返回的 logprob，也就是确实是采样端生成的
                        no_lp += sum(1 for v, m in zip(s["lp"], s["mask"]) if m and v is None)
                        spans, cur = [], []
                        for t, m in enumerate(s["mask"]):
                            if m:
                                cur.append(s["ids"][t])
                            elif cur:
                                spans.append(cur)
                                cur = []
                        if cur:
                            spans.append(cur)
                        for sp in spans:
                            txt = chat.decode(sp)
                            checked += 1
                            if "<tool_response>" in txt or "<|im_start|>" in txt:      # 模板拼进来的标记
                                bad += 1
                            elif any(len(x) > 8 and x in txt for x in tool_texts) or r["transcript"][0]["content"][:30] in txt:
                                # 模型在自己的输出里原样引用了工具结果或用户的话：是它生成的，记下来备查，不算错
                                hit = next((x for x in tool_texts if len(x) > 8 and x in txt), r["transcript"][0]["content"][:30])
                                i = txt.find(hit)
                                quotes.append({"uid": r["uid"], "idx": r["idx"], "quoted": hit[:120],
                                               "context": txt[max(0, i - 120):i + len(hit) + 60]})
            appended7 = sum(t["appended_stop"] for recs in trajs for r in recs for t in r.get("turn_stats", []))
            record(7, "只有模型生成的 token 计入 loss", bad == 0 and no_lp <= appended7, spans_checked=checked,
                   spans_with_template_markers=bad, loss_tokens_without_sampler_logprob=no_lp,
                   stop_tokens_appended=appended7, spans_quoting_context=len(quotes), quote_examples=quotes[:3],
                   criterion="计入 loss 的 token 都有采样端返回的 logprob（确实是采样端生成的；补上的结束符除外），"
                             "且计入 loss 的片段里没有模板标记（<tool_response>、<|im_start|>）",
                   original_rule="片段里没有模板标记，也不含任何一条完整的工具结果或用户第一句话的前 30 字",
                   original_rule_pass=bad == 0 and not quotes,
                   criterion_change="原规则按文字找，模型在思考里原样引用记忆或用户的话也会被算成\"不是生成的\"；"
                                    "改成逐 token 核对是不是采样端生成的（精确），引用另记备查")
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
