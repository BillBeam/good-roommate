"""本地 agent：Qwen3.5-4B 跑在自己的 vLLM 服务上，逐 token 采样、逐 token 记录，供评测和 GRPO 共用。

和 API 基线（roommate.agent_llm.LLMAgent）一致的地方：同一段系统提示、同一组工具、同一个工具执行函数
（LLMAgent._exec）、同样最多 14 轮、没调工具时插同一句提醒、同样的格式问题分类。
不同的地方：
- 不经服务商的聊天接口，而是自己套 Qwen3.5 官方模板、按 token id 发给 vLLM；模型生成的 token 原样接在前文后面，
  只有在插入真实用户消息（提醒）时才按模板重新渲染前文（这时模板会删掉之前各轮的思考），见 qwen_chat 的"段"。
- 思考上限：思考用满 think_budget 个 token 还没结束，就补上"\\n</think>\\n\\n"（补的 token 不计入 loss），
  再给回答 answer_cap 个 token。和体检时硅基流动的 thinking_budget 4096 对应（它截断后回答会被思考污染，这里不会）。
- 每次会话在 ctx.notes 里留下：各段的 token、loss 掩码、采样端 logprob；每轮的思考长度、是否超上限、是否截断；
  以及只用于统计捷径指标的"会话开始时用户已经告诉过 agent 什么"（来自模拟用户，从不给模型看）。
"""

import hashlib

from ..agent_llm import FORMAT_KINDS, MAX_TURNS, NUDGE, _TOOL_TEXT, LLMAgent, system_prompt, tools


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class LocalAgent:
    def __init__(self, sampler, chat, temperature, think_budget=4096, answer_cap=8192, max_model_len=32768,
                 seed=0, rep=0, label="qwen3.5-4b", keep_tokens=True, top_p=1.0):
        self.sampler = sampler
        self.chat = chat
        self.temperature = temperature
        self.top_p = top_p
        self.think_budget = think_budget
        self.answer_cap = answer_cap
        self.max_model_len = max_model_len
        self.seed = seed
        self.rep = rep
        self.name = f"local:{label}"
        self.keep_tokens = keep_tokens
        self.raise_errors = True   # 采样端出错是基础设施问题，不能记成"未作答"

    # ---------- 一轮生成（带思考上限） ----------
    def _generate(self, prompt, tag):
        c = self.chat
        room = self.max_model_len - len(prompt) - 8
        if room < 64:
            return None
        ids, lps, gen = [], [], []
        r = self.sampler.complete(prompt, min(self.think_budget, room), self.temperature, self.top_p,
                                  seed=_seed(tag, 0), stop_ids=c.stop_ids)
        calls = [{"at": len(prompt), **{k: r[k] for k in ("sent", "recv") if k in r}}]
        ids += r["ids"]
        lps += r["lp"]
        gen += [1] * len(r["ids"])
        think_hit = truncated = False
        appended = r["appended_stop"]
        if r["finish_reason"] == "length":
            if c.think_end_id not in r["ids"]:
                think_hit = True
                forced = c.encode("\n</think>\n\n")
                ids += forced
                lps += [None] * len(forced)
                gen += [0] * len(forced)
                remain = self.answer_cap
            else:
                remain = self.answer_cap - (len(r["ids"]) - r["ids"].index(c.think_end_id) - 1)
            remain = min(remain, self.max_model_len - len(prompt) - len(ids) - 8)
            if remain > 0:
                r2 = self.sampler.complete(prompt + ids, remain, self.temperature, self.top_p,
                                           seed=_seed(tag, 1), stop_ids=c.stop_ids)
                calls.append({"at": len(prompt) + len(ids), **{k: r2[k] for k in ("sent", "recv") if k in r2}})
                ids += r2["ids"]
                lps += r2["lp"]
                gen += [1] * len(r2["ids"])
                truncated = r2["finish_reason"] == "length"
                appended = appended or r2["appended_stop"]
            else:
                truncated = True
        think = ids.index(c.think_end_id) if c.think_end_id in ids else len(ids)
        think_gen = sum(gen[:think])
        return {"ids": ids, "lp": lps, "gen": gen, "think_tokens": think_gen, "think_hit": think_hit,
                "truncated": truncated, "appended_stop": appended, "calls": calls}

    def _new_segment(self, messages):
        ids = self.chat.encode(self.chat.render(messages, add_generation_prompt=True))
        return {"ids": ids, "mask": [0] * len(ids), "lp": [None] * len(ids), "prompts": [], "calls": []}

    # ---------- 一次会话 ----------
    def run_session(self, ctx):
        c = self.chat
        cfg = ctx.ep.config
        # 只用于统计（捷径指标），从不进模型的输入
        ctx.notes["told_at_start"] = dict(ctx.ep.user.told)
        ctx.notes["state_long"] = dict(ctx.ep.user.state)
        ctx.notes["temp_attr"] = ctx.sess["temp"]["attr"] if ctx.sess.get("temp") else None
        messages = [{"role": "system", "content": system_prompt(cfg["question_cost"], cfg["repeat_mode"])},
                    {"role": "user", "content": ctx.prompt}]
        tag = f"{self.seed}|{ctx.ep.seed}|{ctx.ep.plan['uid']}|{ctx.sess['idx']}|rep{self.rep}"
        seg = self._new_segment(messages)
        segments, issues, turn_stats = [], [], []
        usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        turns, overflow = 0, False
        for turns in range(1, MAX_TURNS + 1):
            seg["prompts"].append(len(seg["ids"]))
            g = self._generate(seg["ids"], f"{tag}|t{turns}")
            if g is None:
                overflow = True
                issues.append({"turn": turns, "kind": "context_overflow"})
                break
            seg["calls"] += g["calls"]          # 每次调用采样端：发出去的、收回来的 token 序列指纹（自检第 6 项核对）
            seg["ids"] += g["ids"]
            seg["mask"] += g["gen"]
            seg["lp"] += [lp if m else None for lp, m in zip(g["lp"], g["gen"])]
            usage["calls"] += 1
            usage["prompt_tokens"] += seg["prompts"][-1]
            usage["completion_tokens"] += sum(g["gen"])
            usage["reasoning_tokens"] += g["think_tokens"]
            turn_stats.append({k: g[k] for k in ("think_tokens", "think_hit", "truncated", "appended_stop")} |
                              {"gen_tokens": sum(g["gen"])})
            body = g["ids"][:-1] if g["ids"] and g["ids"][-1] in c.stop_ids else g["ids"]
            msg, calls, parse_issues = c.parse(c.decode(body))
            if g["truncated"]:
                issues.append({"turn": turns, "kind": "truncated"})
            if g["think_hit"]:
                issues.append({"turn": turns, "kind": "think_runaway", "reasoning_tokens": g["think_tokens"]})
            issues += [dict(x, turn=turns) for x in parse_issues]
            messages.append(msg)
            if not calls:
                kind = "tool_text" if _TOOL_TEXT.search(msg["content"] or "") or "<tool_call>" in c.decode(body) \
                    else "no_tool_call"
                issues.append({"turn": turns, "kind": kind, "text": (msg["content"] or "")[:300]})
                messages.append({"role": "user", "content": NUDGE})
                segments.append(seg)
                seg = self._new_segment(messages)   # 真实用户消息：按模板重新渲染（删掉之前的思考）
                continue
            results = []
            for i, (name, raw) in enumerate(calls):
                res = LLMAgent._exec(ctx, name, raw)
                results.append(res)
                messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": res})
            if ctx.done:
                break
            if not g["ids"] or g["ids"][-1] not in c.stop_ids:   # 截断的输出里也有完整的工具调用：补上模板的 <|im_end|>
                seg["ids"].append(c.im_end_id)
                seg["mask"].append(0)
                seg["lp"].append(None)
            suffix = c.encode(c.tool_suffix(results))
            seg["ids"] += suffix
            seg["mask"] += [0] * len(suffix)
            seg["lp"] += [None] * len(suffix)
        segments.append(seg)
        ctx.notes["turns"] = turns
        ctx.notes["format_issues"] = issues
        ctx.notes["usage"] = usage
        ctx.notes["turn_stats"] = turn_stats
        ctx.notes["hit_max_turns"] = turns == MAX_TURNS and not ctx.done
        ctx.notes["context_overflow"] = overflow
        ctx.notes["transcript"] = messages[1:]
        if self.keep_tokens:
            ctx.notes["segments"] = segments


__all__ = ["LocalAgent", "FORMAT_KINDS", "tools", "system_prompt"]
