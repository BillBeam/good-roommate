"""Claude agent：和 agent_llm.LLMAgent 用同一段系统提示、同一组工具（名字、说明、参数原样搬过来）、同样的轮数上限和提醒，
只是请求走 Anthropic Messages API 的原生格式。

和 DeepSeek 基线对不齐的地方（报告里逐条列出）：
- 不能设温度 0：这几个 Claude 模型不接受 temperature / top_p / top_k（发了就 400），只能用默认采样；也没有 seed。
  复现靠缓存；缓存键另带"种子|用户|会话"（cache_tag），不同用户碰巧相同的请求也各自独立采样。
- 思考：DeepSeek V4 默认开思考；这里四个模型都开自适应思考（thinking.type = adaptive），努力程度统一显式设成 EFFORT。
  Opus 5.5、Fable 5.1 本来就关不掉思考。思考内容只返回摘要（display = summarized），和 DeepSeek 一样，
  工具调用循环里把上一轮的思考块原样带回。
- 提示缓存：DeepSeek 自动缓存；这里开顶层自动缓存（cache_control）。只影响花费，不影响模型看到的内容。
- 遇到安全分类器拒答（stop_reason = refusal）不换模型重试：换模型会把别的模型的回答混进来；拒答记进 format_issues，这次会话按没作答计。
"""

import copy
import json

from .agent_llm import MAX_TURNS, NUDGE, TOOL_NAMES, _TOOL_TEXT, LLMAgent, system_prompt, tools
from .llm import usage_cost

EFFORT = "high"
MAX_TOKENS = 32000


def claude_tools(question_cost):
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in tools(question_cost)]


def _replay_block(b):
    """响应里的内容块 → 下一轮请求里原样带回的块（只留接口要的字段）。"""
    t = b["type"]
    if t == "text":
        return {"type": "text", "text": b["text"]}
    if t == "thinking":
        return {"type": "thinking", "thinking": b.get("thinking", ""), "signature": b["signature"]}
    if t == "redacted_thinking":
        return {"type": "redacted_thinking", "data": b["data"]}
    if t == "tool_use":
        return {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}}
    return b


def _strip_signatures(messages):
    """存进 transcripts.jsonl 的副本去掉思考块的签名（不透明的长串，只有带回接口时有用；缓存里仍是完整响应）。"""
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "thinking":
                    b.pop("signature", None)
                elif b.get("type") == "redacted_thinking":
                    b["data"] = f"<{len(b.get('data', ''))} 字节>"
    return out


class ClaudeAgent(LLMAgent):
    def __init__(self, client, model, seed=0, inject_true_memory=False, truncate_on_cache_miss=False,
                 effort=EFFORT, **_):
        super().__init__(client, model, seed=seed, inject_true_memory=inject_true_memory,
                         truncate_on_cache_miss=truncate_on_cache_miss, track_format=True)
        self.effort = effort

    def run_session(self, ctx):
        cfg = ctx.ep.config
        system = system_prompt(cfg["question_cost"], cfg["repeat_mode"])
        messages = [{"role": "user", "content": ctx.prompt}]
        tag = f"{self.seed}|{ctx.ep.plan['uid']}|{ctx.sess['idx']}"
        role = "control" if self.inject_true_memory else "agent"
        issues = []
        usage = {"calls": 0, "input_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0,
                 "output_tokens": 0, "cost_usd": 0.0}
        turns = 0
        refused = False
        for turns in range(1, MAX_TURNS + 1):
            body = {"model": self.model, "max_tokens": MAX_TOKENS, "system": system,
                    "tools": claude_tools(cfg["question_cost"]), "tool_choice": {"type": "auto"},
                    "thinking": {"type": "adaptive", "display": "summarized"},
                    "output_config": {"effort": self.effort},
                    "cache_control": {"type": "ephemeral"}, "messages": messages}
            resp = self.client.chat(body, role=role, cache_tag=tag)
            u = resp.get("usage") or {}
            usage["calls"] += 1
            usage["input_tokens"] += u.get("input_tokens") or 0
            usage["cache_write_tokens"] += u.get("cache_creation_input_tokens") or 0
            usage["cache_read_tokens"] += u.get("cache_read_input_tokens") or 0
            usage["output_tokens"] += u.get("output_tokens") or 0
            usage["cost_usd"] += usage_cost(self.model, u)[0]
            content = resp.get("content") or []
            calls = [b for b in content if b["type"] == "tool_use"]
            stop = resp.get("stop_reason")
            if stop == "max_tokens":
                issues.append({"turn": turns, "kind": "truncated"})
            for c in calls:
                if c["name"] not in TOOL_NAMES:
                    issues.append({"turn": turns, "kind": "unknown_tool", "name": c["name"][:80]})
            if stop == "refusal":
                issues.append({"turn": turns, "kind": "refusal",
                               "category": (resp.get("stop_details") or {}).get("category")})
                refused = True
                break
            messages.append({"role": "assistant", "content": [_replay_block(b) for b in content]})
            if not calls:
                text = "".join(b.get("text", "") for b in content if b["type"] == "text")
                issues.append({"turn": turns, "kind": "tool_text" if _TOOL_TEXT.search(text) else "no_tool_call",
                               "text": text[:300]})
                messages.append({"role": "user", "content": NUDGE})
                continue
            results = []
            for c in calls:
                out = self._exec(ctx, c["name"], json.dumps(c.get("input") or {}, ensure_ascii=False))
                results.append({"type": "tool_result", "tool_use_id": c["id"], "content": out})
            messages.append({"role": "user", "content": results})
            if ctx.done:
                break
        ctx.notes["turns"] = turns
        ctx.notes["format_issues"] = issues
        ctx.notes["usage"] = usage
        ctx.notes["hit_max_turns"] = turns == MAX_TURNS and not ctx.done and not refused
        ctx.notes["refused"] = refused
        ctx.notes["transcript"] = _strip_signatures(messages)
