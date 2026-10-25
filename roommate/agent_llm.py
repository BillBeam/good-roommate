"""大模型 agent：通过函数调用使用记忆工具、提问工具和选择工具。

每次会话从零开始（不带以前的对话），跨会话信息只有记忆工具里的内容。
提问时必须声明问哪几项属性（一次最多两项），用户只回答声明的这几项。

M2 起点测量另加两样（默认关闭，M1 的记录因此逐字不变）：
- rep：同一用户重复跑第几遍。rep 不为 0 时，它进入请求的 seed 和缓存键，使每一遍是独立的一次采样。
- track_format：逐轮记下工具调用格式的问题，写进记录的 format_issues（种类见 FORMAT_KINDS），
  另记这次会话的 token 用量（usage）和是否用满了轮数（hit_max_turns）。
"""

import re

import hashlib
import json

from .memory import MAX_ENTRIES
from .schema import ATTR_CN, MAX_ATTRS_PER_QUESTION

ATTR_LIST = "、".join(ATTR_CN.values())


def system_prompt(question_cost, repeat_mode):
    repeat_rule = (
        "- 用户已经告诉过你的事（在回答里说过的，或者聊天时提到、能听出来的），你再问，用户只会说"
        "\"这个我之前跟你说过了\"，不会再说一遍，这次提问照样扣分。只有这次的临时情况会照实再说。"
        "所以听到的重要信息要自己记下来。"
        if repeat_mode == "refuse_told" else
        "- 用户每次被问到都会照实回答。")
    return f"""你是用户的长期私人助理，帮用户处理日常事务：订晚饭的餐厅、找午饭的地方，以及算账、换算之类的小事。
你和同一个用户会有很多次会话，时间跨度几个月。

规则：
- 每次会话开始时你不记得以前的对话。跨会话的信息只能靠记忆工具：memory_read 读出全部记忆，memory_write 写入或覆盖一条（最多 {MAX_ENTRIES} 条），memory_delete 删除一条。
- 你只能通过工具和用户打交道：直接输出的文字用户看不到。
- 要提问就调用 ask_user，并在 attributes 里声明这次问的是哪几项（一次最多 {MAX_ATTRS_PER_QUESTION} 项，可选：{ATTR_LIST}）。用户只回答声明的这几项。每问一次扣 {question_cost} 分。
{repeat_rule}
- 每次会话最后必须调用 choose_option 选出一个选项（只能选一次）。选中真正合适的得 1 分，选错得 0 分。
- 用户的生活会变化，原来的偏好可能随之改变，用户不一定会明说；也有些情况只针对这一次。
- 提问请用中文，简短直接。"""


def tools(question_cost):
    return [
        {"type": "function", "function": {
            "name": "memory_read", "description": "读出全部长期记忆。",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "memory_write", "description": "写入或覆盖一条长期记忆。",
            "parameters": {"type": "object", "properties": {
                "key": {"type": "string", "description": "简短的条目名"},
                "value": {"type": "string", "description": "内容"}}, "required": ["key", "value"]}}},
        {"type": "function", "function": {
            "name": "memory_delete", "description": "删除一条长期记忆。",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
        {"type": "function", "function": {
            "name": "ask_user",
            "description": f"向用户提一个问题（每次扣 {question_cost} 分）。必须在 attributes 里声明问哪几项，"
                           f"一次最多 {MAX_ATTRS_PER_QUESTION} 项；用户只回答声明的项。",
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string", "description": "问题本身"},
                "attributes": {"type": "array", "items": {"type": "string", "enum": list(ATTR_CN.values())},
                               "description": f"这次问的是哪几项，最多 {MAX_ATTRS_PER_QUESTION} 项"}},
                "required": ["question", "attributes"]}}},
        {"type": "function", "function": {
            "name": "choose_option", "description": "选出最终方案，结束本次会话。",
            "parameters": {"type": "object", "properties": {
                "option": {"type": "string", "description": "选项字母，如 A"}}, "required": ["option"]}}},
    ]


MAX_TURNS = 14
TOOL_NAMES = {"memory_read", "memory_write", "memory_delete", "ask_user", "choose_option"}
# 工具调用格式问题的种类：
# bad_json 参数不是合法 JSON；unknown_tool 调了不存在的工具；truncated 输出被 max_tokens 截断；
# tool_text 没有结构化的工具调用，但正文里像是写了工具调用（接口没解析出来，或模型写错了格式）；
# think_runaway 思考用满了 thinking_budget 被截断（思考停不下来）；
# no_tool_call 这一轮只输出文字、没调工具（会收到提醒，继续下一轮）。
FORMAT_KINDS = ["bad_json", "unknown_tool", "truncated", "think_runaway", "tool_text", "no_tool_call"]
_TOOL_TEXT = re.compile(r"<tool_call>|<function[=>]|</?tool_call|\"name\"\s*:\s*\"(%s)\"|\b(%s)\s*\(" %
                        ("|".join(TOOL_NAMES), "|".join(TOOL_NAMES)))
NUDGE = "（系统提示：用户看不到你直接输出的文字。要提问请调用 ask_user；完成任务请调用 choose_option。）"


class LLMAgent:
    def __init__(self, client, model, extra=None, seed=0, inject_true_memory=False, echo_reasoning=False,
                 truncate_on_cache_miss=False, rep=0, track_format=False):
        self.client = client
        self.truncate_on_cache_miss = truncate_on_cache_miss  # 只用缓存重建已有结果：缺缓存时截断这个用户
        self.echo_reasoning = echo_reasoning   # 有的接口要求工具调用循环里把思考内容原样带回
        self.model = model
        self.inject_true_memory = inject_true_memory   # "知道真实状态"对照组
        self.name = f"llm:{model}" + (":true_memory" if inject_true_memory else "")
        self.extra = extra or {}
        self.seed = seed
        self.rep = rep
        self.track_format = track_format

    def run_session(self, ctx):
        cfg = ctx.ep.config
        messages = [{"role": "system", "content": system_prompt(cfg["question_cost"], cfg["repeat_mode"])},
                    {"role": "user", "content": ctx.prompt}]
        tag = f"{self.seed}|{ctx.ep.plan['uid']}|{ctx.sess['idx']}" + (f"|rep{self.rep}" if self.rep else "")
        sid = 1 + int(hashlib.sha256(tag.encode()).hexdigest()[:7], 16)
        role = "control" if self.inject_true_memory else "agent"
        if self.rep:
            role += f"_rep{self.rep}"
        issues = []
        usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        turns = 0
        for turns in range(1, MAX_TURNS + 1):
            body = {"model": self.model, "messages": messages, "tools": tools(cfg["question_cost"]),
                    "tool_choice": "auto", "temperature": 0, "seed": sid, **self.extra}
            if self.model.startswith("deepseek"):
                body.pop("seed")  # DeepSeek 接口没有 seed 参数；复现靠缓存
            resp = self.client.chat(body, role=role)
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if self.track_format:
                issues += self._format_issues(turns, resp["choices"][0], calls, resp.get("usage"),
                                              self.extra.get("thinking_budget"))
                u = resp.get("usage") or {}
                usage["calls"] += 1
                usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage["completion_tokens"] += u.get("completion_tokens", 0)
                usage["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            out = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content") and self.echo_reasoning:
                out["reasoning_content"] = msg["reasoning_content"]
            if calls:
                out["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["function"]["name"],
                                                   "arguments": c["function"]["arguments"]}} for c in calls]
            messages.append(out)
            if not calls:
                messages.append({"role": "user", "content": NUDGE})
                continue
            for c in calls:
                result = self._exec(ctx, c["function"]["name"], c["function"]["arguments"])
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            if ctx.done:
                break
        ctx.notes["turns"] = turns
        if self.track_format:
            ctx.notes["format_issues"] = issues
            ctx.notes["usage"] = usage
            ctx.notes["hit_max_turns"] = turns == MAX_TURNS and not ctx.done
        ctx.notes["transcript"] = messages[1:]

    @staticmethod
    def _format_issues(turn, choice, calls, usage=None, thinking_budget=None):
        msg = choice["message"]
        out = []
        if choice.get("finish_reason") == "length":
            out.append({"turn": turn, "kind": "truncated"})
        rt = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        if thinking_budget and rt >= thinking_budget:
            out.append({"turn": turn, "kind": "think_runaway", "reasoning_tokens": rt})
        for c in calls:
            name, raw = c["function"]["name"], c["function"]["arguments"]
            if name not in TOOL_NAMES:
                out.append({"turn": turn, "kind": "unknown_tool", "name": name[:80]})
            try:
                json.loads(raw) if raw else {}
            except ValueError:
                out.append({"turn": turn, "kind": "bad_json", "name": name[:80], "raw": raw[:300]})
        if not calls:
            text = msg.get("content") or ""
            kind = "tool_text" if _TOOL_TEXT.search(text) else "no_tool_call"
            out.append({"turn": turn, "kind": kind, "text": text[:300]})
        return out

    @staticmethod
    def _exec(ctx, name, raw):
        try:
            args = json.loads(raw) if raw else {}
        except ValueError:
            return "错误：参数不是合法 JSON"
        if ctx.done:
            return "本次会话已经结束。"
        if name == "memory_read":
            return ctx.memory_read()
        if name == "memory_write":
            return ctx.memory_write(args.get("key", ""), args.get("value", ""))
        if name == "memory_delete":
            return ctx.memory_delete(args.get("key", ""))
        if name == "ask_user":
            return ctx.ask(str(args.get("question", "")), args.get("attributes") or [])
        if name == "choose_option":
            return ctx.choose(args.get("option", ""))
        return f"错误：没有工具 {name}"
