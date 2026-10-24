"""大模型 agent：通过函数调用使用记忆工具、提问工具和选择工具。

每次会话从零开始（不带以前的对话），跨会话信息只有记忆工具里的内容。
"""

import hashlib
import json

from .grading import QUESTION_COST
from .memory import MAX_ENTRIES

SYSTEM_PROMPT = f"""你是用户的长期私人助理，帮用户处理日常事务：订餐厅、找午饭、订酒店，以及算账、换算之类的小事。
你和同一个用户会有很多次会话，时间跨度几个月。

规则：
- 每次会话开始时你不记得以前的对话。跨会话的信息只能靠记忆工具：memory_read 读出全部记忆，memory_write 写入或覆盖一条（最多 {MAX_ENTRIES} 条），memory_delete 删除一条。
- 你只能通过工具和用户打交道：直接输出的文字用户看不到。要提问就调用 ask_user，每问一次扣 {QUESTION_COST} 分；一个问题里用户最多回答两个方面。
- 每次会话最后必须调用 choose_option 选出一个选项（只能选一次）。选中真正合适的得 1 分，选错得 0 分。
- 用户的生活会变化，原来的偏好可能随之改变，用户不一定会明说；也有些情况只针对这一次。
- 提问请用中文，简短直接。"""

TOOLS = [
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
        "name": "ask_user", "description": f"向用户提一个问题（每次扣 {QUESTION_COST} 分）。",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "choose_option", "description": "选出最终方案，结束本次会话。",
        "parameters": {"type": "object", "properties": {
            "option": {"type": "string", "description": "选项字母，如 A"}}, "required": ["option"]}}},
]

MAX_TURNS = 12
NUDGE = "（系统提示：用户看不到你直接输出的文字。要提问请调用 ask_user；完成任务请调用 choose_option。）"


class LLMAgent:
    def __init__(self, client, model, extra=None, seed=0):
        self.client = client
        self.model = model
        self.name = f"llm:{model}"
        self.extra = extra or {}
        self.seed = seed

    def run_session(self, ctx):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ctx.prompt}]
        sid = 1 + int(hashlib.sha256(f"{self.seed}|{ctx.ep.plan['uid']}|{ctx.sess['idx']}".encode()).hexdigest()[:7], 16)
        turns = 0
        for turns in range(1, MAX_TURNS + 1):
            body = {"model": self.model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
                    "temperature": 0, "seed": sid, **self.extra}
            resp = self.client.chat(body, role="agent")
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            out = {"role": "assistant", "content": msg.get("content") or ""}
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
        ctx.notes["transcript"] = messages[1:]

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
            return ctx.ask(str(args.get("question", "")))
        if name == "choose_option":
            return ctx.choose(args.get("option", ""))
        return f"错误：没有工具 {name}"
