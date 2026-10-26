"""测试用的确定性"假模型"：回复只由消息内容的哈希决定，不联网。"""

import hashlib
import json

from roommate.llm import LLMClient
from roommate.schema import ATTR_CN


def _h(x):
    return int(hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12], 16)


def fake_response(body):
    msgs = body["messages"]
    if "tools" not in body:  # 台词校验
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps({"changes": {}})}}],
                "usage": {"cost_in_usd_ticks": 0}}
    n_tool = sum(m["role"] == "tool" for m in msgs)
    h = _h(msgs)
    names = list(ATTR_CN.values())
    if n_tool == 0:
        call = ("memory_read", {})
    elif n_tool == 1 and h % 3 == 0:
        call = ("ask_user", {"question": "问一下", "attributes": [names[h % 11], names[(h // 11) % 11]]})
    elif n_tool <= 2:
        call = ("memory_write", {"key": f"k{h % 5}", "value": msgs[-1]["content"][:30]})
    else:
        call = ("choose_option", {"option": "ABCDEF"[h % 6]})
    tc = {"id": f"c{n_tool}", "type": "function", "function": {"name": call[0], "arguments": json.dumps(call[1], ensure_ascii=False)}}
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tc]}}],
            "usage": {"cost_in_usd_ticks": 0}}


class FakeClient(LLMClient):
    """走真实的缓存逻辑，只把网络请求换成 fake_response，并统计真正"发出"的请求数。"""

    def __init__(self, cache_dir, ledger, offline=False):
        super().__init__(cache_dir, ledger, offline=offline)
        self.posts = 0

    def _post(self, body):
        self.posts += 1
        return fake_response(body)


def fake_claude_response(body):
    """Messages API 形状的确定性回复：带思考块（有签名）和 tool_use，usage 按 Claude 的四类 token 给。"""
    msgs = body["messages"]
    n_tool = sum(1 for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
                 and any(b.get("type") == "tool_result" for b in m["content"]))
    h = _h(msgs)
    names = list(ATTR_CN.values())
    if n_tool == 0 and h % 7 == 0:
        return {"type": "message", "role": "assistant", "model": body["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "我先想想。"}],
                "usage": {"input_tokens": 50, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0,
                          "output_tokens": 20}}
    if n_tool == 0:
        call = ("memory_read", {})
    elif n_tool == 1 and h % 3 == 0:
        call = ("ask_user", {"question": "问一下", "attributes": [names[h % 11], names[(h // 11) % 11]]})
    elif n_tool <= 2:
        call = ("memory_write", {"key": f"k{h % 5}", "value": "记一下"})
    else:
        call = ("choose_option", {"option": "ABCDEF"[h % 6]})
    return {"type": "message", "role": "assistant", "model": body["model"], "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "摘要", "signature": f"sig{h % 997}"},
                        {"type": "tool_use", "id": f"toolu_{n_tool}_{h % 1000}", "name": call[0], "input": call[1]}],
            "usage": {"input_tokens": 30, "cache_creation_input_tokens": 200, "cache_read_input_tokens": 1500 * n_tool,
                      "output_tokens": 120}}


class FakeClaudeClient(FakeClient):
    def _post(self, body):
        self.posts += 1
        self.bodies = getattr(self, "bodies", []) + [body]
        return fake_claude_response(body)
