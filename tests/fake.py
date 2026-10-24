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
