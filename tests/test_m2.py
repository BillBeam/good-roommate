"""M2 新加的代码：硅基流动计费与人民币上限、格式问题的计数、重复采样的 seed 与缓存键、gzip 缓存、默认关闭时记录不变。"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

from roommate import llm
from roommate.agent_llm import LLMAgent
from roommate.env import Env
from roommate.llm import BudgetExceeded, Ledger
from roommate.world import generate_world

from .fake import FakeClient, fake_response


def small_world():
    w = generate_world(2)
    w["users"] = w["users"][:2]
    w["n_users"] = 2
    return w


class BadFormatClient(FakeClient):
    """第一轮只说话（正文里写了一个没被解析的工具调用），第二轮参数不是 JSON，之后正常。"""

    def _post(self, body):
        self.posts += 1
        n_asst = sum(m["role"] == "assistant" for m in body["messages"])
        if n_asst == 0:
            return {"choices": [{"message": {"role": "assistant", "content": "<tool_call>\n<function=memory_read>"},
                                 "finish_reason": "length"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        if n_asst == 2:
            tc = {"id": "x", "type": "function", "function": {"name": "memory_write", "arguments": "{key: 1"}}
            return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tc]},
                                 "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return dict(fake_response(body), usage={"prompt_tokens": 10, "completion_tokens": 5})


class StuckClient(FakeClient):
    """永远只读记忆，从不选择。"""

    def _post(self, body):
        self.posts += 1
        tc = {"id": f"r{len(body['messages'])}", "type": "function", "function": {"name": "memory_read", "arguments": "{}"}}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tc]},
                             "finish_reason": "tool_calls"}], "usage": {}}


class M2Test(unittest.TestCase):
    def test_siliconflow_cost_and_cny_limit(self):
        llm.SILICONFLOW_PRICES["Qwen/X"] = {"input": 2.0, "output": 8.0}
        llm.SILICONFLOW_FX["cny_per_usd"] = 8.0
        u = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
        self.assertEqual(llm.usage_cost("Qwen/X", u), (0.75, 6.0))
        self.assertEqual(llm.provider_of("Qwen/Qwen3.5-4B"), "siliconflow")
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "ledger.json")
            with open(p, "w") as f:
                json.dump({"total_usd": 0.0, "by_model_role": {}, "siliconflow_cny_total": 0.0,
                           "siliconflow_cny_limit": 5.0}, f)
            led = Ledger(p, 100)
            led.check("Qwen/X")
            led.add("Qwen/X", "agent", u, cached=False, latency=1.234)
            self.assertAlmostEqual(led._load()["siliconflow_cny_total"], 6.0)
            with self.assertRaises(BudgetExceeded):
                led.check("Qwen/X")
            led.check("deepseek-flash")   # 另一家服务商的上限不受影响
            line = json.loads(open(os.path.join(td, "spend_calls.jsonl")).readline())
            self.assertEqual(line["cost_cny"], 6.0)
            self.assertEqual(line["latency_s"], 1.23)

    def test_format_issues_counted(self):
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            c = BadFormatClient(os.path.join(td, "c"), Ledger(None, 1))
            env = Env(w)
            env.run(LLMAgent(c, "fake-model", track_format=True), until=1)
            r = env.records()[0]
            kinds = [i["kind"] for i in r["format_issues"]]
            self.assertEqual(kinds, ["truncated", "tool_text", "bad_json"])
            self.assertEqual(r["usage"]["calls"], r["turns"])
            self.assertEqual(r["usage"]["prompt_tokens"], 10 * r["turns"])
            self.assertFalse(r["hit_max_turns"])

    def test_hit_max_turns(self):
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            c = StuckClient(os.path.join(td, "c"), Ledger(None, 1))
            env = Env(w)
            env.run(LLMAgent(c, "fake-model", track_format=True), until=1)
            r = env.records()[0]
            self.assertTrue(r["hit_max_turns"])
            self.assertIsNone(r["chosen"])
            self.assertEqual(r["turns"], 14)

    def test_defaults_leave_records_unchanged(self):
        """不开 track_format、rep=0 时，请求和记录与 M1 完全相同（M1 的缓存因此照样有效）。"""
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            c = FakeClient(os.path.join(td, "c"), Ledger(None, 1))
            env = Env(w)
            env.run(LLMAgent(c, "fake-model"))
            for r in env.records():
                self.assertNotIn("format_issues", r)
                self.assertNotIn("usage", r)
            c2 = FakeClient(os.path.join(td, "c"), Ledger(None, 1), offline=True)
            env2 = Env(w)
            env2.run(LLMAgent(c2, "fake-model", rep=0))
            self.assertEqual(json.dumps(env.records(), sort_keys=True), json.dumps(env2.records(), sort_keys=True))

    def test_rep_changes_seed_and_cache_key(self):
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            c = FakeClient(os.path.join(td, "c"), Ledger(None, 1))
            Env(w).run(LLMAgent(c, "fake-model"), until=1)
            n0 = c.posts
            Env(w).run(LLMAgent(c, "fake-model", rep=1), until=1)
            self.assertGreater(c.posts, n0)            # rep=1 不会命中 rep=0 的缓存
            n1 = c.posts
            Env(w).run(LLMAgent(c, "fake-model", rep=1), until=1)
            self.assertEqual(c.posts, n1)              # 同一个 rep 重跑全部命中缓存

    def test_gzip_cache_is_read(self):
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, "c")
            c = FakeClient(cache, Ledger(None, 1))
            env = Env(w)
            env.run(LLMAgent(c, "fake-model"))
            src = os.path.join(cache, "fake-model.jsonl")
            with open(src, "rb") as f, gzip.open(src + ".gz", "wb") as g:
                shutil.copyfileobj(f, g)
            os.remove(src)
            c2 = FakeClient(cache, Ledger(None, 1), offline=True)
            env2 = Env(w)
            env2.run(LLMAgent(c2, "fake-model"))
            self.assertEqual(c2.posts, 0)
            self.assertEqual(json.dumps(env.records(), sort_keys=True), json.dumps(env2.records(), sort_keys=True))


if __name__ == "__main__":
    unittest.main()
