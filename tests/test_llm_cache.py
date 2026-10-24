"""接口与缓存：停用的服务商不发请求；只用缓存重建时缺缓存就截断；同一个键有多个响应时能挑对；按时段计价。"""

import datetime as dt
import json
import os
import tempfile
import unittest

from roommate import llm
from roommate.agent_llm import LLMAgent
from roommate.env import Env
from roommate.llm import CacheMiss, Ledger
from roommate.world import generate_world

from .fake import FakeClient


def small_world():
    w = generate_world(2)
    w["users"] = w["users"][:3]
    w["n_users"] = 3
    return w


class LLMCacheTest(unittest.TestCase):
    def test_blocked_provider_never_posts(self):
        with tempfile.TemporaryDirectory() as td:
            c = FakeClient(os.path.join(td, "c"), Ledger(None, 1))
            c.blocked = {"xai"}
            with self.assertRaises(CacheMiss):
                c.chat({"model": "grok-4.7", "messages": [{"role": "user", "content": "hi"}]}, role="agent")
            self.assertEqual(c.posts, 0)

    def test_cache_only_rebuild_truncates(self):
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, "c")
            c1 = FakeClient(cache, Ledger(None, 1))
            env = Env(w)
            env.run(LLMAgent(c1, "fake-model"), until=3)   # 只跑前 3 次会话
            c2 = FakeClient(cache, Ledger(None, 1), offline=True)
            env2 = Env(w)
            env2.run(LLMAgent(c2, "fake-model", truncate_on_cache_miss=True))
            self.assertEqual(c2.posts, 0)
            self.assertEqual(json.dumps(env2.records(), sort_keys=True, ensure_ascii=False),
                             json.dumps(env.records(), sort_keys=True, ensure_ascii=False))
            self.assertTrue(all(e.truncated_at == 3 for e in env2.episodes))

    def test_duplicate_key_backtracks(self):
        """同一个键被写了两个响应时，重建会试另一个，直到这次会话的后续请求都在缓存里。"""
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, "c")
            c1 = FakeClient(cache, Ledger(None, 1))
            env = Env(w)
            env.run(LLMAgent(c1, "fake-model"))
            ref = json.dumps(env.records(), sort_keys=True, ensure_ascii=False)
            # 往每个键前面插一个"别的运行"写下的假响应：工具调用 id 不同，后续请求的键就对不上
            path = os.path.join(cache, "fake-model.jsonl")
            lines = [json.loads(x) for x in open(path, encoding="utf-8")]
            out = []
            for d in lines:
                fake = json.loads(json.dumps(d))
                for tc in fake["response"]["choices"][0]["message"].get("tool_calls") or []:
                    tc["id"] = "other-" + tc["id"]
                    tc["function"]["name"] = "memory_read"
                    tc["function"]["arguments"] = "{}"
                out += [fake, d]
            with open(path, "w", encoding="utf-8") as f:
                for d in out:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            c2 = FakeClient(cache, Ledger(None, 1), offline=True)
            env2 = Env(w)
            env2.run(LLMAgent(c2, "fake-model", truncate_on_cache_miss=True))
            self.assertEqual(json.dumps(env2.records(), sort_keys=True, ensure_ascii=False), ref)

    def test_deepseek_peak_pricing(self):
        utc = dt.timezone.utc
        self.assertTrue(llm.is_peak(dt.datetime(2026, 10, 26, 2, 0, tzinfo=utc)))     # 周一 北京 10:00
        self.assertFalse(llm.is_peak(dt.datetime(2026, 10, 26, 5, 0, tzinfo=utc)))    # 周一 北京 13:00
        self.assertTrue(llm.is_peak(dt.datetime(2026, 10, 26, 9, 30, tzinfo=utc)))    # 周一 北京 17:30
        self.assertFalse(llm.is_peak(dt.datetime(2026, 10, 24, 2, 0, tzinfo=utc)))    # 周六
        llm.DEEPSEEK_PRICES["deepseek-x"] = {
            "usd": {"peak": {"input_cache_hit": 2, "input_cache_miss": 4, "output": 8},
                    "off_peak": {"input_cache_hit": 1, "input_cache_miss": 2, "output": 4}},
            "cny": {"peak": {"input_cache_hit": 20, "input_cache_miss": 40, "output": 80},
                    "off_peak": {"input_cache_hit": 10, "input_cache_miss": 20, "output": 40}}}
        u = {"prompt_cache_hit_tokens": 1_000_000, "prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 1_000_000}
        self.assertEqual(llm.usage_cost("deepseek-x", u, peak=True), (14.0, 140.0))
        self.assertEqual(llm.usage_cost("deepseek-x", u, peak=False), (7.0, 70.0))


if __name__ == "__main__":
    unittest.main()
