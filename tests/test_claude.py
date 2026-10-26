"""Claude 接入：请求的形状、计价、缓存复现、账本上限与余量、"不进训练数据"的标记。不联网。"""

import json
import os
import tempfile
import unittest

from roommate import data_policy, llm
from roommate.agent_claude import ClaudeAgent
from roommate.agent_llm import tools
from roommate.env import Env
from roommate.llm import BudgetExceeded, Ledger
from roommate.world import generate_world

from .fake import FakeClaudeClient

PRICES = {"claude-opus-5-5": {"input": 4, "cache_write_5m": 5, "cache_write_1h": 8, "cache_read": 0.2, "output": 20},
          "claude-haiku-5-5": {"input": 0.1, "cache_write_5m": 0.125, "cache_write_1h": 0.2, "cache_read": 0.01,
                               "output": 0.5, "long": {"over_prompt_tokens": 100000, "input": 0.5,
                                                       "cache_write_5m": 0.625, "cache_write_1h": 1,
                                                       "cache_read": 0.05, "output": 2.5}}}


def small_world(n=3):
    w = generate_world(1)
    w["users"] = w["users"][:n]
    w["n_users"] = n
    return w


class ClaudeTest(unittest.TestCase):
    def setUp(self):
        self._old = dict(llm.ANTHROPIC_PRICES)
        llm.ANTHROPIC_PRICES.update(PRICES)

    def tearDown(self):
        llm.ANTHROPIC_PRICES.clear()
        llm.ANTHROPIC_PRICES.update(self._old)

    def test_pricing(self):
        u = {"input_tokens": 1000, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 10000,
             "output_tokens": 500}
        usd, cny = llm.usage_cost("claude-opus-5-5", u)
        self.assertAlmostEqual(usd, (1000 * 4 + 2000 * 5 + 10000 * 0.2 + 500 * 20) / 1e6)
        self.assertIsNone(cny)
        u1h = dict(u, cache_creation={"ephemeral_5m_input_tokens": 500, "ephemeral_1h_input_tokens": 1500})
        self.assertAlmostEqual(llm.usage_cost("claude-opus-5-5", u1h)[0],
                               (1000 * 4 + 500 * 5 + 1500 * 8 + 10000 * 0.2 + 500 * 20) / 1e6)
        big = {"input_tokens": 60000, "cache_read_input_tokens": 50000, "output_tokens": 10}
        self.assertAlmostEqual(llm.usage_cost("claude-haiku-5-5", big)[0], (60000 * 0.5 + 50000 * 0.05 + 10 * 2.5) / 1e6)
        with self.assertRaises(RuntimeError):
            llm.usage_cost("claude-unknown", u)

    def test_request_shape_and_same_tools(self):
        with tempfile.TemporaryDirectory() as td:
            c = FakeClaudeClient(os.path.join(td, "c"), Ledger(None, 1))
            Env(small_world(1)).run(ClaudeAgent(c, "claude-opus-5-5", seed=1), until=2)
            b = c.bodies[0]
            for k in ("temperature", "top_p", "top_k", "seed"):
                self.assertNotIn(k, b)
            self.assertEqual(b["thinking"]["type"], "adaptive")
            self.assertEqual(b["tool_choice"], {"type": "auto"})
            self.assertEqual([(t["name"], t["description"], t["input_schema"]) for t in b["tools"]],
                             [(t["function"]["name"], t["function"]["description"], t["function"]["parameters"])
                              for t in tools(0.15)])
            # 后续轮次把上一轮的思考块（含签名）原样带回，工具结果放在同一条 user 消息里
            later = [x for x in c.bodies if len(x["messages"]) > 2][0]
            asst = later["messages"][1]
            self.assertEqual(asst["role"], "assistant")
            if asst["content"][0]["type"] == "thinking":
                self.assertTrue(asst["content"][0]["signature"].startswith("sig"))

    def test_cache_replay_identical_and_ledger(self):
        w = small_world()
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, "c")
            led = Ledger(os.path.join(td, "led", "spend_ledger.json"), 10)
            os.makedirs(os.path.join(td, "led"))
            c1 = FakeClaudeClient(cache, led)
            e1 = Env(w)
            e1.run(ClaudeAgent(c1, "claude-opus-5-5", seed=1), workers=3)
            r1 = e1.records()
            self.assertGreater(c1.posts, 0)
            with open(os.path.join(td, "led", "spend_ledger.json")) as f:
                tot = json.load(f)["total_usd"]
            self.assertAlmostEqual(tot, sum(r["usage"]["cost_usd"] for r in r1), places=9)
            c2 = FakeClaudeClient(cache, Ledger(None, 0), offline=True)
            e2 = Env(w)
            e2.run(ClaudeAgent(c2, "claude-opus-5-5", seed=1, truncate_on_cache_miss=True), workers=3)
            self.assertEqual(c2.posts, 0)
            strip = lambda rs: json.dumps(rs, sort_keys=True, ensure_ascii=False)
            self.assertEqual(strip(r1), strip(e2.records()))
            self.assertTrue(all("signature" not in json.dumps(r["transcript"]) for r in r1))
            self.assertTrue(any(r["chosen"] for r in r1))

    def test_cache_tag_makes_users_independent(self):
        with tempfile.TemporaryDirectory() as td:
            c = FakeClaudeClient(os.path.join(td, "c"), Ledger(None, 1))
            body = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "同一句话"}], "tools": []}
            c.chat(body, role="agent", cache_tag="1|u00|0")
            c.chat(body, role="agent", cache_tag="1|u01|0")
            c.chat(body, role="agent", cache_tag="1|u00|0")
            self.assertEqual(c.posts, 2)

    def test_budget_reserve_stops_before_limit(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "spend_ledger.json")
            with open(p, "w") as f:
                json.dump({"total_usd": 146.0, "by_model_role": {}}, f)
            with self.assertRaises(BudgetExceeded):
                Ledger(p, 150, reserve_usd=5).check("claude-opus-5-5")
            Ledger(p, 150, reserve_usd=3).check("claude-opus-5-5")

    def test_not_for_training(self):
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "runs", "x")
            data_policy.mark_not_for_training(d)
            f = os.path.join(d, "sub", "sessions.jsonl")
            os.makedirs(os.path.dirname(f))
            open(f, "w").close()
            with self.assertRaises(data_policy.NotForTraining):
                data_policy.assert_trainable(f)
            self.assertFalse(data_policy.trainable(os.path.join(td, "llm_cache", "claude-opus-5-5.jsonl")))
            self.assertFalse(data_policy.trainable(os.path.join(td, "seed1", "policies", "llm_claude-opus-5-5", "a.jsonl")))
            self.assertTrue(data_policy.trainable(os.path.join(td, "runs", "y", "a.jsonl")))


if __name__ == "__main__":
    unittest.main()
