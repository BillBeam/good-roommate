"""在任意会话边界保存并恢复完整状态，恢复后续跑的结果与不中断时逐字一致。"""

import json
import os
import tempfile
import unittest

from roommate.agent_llm import LLMAgent
from roommate.env import Env
from roommate.llm import Ledger, make_classifier
from roommate.policies import SCRIPTED
from roommate.world import generate_world

from .fake import FakeClient


def dumps(recs):
    return json.dumps(recs, ensure_ascii=False, sort_keys=True)


class CheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = generate_world(7, n_users=6, n_sessions=8)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _client(self):
        return FakeClient(os.path.join(self.tmp.name, "cache"), Ledger(None, 1))

    def _policies(self):
        c = self._client()
        yield "remember_no_update", SCRIPTED["remember_no_update"](), None
        yield "fake_llm", LLMAgent(c, "fake-model"), make_classifier(c)

    def test_every_boundary(self):
        n = self.world["n_sessions"]
        for name, pol, clf in self._policies():
            full = Env(self.world, clf)
            full.run(pol)
            ref = dumps(full.records())
            for k in range(0, n + 1):
                with self.subTest(policy=name, k=k):
                    env = Env(self.world, clf)
                    env.run(pol, until=k)
                    path = os.path.join(self.tmp.name, f"{name}_{k}.json")
                    env.save(path)
                    del env
                    env2 = Env.load(path, clf)
                    self.assertTrue(all(e.next_idx == k for e in env2.episodes))
                    env2.run(pol)
                    self.assertEqual(dumps(env2.records()), ref)

    def test_checkpoint_contains_true_state_and_memory(self):
        env = Env(self.world, None)
        env.run(SCRIPTED["remember_no_update"](), until=self.world["n_sessions"] - 1)
        path = os.path.join(self.tmp.name, "inspect.json")
        env.save(path)
        data = json.load(open(path, encoding="utf-8"))
        ep = data["episodes"][0]
        self.assertIn("state", ep["user"])
        self.assertTrue(ep["memory"]["entries"])
        changed = any(e["user"]["state"] != u["init_state"] for e, u in zip(data["episodes"], self.world["users"]))
        self.assertTrue(changed, "至少有一个用户的真实状态被事件改动过")

    def test_dropping_memory_changes_results(self):
        """反例：恢复时丢掉 agent 记忆，结果就不同——说明上面的一致性检验是有效的。"""
        pol = SCRIPTED["remember_no_update"]()
        full = Env(self.world, None)
        full.run(pol)
        env = Env(self.world, None)
        env.run(pol, until=3)
        path = os.path.join(self.tmp.name, "nomem.json")
        env.save(path)
        data = json.load(open(path, encoding="utf-8"))
        for e in data["episodes"]:
            e["memory"]["entries"] = {}
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        env2 = Env.load(path, None)
        env2.run(pol)
        self.assertNotEqual(dumps(env2.records()), dumps(full.records()))


if __name__ == "__main__":
    unittest.main()
