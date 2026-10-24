import json
import os
import tempfile
import unittest

from roommate.agent_llm import LLMAgent
from roommate.env import Env
from roommate.llm import CacheMiss, Ledger, make_classifier
from roommate.policies import SCRIPTED
from roommate.world import generate_world, world_hash

from .fake import FakeClient


def run(world, pol, clf=None):
    env = Env(world, clf)
    env.run(pol, workers=4)
    return json.dumps(env.records(), ensure_ascii=False, sort_keys=True)


class ReproTest(unittest.TestCase):
    def test_world_depends_only_on_seed(self):
        self.assertEqual(world_hash(generate_world(3)), world_hash(generate_world(3)))
        self.assertNotEqual(world_hash(generate_world(3)), world_hash(generate_world(4)))

    def test_scripted_identical(self):
        for name, P in SCRIPTED.items():
            with self.subTest(policy=name):
                self.assertEqual(run(generate_world(1), P()), run(generate_world(1), P()))

    def test_llm_rerun_hits_cache(self):
        w = generate_world(2, n_users=5, n_sessions=6)
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, "cache")
            c1 = FakeClient(cache, Ledger(None, 1))
            a = run(w, LLMAgent(c1, "fake-model"), make_classifier(c1))
            self.assertGreater(c1.posts, 0)
            c2 = FakeClient(cache, Ledger(None, 1), offline=True)
            b = run(w, LLMAgent(c2, "fake-model"), make_classifier(c2))
            self.assertEqual(c2.posts, 0)
            self.assertEqual(a, b)

    def test_offline_cache_miss_raises(self):
        w = generate_world(2, n_users=2, n_sessions=6)
        with tempfile.TemporaryDirectory() as td:
            c = FakeClient(os.path.join(td, "empty"), Ledger(None, 1), offline=True)
            with self.assertRaises(CacheMiss):
                run(w, LLMAgent(c, "fake-model"), make_classifier(c))


if __name__ == "__main__":
    unittest.main()
