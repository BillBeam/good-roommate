import unittest

from roommate.env import Env
from roommate.grading import QUESTION_COST, grade_session, satisfies, summarize
from roommate.policies import SCRIPTED
from roommate.world import generate_world, world_stats


def run(world, name):
    env = Env(world, classify=None)
    env.run(SCRIPTED[name]())
    return summarize(env.records())


class GradingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = generate_world(0)

    def test_oracle_full_score(self):
        s = run(self.world, "oracle")
        self.assertEqual(s["mean_score"], 1.0)
        self.assertEqual(s["success_rate"], 1.0)

    def test_nothing_zero(self):
        s = run(self.world, "nothing")
        self.assertEqual(s["mean_score"], 0.0)
        self.assertEqual(s["no_choice"], s["n_sessions"])

    def test_ask_all_solves_everything_but_pays(self):
        s = run(self.world, "ask_all")
        self.assertEqual(s["success_rate"], 1.0)
        self.assertAlmostEqual(s["mean_score"], 1.0 - QUESTION_COST * s["avg_questions"])

    def test_exactly_one_correct_and_traps_valid(self):
        for u in self.world["users"]:
            for s in u["sessions"]:
                if s["domain"] == "util":
                    continue
                ok = [o["id"] for o in s["options"] if satisfies(o, s["eff_state"], s["domain"])]
                self.assertEqual(ok, [s["correct"]])
                for role, st in (("stale_trap", s["stale_state"]), ("temp_trap", s["temp_trap_state"])):
                    if st is None:
                        continue
                    o = next(o for o in s["options"] if s["roles"][o["id"]] == role)
                    self.assertTrue(satisfies(o, st, s["domain"]))
                    self.assertFalse(satisfies(o, s["eff_state"], s["domain"]))

    def test_stale_error_counted(self):
        s = next(s for u in self.world["users"] for s in u["sessions"] if s["stale_state"] is not None)
        trap = next(i for i, r in s["roles"].items() if r == "stale_trap")
        g = grade_session(s, trap, 1)
        self.assertFalse(g["success"])
        self.assertTrue(g["stale_error"])
        self.assertAlmostEqual(g["score"], -QUESTION_COST)

    def test_scenario_mix(self):
        st = world_stats(self.world)
        self.assertGreaterEqual(st["n_users"], 20)
        self.assertGreaterEqual(st["n_sessions_per_user"], 6)
        self.assertGreater(st["events_multi_attr"], 0)
        self.assertGreater(st["temp_situations"], 0)
        self.assertGreater(st["unrelated_sessions"], 0)

    def test_agent_prompt_hides_true_state(self):
        from roommate.env import render_session
        for u in self.world["users"]:
            for s in u["sessions"]:
                p = render_session(s)
                for word in ("eff_state", "long_state", "stale", "correct", "过敏原：" + u["init_state"]["allergy"]):
                    self.assertNotIn(word, p)


if __name__ == "__main__":
    unittest.main()
