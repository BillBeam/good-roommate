import unittest

from roommate.env import Env, render_session
from roommate.grading import grade_session, satisfies, summarize
from roommate.policies import SCRIPTED
from roommate.world import generate_world, world_stats


def run(world, name, config=None):
    env = Env(world, config)
    env.run(SCRIPTED[name]())
    return summarize(env.records()), env.records()


class GradingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = generate_world(1)

    def test_oracle_full_score(self):
        s, _ = run(self.world, "oracle")
        self.assertEqual(s["mean_score"], 1.0)
        self.assertEqual(s["util"]["success_rate"], 1.0)

    def test_nothing_zero(self):
        s, _ = run(self.world, "nothing")
        self.assertEqual(s["mean_score"], 0.0)
        self.assertEqual(s["no_choice"], s["n_sessions"])

    def test_ideal_solves_everything_and_never_reasks(self):
        s, recs = run(self.world, "ideal")
        self.assertEqual(s["success_rate"], 1.0)
        self.assertEqual(s["told_before_rate"], 0.0)
        asked = {}
        for r in recs:
            for q in r["questions"]:
                for a in q["attributes"]:
                    asked.setdefault((r["uid"], a), 0)
                    asked[(r["uid"], a)] += 1
        self.assertEqual(max(asked.values()), 1, "理想记忆策略每项属性最多问一次")

    def test_ask_all_v1_mode_solves_everything(self):
        """第一版的行为（总会重新回答）作为配置项保留：每次都问一遍就能全部做对。"""
        s, _ = run(self.world, "ask_all", {"repeat_mode": "always_answer"})
        self.assertEqual(s["success_rate"], 1.0)

    def test_memory_needed_in_v2_mode(self):
        s_ask, _ = run(self.world, "ask_all")
        s_ideal, _ = run(self.world, "ideal")
        self.assertLess(s_ask["mean_score"], 0.5 * s_ideal["mean_score"])

    def test_question_cost_is_config(self):
        s, _ = run(self.world, "ask_all", {"repeat_mode": "always_answer", "question_cost": 0.0})
        self.assertEqual(s["mean_score"], 1.0)

    def test_exactly_one_correct_and_near_miss_valid(self):
        for u in self.world["users"]:
            for s in u["sessions"]:
                if s["domain"] == "util":
                    continue
                ok = [o["id"] for o in s["options"] if satisfies(o, s["eff_state"], s["domain"])]
                self.assertEqual(ok, [s["correct"]])
                nm = [o for o in s["options"] if s["roles"][o["id"]] == "near_miss"]
                self.assertEqual(len(nm), 1)
                st = dict(s["eff_state"], **s["near_miss_state"])
                self.assertTrue(satisfies(nm[0], st, s["domain"]))

    def test_near_miss_error_counted(self):
        s = next(s for u in self.world["users"] for s in u["sessions"] if s["domain"] != "util")
        nm = next(i for i, r in s["roles"].items() if r == "near_miss")
        g = grade_session(s, nm, 1, 0.15)
        self.assertFalse(g["success"])
        self.assertTrue(g["near_miss_error"])
        self.assertAlmostEqual(g["score"], -0.15)

    def test_scenario_mix_same_as_v1(self):
        st = world_stats(self.world)
        self.assertEqual(st["n_users"], 24)
        self.assertEqual(st["n_sessions_per_user"], 8)
        self.assertEqual(st["unrelated_sessions"], 24)
        self.assertEqual(st["temp_situations"], 24)
        self.assertGreater(st["events_multi_attr"], 0)
        self.assertAlmostEqual(st["correct_is_post_share"], 0.5, delta=0.03)

    def test_agent_prompt_hides_true_state(self):
        for u in self.world["users"]:
            for s in u["sessions"]:
                p = render_session(s)
                for word in ("eff_state", "long_state", "near_miss", "correct", "decoy"):
                    self.assertNotIn(word, p)

    def test_holdout_split(self):
        from roommate.world import event_split
        train, held = event_split(["pregnancy", "dog"])
        w = generate_world(5, events=train)
        seen = {s["event"]["type"] for u in w["users"] for s in u["sessions"] if s["event"]}
        self.assertFalse(seen & set(held))
        w2 = generate_world(5, events=held)
        seen2 = {s["event"]["type"] for u in w2["users"] for s in u["sessions"] if s["event"]}
        self.assertTrue(seen2 <= set(held))


if __name__ == "__main__":
    unittest.main()
