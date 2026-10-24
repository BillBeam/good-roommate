"""模拟用户：只答所问、前后一致、说过的不再说。"""

import copy
import unittest
from unittest import mock

from roommate import checks, texts, usersim
from roommate.env import Env
from roommate.schema import ATTR_ORDER
from roommate.usersim import SimUser
from roommate.world import generate_world


class UserSimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = [generate_world(s) for s in (0, 1, 2, 3)]

    def test_consistency_all_worlds(self):
        """遍历所有世界、用户、会话、属性生成回答，逐条检查（见 roommate/checks.py 的九条）。"""
        for w in self.worlds:
            for mode in usersim.REPEAT_MODES:
                with self.subTest(seed=w["seed"], mode=mode):
                    r = checks.check_world(w, repeat_mode=mode)
                    self.assertEqual(r["n_problems"], 0, r["problems"][:5])
                    self.assertEqual(r["first_answers_truthful"], r["n_first_answers"])

    def test_first_ask_is_truth(self):
        """每项属性第一次被问到（且之前没在台词里说过）时，回答就是用户当时的真实值。"""
        for u in self.worlds[1]["users"]:
            user = SimUser(u)
            user.begin_session(u["sessions"][0])
            for a in ATTR_ORDER:
                claims = texts.parse_claims(user.answer([a])["text"])
                vals = [c for c in claims if c["kind"] == "value" and c["attr"] == a]
                self.assertTrue(vals)
                self.assertEqual(vals[0]["value"], user.effective()[a])

    def test_told_before_then_truthful_after_change(self):
        u = next(u for u in self.worlds[1]["users"] if any(s["event"] for s in u["sessions"]))
        user = SimUser(u)
        user.begin_session(u["sessions"][0])
        r1 = user.answer(["home", "budget"])
        self.assertEqual(set(r1["status"].values()), {"answered"})
        r2 = user.answer(["home", "budget"])
        self.assertEqual(set(r2["status"].values()), {"told_before"})
        self.assertIn("这个我之前跟你说过了", r2["text"])
        always = SimUser(u, "always_answer")
        always.begin_session(u["sessions"][0])
        always.answer(["home"])
        self.assertEqual(always.answer(["home"])["status"]["home"], "answered_again")

    def test_line_counts_as_told(self):
        """事件台词里明说或可推断的改动，再问时用户说"之前说过了"。"""
        for u in self.worlds[1]["users"]:
            for s in u["sessions"]:
                if s["event"]:
                    user = SimUser(u)
                    for s2 in u["sessions"][:s["idx"] + 1]:
                        user.begin_session(s2)
                        if s2 is not s:
                            user.end_session()
                    for a in s["event"]["changes"]:
                        self.assertEqual(user.answer([a])["status"][a], "told_before")
                    return
        self.fail("没有事件")

    def test_temp_answered_truthfully(self):
        u, s = next((u, s) for u in self.worlds[1]["users"] for s in u["sessions"] if s["temp"])
        user = SimUser(u)
        for s2 in u["sessions"][:s["idx"] + 1]:
            user.begin_session(s2)
            if s2 is not s:
                user.end_session()
        a = s["temp"]["attr"]
        user.answer([a])
        r = user.answer([a])  # 即使长期值已经说过，临时值仍照实回答
        self.assertEqual(r["status"][a], "temp")
        claims = texts.parse_claims(r["text"])
        self.assertTrue(any(c["kind"] == "value" and c["scope"] == "temp" and c["value"] == s["temp"]["value"]
                            for c in claims))

    def test_max_two_attrs(self):
        u = self.worlds[1]["users"][0]
        user = SimUser(u)
        user.begin_session(u["sessions"][0])
        r = user.answer(["home", "budget", "pet"])
        self.assertEqual(r["answered"], ["home", "budget"])
        self.assertTrue(r["too_many"])
        self.assertTrue(r["text"].endswith(texts.TOO_MANY_ANSWER))

    def test_undeclared_question_rejected_free(self):
        env = Env(self.worlds[1])
        ep = env.episodes[0]
        from roommate.env import SessionContext
        ep.user.begin_session(ep.plan["sessions"][0])
        ctx = SessionContext(ep, ep.plan["sessions"][0])
        out = ctx.ask("你最近怎么样？", [])
        self.assertIn("没有声明", out)
        self.assertEqual(ctx.questions, [])
        out = ctx.ask("近况？", ["近况"])
        self.assertEqual(ctx.questions, [])

    def test_no_recent_news_topic(self):
        self.assertNotIn("近况", texts.PHRASES)
        from roommate.schema import CN_ATTR
        self.assertNotIn("近况", CN_ATTR)

    # ---------- 反例：检查确实能抓住问题 ----------
    def test_checker_catches_missing_health_note(self):
        orig = usersim.SimUser._answer_attr

        def drop(self, a):
            t, st = orig(self, a)
            return t.split("；另外")[0], st
        with mock.patch.object(usersim.SimUser, "_answer_attr", drop):
            r = checks.check_world(self.worlds[1], histories=("never",))
        self.assertTrue(any("健康原因" in p[-1] for p in r["problems"]))

    def test_checker_catches_misrouting(self):
        orig = usersim.SimUser._answer_attr
        with mock.patch.object(usersim.SimUser, "_answer_attr",
                               lambda self, a: orig(self, "diet" if a == "spicy" else a)):
            r = checks.check_world(self.worlds[1], histories=("never",))
        self.assertTrue(any("答非所问" in p[-1] for p in r["problems"]))

    def test_checker_catches_blanket_negation(self):
        """第一版的问题：'饮食上没什么特别限制'这类全称否定，在有约束时出现。"""
        phrases = copy.deepcopy(texts.PHRASES)
        phrases["diet"]["普通"] = "饮食上没什么特别限制"
        blanket = dict(texts.BLANKET, **{"饮食上没什么特别限制": lambda s: s["allergy"] == "无" and s["diet"] == "普通"
                                         and s["spicy"] == "爱吃辣"})
        with mock.patch.object(texts, "PHRASES", phrases), mock.patch.object(texts, "BLANKET", blanket), \
                mock.patch.object(texts, "_INV", sorted([(p, a, v) for a, t in phrases.items() for v, p in t.items()],
                                                       key=lambda x: -len(x[0]))):
            r = checks.check_world(self.worlds[1], histories=("never",))
        self.assertTrue(any("全称否定" in p[-1] for p in r["problems"]))

    def test_checker_catches_cueless_change_and_stale_told(self):
        w = copy.deepcopy(self.worlds[1])
        for u in w["users"]:
            for s in u["sessions"]:
                if s["event"]:
                    a = next(iter(s["event"]["changes"]))
                    s["event"]["cues"].pop(a)
        r = checks.check_world(w, histories=("every_session",))
        msgs = [p[-1] for p in r["problems"]]
        self.assertTrue(any("没有线索" in m for m in msgs))
        with mock.patch.object(usersim.SimUser, "_refuse",
                               lambda self, a: self.repeat_mode == "refuse_told" and a in self.told):
            r = checks.check_world(w, histories=("every_session",))
        self.assertTrue(any("之前说过了" in p[-1] for p in r["problems"]))


if __name__ == "__main__":
    unittest.main()
