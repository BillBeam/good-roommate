"""模拟用户：由程序维护真实状态，按模板回答提问。

agent 的自由文本问题先被归到话题（schema.TOPICS）上，再由代码按用户当时的真实状态生成回答。
归类这一步对大模型 agent 用的是 xAI 模型（见 llm.classify_topics），它只决定"问的是哪方面"，
不参与判分，也看不到用户的真实状态。
"""

from . import texts
from .schema import MAX_TOPICS_PER_QUESTION, TOPICS


class SimUser:
    def __init__(self, plan):
        self.plan = plan
        self.state = dict(plan["init_state"])   # 长期真实状态
        self.temp = None                         # 本次会话的临时情况
        self.unreported = []                     # 还没在"近况"里讲过的事件台词
        self.last_answer = {}                    # 话题 -> 上次给 agent 的回答（用来判定重复提问）

    # ---------- 会话边界 ----------
    def begin_session(self, sess):
        ev = sess.get("event")
        if ev:
            for a, (old, new) in ev["changes"].items():
                assert self.state[a] == old, (self.plan["uid"], sess["idx"], a, self.state[a], old)
                self.state[a] = new
            self.unreported.append(ev["mention"])
        self.temp = sess.get("temp")
        if sess["eff_state"] is not None:
            assert self.effective() == sess["eff_state"], (self.plan["uid"], sess["idx"])

    def end_session(self):
        self.temp = None

    def effective(self):
        s = dict(self.state)
        if self.temp:
            s[self.temp["attr"]] = self.temp["value"]
        return s

    # ---------- 回答 ----------
    def topic_answer(self, topic):
        if topic == "近况":
            if not self.unreported:
                return texts.NOTHING_NEW
            return "".join(self.unreported)
        eff = self.effective()
        temp_cn = self.temp["cn"] if self.temp else None
        parts = [texts.answer_attr(a, self.state[a], eff[a], temp_cn) for a in TOPICS[topic]]
        return "；".join(parts) + "。"

    def answer(self, topics):
        """返回 (回答文本, 实际回答的话题, 每个话题是否属于重复提问)。"""
        topics = [t for t in dict.fromkeys(topics) if t in TOPICS]
        if not topics:
            return texts.NO_TOPIC_ANSWER, [], []
        answered = topics[:MAX_TOPICS_PER_QUESTION]
        pieces, repeated = [], []
        for t in answered:
            a = self.topic_answer(t)
            repeated.append(self.last_answer.get(t) == a)
            self.last_answer[t] = a
            if t == "近况":
                self.unreported = []
            pieces.append(a)
        text = "".join(pieces)
        if len(topics) > MAX_TOPICS_PER_QUESTION:
            text += texts.TOO_MANY_ANSWER
        return text, answered, repeated

    def structured(self, topics):
        """供脚本策略使用：直接给出这些话题对应属性的当前有效值（相当于完美理解了回答）。"""
        eff = self.effective()
        return {a: eff[a] for t in topics for a in TOPICS[t]}

    # ---------- 存档 ----------
    def state_dict(self):
        return {"state": self.state, "temp": self.temp, "unreported": self.unreported,
                "last_answer": self.last_answer}

    def load_state_dict(self, d):
        self.state = dict(d["state"])
        self.temp = d["temp"]
        self.unreported = list(d["unreported"])
        self.last_answer = dict(d["last_answer"])
