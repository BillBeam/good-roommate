"""模拟用户（第二版）：由程序维护真实状态，按 agent 声明的属性、用模板回答。

- 只答所问：agent 提问时声明问哪几项属性（一次最多两项），用户只回答这几项；不再用模型猜问题问的是什么。
- 前后一致：每个回答由 texts.PHRASES 里的短句拼成；健康原因造成的不吃辣，被问到饮食限制时一定说出来。
- 说过的不再说：用户记着每项属性"上次告诉 agent 的值"（回答里说过的，或事件台词里明说、可推断的）。
  再问时，当前值等于上次告诉的值，只答"这个我之前跟你说过了"（照常扣分）；从没告诉过的照实回答；
  本次临时情况改到的属性，照实回答临时值。是否这样做由世界配置 repeat_mode 决定：
  "refuse_told"（第二版默认）或 "always_answer"（第一版的行为：总会重新回答）。
"""

from . import texts
from .schema import MAX_ATTRS_PER_QUESTION, ATTR_CN, ATTRS

REPEAT_MODES = ("refuse_told", "always_answer")


class SimUser:
    def __init__(self, plan, repeat_mode="refuse_told"):
        assert repeat_mode in REPEAT_MODES, repeat_mode
        self.plan = plan
        self.repeat_mode = repeat_mode
        self.state = dict(plan["init_state"])   # 长期真实状态
        self.spicy_reason = plan.get("init_spicy_reason", "taste")
        self.temp = None                         # 本次会话的临时情况
        self.told = {}                           # 属性 -> 上次告诉 agent 的长期值

    # ---------- 会话边界 ----------
    def begin_session(self, sess):
        ev = sess.get("event")
        if ev:
            for a, (old, new) in ev["changes"].items():
                assert self.state[a] == old, (self.plan["uid"], sess["idx"], a, self.state[a], old)
                self.state[a] = new
            if ev["type"] in texts.HEALTH_SPICY_EVENTS:
                self.spicy_reason = ev["type"]
            for a in ev["cues"]:          # 台词里明说或可推断的属性，算作已经告诉过 agent
                self.told[a] = self.state[a]
        self.temp = sess.get("temp")
        if sess.get("eff_state") is not None:
            assert self.effective() == sess["eff_state"], (self.plan["uid"], sess["idx"])

    def end_session(self):
        self.temp = None

    def effective(self):
        s = dict(self.state)
        if self.temp:
            s[self.temp["attr"]] = self.temp["value"]
        return s

    # ---------- 回答 ----------
    def _refuse(self, a):
        return self.repeat_mode == "refuse_told" and self.told.get(a) == self.state[a]

    def _answer_attr(self, a):
        """返回 (文字, 状态)。状态："answered" | "told_before" | "temp"。"""
        long = self.state[a]
        t = self.temp
        if t and t["attr"] == a:
            cn = ATTR_CN[a]
            if a == "home":
                head = texts.TEMP_LONG_TOLD.format(attr_cn=cn) if self._refuse(a) else texts.phrase(a, long)
                text = head + "，" + texts.HOME_TEMP_PART.format(area=t["value"])
            else:
                head = (texts.TEMP_LONG_TOLD.format(attr_cn=cn) if self._refuse(a)
                        else "平时" + texts.phrase(a, long, self.spicy_reason))
                text = head + "；" + texts.TEMP_PART.format(temp_cn=t["reason"], phrase=texts.phrase(a, t["value"]))
            self.told[a] = long
            return text, "temp"
        if self._refuse(a):
            text, status = texts.TOLD_BEFORE.format(attr_cn=ATTR_CN[a]), "told_before"
        else:
            text, status = texts.phrase(a, long, self.spicy_reason), "answered"
            self.told[a] = long
        if a == "diet":
            # 健康原因不吃辣也是饮食限制，被问到时一定说出来
            if status == "answered" and self.state["spicy"] == "不吃辣" and self.spicy_reason in texts.SPICY_REASON:
                text += "；" + texts.DIET_HEALTH_APPENDIX.format(reason=texts.SPICY_REASON[self.spicy_reason])
                self.told["spicy"] = self.state["spicy"]
            if t and t["attr"] == "spicy":
                text += "；" + texts.DIET_TEMP_APPENDIX.format(temp_cn=t["reason"])
        return text, status

    def answer(self, attrs):
        """attrs：属性键的列表（已去掉无效项）。返回 dict(text, answered, status, too_many)。"""
        attrs = [a for a in dict.fromkeys(attrs) if a in ATTRS]
        assert attrs, "调用前应先检查至少有一项有效属性"
        answered = attrs[:MAX_ATTRS_PER_QUESTION]
        pieces, status = [], {}
        for a in answered:
            before = self.told.get(a)
            text, st = self._answer_attr(a)
            if self.repeat_mode == "always_answer" and st == "answered" and before == self.state[a]:
                st = "answered_again"
            pieces.append(text)
            status[a] = st
        text = "；".join(pieces) + "。"
        too_many = len(attrs) > MAX_ATTRS_PER_QUESTION
        if too_many:
            text += texts.TOO_MANY_ANSWER
        return {"text": text, "answered": answered, "status": status, "too_many": too_many}

    # ---------- 存档 ----------
    def state_dict(self):
        return {"state": self.state, "temp": self.temp, "spicy_reason": self.spicy_reason, "told": self.told}

    def load_state_dict(self, d):
        self.state = dict(d["state"])
        self.temp = d["temp"]
        self.spicy_reason = d["spicy_reason"]
        self.told = dict(d["told"])
