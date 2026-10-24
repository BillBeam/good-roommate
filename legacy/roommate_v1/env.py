"""会话循环、agent 工具、存档与恢复。

一个 Episode 对应一个模拟用户的一生；Env 管理全部用户，按会话序号一轮一轮推进，
因此可以在任意会话边界保存完整状态（用户真实状态、agent 记忆、已产生的判分记录）并恢复。
策略本身不保存跨会话状态：大模型 agent 和脚本策略的跨会话信息都只在记忆里。
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor

from .grading import grade_session
from .llm import BudgetExceeded, CacheMiss
from .memory import MemoryStore
from .usersim import SimUser

CHECKPOINT_VERSION = 1


def render_option(o, domain):
    if domain == "util":
        return f"{o['id']}. {o['name']}"
    yn = lambda b: "有" if b else "无"
    if domain == "hotel":
        parts = [o["name"], f"每晚 {o['price']} 元", "早餐：" + ("、".join(o["diet_tags"]) or "普通"),
                 f"停车位：{yn(o['parking'])}", f"婴儿床：{yn(o['kid_ok'])}",
                 "可带宠物：" + ("是" if o["pet_ok"] else "否")]
    else:
        parts = [o["name"], o["cuisine"], o["area"], f"人均 {o['price']} 元", f"口味：{o['spice']}",
                 "饮食：" + ("、".join(o["diet_tags"]) or "无特别标注"),
                 "含：" + ("、".join(o["allergens"]) or "无常见过敏原")]
        if domain == "dinner":
            parts[4:4] = [f"可订 {o['slot']}"]
            parts += [f"停车位：{yn(o['parking'])}", f"宝宝椅：{yn(o['kid_ok'])}"]
    return f"{o['id']}. " + "｜".join(parts)


def render_session(sess):
    head = f"【第 {sess['week']} 周 · {sess['date']} 周{sess['weekday']}】"
    opts = "\n".join(render_option(o, sess["domain"]) for o in sess["options"])
    return f"{head}\n用户：{sess['user_text']}\n\n候选选项：\n{opts}"


class SessionContext:
    """一次会话里 agent 能做的事。"""

    def __init__(self, episode, sess, classify):
        self.ep = episode
        self.sess = sess
        self.classify = classify
        self.prompt = render_session(sess)
        self.questions = []
        self.memory_ops = []
        self.chosen = None
        self.done = False
        self.notes = {}

    # 记忆工具
    def memory_read(self):
        self.memory_ops.append(["read"])
        return self.ep.memory.read()

    def memory_write(self, key, value):
        self.memory_ops.append(["write", key, value])
        return self.ep.memory.write(key, value, self.sess["week"])

    def memory_delete(self, key):
        self.memory_ops.append(["delete", key])
        return self.ep.memory.delete(key)

    # 提问工具（计成本）
    def ask(self, question, topics=None):
        if topics is None:
            topics = self.classify(question)
        text, answered, rep = self.ep.user.answer(topics)
        self.questions.append({"text": question, "topics": answered,
                               "repeated": bool(answered) and all(rep), "answer": text})
        return text

    def ask_structured(self, topics):
        """脚本策略专用：提问照常计数，额外拿到结构化取值。"""
        self.ask("／".join(topics), topics=topics)
        return self.ep.user.structured(topics[:2])

    def choose(self, option_id):
        if self.done:
            return "本次会话已经结束。"
        ids = [o["id"] for o in self.sess["options"]]
        option_id = str(option_id).strip().upper()[:1]
        if option_id not in ids:
            return f"错误：没有选项 {option_id}，可选 {'、'.join(ids)}"
        self.chosen = option_id
        self.done = True
        return f"已选择 {option_id}，本次会话结束。"

    # 特权接口：只有"标准策略"会调用，用于判分自检
    def oracle_state(self):
        return self.ep.user.effective()

    def rng(self, salt):
        return random.Random(f"{self.ep.seed}|{salt}|{self.ep.plan['uid']}|{self.sess['idx']}")


class Episode:
    def __init__(self, plan, seed):
        self.plan = plan
        self.seed = seed
        self.user = SimUser(plan)
        self.memory = MemoryStore()
        self.next_idx = 0
        self.records = []

    @property
    def finished(self):
        return self.next_idx >= len(self.plan["sessions"])

    def run_session(self, policy, classify):
        sess = self.plan["sessions"][self.next_idx]
        self.user.begin_session(sess)
        ctx = SessionContext(self, sess, classify)
        error = None
        try:
            policy.run_session(ctx)
        except Exception as e:  # noqa: BLE001 —— agent 出错按"未作答"记，并留下原因
            if getattr(policy, "raise_errors", False) or isinstance(e, (BudgetExceeded, CacheMiss)):
                raise
            error = f"{type(e).__name__}: {e}"
        self.user.end_session()
        rec = {
            "uid": self.plan["uid"], "idx": sess["idx"], "week": sess["week"], "type": sess["type"],
            "domain": sess["domain"], "event": sess["event"]["cn"] if sess["event"] else None,
            "temp": sess["temp"]["cn"] if sess["temp"] else None,
            "chosen": ctx.chosen, "correct": sess["correct"],
            "chosen_role": sess["roles"].get(ctx.chosen) if ctx.chosen else None,
            "questions": ctx.questions, "memory_ops": ctx.memory_ops,
            "memory_after": {k: v["value"] for k, v in self.memory.entries.items()},
            "grade": grade_session(sess, ctx.chosen, len(ctx.questions)),
            "error": error, **ctx.notes,
        }
        self.records.append(rec)
        self.next_idx += 1
        return rec

    def state_dict(self):
        return {"uid": self.plan["uid"], "next_idx": self.next_idx, "user": self.user.state_dict(),
                "memory": self.memory.state_dict(), "records": self.records}

    def load_state_dict(self, d):
        assert d["uid"] == self.plan["uid"]
        self.next_idx = d["next_idx"]
        self.user.load_state_dict(d["user"])
        self.memory.load_state_dict(d["memory"])
        self.records = list(d["records"])


class Env:
    def __init__(self, world, classify):
        self.world = world
        self.classify = classify
        self.episodes = [Episode(u, world["seed"]) for u in world["users"]]

    def run(self, policy, until=None, workers=1):
        """推进到会话序号 until（不含）；None 表示跑完。按会话序号逐轮推进，各用户之间并行。"""
        n = self.world["n_sessions"] if until is None else until
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                todo = [e for e in self.episodes if e.next_idx < n]
                if not todo:
                    break
                list(pool.map(lambda e: e.run_session(policy, self.classify), todo))

    def records(self):
        return sorted((r for e in self.episodes for r in e.records), key=lambda r: (r["uid"], r["idx"]))

    def save(self, path):
        data = {"version": CHECKPOINT_VERSION, "world": self.world,
                "episodes": [e.state_dict() for e in self.episodes]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path, classify):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == CHECKPOINT_VERSION
        env = cls(data["world"], classify)
        for e, d in zip(env.episodes, data["episodes"]):
            e.load_state_dict(d)
        return env
