"""会话循环、agent 工具、存档与恢复。

一个 Episode 对应一个模拟用户的一生；Env 管理全部用户，按会话序号一轮一轮推进，
因此可以在任意会话边界保存完整状态（用户真实状态、用户记着的"告诉过 agent 的值"、agent 记忆、判分记录）并恢复。
策略本身不保存跨会话状态：大模型 agent 和脚本策略的跨会话信息都只在记忆里。
"""

import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor

from . import texts
from .grading import grade_session
from .llm import ApiError, BudgetExceeded, CacheMiss
from .memory import MemoryStore
from .schema import ATTR_CN, ATTR_ORDER, CN_ATTR, DOMAINS
from .usersim import SimUser

CHECKPOINT_VERSION = 2


def render_option(o, domain):
    if domain == "util":
        return f"{o['id']}. {o['name']}"
    yn = lambda b: "有" if b else "无"
    parts = [o["name"], o["cuisine"], o["area"], f"人均 {o['price']} 元", f"口味：{o['spice']}",
             "饮食：" + "、".join(o["diet_tags"]), "含：" + "、".join(o["allergens"])]
    if domain == "dinner":
        parts[4:4] = [f"可订 {o['slot']}"]
        parts += [f"停车位：{yn(o['parking'])}", f"宝宝椅：{yn(o['kid_ok'])}",
                  "可带宠物：" + ("是" if o["pet_ok"] else "否")]
    return f"{o['id']}. " + "｜".join(parts)


def render_session(sess):
    head = f"【第 {sess['week']} 周 · {sess['date']} 周{sess['weekday']}】"
    opts = "\n".join(render_option(o, sess["domain"]) for o in sess["options"])
    return f"{head}\n用户：{sess['user_text']}\n\n候选选项：\n{opts}"


def render_true_state(state, temp, spicy_reason=None):
    """"知道真实状态"对照组：把用户当前的真实情况写成记忆条目。"""
    out = {}
    for a in ATTR_ORDER:
        v = texts.phrase(a, state[a], spicy_reason)
        if temp and temp["attr"] == a:
            v = f"平时：{v}；本次（{temp['cn']}）：{texts.phrase(a, temp['value'])}"
        out[f"用户当前真实情况·{ATTR_CN[a]}"] = v
    return out


def parse_attrs(raw):
    """把 agent 声明的属性（中文名或英文键）解析成属性键；返回 (有效键, 无效项)。"""
    if isinstance(raw, str):
        raw = [x for x in raw.replace("，", ",").replace("、", ",").split(",")]
    good, bad = [], []
    for x in raw or []:
        x = str(x).strip()
        if not x:
            continue
        k = CN_ATTR.get(x) or (x if x in ATTR_CN else None)
        (good if k else bad).append(k or x)
    return good, bad


def _advance(choices, keys, client):
    """把"每个多响应的键用第几个"当作里程表进一位；全部试过返回 False。"""
    for k in reversed(keys):
        start = client.default_choice.get(k, 0)
        n = len(client.alts[k])
        i = choices.get(k, start)
        if (i + 1) % n != start:
            choices[k] = (i + 1) % n
            return True
        choices[k] = start
    return False


class SessionContext:
    """一次会话里 agent 能做的事。"""

    def __init__(self, episode, sess):
        self.ep = episode
        self.sess = sess
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

    # 提问工具（计成本）：必须声明问哪几项属性
    def ask(self, question, attributes):
        good, bad = parse_attrs(attributes)
        if not good:
            return texts.NO_ATTR_ERROR + (f"（无法识别：{'、'.join(bad)}）" if bad else "")
        r = self.ep.user.answer(good)
        self.questions.append({"text": question, "attributes": r["answered"], "declared": good,
                               "invalid": bad, "answer": r["text"], "status": r["status"],
                               "too_many": r["too_many"],
                               "repeated": all(st in ("told_before", "answered_again") for st in r["status"].values())})
        return r["text"]

    def ask_structured(self, attrs):
        """脚本策略专用：照常提问计数，并按回答的文字（texts.parse_claims）拿到其中的陈述。"""
        text = self.ask("／".join(ATTR_CN[a] for a in attrs), [ATTR_CN[a] for a in attrs])
        return texts.parse_claims(text)

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

    # 特权接口：只有标准策略、理想记忆策略（只用于更新）和对照组会用到，用于判分自检与参照
    def oracle_state(self):
        return self.ep.user.effective()

    def rng(self, salt):
        return random.Random(f"{self.ep.seed}|{salt}|{self.ep.plan['uid']}|{self.sess['idx']}")


class Episode:
    def __init__(self, plan, seed, config):
        self.plan = plan
        self.seed = seed
        self.config = config
        self.user = SimUser(plan, config["repeat_mode"])
        self.memory = MemoryStore()
        self.next_idx = 0
        self.records = []
        self.truncated_at = None

    @property
    def finished(self):
        return self.next_idx >= len(self.plan["sessions"])

    def run_session(self, policy):
        sess = self.plan["sessions"][self.next_idx]
        rebuild = getattr(policy, "truncate_on_cache_miss", False)
        client = getattr(policy, "client", None) if rebuild else None
        snap = (copy.deepcopy(self.user.state_dict()), copy.deepcopy(self.memory.state_dict())) if rebuild else None
        choices = {}
        while True:
            if client is not None:
                client.set_choices(choices)
            self.user.begin_session(sess)
            if getattr(policy, "inject_true_memory", False):
                self.memory.entries = {k: {"value": v, "week": sess["week"]}
                                       for k, v in render_true_state(self.user.state, sess.get("temp"), self.user.spicy_reason).items()}
            ctx = SessionContext(self, sess)
            error = None
            try:
                policy.run_session(ctx)
                break
            except CacheMiss:
                if not rebuild:
                    raise
                # 只用缓存重建。同一个键有多个缓存响应（并发运行时写过两次）：换下一个组合重试这次会话；
                # 都试过还缺缓存，说明当时没跑完，这个用户从这里起截断，不记录
                self.user.load_state_dict(copy.deepcopy(snap[0]))
                self.memory.load_state_dict(copy.deepcopy(snap[1]))
                if not _advance(choices, client.ambiguous_used() if client else [], client):
                    self.truncated_at = self.next_idx
                    self.next_idx = len(self.plan["sessions"])
                    if client is not None:
                        client.set_choices(None)
                    return None
            except Exception as e:  # noqa: BLE001 —— agent 出错按"未作答"记，并留下原因
                if getattr(policy, "raise_errors", False) or isinstance(e, (BudgetExceeded, ApiError)):
                    raise
                error = f"{type(e).__name__}: {e}"
                break
        if client is not None:
            client.set_choices(None)
        self.user.end_session()
        rec = {
            "uid": self.plan["uid"], "idx": sess["idx"], "week": sess["week"], "type": sess["type"],
            "domain": sess["domain"], "event": sess["event"]["cn"] if sess["event"] else None,
            "temp": sess["temp"]["cn"] if sess["temp"] else None,
            "chosen": ctx.chosen, "correct": sess["correct"],
            "chosen_role": sess["roles"].get(ctx.chosen) if ctx.chosen else None,
            "questions": ctx.questions, "memory_ops": ctx.memory_ops,
            "memory_after": {k: v["value"] for k, v in self.memory.entries.items()},
            "told_after": dict(self.user.told),
            "grade": grade_session(sess, ctx.chosen, len(ctx.questions), self.config["question_cost"]),
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
    def __init__(self, world, config=None):
        """config 覆盖世界自带的配置（repeat_mode、question_cost）。"""
        self.world = world
        self.config = dict(world["config"], **(config or {}))
        self.episodes = [Episode(u, world["seed"], self.config) for u in world["users"]]

    def run(self, policy, until=None, workers=1):
        """推进到会话序号 until（不含）；None 表示跑完。按会话序号逐轮推进，各用户之间并行。"""
        n = self.world["n_sessions"] if until is None else until
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                todo = [e for e in self.episodes if e.next_idx < n]
                if not todo:
                    break
                list(pool.map(lambda e: e.run_session(policy), todo))

    def records(self):
        return sorted((r for e in self.episodes for r in e.records), key=lambda r: (r["uid"], r["idx"]))

    def save(self, path):
        data = {"version": CHECKPOINT_VERSION, "world": self.world, "config": self.config,
                "episodes": [e.state_dict() for e in self.episodes]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == CHECKPOINT_VERSION
        env = cls(data["world"], data["config"])
        for e, d in zip(env.episodes, data["episodes"]):
            e.load_state_dict(d)
        return env
