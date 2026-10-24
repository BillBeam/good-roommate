"""xAI 接口：带磁盘缓存（保证复现）与计费账本（控制总花费）。

- API 密钥只从环境变量 XAI_API_KEY 读取，不写入任何文件或日志。
- 每个请求按（模型、消息、工具、参数）的哈希缓存响应；同一种子重跑时全部命中缓存，结果逐字一致。
- 账本按 xAI 返回的 usage.cost_in_usd_ticks（1 tick = 1e-10 美元）记真实花费；累计花费超过上限就停止。
"""

import fcntl
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

API_URL = "https://api.x.ai/v1/chat/completions"
TICKS_PER_USD = 1e10


class BudgetExceeded(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


class Ledger:
    """累计花费账本，跨多次运行、多个进程持久化（写入时加文件锁并重新读取）。"""

    def __init__(self, path, limit_usd):
        self.path = path
        self.limit = limit_usd
        self.lock = threading.Lock()
        self.data = self._load()
        self.session = {}  # 本进程内的统计（含缓存命中），用于报告

    def _load(self):
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"total_usd": 0.0, "by_model_role": {}}

    def check(self):
        with self.lock:
            if self.path:
                self.data = self._load()
            if self.data["total_usd"] >= self.limit:
                raise BudgetExceeded(f"累计花费 {self.data['total_usd']:.2f} 美元已达上限 {self.limit} 美元")

    @staticmethod
    def _bump(table, k, usage, cached):
        e = table.setdefault(k, {
            "api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "cached_prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0})
        if cached:
            e["cache_hits"] += 1
            return 0.0
        cost = usage.get("cost_in_usd_ticks", 0) / TICKS_PER_USD
        e["api_calls"] += 1
        e["prompt_tokens"] += usage.get("prompt_tokens", 0)
        e["cached_prompt_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        e["completion_tokens"] += usage.get("completion_tokens", 0)
        e["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        e["cost_usd"] += cost
        return cost

    def add(self, model, role, usage, cached):
        k = f"{model}|{role}"
        with self.lock:
            self._bump(self.session, k, usage, cached)
            if cached or not self.path:
                return
            with open(self.path + ".lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                self.data = self._load()
                self.data["total_usd"] += self._bump(self.data["by_model_role"], k, usage, cached)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)


class XAIClient:
    def __init__(self, cache_dir, ledger, offline=False, role_prefix=""):
        self.role_prefix = role_prefix
        self.cache_dir = cache_dir
        self.ledger = ledger
        self.offline = offline
        self.lock = threading.Lock()
        self.cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        for fn in sorted(os.listdir(cache_dir)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(cache_dir, fn), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            self.cache[d["key"]] = d["response"]

    @staticmethod
    def key(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def chat(self, body, role):
        k = self.key(body)
        model = body["model"]
        role = self.role_prefix + role
        if k in self.cache:
            self.ledger.add(model, role, self.cache[k].get("usage", {}), cached=True)
            return self.cache[k]
        if self.offline:
            raise CacheMiss(k)
        self.ledger.check()
        resp = self._post(body)
        resp.pop("id", None)
        resp.pop("created", None)
        with self.lock:
            self.cache[k] = resp
            with open(os.path.join(self.cache_dir, f"{model}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "response": resp}, ensure_ascii=False) + "\n")
        self.ledger.add(model, role, resp.get("usage", {}), cached=False)
        return resp

    def _post(self, body):
        key = os.environ.get("XAI_API_KEY")
        if not key:
            raise RuntimeError("环境变量 XAI_API_KEY 未设置")
        data = json.dumps(body, ensure_ascii=False).encode()
        last = None
        for attempt in range(6):
            req = urllib.request.Request(API_URL, data=data, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read()[:500].decode(errors="replace")
                last = f"HTTP {e.code}: {msg}"
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(last) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = repr(e)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"xAI 请求失败：{last}")


# ---------- 模拟用户的提问归类 ----------

CLASSIFY_PROMPT = """下面是私人助理向用户提出的一个问题。判断它在问用户哪些方面，只能从这些话题里选：
- 饮食：过敏、忌口、吃素、控碳水、控糖等饮食限制
- 口味：爱吃的菜系、能不能吃辣
- 预算：花多少钱、价位档次
- 住处：住在哪里、家在哪个区
- 公司：在哪里上班、公司在哪个区
- 出行：交通方式、是否开车、要不要停车位
- 家庭：小孩、宠物、晚上几点要回家等家庭情况与作息
- 近况：最近生活有没有变化、发生了什么事（泛泛地问近况）
注意：问停车位属于"出行"，不是"住处"；问酒店早餐属于"饮食"；问几个人、哪天、什么时候去这类与用户偏好无关的内容不算任何话题。
按问题里出现的先后排序；如果问题不涉及以上任何方面，返回空列表。
只输出 JSON，例如 {"topics": ["饮食", "预算"]}。

问题：%s"""

USERSIM_MODEL = "grok-4.20-0309-non-reasoning"


def make_classifier(client, model=USERSIM_MODEL):
    from .schema import TOPICS

    def classify(question):
        body = {"model": model, "temperature": 0, "seed": 1,
                "messages": [{"role": "user", "content": CLASSIFY_PROMPT % question}],
                "response_format": {"type": "json_object"}}
        resp = client.chat(body, role="usersim")
        try:
            topics = json.loads(resp["choices"][0]["message"]["content"])["topics"]
        except (ValueError, KeyError, TypeError):
            return []
        return [t for t in topics if t in TOPICS]

    return classify
