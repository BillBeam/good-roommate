"""大模型接口（xAI 与 DeepSeek，都是 OpenAI 兼容的 chat/completions）：带磁盘缓存（保证复现）与计费账本（控制总花费）。

- API 密钥只从环境变量读取（XAI_API_KEY、DEEPSEEK_API_KEY），不写入任何文件或日志。
- 每个请求按（模型、消息、工具、参数）的哈希缓存响应；同一种子重跑时全部命中缓存，结果逐字一致。
- 账本记真实花费：xAI 用响应里的 usage.cost_in_usd_ticks（1 tick = 1e-10 美元）；
  DeepSeek 的响应不带金额，按官方价目表（runs/m1v2/pricing/ 下的快照）和 usage 里的 token 数计算。
  累计花费超过上限就停止。
"""

import fcntl
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

TICKS_PER_USD = 1e10
PROVIDERS = {
    "xai": {"url": "https://api.x.ai/v1/chat/completions", "key": "XAI_API_KEY"},
    "deepseek": {"url": "https://api.deepseek.com/chat/completions", "key": "DEEPSEEK_API_KEY"},
}
# DeepSeek 单价（美元 / 百万 tokens）。由 run.py 从官方价目快照读入；没有快照时拒绝调用。
DEEPSEEK_PRICES = {}


def provider_of(model):
    return "deepseek" if model.startswith("deepseek") else "xai"


class BudgetExceeded(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


def usage_cost(model, usage):
    if provider_of(model) == "xai":
        return usage.get("cost_in_usd_ticks", 0) / TICKS_PER_USD
    p = DEEPSEEK_PRICES.get(model)
    if p is None:
        raise RuntimeError(f"没有 {model} 的官方单价快照，不能计费")
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    return (hit * p["input_cache_hit"] + miss * p["input_cache_miss"] + usage.get("completion_tokens", 0) * p["output"]) / 1e6


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
    def _bump(table, k, model, usage, cached):
        e = table.setdefault(k, {
            "api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "cached_prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0})
        if cached:
            e["cache_hits"] += 1
            return 0.0
        cost = usage_cost(model, usage)
        e["api_calls"] += 1
        e["prompt_tokens"] += usage.get("prompt_tokens", 0)
        e["cached_prompt_tokens"] += ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                      or usage.get("prompt_cache_hit_tokens", 0))
        e["completion_tokens"] += usage.get("completion_tokens", 0)
        e["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        e["cost_usd"] += cost
        return cost

    def add(self, model, role, usage, cached):
        k = f"{model}|{role}"
        with self.lock:
            self._bump(self.session, k, model, usage, cached)
            if cached or not self.path:
                return
            with open(self.path + ".lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                self.data = self._load()
                self.data["total_usd"] += self._bump(self.data["by_model_role"], k, model, usage, cached)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)


class LLMClient:
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
        resp.pop("system_fingerprint", None)
        with self.lock:
            self.cache[k] = resp
            with open(os.path.join(self.cache_dir, f"{model}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "response": resp}, ensure_ascii=False) + "\n")
        self.ledger.add(model, role, resp.get("usage", {}), cached=False)
        return resp

    def _post(self, body):
        prov = PROVIDERS[provider_of(body["model"])]
        key = os.environ.get(prov["key"])
        if not key:
            raise RuntimeError(f"环境变量 {prov['key']} 未设置")
        data = json.dumps(body, ensure_ascii=False).encode()
        last = None
        for attempt in range(6):
            req = urllib.request.Request(prov["url"], data=data, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read()[:500].decode(errors="replace")
                last = f"HTTP {e.code}: {msg}"
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(last) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = repr(e)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"请求失败：{last}")


XAIClient = LLMClient  # 第一版的名字，保留兼容
