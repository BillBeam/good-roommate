"""大模型接口（xAI 与 DeepSeek，都是 OpenAI 兼容的 chat/completions）：带磁盘缓存（保证复现）与计费账本（控制总花费）。

- API 密钥只从环境变量读取（XAI_API_KEY、DEEPSEEK_API_KEY），不写入任何文件或日志。
- 每个请求按（模型、消息、工具、参数）的哈希缓存响应；同一种子重跑时全部命中缓存，结果逐字一致。
- 账本记真实花费：xAI 用响应里的 usage.cost_in_usd_ticks（1 tick = 1e-10 美元）；
  DeepSeek 的响应不带金额，按官方价目表（runs/m1v2/pricing/ 下的快照）、调用时刻（高峰 / 空闲时段）
  和 usage 里的 token 数计算，同时记人民币和美元。每次实际调用另记一行到 spend_calls.jsonl，便于和服务商后台对账。
- 累计花费超过上限就停止；DeepSeek 另有人民币上限（账本里的 deepseek_cny_limit）。
- blocked_providers 里的服务商只能读缓存，不会发出任何请求。
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
# DeepSeek 单价（每百万 tokens）：{模型: {"usd"|"cny": {"peak"|"off_peak": {input_cache_hit, input_cache_miss, output}}}}。
# 由 run.py 从官方价目快照读入；没有快照时拒绝调用。
DEEPSEEK_PRICES = {}


def is_peak(t=None):
    """DeepSeek 高峰时段：北京时间周一至周五 9:00–12:00、14:00–18:00，即 UTC 01:00–04:00、06:00–10:00（同一天）。
    中国法定节假日也算空闲时段，这里没有节假日表，节假日按高峰算（偏高估）。"""
    import datetime as dt
    t = t or dt.datetime.now(dt.timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def provider_of(model):
    return "deepseek" if model.startswith("deepseek") else "xai"


class BudgetExceeded(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


class ApiError(RuntimeError):
    """接口调用失败（额度用完、权限、重试仍失败等）。这不是 agent 的错，整次运行要停下，不能记成"未作答"。"""


def usage_cost(model, usage, peak=None):
    """返回 (美元, 人民币)。xAI 只有美元。"""
    if provider_of(model) == "xai":
        return usage.get("cost_in_usd_ticks", 0) / TICKS_PER_USD, None
    p = DEEPSEEK_PRICES.get(model)
    if p is None:
        raise RuntimeError(f"没有 {model} 的官方单价快照，不能计费")
    tier = "peak" if (is_peak() if peak is None else peak) else "off_peak"
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    out = usage.get("completion_tokens", 0)
    f = lambda q: (hit * q["input_cache_hit"] + miss * q["input_cache_miss"] + out * q["output"]) / 1e6
    return f(p["usd"][tier]), f(p["cny"][tier])


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
        return {"total_usd": 0.0, "deepseek_cny_total": 0.0, "by_model_role": {}}

    def check(self, model=None):
        with self.lock:
            if self.path:
                self.data = self._load()
            if self.data["total_usd"] >= self.limit:
                raise BudgetExceeded(f"累计花费 {self.data['total_usd']:.2f} 美元已达上限 {self.limit} 美元")
            lim = self.data.get("deepseek_cny_limit")
            if model and provider_of(model) == "deepseek" and lim is not None and \
                    self.data.get("deepseek_cny_total", 0.0) >= lim:
                raise BudgetExceeded(f"DeepSeek 累计 {self.data['deepseek_cny_total']:.2f} 元已达上限 {lim:.2f} 元")

    @staticmethod
    def _bump(table, k, model, usage, cached):
        e = table.setdefault(k, {
            "api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "cached_prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0})
        if cached:
            e["cache_hits"] += 1
            return 0.0, None
        cost, cny = usage_cost(model, usage)
        e["api_calls"] += 1
        e["prompt_tokens"] += usage.get("prompt_tokens", 0)
        e["cached_prompt_tokens"] += ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                      or usage.get("prompt_cache_hit_tokens", 0))
        e["completion_tokens"] += usage.get("completion_tokens", 0)
        e["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        e["cost_usd"] += cost
        if cny is not None:
            e["cost_cny"] = e.get("cost_cny", 0.0) + cny
        return cost, cny

    def add(self, model, role, usage, cached):
        import datetime as dt
        k = f"{model}|{role}"
        with self.lock:
            self._bump(self.session, k, model, usage, cached)
            if cached or not self.path:
                return
            with open(self.path + ".lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                self.data = self._load()
                usd, cny = self._bump(self.data["by_model_role"], k, model, usage, cached)
                self.data["total_usd"] += usd
                if cny is not None:
                    self.data["deepseek_cny_total"] = self.data.get("deepseek_cny_total", 0.0) + cny
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
                calls = os.path.join(os.path.dirname(self.path), "spend_calls.jsonl")
                with open(calls, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        "model": model, "role": role, "peak": is_peak() if cny is not None else None,
                                        "usage": {x: usage.get(x) for x in ("prompt_tokens", "completion_tokens",
                                                                            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                                                                            "cost_in_usd_ticks") if x in usage},
                                        "cost_usd": usd, "cost_cny": cny}, ensure_ascii=False) + "\n")


class LLMClient:
    def __init__(self, cache_dir, ledger, offline=False, role_prefix="", blocked_providers=()):
        self.role_prefix = role_prefix
        self.blocked = set(blocked_providers)
        self.cache_dir = cache_dir
        self.ledger = ledger
        self.offline = offline
        self.lock = threading.Lock()
        self.cache = {}   # 键 -> 第一次写入的响应
        self.alts = {}    # 键 -> 全部响应（按写入顺序）；只有并发运行时同一个键被写过两次才会多于一个
        self._tl = threading.local()
        self.chosen = {}
        self.default_choice = {}
        os.makedirs(cache_dir, exist_ok=True)
        for fn in sorted(os.listdir(cache_dir)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(cache_dir, fn), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            self.alts.setdefault(d["key"], []).append(d["response"])
                            self.cache.setdefault(d["key"], d["response"])

    # 从缓存重建时，同一个键有多个响应：由调用方（Episode）指定这次用第几个，并记下这次会话碰到了哪些这样的键
    def set_choices(self, choices):
        self._tl.choices = choices
        self._tl.ambiguous = []

    def ambiguous_used(self):
        return list(dict.fromkeys(getattr(self._tl, "ambiguous", [])))

    def _lookup(self, k):
        alts = self.alts.get(k)
        if alts and len(alts) > 1 and getattr(self._tl, "choices", None) is not None:
            self._tl.ambiguous.append(k)
            i = self._tl.choices.get(k, self.default_choice.get(k, 0))
            self.chosen[k] = i
            return alts[i]
        return self.cache.get(k)

    # 记录重建时每个多响应的键最终用了第几个；另一种运行可以据此指定"用另一个"（default_choice）
    chosen = {}
    default_choice = {}

    @staticmethod
    def key(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def chat(self, body, role):
        # 新请求的键带上用途（agent / control / …），免得两种运行的第一轮请求完全相同、并发时互相覆盖；
        # 旧的键（不带用途）照样能读到，已有缓存全部有效
        k_old = self.key(body)
        k = self.key(dict(body, __cache_role=role))
        model = body["model"]
        role = self.role_prefix + role
        for kk in (k, k_old):
            hit = self._lookup(kk)
            if hit is not None:
                self.ledger.add(model, role, hit.get("usage", {}), cached=True)
                return hit
        if self.offline or provider_of(model) in self.blocked:
            raise CacheMiss(k)
        self.ledger.check(model)
        resp = self._post(body)
        resp.pop("id", None)
        resp.pop("created", None)
        resp.pop("system_fingerprint", None)
        with self.lock:
            self.cache[k] = resp
            self.alts[k] = [resp]
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
                    raise ApiError(last) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = repr(e)
            time.sleep(2 ** attempt)
        raise ApiError(f"请求失败：{last}")


XAIClient = LLMClient  # 第一版的名字，保留兼容
