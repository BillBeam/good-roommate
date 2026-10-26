"""大模型接口（xAI、DeepSeek 与硅基流动，都是 OpenAI 兼容的 chat/completions；Claude 走 Anthropic 官方 SDK 的 Messages API）：带磁盘缓存（保证复现）与计费账本（控制总花费）。

- API 密钥只从环境变量读取（XAI_API_KEY、DEEPSEEK_API_KEY、SILICONFLOW_API_KEY），不写入任何文件或日志。
- 每个请求按（模型、消息、工具、参数）的哈希缓存响应；同一种子重跑时全部命中缓存，结果逐字一致。
- 账本记真实花费：xAI 用响应里的 usage.cost_in_usd_ticks（1 tick = 1e-10 美元）；
  DeepSeek 的响应不带金额，按官方价目表（runs/m1v2/pricing/ 下的快照）、调用时刻（高峰 / 空闲时段）
  和 usage 里的 token 数计算，同时记人民币和美元。每次实际调用另记一行到 spend_calls.jsonl，便于和服务商后台对账。
- 硅基流动（模型名以 "Qwen/" 开头）同样按价目快照（runs/m2/pricing/）和 token 数记人民币，美元按快照里的汇率折算。
- Claude（模型名以 "claude-" 开头）的密钥只读 CLAUDE_EVAL_API_KEY；请求体是 Messages API 的原生格式，响应按 model_dump 存缓存。
  金额按官方价目快照（runs/m1v2/pricing/anthropic_*.json）和 usage 里的四类 token（未缓存输入、缓存写入、缓存命中、输出）计算。
- 累计花费超过上限就停止；按人民币计价的服务商另有人民币上限（账本里的 deepseek_cny_limit、siliconflow_cny_limit）。
- blocked_providers 里的服务商只能读缓存，不会发出任何请求。
"""

import fcntl
import gzip
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
    "siliconflow": {"url": "https://api.siliconflow.cn/v1/chat/completions", "key": "SILICONFLOW_API_KEY"},
}
# DeepSeek 单价（每百万 tokens）：{模型: {"usd"|"cny": {"peak"|"off_peak": {input_cache_hit, input_cache_miss, output}}}}。
# 由 run.py 从官方价目快照读入；没有快照时拒绝调用。
DEEPSEEK_PRICES = {}
# 硅基流动单价（每百万 tokens，人民币）：{模型: {"input", "output"}}，另有折美元的汇率。由 m2.py 从价目快照读入。
SILICONFLOW_PRICES = {}
SILICONFLOW_FX = {"cny_per_usd": None}
# Claude 单价（每百万 tokens，美元）：{模型: {input, cache_write_5m, cache_write_1h, cache_read, output}}，
# 有长上下文分档的模型另有 "long": {"over_prompt_tokens": N, 同样五项}。由 claude_run.py 从官方价目快照读入。
ANTHROPIC_PRICES = {}
ANTHROPIC_KEY_ENV = "CLAUDE_EVAL_API_KEY"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"   # 显式指定：环境里的 ANTHROPIC_BASE_URL 不是给评测用的


def is_peak(t=None):
    """DeepSeek 高峰时段：北京时间周一至周五 9:00–12:00、14:00–18:00，即 UTC 01:00–04:00、06:00–10:00（同一天）。
    中国法定节假日也算空闲时段，这里没有节假日表，节假日按高峰算（偏高估）。"""
    import datetime as dt
    t = t or dt.datetime.now(dt.timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def provider_of(model):
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("Qwen/"):
        return "siliconflow"
    return "deepseek" if model.startswith("deepseek") else "xai"


class BudgetExceeded(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


class ApiError(RuntimeError):
    """接口调用失败（额度用完、权限、重试仍失败等）。这不是 agent 的错，整次运行要停下，不能记成"未作答"。"""


def anthropic_tokens(usage):
    """Claude 的 usage → (未缓存输入, 5 分钟缓存写入, 1 小时缓存写入, 缓存命中, 输出)。"""
    cw = usage.get("cache_creation_input_tokens") or 0
    split = usage.get("cache_creation") or {}
    w1h = split.get("ephemeral_1h_input_tokens") or 0
    return (usage.get("input_tokens") or 0, cw - w1h, w1h, usage.get("cache_read_input_tokens") or 0,
            usage.get("output_tokens") or 0)


def usage_cost(model, usage, peak=None):
    """返回 (美元, 人民币)。xAI 与 Claude 只有美元。"""
    if provider_of(model) == "anthropic":
        p = ANTHROPIC_PRICES.get(model)
        if p is None:
            raise RuntimeError(f"没有 {model} 的官方单价快照，不能计费")
        inp, w5, w1, rd, out = anthropic_tokens(usage)
        if "long" in p and inp + w5 + w1 + rd > p["long"]["over_prompt_tokens"]:
            p = p["long"]
        return (inp * p["input"] + w5 * p["cache_write_5m"] + w1 * p["cache_write_1h"] + rd * p["cache_read"]
                + out * p["output"]) / 1e6, None
    if provider_of(model) == "xai":
        return usage.get("cost_in_usd_ticks", 0) / TICKS_PER_USD, None
    if provider_of(model) == "siliconflow":
        p = SILICONFLOW_PRICES.get(model)
        if p is None or not SILICONFLOW_FX["cny_per_usd"]:
            raise RuntimeError(f"没有 {model} 的官方单价快照，不能计费")
        cny = (usage.get("prompt_tokens", 0) * p["input"] + usage.get("completion_tokens", 0) * p["output"]) / 1e6
        return cny / SILICONFLOW_FX["cny_per_usd"], cny
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

    def __init__(self, path, limit_usd, reserve_usd=0.0):
        self.path = path
        self.limit = limit_usd
        # 留给并发中、还没记账的请求的余量：累计 + 余量到上限就不再发新请求，保证最终不超过上限
        self.reserve = reserve_usd
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
            if self.data["total_usd"] + self.reserve >= self.limit:
                raise BudgetExceeded(f"累计花费 {self.data['total_usd']:.2f} 美元（另留 {self.reserve} 美元给并发中的请求）"
                                     f"已达上限 {self.limit} 美元")
            prov = provider_of(model) if model else None
            lim = self.data.get(f"{prov}_cny_limit")
            if prov in ("deepseek", "siliconflow") and lim is not None and \
                    self.data.get(f"{prov}_cny_total", 0.0) >= lim:
                raise BudgetExceeded(f"{prov} 累计 {self.data[f'{prov}_cny_total']:.2f} 元已达上限 {lim:.2f} 元")

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
        if provider_of(model) == "anthropic":
            inp, w5, w1, rd, out = anthropic_tokens(usage)
            e["prompt_tokens"] += inp + w5 + w1 + rd
            e["cached_prompt_tokens"] += rd
            e["cache_write_tokens"] = e.get("cache_write_tokens", 0) + w5 + w1
            e["completion_tokens"] += out
            e["cost_usd"] += cost
            return cost, cny
        e["prompt_tokens"] += usage.get("prompt_tokens", 0)
        e["cached_prompt_tokens"] += ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                      or usage.get("prompt_cache_hit_tokens", 0))
        e["completion_tokens"] += usage.get("completion_tokens", 0)
        e["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        e["cost_usd"] += cost
        if cny is not None:
            e["cost_cny"] = e.get("cost_cny", 0.0) + cny
        return cost, cny

    def add(self, model, role, usage, cached, latency=None):
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
                    ck = f"{provider_of(model)}_cny_total"
                    self.data[ck] = self.data.get(ck, 0.0) + cny
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
                calls = os.path.join(os.path.dirname(self.path), "spend_calls.jsonl")
                with open(calls, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                        "model": model, "role": role,
                                    "peak": is_peak() if provider_of(model) == "deepseek" else None,
                                        "usage": {x: usage.get(x) for x in ("prompt_tokens", "completion_tokens",
                                                                            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                                                                            "cost_in_usd_ticks", "input_tokens",
                                                                            "output_tokens", "cache_creation_input_tokens",
                                                                            "cache_read_input_tokens") if x in usage},
                                        "cost_usd": usd, "cost_cny": cny,
                                    **({"latency_s": round(latency, 2)} if latency is not None else {})},
                                   ensure_ascii=False) + "\n")


class LLMClient:
    def __init__(self, cache_dir, ledger, offline=False, role_prefix="", blocked_providers=(), max_inflight=None):
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
        # 同时发出的请求数上限：超过服务商能并发处理的数量时，请求只会在对方排队、超时（硅基流动的免费模型）
        self.inflight = threading.BoundedSemaphore(max_inflight) if max_inflight else None
        self.retries = {}  # 重试原因 -> 次数（如 HTTP 429），只用于观察限流
        os.makedirs(cache_dir, exist_ok=True)
        for fn in sorted(os.listdir(cache_dir)):
            if fn.endswith(".jsonl") or fn.endswith(".jsonl.gz"):
                opener = gzip.open if fn.endswith(".gz") else open
                with opener(os.path.join(cache_dir, fn), "rt", encoding="utf-8") as f:
                    for line in f:
                        # 中途被打断的进程可能留下没写完的最后一行（没有换行）：跳过，那次请求重跑时会重新发
                        if line.strip() and line.endswith("\n"):
                            d = json.loads(line)
                            alts = self.alts.setdefault(d["key"], [])
                            if d["response"] not in alts:   # 同一个响应在 .jsonl 和 .jsonl.gz 里各存了一份时只算一个
                                alts.append(d["response"])
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

    def chat(self, body, role, cache_tag=None):
        # 新请求的键带上用途（agent / control / …），免得两种运行的第一轮请求完全相同、并发时互相覆盖；
        # 旧的键（不带用途）照样能读到，已有缓存全部有效。
        # cache_tag（如"种子|用户|会话"）只进缓存键、不发给服务商：不设温度 0 的模型，不同用户碰巧相同的请求也各自独立采样
        k_old = self.key(body)
        k = self.key(dict(body, __cache_role=role, **({"__cache_tag": cache_tag} if cache_tag else {})))
        if cache_tag:
            k_old = k
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
        if self.inflight:
            self.inflight.acquire()
        try:
            t0 = time.time()
            resp = self._post(body)
            latency = time.time() - t0
        finally:
            if self.inflight:
                self.inflight.release()
        resp.pop("id", None)
        resp.pop("created", None)
        resp.pop("system_fingerprint", None)
        with self.lock:
            self.cache[k] = resp
            self.alts[k] = [resp]
            with open(os.path.join(self.cache_dir, f"{model.replace('/', '_')}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "response": resp}, ensure_ascii=False) + "\n")
        self.ledger.add(model, role, resp.get("usage", {}), cached=False, latency=latency)
        return resp

    def _post(self, body):
        if provider_of(body["model"]) == "anthropic":
            return self._post_anthropic(body)
        prov = PROVIDERS[provider_of(body["model"])]
        key = os.environ.get(prov["key"])
        if not key:
            raise RuntimeError(f"环境变量 {prov['key']} 未设置")
        data = json.dumps(body, ensure_ascii=False).encode()
        last = None
        for attempt in range(20):
            req = urllib.request.Request(prov["url"], data=data, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read()[:500].decode(errors="replace")
                last = f"HTTP {e.code}: {msg}"
                with self.lock:
                    self.retries[f"HTTP {e.code}"] = self.retries.get(f"HTTP {e.code}", 0) + 1
                # 400 一般是请求本身有错，直接停；但硅基流动偶尔对同一个合法请求返回 400（重发即成功），所以先重试两次
                if e.code not in (408, 409, 429, 500, 502, 503, 504) and not (e.code == 400 and attempt < 2):
                    raise ApiError(last) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = repr(e)
                with self.lock:
                    self.retries[type(e).__name__] = self.retries.get(type(e).__name__, 0) + 1
            time.sleep(min(60, 2 ** attempt))
        raise ApiError(f"请求失败：{last}")


    _anthropic = None

    def _post_anthropic(self, body):
        """用官方 SDK 发一次 Messages 请求（流式，取最终消息）。限流、过载、5xx、断线时退避重试；其余错误整次停下。"""
        import anthropic
        with self.lock:
            if self._anthropic is None:
                key = os.environ.get(ANTHROPIC_KEY_ENV)
                if not key:
                    raise RuntimeError(f"环境变量 {ANTHROPIC_KEY_ENV} 未设置")
                self._anthropic = anthropic.Anthropic(api_key=key, base_url=ANTHROPIC_BASE_URL,
                                                      max_retries=0, timeout=900)
        last = None
        for attempt in range(20):
            wait = min(60, 2 ** attempt)
            try:
                with self._anthropic.messages.stream(**body) as s:
                    return s.get_final_message().model_dump(mode="json", exclude_none=True)
            except anthropic.APIStatusError as e:
                last = f"HTTP {e.status_code}: {str(e.message)[:300]}"
                with self.lock:
                    self.retries[f"HTTP {e.status_code}"] = self.retries.get(f"HTTP {e.status_code}", 0) + 1
                if e.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                    raise ApiError(last) from None
                ra = e.response.headers.get("retry-after") if e.response is not None else None
                if ra and ra.replace(".", "", 1).isdigit():
                    wait = max(wait, min(120, float(ra)))
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                last = type(e).__name__
                with self.lock:
                    self.retries[last] = self.retries.get(last, 0) + 1
            time.sleep(wait)
        raise ApiError(f"请求失败：{last}")


XAIClient = LLMClient  # 第一版的名字，保留兼容
