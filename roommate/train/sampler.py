"""采样端：vLLM 的 OpenAI 兼容服务（/v1/completions，按 token id 输入、逐 token 返回 logprob）。

- 每个请求显式给出全部采样参数（温度、top_p=1、top_k 关、min_p=0、各种惩罚为 0），
  服务端用 --generation-config vllm 启动，不套用模型自带的默认采样参数（Qwen 的 generation_config 里是温度 0.6 等）。
- 返回采样 token 的 logprob（vLLM 默认 raw_logprobs：温度和截断之前的原始分布；训练采样用温度 1、不截断，两者相同）。
- 权重同步：训练端把合并了 LoRA 的权重写进 /dev/shm 的 safetensors，经 /collective_rpc 调
  roommate.train.vllm_ext 里的 update_weights_from_path 载入，然后清前缀缓存。
"""

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request


def ids_sha(ids):
    """token 序列的指纹（自检第 6 项用）。"""
    return hashlib.sha1(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:16]


class SamplerError(RuntimeError):
    pass


class VLLMClient:
    def __init__(self, base_url, model, timeout=1800):
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "prompt_tokens": 0, "gen_tokens": 0, "seconds": 0.0}

    def _post(self, path, body, timeout=None):
        data = json.dumps(body).encode()
        last = None
        for attempt in range(6):
            req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    raw = r.read()
                    return json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}"
                if e.code < 500:
                    raise SamplerError(last) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = repr(e)
            time.sleep(2 * (attempt + 1))
        raise SamplerError(last)

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw.strip() else {}

    def healthy(self):
        try:
            self._get("/health")
            return True
        except Exception:
            return False

    def complete(self, prompt_ids, max_tokens, temperature, top_p=1.0, seed=None, stop_ids=None, top_logprobs=1,
                 ignore_eos=False):
        """返回里另有 sent / recv：发给采样端的 token 序列、采样端返回的 token 序列的指纹（在 HTTP 这一层算），
        自检第 6 项拿它们逐 token 核对训练样本和采样时模型看到的是否一致。"""
        body = {"model": self.model, "prompt": list(prompt_ids), "max_tokens": int(max_tokens),
                "temperature": float(temperature), "top_p": float(top_p), "top_k": -1, "min_p": 0.0,
                "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "logprobs": int(top_logprobs), "return_tokens_as_token_ids": True, "skip_special_tokens": False,
                "spaces_between_special_tokens": False, "include_stop_str_in_output": True,
                "stop_token_ids": list(stop_ids or []), "ignore_eos": bool(ignore_eos)}
        if seed is not None:
            body["seed"] = int(seed)
        t0 = time.time()
        resp = self._post("/v1/completions", body)
        ch = resp["choices"][0]
        lp = ch.get("logprobs") or {}
        ids = [int(str(t).split(":")[-1]) for t in lp.get("tokens") or []]
        lps = list(lp.get("token_logprobs") or [])
        tops = lp.get("top_logprobs") or []
        appended = False
        stop_reason = ch.get("stop_reason")
        if ch.get("finish_reason") == "stop" and isinstance(stop_reason, int) and (not ids or ids[-1] != stop_reason):
            ids.append(stop_reason)       # 停止 token 没在输出里时补上（采样端 logprob 记为 None）
            lps.append(None)
            appended = True
        with self.lock:
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += len(prompt_ids)
            self.stats["gen_tokens"] += len(ids)
            self.stats["seconds"] += time.time() - t0
        return {"ids": ids, "lp": lps, "top": tops, "finish_reason": ch.get("finish_reason"),
                "stop_reason": stop_reason, "appended_stop": appended,
                "sent": {"n": len(body["prompt"]), "sha": ids_sha(body["prompt"])},
                "recv": {"n": len(ids) - int(appended), "sha": ids_sha(ids[:len(ids) - int(appended)])}}

    def prompt_logprobs(self, ids):
        """整段序列每个位置（从第 2 个 token 起）在采样端模型下的 logprob。"""
        body = {"model": self.model, "prompt": list(ids), "max_tokens": 1, "temperature": 0.0,
                "prompt_logprobs": 0, "return_tokens_as_token_ids": True, "skip_special_tokens": False}
        resp = self._post("/v1/completions", body)
        pl = resp["choices"][0].get("prompt_logprobs") or []
        out = []
        for pos, d in enumerate(pl):
            if not d:
                out.append(None)
                continue
            want = ids[pos]
            val = None
            for k, v in d.items():
                kid = int(str(k).split(":")[-1]) if str(k).split(":")[-1].lstrip("-").isdigit() else None
                if kid == want:
                    val = v["logprob"] if isinstance(v, dict) else v
            out.append(val)
        return out

    def update_weights(self, path):
        self._post("/collective_rpc", {"method": "update_weights_from_path", "args": [path]}, timeout=1800)
        self.reset_prefix_cache()

    def reset_prefix_cache(self):
        self._post("/reset_prefix_cache", {}, timeout=300)

    def sleep(self, level=1):
        self._post(f"/sleep?level={level}", {}, timeout=600)

    def wake_up(self):
        self._post("/wake_up", {}, timeout=600)
