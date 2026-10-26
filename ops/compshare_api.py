"""Minimal read-only client for the Compshare (UCloud) OpenAPI.

Keys come only from COMPSHARE_PUBLIC_KEY / COMPSHARE_PRIVATE_KEY. call() only runs
actions in READ_ONLY. call_write() runs the few lifecycle actions in WRITE and is
used only by gpu_grab.py (one test instance, always destroyed) and train_relay.py (one
training instance, platform shutdown at most 2 h after the session's last request).
Signing follows ucloud-sdk-python3 (sorted flattened params + private key, sha1).
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://api.compshare.cn"
READ_ONLY = frozenset({
    "GetBalance",
    "GetProjectList",
    "DescribeCompShareSupportZone",
    "DescribeAvailableCompShareInstanceTypes",
    "DescribeCompShareMachineTypeFamilies",
    "DescribeCompShareImages",
    "DescribeCompShareInstance",
    "CheckCompShareResourceCapacity",
    "GetCompShareInstancePrice",
    "GetCompShareImageCreateProgress",
    "DescribeCompShareCustomImages",
    "GetSoftwareURL",
    "DescribeCompShareSoftwarePort",
    "GetCompShareInstanceMonitor",
})
WRITE = frozenset({
    "CreateCompShareInstance",
    "UpdateCompShareStopScheduler",
    "StopCompShareInstance",
    "TerminateCompShareInstance",
    "StartCompShareInstance",
    "RebootCompShareInstance",
    "CreateCompShareCustomImage",
})


def _encode_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v % 1 == 0:
        return str(int(v))
    return str(v)


def _flatten(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for ek, ev in _flatten(v).items():
                out[f"{k}.{ek}"] = ev
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    for ek, ev in _flatten(item).items():
                        out[f"{k}.{i}.{ek}"] = ev
                else:
                    out[f"{k}.{i}"] = _encode_value(item)
        else:
            out[k] = _encode_value(v)
    return out


def call(action, params=None, timeout=30, json_body=False):
    """Return (response_json, elapsed_seconds). Never returns or logs the request.
    json_body: send a JSON body (arrays stay arrays) instead of form fields; some actions only accept that."""
    if action not in READ_ONLY:
        raise ValueError(f"{action} is not in the read-only allowlist")
    return _send(action, params, timeout, json_body=json_body)


NO_RETRY = {"CreateCompShareInstance", "CreateCompShareCustomImage"}   # 重发可能多建一台/一个镜像


def call_write(action, params=None, timeout=60):
    if action not in WRITE:
        raise ValueError(f"{action} is not in the write allowlist")
    return _send(action, params, timeout, tries=1 if action in NO_RETRY else None)


TRIES = 6   # 默认重试次数；中转可以调低（连不上时要在几分钟内收尾）


def _send(action, params, timeout, tries=None, json_body=False):
    tries = tries or TRIES
    payload = _flatten({**(params or {}), "Action": action})
    payload["PublicKey"] = os.environ["COMPSHARE_PUBLIC_KEY"]
    sig_src = payload
    if json_body in ("json_compact", "join"):    # JSON 请求体时，签名里的数组怎么写（几种约定，按需试）
        sig_src = {k: v for k, v in payload.items() if not any(k.startswith(a + ".") for a, x in (params or {}).items()
                                                               if isinstance(x, list))}
        for a, x in (params or {}).items():
            if isinstance(x, list):
                sig_src[a] = json.dumps(x, separators=(",", ":")) if json_body == "json_compact" else ",".join(map(str, x))
    raw = "".join(k + sig_src[k] for k in sorted(sig_src)) + os.environ["COMPSHARE_PRIVATE_KEY"]
    payload["Signature"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    if json_body:   # 签名照旧按展开后的参数算；请求体用 JSON，数组保持数组
        data = json.dumps({**(params or {}), "Action": action, "PublicKey": payload["PublicKey"],
                           "Signature": payload["Signature"]}).encode()
        ctype = "application/json"
    else:
        data, ctype = urllib.parse.urlencode(payload).encode(), "application/x-www-form-urlencoded"
    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={"Content-Type": ctype, "U-Timestamp-Ms": str(int(time.time() * 1000))},
        method="POST",
    )
    t0 = time.monotonic()
    last = None
    for attempt in range(tries):   # 运行器在美国，跨境偶尔握手超时：退避重试（读写都是幂等或带名字可查）
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body, round(time.monotonic() - t0, 3)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            if attempt + 1 < tries:
                time.sleep(min(60, 5 * 2 ** attempt))
    raise last
