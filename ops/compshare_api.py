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


def call(action, params=None, timeout=30):
    """Return (response_json, elapsed_seconds). Never returns or logs the request."""
    if action not in READ_ONLY:
        raise ValueError(f"{action} is not in the read-only allowlist")
    return _send(action, params, timeout)


NO_RETRY = {"CreateCompShareInstance", "CreateCompShareCustomImage"}   # 重发可能多建一台/一个镜像


def call_write(action, params=None, timeout=60):
    if action not in WRITE:
        raise ValueError(f"{action} is not in the write allowlist")
    return _send(action, params, timeout, tries=1 if action in NO_RETRY else None)


TRIES = 6   # 默认重试次数；中转可以调低（连不上时要在几分钟内收尾）


def _send(action, params, timeout, tries=None):
    tries = tries or TRIES
    payload = _flatten({**(params or {}), "Action": action})
    payload["PublicKey"] = os.environ["COMPSHARE_PUBLIC_KEY"]
    raw = "".join(k + payload[k] for k in sorted(payload)) + os.environ["COMPSHARE_PRIVATE_KEY"]
    payload["Signature"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    req = urllib.request.Request(
        BASE_URL,
        data=urllib.parse.urlencode(payload).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "U-Timestamp-Ms": str(int(time.time() * 1000))},
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
