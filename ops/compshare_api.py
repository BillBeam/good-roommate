"""Minimal read-only client for the Compshare (UCloud) OpenAPI.

Keys come only from COMPSHARE_PUBLIC_KEY / COMPSHARE_PRIVATE_KEY. Only actions in
READ_ONLY can be called, so this module cannot create, start or bill anything.
Signing follows ucloud-sdk-python3 (sorted flattened params + private key, sha1).
"""
import hashlib
import json
import os
import time
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, round(time.monotonic() - t0, 3)
