"""优云智算（Compshare）只读接口的最小客户端。

只放行 Describe*/Get*/Check*/List* 开头的接口，任何会开机、建盘、扣费的接口都调不出去。
密钥从环境变量 COMPSHARE_PUBLIC_KEY / COMPSHARE_PRIVATE_KEY 读取，不打印、不落盘。

用法：python tools/compshare_ro.py <Action> [Key=Value ...]
"""
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

BASE_URL = "https://api.compshare.cn/"
READ_ONLY_PREFIXES = ("Describe", "Get", "Check", "List")


def _sign(params: dict, private_key: str) -> str:
    s = "".join(f"{k}{params[k]}" for k in sorted(params))
    return hashlib.sha1((s + private_key).encode("utf-8")).hexdigest()


def call(action: str, **params) -> dict:
    if not action.startswith(READ_ONLY_PREFIXES):
        raise ValueError(f"{action} 不是只读接口，拒绝调用")
    pub = os.environ.get("COMPSHARE_PUBLIC_KEY")
    pri = os.environ.get("COMPSHARE_PRIVATE_KEY")
    if not pub or not pri:
        raise RuntimeError("缺少 COMPSHARE_PUBLIC_KEY 或 COMPSHARE_PRIVATE_KEY")
    p = {"Action": action, "PublicKey": pub}
    p.update({k: str(v) for k, v in params.items()})
    p["Signature"] = _sign(p, pri)
    req = urllib.request.Request(
        BASE_URL,
        data=urllib.parse.urlencode(p).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[2:])
    print(json.dumps(call(sys.argv[1], **kv), ensure_ascii=False, indent=1))
