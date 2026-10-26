"""Zero-cost probe of the session -> GitHub Actions -> Compshare chain.

Calls only read-only Compshare actions (see compshare_api.READ_ONLY), checks that
the launch parameters in launch_config.json are legal, measures SSH handshakes to
public hosts in mainland China, and saves Compshare billing docs as text.
Writes one JSON file; keys never appear in it.
"""
import html.parser
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import compshare_api  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SSH_HOSTS = ["gitee.com", "codeup.aliyun.com", "git.code.tencent.com"]
DOC_KEYWORDS = ("charge", "bill", "arrear", "balance", "recharge", "postpay", "createcompshareinstance",
                "startcompshareinstance", "quota", "level")


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k in ("ProjectId", "request_uuid") else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def compshare(out):
    cfg = json.load(open(os.path.join(HERE, "launch_config.json")))
    loc = {"Region": cfg["Region"], "Zone": cfg["Zone"]}
    res = out["compshare"] = {"launch_config": cfg, "calls": {}}

    def run(name, action, params=None):
        try:
            body, secs = compshare_api.call(action, params)
            res["calls"][name] = {"action": action, "seconds": secs, "response": _redact(body)}
            return body
        except Exception as e:  # keep going; record the failure
            res["calls"][name] = {"action": action, "error": f"{type(e).__name__}: {e}"}
            return {}

    bal = run("balance", "GetBalance")
    run("zones", "DescribeCompShareSupportZone")
    types = run("instance_types", "DescribeAvailableCompShareInstanceTypes",
                {**loc, "InstanceType": "uhost", "MachineTypes": [cfg["GpuType"]]})
    images = run("images", "DescribeCompShareImages", {**loc, "Limit": 100, "Offset": 0})
    while images.get("RetCode") == 0 and len(images.get("ImageSet") or []) < (images.get("TotalCount") or 0):
        more, _ = compshare_api.call("DescribeCompShareImages",
                                     {**loc, "Limit": 100, "Offset": len(images["ImageSet"])})
        if not more.get("ImageSet"):
            break
        images["ImageSet"] += more["ImageSet"]
    # keep the image list short in the output
    if "response" in res["calls"]["images"]:
        res["calls"]["images"]["response"] = {
            "RetCode": images.get("RetCode"), "TotalCount": images.get("TotalCount"),
            "selected": [i for i in images.get("ImageSet") or []
                         if i.get("CompShareImageId") == cfg["CompShareImageId"]]}
    cap = run("capacity", "CheckCompShareResourceCapacity", {
        **loc, "GpuType": cfg["GpuType"], "MachineType": "G", "MinimalCpuPlatform": cfg["MinimalCpuPlatform"],
        "CompShareImageId": cfg["CompShareImageId"], "ChargeType": cfg["ChargeType"], "Disks": cfg["Disks"]})
    price_params = {**loc, "GpuType": cfg["GpuType"], "Gpu": cfg["Gpu"], "Cpu": cfg["Cpu"],
                    "Memory": cfg["MemoryGiB"] * 1024, "ChargeType": cfg["ChargeType"], "Disks": cfg["Disks"],
                    "CompShareImageId": cfg["CompShareImageId"], "Quantity": cfg["Quantity"]}
    price = run("price", "GetCompShareInstancePrice", price_params)
    # control: an illegal CPU count, to learn whether the price API validates specs
    run("price_illegal_cpu_control", "GetCompShareInstancePrice", {**price_params, "Cpu": cfg["Cpu"] + 1})
    run("price_spot_same_spec", "GetCompShareInstancePrice", {**price_params, "ChargeType": "Spot"})

    sizes = []
    for t in types.get("AvailableInstanceTypes") or []:
        for s in t.get("MachineSizes") or []:
            for col in s.get("Collection") or []:
                for mem in col.get("Memory") or []:
                    sizes.append((s.get("Gpu"), col.get("Cpu"), mem))
    spec = (cfg["Gpu"], cfg["Cpu"], cfg["MemoryGiB"])
    cap_specs = {(s.get("Gpu"), s.get("Cpu"), s.get("Mem")): s.get("ResourceEnough") for s in cap.get("Specs") or []}
    img = (res["calls"]["images"].get("response") or {}).get("selected") or []
    details = (price.get("PriceDetails") or [{}])[0]
    hourly = sum(float(details.get(k) or 0) for k in ("Instance", "Disks", "SystemDisks", "CompShareImage"))
    res["checks"] = {
        "spec_in_machine_sizes": spec in sizes,
        "spec_in_capacity_list": spec in cap_specs,
        "spec_in_stock_now": cap_specs.get(spec),
        "image_available": bool(img) and img[0].get("Status") == "Available",
        "price_retcode": price.get("RetCode"),
        "hourly_cny": round(hourly, 4),
        "balance": (bal.get("AccountInfo") or {}),
    }


def ssh(out):
    rows = out["ssh"] = []
    for host in SSH_HOSTS:
        row = {"host": host}
        try:
            row["ip"] = socket.gethostbyname(host)
            rtts, banner = [], None
            for _ in range(5):
                t0 = time.monotonic()
                with socket.create_connection((host, 22), timeout=10) as s:
                    rtts.append(round((time.monotonic() - t0) * 1000, 1))
                    s.settimeout(10)
                    banner = s.recv(256).decode("utf-8", "replace").strip()
            row.update(tcp_connect_ms=rtts, tcp_connect_median_ms=statistics.median(rtts), banner=banner)
            t0 = time.monotonic()
            ks = subprocess.run(["ssh-keyscan", "-T", "15", "-t", "ed25519,rsa", host],
                                capture_output=True, text=True, timeout=40)
            row["keyscan_seconds"] = round(time.monotonic() - t0, 3)
            row["host_keys"] = [" ".join(l.split()[1:2]) for l in ks.stdout.splitlines() if l and not l.startswith("#")]
            row["handshake_ok"] = bool(row["host_keys"])
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            row["handshake_ok"] = False
        rows.append(row)
    try:
        t0 = time.monotonic()
        with socket.create_connection(("api.compshare.cn", 443), timeout=10):
            out["api_compshare_tcp443_ms"] = round((time.monotonic() - t0) * 1000, 1)
    except Exception as e:
        out["api_compshare_tcp443_ms"] = f"error: {e}"


class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 good-roommate-probe"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def docs(out, out_dir):
    d = out["docs"] = {"pages": {}}
    try:
        sm = _get("https://www.compshare.cn/sitemap.xml")
        urls = re.findall(r"<loc>([^<]+)</loc>", sm)
        d["sitemap_url_count"] = len(urls)
    except Exception as e:
        d["sitemap_error"] = str(e)
        urls = []
    picked = [u for u in urls if "/docs/" in u and any(k in u.lower() for k in DOC_KEYWORDS)]
    must = ["https://www.compshare.cn/docs/operation/charge/billdescribe",
            "https://www.compshare.cn/docs/uaccount/bill/charge",
            "https://www.compshare.cn/docs/gpus/instance/createcompshareinstance",
            "https://www.compshare.cn/docs/gpus/instance/startcompshareinstance"]
    picked = list(dict.fromkeys(must + picked))[:40]
    d["picked"] = picked
    os.makedirs(os.path.join(out_dir, "docs"), exist_ok=True)
    for u in picked:
        try:
            p = _Text()
            p.feed(_get(u))
            text = "\n".join(p.parts)
            name = re.sub(r"[^A-Za-z0-9]+", "_", u.split("/docs/", 1)[-1]).strip("_") + ".txt"
            with open(os.path.join(out_dir, "docs", name), "w") as f:
                f.write(u + "\n\n" + text)
            d["pages"][u] = {"file": f"docs/{name}", "chars": len(text)}
        except Exception as e:
            d["pages"][u] = {"error": str(e)}


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    out = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "dispatched_utc": os.environ.get("DISPATCHED_AT") or None,
           "run_id": os.environ.get("GITHUB_RUN_ID"),
           "runner": {"os": os.environ.get("RUNNER_OS"), "arch": os.environ.get("RUNNER_ARCH")}}
    for step in (compshare, ssh):
        try:
            step(out)
        except Exception as e:
            out[f"{step.__name__}_error"] = f"{type(e).__name__}: {e}"
    try:
        docs(out, out_dir)
    except Exception as e:
        out["docs_error"] = f"{type(e).__name__}: {e}"
    out["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = json.dumps(out, ensure_ascii=False, indent=2)
    for secret in (os.environ.get("COMPSHARE_PUBLIC_KEY"), os.environ.get("COMPSHARE_PRIVATE_KEY")):
        if secret and secret in text:
            sys.exit("refusing to write output: it contains a key")
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        f.write(text + "\n")
    c = out.get("compshare", {}).get("checks", {})
    print(json.dumps({"checks": c, "ssh": [{k: r.get(k) for k in ("host", "handshake_ok", "tcp_connect_median_ms")}
                                           for r in out.get("ssh", [])]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
