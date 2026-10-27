"""Destroy every instance named gr-probe-* (safety net run after the grab job,
whatever happened to it). Prints what it found and did; no account data."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import compshare_api as api  # noqa: E402

REGIONS = ("cn-wlcb", "cn-sh2", "cn-bj2")


def ours():
    found = []
    for region in REGIONS:
        r, _ = api.call("DescribeCompShareInstance", {"Region": region, "Limit": 100})
        found += [(region, h) for h in r.get("UHostSet") or [] if str(h.get("Name", "")).startswith("gr-probe-")]
    uniq = {}
    for region, h in found:      # 接口不按 Region 过滤：去重，地域取自可用区
        uniq.setdefault(h["UHostId"], (str(h.get("Zone") or region).rsplit("-", 1)[0], h))
    return list(uniq.values())


report = {"found": [], "actions": []}
deadline = time.time() + 900
while time.time() < deadline:
    hosts = ours()
    if not report["found"]:
        report["found"] = [{"id": h.get("UHostId"), "name": h.get("Name"), "state": h.get("State")} for _, h in hosts]
    if not hosts:
        break
    for region, h in hosts:
        loc = {"Region": region, "Zone": h.get("Zone"), "UHostId": h.get("UHostId")}
        if h.get("State") in ("Stopping", "Starting", "Rebooting"):
            continue
        r, _ = api.call_write("TerminateCompShareInstance", {**loc, "ReleaseUDisk": True})
        report["actions"].append({"id": h.get("UHostId"), "action": "Terminate", "RetCode": r.get("RetCode")})
        if r.get("RetCode") != 0 and h.get("State") == "Running":
            r, _ = api.call_write("StopCompShareInstance", loc)
            report["actions"].append({"id": h.get("UHostId"), "action": "Stop", "RetCode": r.get("RetCode")})
    time.sleep(10)
report["remaining"] = [h.get("UHostId") for _, h in ours()]
print(json.dumps(report))
sys.exit(1 if report["remaining"] else 0)
