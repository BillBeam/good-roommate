"""Grab one GPU machine, run gpu_remote_check.py on it over SSH, destroy it.

Runs on a GitHub Actions runner (push to chain-request with mode "grab").
Limits enforced here:
  - stock is swept at most once every SWEEP_EVERY seconds, for at most window_minutes;
  - only cards with >= 24 GB VRAM and native BF16, pay-as-you-go, <= MAX_HOURLY CNY/h;
  - at most one instance (refuses if any instance already exists, one create call
    per run apart from retries after a sold-out error, MaxCount 1);
  - the instance is destroyed by DESTROY_AT minutes after creation; the platform's
    scheduled shutdown is set for STOP_AT minutes as a backstop, and the workflow's
    cleanup job destroys anything named gr-probe-* even if this job dies.
Public output (OUT_DIR) never contains balance, charges, keys or the login password.

usage: python3 gpu_grab.py OUT_DIR
env: COMPSHARE_*_KEY, GRAB_WINDOW_MINUTES, GRAB_REQUEST_SHA, GITHUB_RUN_ID, PR_NUMBER, GH_TOKEN
"""
import base64
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import compshare_api as api  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = sys.argv[1]
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")
NAME = f"gr-probe-{RUN_ID}"
WINDOW = float(os.environ.get("GRAB_WINDOW_MINUTES") or 300) * 60
SWEEP_EVERY = 30
MAX_HOURLY = 11.0
BF16_GPUS = ("3090", "4090", "4090D", "4090_48G", "5090", "A100", "A800", "H20", "H100", "H800",
             "L20", "L40", "L40S", "A40", "A6000")
MIN_VRAM = 24
IMAGE_PREFERENCE = ("vLLM v0.25", "cuda132_torch", "cuda130_torch", "cuda128_torch")
DESTROY_AT = 40 * 60     # start teardown 40 min after create (must be gone by 45)
STOP_AT = 44 * 60        # platform-side scheduled shutdown backstop
REGIONS = ("cn-wlcb", "cn-sh2", "cn-bj2")
os.makedirs(OUT, exist_ok=True)

T_START = time.time()
result = {"run_id": RUN_ID, "instance_name": NAME, "started_utc": None, "events": []}
stock_log = open(os.path.join(OUT, "stock_log.jsonl"), "a")


def now_iso(t=None):
    return datetime.fromtimestamp(t or time.time(), timezone.utc).isoformat(timespec="seconds")


def event(msg, **kw):
    e = {"t": now_iso(), "msg": msg, **kw}
    result["events"].append(e)
    print(json.dumps(e, ensure_ascii=False), flush=True)
    save()


def save():
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


def redact(text):
    return re.sub(r"\d+(\.\d+)?", "#", str(text))  # platform messages may carry amounts


def comment(body):
    pr = os.environ.get("PR_NUMBER")
    if not pr:
        return
    subprocess.run(["gh", "pr", "comment", pr, "--body", body], capture_output=True, text=True)


def stop_requested():
    """A later push to chain-request with mode "stop" ends the grab window early."""
    try:
        head = subprocess.run(["git", "ls-remote", "origin", "refs/heads/chain-request"],
                              capture_output=True, text=True, timeout=20).stdout.split()[0]
    except Exception:
        return False
    if head == os.environ.get("GRAB_REQUEST_SHA"):
        return False
    try:
        subprocess.run(["git", "fetch", "-q", "--depth=1", "origin", head], timeout=60, check=True)
        req = json.loads(subprocess.run(["git", "show", f"{head}:ops/probe_request.json"],
                                        capture_output=True, text=True).stdout)
        return req.get("mode") == "stop"
    except Exception:
        return False


# ---------------- account checks (no numbers leave this function) ----------------

def balance_ok(hourly):
    info = (api.call("GetBalance")[0].get("AccountInfo") or {})
    avail = float(info.get("AmountAvailable") or 0)
    return avail >= hourly  # the team docs require one hour; treat it the same


def existing_instances():
    found = []
    for region in REGIONS:
        r, _ = api.call("DescribeCompShareInstance", {"Region": region, "Limit": 100})
        for h in r.get("UHostSet") or []:
            found.append({"region": region, "id": h.get("UHostId"), "name": h.get("Name"),
                          "state": h.get("State"), "zone": h.get("Zone")})
    return found


# ---------------- stock sweep ----------------

_types, _images, _prices = {}, {}, {}


def zone_catalog():
    zones, _ = api.call("DescribeCompShareSupportZone")
    out = []
    for z in zones.get("ZoneInfo") or []:
        key = z["Zone"]
        if key not in _types or time.time() - _types[key][0] > 1800:
            t, _ = api.call("DescribeAvailableCompShareInstanceTypes",
                            {"Region": z["Region"], "Zone": key, "InstanceType": "uhost"})
            _types[key] = (time.time(), [x for x in t.get("AvailableInstanceTypes") or []
                                         if x.get("Name") in BF16_GPUS
                                         and ((x.get("GraphicsMemory") or {}).get("Value") or 0) >= MIN_VRAM])
            imgs, page = [], 0
            while True:
                r, _ = api.call("DescribeCompShareImages",
                                {"Region": z["Region"], "Zone": key, "Limit": 100, "Offset": page})
                imgs += r.get("ImageSet") or []
                page += 100
                if page >= (r.get("TotalCount") or 0) or not r.get("ImageSet"):
                    break
            _images[key] = imgs
        out.append((z, _types[key][1]))
    return out


def pick_image(zone, gpu):
    imgs = [i for i in _images.get(zone, []) if i.get("Status") == "Available"]
    for pref in IMAGE_PREFERENCE:
        matches = [i for i in imgs if (i.get("Name") or "").startswith(pref)]
        if gpu == "5090":  # use a 5090 build of the image when the platform ships one
            special = [i for i in matches if "5090" in i["Name"]]
            if special:
                return special[0]
        plain = [i for i in matches if "5090" not in i["Name"]]
        if plain:
            return plain[0]
    return None


def price_of(loc, gpu, cpu, mem, image):
    key = (loc["Zone"], gpu, cpu, mem, image)
    if key not in _prices:
        r, _ = api.call("GetCompShareInstancePrice", {
            **loc, "GpuType": gpu, "Gpu": 1, "Cpu": cpu, "Memory": mem * 1024, "ChargeType": "Postpay",
            "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": 100}], "CompShareImageId": image,
            "Quantity": 1})
        d = (r.get("PriceDetails") or [{}])[0]
        total = sum(float(d.get(k) or 0) for k in ("Instance", "Disks", "SystemDisks", "CompShareImage"))
        _prices[key] = round(total, 4) if r.get("RetCode") == 0 and total > 0 else None
    return _prices[key]


def sweep():
    rec = {"t": now_iso(), "checked": [], "in_stock": []}
    cands = []
    for z, types in zone_catalog():
        loc = {"Region": z["Region"], "Zone": z["Zone"]}
        for t in types:
            gpu = t["Name"]
            img = pick_image(z["Zone"], gpu)
            if img is None:
                rec["checked"].append([z["Zone"], gpu, "no image"])
                continue
            cap, _ = api.call("CheckCompShareResourceCapacity", {
                **loc, "GpuType": gpu, "MachineType": "G", "MinimalCpuPlatform": "Auto",
                "CompShareImageId": img["CompShareImageId"], "ChargeType": "Postpay",
                "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": 100}]})
            if cap.get("RetCode") != 0:
                rec["checked"].append([z["Zone"], gpu, f"RetCode {cap.get('RetCode')}"])
                continue
            ok = [s for s in cap.get("Specs") or [] if s.get("ResourceEnough") and s.get("Gpu") == 1]
            rec["checked"].append([z["Zone"], gpu, len(ok)])
            for s in ok:
                p = price_of(loc, gpu, s["Cpu"], s["Mem"], img["CompShareImageId"])
                row = {"zone": z["Zone"], "zone_name": z.get("Describe"), "region": z["Region"], "gpu": gpu,
                       "vram": (t.get("GraphicsMemory") or {}).get("Value"), "cpu": s["Cpu"], "mem": s["Mem"],
                       "hourly": p, "image": img.get("Name"), "image_id": img["CompShareImageId"]}
                rec["in_stock"].append({k: row[k] for k in ("zone", "gpu", "cpu", "mem", "hourly")})
                if p is not None and p <= MAX_HOURLY:
                    cands.append(row)
    stock_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stock_log.flush()
    cands.sort(key=lambda r: (r["hourly"], -r["vram"]))
    return rec, cands


# ---------------- instance lifecycle ----------------

def describe(region, uid):
    r, _ = api.call("DescribeCompShareInstance", {"Region": region, "UHostIds": [uid]})
    hosts = r.get("UHostSet") or []
    return hosts[0] if hosts else None


def destroy(region, zone, uid):
    """Terminate; if refused, stop then terminate. Returns when the instance is gone."""
    t0 = time.time()
    tries = []
    while time.time() - t0 < 600:
        h = describe(region, uid)
        if h is None:
            event("destroyed", seconds=round(time.time() - t0, 1), tries=tries)
            return True
        state = h.get("State")
        if state not in ("Stopping", "Starting", "Rebooting"):
            r, _ = api.call_write("TerminateCompShareInstance",
                                  {"Region": region, "Zone": zone, "UHostId": uid, "ReleaseUDisk": True})
            tries.append({"action": "Terminate", "state": state, "RetCode": r.get("RetCode"),
                          "Message": redact(r.get("Message") or "")})
            if r.get("RetCode") != 0 and state == "Running":
                r, _ = api.call_write("StopCompShareInstance", {"Region": region, "Zone": zone, "UHostId": uid})
                tries.append({"action": "Stop", "RetCode": r.get("RetCode"), "Message": redact(r.get("Message") or "")})
        time.sleep(10)
    event("destroy_timeout", tries=tries)
    return False


def parse_login(cmd):
    argv = shlex.split(cmd)
    port, dest = 22, None
    i = 1
    while i < len(argv):
        if argv[i] == "-p":
            port = int(argv[i + 1])
            i += 2
            continue
        if "@" in argv[i] and not argv[i].startswith("-"):
            dest = argv[i]
        i += 1
    user, host = dest.split("@", 1)
    return user, host, port


def ssh_base(user, host, port):
    return ["sshpass", "-e", "ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "PreferredAuthentications=password,keyboard-interactive",
            "-o", "PubkeyAuthentication=no", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
            "-o", "LogLevel=ERROR", f"{user}@{host}"]


def ssh_stability(host, port, n=20):
    rtts, fails = [], 0
    for _ in range(n):
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=10) as s:
                rtts.append(round((time.monotonic() - t0) * 1000, 1))
                s.settimeout(10)
                s.recv(64)
        except Exception:
            fails += 1
        time.sleep(0.5)
    ks = []
    for _ in range(3):
        t0 = time.monotonic()
        p = subprocess.run(["ssh-keyscan", "-T", "10", "-p", str(port), host], capture_output=True, text=True)
        ks.append({"ok": bool(p.stdout.strip()), "seconds": round(time.monotonic() - t0, 2)})
    return {"tcp_attempts": n, "tcp_failures": fails, "tcp_connect_ms": rtts,
            "tcp_median_ms": statistics.median(rtts) if rtts else None,
            "tcp_max_ms": max(rtts) if rtts else None, "keyscan": ks}


def run_tests(host_info, t_create, env):
    user, host, port = host_info
    base = ssh_base(user, host, port)
    # login (retry until it works or 10 minutes pass)
    t0 = time.time()
    login = None
    while time.time() - t0 < 600:
        p = subprocess.run(base + ["echo LOGIN_OK; whoami"], capture_output=True, text=True, env=env, timeout=60)
        if "LOGIN_OK" in p.stdout:
            login = time.time()
            break
        time.sleep(5)
    if login is None:
        event("login_failed", stderr=p.stderr[-300:])
        return
    result["create_to_login_seconds"] = round(login - t_create, 1)
    event("login_ok", create_to_login_seconds=result["create_to_login_seconds"], user=p.stdout.split()[-1])

    bundle = os.path.join(OUT, "_bundle")
    os.makedirs(os.path.join(bundle, "probes"), exist_ok=True)
    sp = os.path.join(REPO, "runs/m2/siliconflow_probe")
    for f in os.listdir(sp):
        if f.startswith("p") and f.endswith(".json"):
            subprocess.run(["cp", os.path.join(sp, f), os.path.join(bundle, "probes", f)])
    for src in ("tools/token_check.py", "template_token_check.json", "hf_ref/chat_template.jinja.sha256"):
        subprocess.run(["cp", os.path.join(sp, src), bundle])
    subprocess.run(["cp", os.path.join(HERE, "gpu_remote_check.py"), bundle])
    tar = os.path.join(OUT, "_bundle.tgz")
    subprocess.run(["tar", "czf", tar, "-C", bundle, "."], check=True)
    scp = ["sshpass", "-e", "scp", "-P", str(port), "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    p = subprocess.run(scp + [tar, f"{user}@{host}:/tmp/gr_bundle.tgz"], capture_output=True, text=True, env=env)
    event("bundle_uploaded", rc=p.returncode)

    deadline = t_create + DESTROY_AT - 120
    remote = (f"mkdir -p /tmp/grb && tar xzf /tmp/gr_bundle.tgz -C /tmp/grb && "
              f"python3 /tmp/grb/gpu_remote_check.py /tmp/grb /tmp/gro {deadline:.0f}")
    try:
        p = subprocess.run(base + [remote], capture_output=True, text=True, env=env,
                           timeout=max(60, deadline - time.time() + 60))
        event("remote_finished", rc=p.returncode, tail=p.stdout[-200:])
    except subprocess.TimeoutExpired:
        event("remote_timeout")
    p = subprocess.run(scp + [f"{user}@{host}:/tmp/gro/remote_result.json", os.path.join(OUT, "remote_result.json")],
                       capture_output=True, text=True, env=env)
    event("remote_result_fetched", rc=p.returncode)


def main():
    result["started_utc"] = now_iso(T_START)
    result["window_minutes"] = WINDOW / 60
    result["limits"] = {"max_hourly": MAX_HOURLY, "min_vram": MIN_VRAM, "bf16_gpus": BF16_GPUS,
                        "sweep_every_s": SWEEP_EVERY, "destroy_at_min": DESTROY_AT / 60, "stop_at_min": STOP_AT / 60}
    existing = existing_instances()
    if existing:
        event("abort_existing_instances", count=len(existing))
        return
    event("grab_window_open")

    chosen, sold_out_retries = None, 0
    sweeps = 0
    while time.time() - T_START < WINDOW:
        t_sweep = time.time()
        try:
            rec, cands = sweep()
        except Exception as e:  # transient API/network error: skip this sweep
            event("sweep_error", error=f"{type(e).__name__}: {e}")
            time.sleep(SWEEP_EVERY)
            continue
        sweeps += 1
        result["sweeps"] = sweeps
        if cands:
            c = cands[0]
            enough = balance_ok(c["hourly"])
            result["balance"] = "余额够" if enough else "余额不够"
            event("candidate", **{k: c[k] for k in ("zone", "zone_name", "gpu", "vram", "cpu", "mem", "hourly", "image")},
                  balance=result["balance"])
            r, secs = api.call_write("CreateCompShareInstance", {
                "Region": c["region"], "Zone": c["zone"], "GpuType": c["gpu"], "GPU": 1, "CPU": c["cpu"],
                "Memory": c["mem"] * 1024, "MachineType": "G", "MinimalCpuPlatform": "Auto",
                "CompShareImageId": c["image_id"], "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": 100}],
                "ChargeType": "Postpay", "Name": NAME, "MaxCount": 1})
            if r.get("RetCode") == 0 and r.get("UHostIds"):
                chosen = c
                chosen["uhost_id"] = r["UHostIds"][0]
                result["t_create"] = time.time()
                event("created", uhost_id=chosen["uhost_id"], api_seconds=secs)
                break
            event("create_failed", RetCode=r.get("RetCode"), Message=redact(r.get("Message")))
            leftover = [h for h in existing_instances() if h["name"] == NAME]
            if leftover:  # create reported failure but an instance exists: take it
                chosen = c
                chosen["uhost_id"] = leftover[0]["id"]
                result["t_create"] = time.time()
                event("created_despite_error", uhost_id=chosen["uhost_id"])
                break
            if r.get("RetCode") == 8357 and sold_out_retries < 3:  # sold out between sweep and create
                sold_out_retries += 1
            else:
                result["outcome"] = "create_failed"
                return
        if sweeps % 10 == 0 and stop_requested():
            event("stop_requested")
            break
        time.sleep(max(0, SWEEP_EVERY - (time.time() - t_sweep)))

    if not chosen:
        result["outcome"] = result.get("outcome") or "no_stock_in_window"
        event("window_closed", sweeps=sweeps)
        return

    region, zone, uid, t_create = chosen["region"], chosen["zone"], chosen["uhost_id"], result["t_create"]
    try:
        result["machine"] = {k: chosen[k] for k in ("zone", "zone_name", "gpu", "vram", "cpu", "mem", "hourly", "image")}
        comment(f"gpu-grab run {RUN_ID}: got {chosen['gpu']} ({chosen['vram']}G) in {chosen['zone_name']} "
                f"({chosen['zone']}) at {chosen['hourly']} CNY/h, created {now_iso(t_create)}. Testing; "
                f"teardown by {now_iso(t_create + DESTROY_AT)}.")
        proj = (api.call("GetProjectList")[0].get("ProjectSet") or [{}])[0].get("ProjectId")
        r, _ = api.call_write("UpdateCompShareStopScheduler", {
            "Region": region, "Zone": zone, "ProjectId": proj, "UHostId": uid,
            "SchedulerStopTime": int(t_create + STOP_AT)})
        event("stop_scheduler_set", RetCode=r.get("RetCode"), at=now_iso(t_create + STOP_AT))

        host = None
        while time.time() - t_create < 900:
            host = describe(region, uid)
            if host and host.get("State") == "Running" and host.get("SshLoginCommand"):
                break
            time.sleep(5)
        result["create_to_running_seconds"] = round(time.time() - t_create, 1)
        if not host or host.get("State") != "Running":
            event("not_running", state=host and host.get("State"))
            return
        pw = host.get("Password") or ""
        try:
            pw = base64.b64decode(pw).decode("utf-8")
        except Exception:
            pass
        print(f"::add-mask::{pw}", flush=True)
        user, ip, port = parse_login(host["SshLoginCommand"])
        print(f"::add-mask::{ip}", flush=True)
        event("running", create_to_running_seconds=result["create_to_running_seconds"], ssh_port=port)
        env = dict(os.environ, SSHPASS=pw)
        result["ssh_stability"] = ssh_stability(ip, port)
        event("ssh_stability", **{k: result["ssh_stability"][k] for k in ("tcp_failures", "tcp_median_ms", "tcp_max_ms")})
        run_tests((user, ip, port), t_create, env)
        result["remote_tests_ran"] = os.path.exists(os.path.join(OUT, "remote_result.json"))
    finally:
        ok = destroy(region, zone, uid)
        result["t_destroyed"] = time.time() if ok else None
        result["instance_lifetime_minutes"] = round((time.time() - t_create) / 60, 2)
        result["outcome"] = "tested_and_destroyed" if ok else "destroy_failed"
        save()


if __name__ == "__main__":
    try:
        main()
    finally:
        result["finished_utc"] = now_iso()
        save()
        stock_log.close()
        text = open(os.path.join(OUT, "result.json")).read()
        for s in (os.environ.get("COMPSHARE_PUBLIC_KEY"), os.environ.get("COMPSHARE_PRIVATE_KEY")):
            if s and s in text:
                os.remove(os.path.join(OUT, "result.json"))
                sys.exit("result contained a key; removed")
