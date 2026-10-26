"""Relay for the M2 pilot: keep one training machine, run commands on it, bring results back.

Runs on a GitHub Actions runner (push to chain-request with mode "train"). The request
(ops/probe_request.json) lists actions, run in order:
  {"do": "dscheck"}                      DeepSeek key present? balance >= 60 CNY? (booleans only)
  {"do": "stock"}                        one stock sweep, appended to the stock log
  {"do": "grab", "max_minutes": 300}     sweep at most every 30 s until a suitable machine is created
  {"do": "start"}                        start the machine if it is stopped
  {"do": "reboot"}                       restart an unreachable machine (same machine and GPU); at most once per machine
  {"do": "upload"}                       copy the code at the request commit to /root/work/repo
  {"do": "run", "name": n, "cmd": c, "background": true, "timeout_min": 30}
  {"do": "wait", "minutes": 80, "jobs": [n, ...]}   sync every SYNC_EVERY s until the jobs end
  {"do": "sync"}                         results -> data branch, checkpoints -> artifact
  {"do": "kill", "name": n}
  {"do": "image", "name": "..."}         custom image from the machine; time and size recorded
  {"do": "destroy", "confirm_nothing_unsynced": "..."}   terminate, then check that no instance is left;
                                         a machine that was ever unreachable needs the confirmation text
Any action may carry "optional": true: if the machine is unreachable it is skipped instead of ending the round.
Shutdown fallback: every heartbeat sets the platform shutdown to now + 2 h, at the start of the round and after
every sync in a wait, so it is always later than the round's expected end (waits end at most ROUND_WAIT_LIMIT
after the round starts) and the machine stops by itself within 2 h of the last heartbeat if the relay is lost.
Unreachable machine (SSH fails 3 times, 15 s each): the round stops at once (no sync, no further actions), the
PR comment opens with "连不上" and names the failed action, the shutdown is set to 30 min ahead, and the event is
recorded in relay/machine_health.json. Rounds that need SSH check it first, so such a round ends within minutes.
Limits: one instance at a time, <= MAX_HOURLY CNY/h, stock swept at most every 30 s,
grab windows add up to at most 24 h (counted in the data branch).
Never written anywhere public: keys, login password, instance IP, balances, charges.
"""
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import compshare_api as api  # noqa: E402

api.TRIES = 4    # 每次接口调用最坏约 2.5 分钟；连不上之后降到 2 次（约 1 分钟）

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/relay"
REQ = json.load(open(os.path.join(HERE, "probe_request.json")))
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")
PR = os.environ.get("PR_NUMBER") or str(REQ.get("pr_number") or "")
DATA_BRANCH = "m2pilot-data"
DATA_PREFIX = "runs/m2pilot"
NAME_PREFIX = "gr-train-"
REGIONS = ("cn-wlcb", "cn-sh2", "cn-bj2")
MAX_HOURLY = 25.0
SWEEP_EVERY = 30
GRAB_TOTAL_MINUTES = 24 * 60
SYNC_EVERY = 600
POLL_EVERY = 300              # 等待时每 5 分钟查一次任务状态（少登录）
HEARTBEAT = 2 * 3600          # 兜底关机：最后一次心跳之后 2 小时
UNREACHABLE_STOP = 30 * 60    # 连不上时只留 30 分钟（够发一轮重启或换机），不再按 2 小时续
ROUND_WAIT_LIMIT = 100 * 60   # 一轮里等待最晚结束于开轮后 100 分钟，加上收尾同步仍早于兜底关机
SSH_TRIES = 3                 # 每次连 15 秒超时，最多 3 次（约 75 秒）；还连不上就算"连不上"，本轮立即收尾
REMOTE = "/root/work"
# (GPU type, count) in order of preference; one instance, >= 48 GB in total so the sampler
# (vLLM) and the trainer fit side by side.
PREFER = [("A800", 1), ("4090_48G", 1), ("H20", 1), ("A100", 1), ("4090", 2), ("5090", 2), ("3090", 2)]
IMAGE_PREFERENCE = ("cuda132_torch", "cuda130_torch", "cuda128_torch", "cuda132_python312", "cuda130_python312")
CODE_PATHS = ["roommate", "ops", "tests", "runs/m1v2/seed1/world.json", "runs/m1v2/seed2/world.json",
              "runs/m1v2/seed3/world.json", "runs/m2/siliconflow_probe", "runs/m2prep", "runs/m2/interim",
              "runs/m2pilot/config"]
DATA_PATHS = ["runs/m2pilot/teacher", "runs/m2pilot/relay/machine.json", "runs/m2pilot/relay/machines_past.jsonl"]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
summary = {"run_id": RUN_ID, "actions": []}
_mask = set()


class Unreachable(RuntimeError):
    """SSH 连不上（连接、认证或连接中途断开）。本轮不再做任何要连机器的动作，直接收尾报告。"""


def now_iso(t=None):
    return datetime.fromtimestamp(t or time.time(), timezone.utc).isoformat(timespec="seconds")


def log(msg, **kw):
    e = {"t": now_iso(), "msg": msg, **kw}
    text = json.dumps(e, ensure_ascii=False)
    for m in _mask:
        text = text.replace(m, "***")
    print(text, flush=True)
    with open(os.path.join(OUT, "relay_log.jsonl"), "a") as f:
        f.write(text + "\n")


def mask(s):
    if s:
        _mask.add(s)
        print(f"::add-mask::{s}", flush=True)


def redact(text):
    text = str(text)
    for m in _mask:                              # 先遮登录地址（主机名、IP），再遮数字
        text = text.replace(m, "***")
    text = re.sub(r"[\w.-]+\.compshare\.cn", "***", text)
    return re.sub(r"\d+(\.\d+)?", "#", text)


def comment(body):
    if not PR:
        return
    for m in _mask:
        body = body.replace(m, "***")
    subprocess.run(["gh", "pr", "comment", PR, "--body", body], capture_output=True, text=True)


def req_time():
    try:
        return datetime.fromisoformat(REQ["requested_at"].replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()


# ---------------- data branch ----------------

def git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def data_checkout():
    d = os.path.join(OUT, "data")
    hdr = git("config", "--get", "http.https://github.com/.extraheader", cwd=REPO, check=False).stdout.strip()
    url = git("remote", "get-url", "origin", cwd=REPO).stdout.strip()
    if not os.path.isdir(os.path.join(d, ".git")):
        r = subprocess.run(["git", "-c", f"http.https://github.com/.extraheader={hdr}", "clone", "-q", "--depth=1",
                            "--branch", DATA_BRANCH, url, d], capture_output=True, text=True)
        if r.returncode != 0:
            os.makedirs(d, exist_ok=True)
            git("init", "-q", "-b", DATA_BRANCH, cwd=d)
            git("remote", "add", "origin", url, cwd=d)
        git("config", "user.name", "github-actions[bot]", cwd=d)
        git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com", cwd=d)
        if hdr:
            git("config", "http.https://github.com/.extraheader", hdr, cwd=d)
    return d


def data_push(message):
    d = data_checkout()
    git("add", "-A", cwd=d)
    if git("diff", "--cached", "--quiet", cwd=d, check=False).returncode == 0:
        return
    git("commit", "-q", "-m", message, cwd=d)
    for i in range(5):
        if git("push", "-q", "origin", DATA_BRANCH, cwd=d, check=False).returncode == 0:
            return
        git("pull", "-q", "--rebase", "origin", DATA_BRANCH, cwd=d, check=False)
        time.sleep(5 * (i + 1))
    log("data_push_failed")


def data_json(rel, default):
    p = os.path.join(data_checkout(), DATA_PREFIX, rel)
    if os.path.exists(p):
        return json.load(open(p))
    return default


def data_write(rel, obj):
    p = os.path.join(data_checkout(), DATA_PREFIX, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def data_append(rel, line):
    p = os.path.join(data_checkout(), DATA_PREFIX, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------- machine ----------------

def machines():
    """账号下全部实例。接口不按 Region 过滤（每个地域都返回同一批），按实例 ID 去重，地域取自可用区。"""
    out = {}
    r, _ = api.call("DescribeCompShareInstance", {"Region": REGIONS[0], "Limit": 100})   # 不按 Region 过滤，查一次即全部
    for h in r.get("UHostSet") or []:
        out.setdefault(h["UHostId"], dict(h, _region=str(h.get("Zone") or REGIONS[0]).rsplit("-", 1)[0]))
    return list(out.values())


def our_machine():
    ours = [h for h in machines() if str(h.get("Name", "")).startswith(NAME_PREFIX)]
    assert len(ours) <= 1, f"more than one training instance: {[h.get('UHostId') for h in ours]}"
    return ours[0] if ours else None


def describe(h):
    r, _ = api.call("DescribeCompShareInstance", {"Region": h["_region"], "UHostIds": [h["UHostId"]]})
    hosts = r.get("UHostSet") or []
    return dict(hosts[0], _region=h["_region"]) if hosts else None


def loc(h):
    return {"Region": h["_region"], "Zone": h["Zone"], "UHostId": h["UHostId"]}


def heartbeat(h, seconds=HEARTBEAT):
    stop_at = int(time.time() + seconds)
    proj = (api.call("GetProjectList")[0].get("ProjectSet") or [{}])[0].get("ProjectId")
    r, _ = api.call_write("UpdateCompShareStopScheduler", {**loc(h), "ProjectId": proj, "SchedulerStopTime": stop_at})
    log("heartbeat", RetCode=r.get("RetCode"), stop_at=now_iso(stop_at))
    summary["stop_at"] = now_iso(stop_at)
    summary.setdefault("heartbeats", []).append({"at": now_iso(), "stop_at": now_iso(stop_at), "ok": r.get("RetCode") == 0})
    return r.get("RetCode") == 0


def wait_running(h, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = describe(h)
        if h and h.get("State") == "Running" and h.get("SshLoginCommand"):
            return h
        time.sleep(5)
    return None


def login(h):
    pw = h.get("Password") or ""
    try:
        pw = base64.b64decode(pw).decode("utf-8")
    except Exception:
        pass
    mask(pw)
    argv = shlex.split(h["SshLoginCommand"])
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
    mask(host)
    return {"user": user, "host": host, "port": port, "env": dict(os.environ, SSHPASS=pw)}


SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "PreferredAuthentications=password,keyboard-interactive", "-o", "PubkeyAuthentication=no",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]


def ssh(c, cmd, timeout=600, stdin=None, binary=False):
    argv = ["sshpass", "-e", "ssh", "-p", str(c["port"]), *SSH_OPTS, f"{c['user']}@{c['host']}", cmd]
    for attempt in range(SSH_TRIES):
        try:
            p = subprocess.run(argv, input=stdin, capture_output=True, text=not binary, env=c["env"], timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 255:  # 255 = ssh itself failed (connect/auth/dropped); retry
            return p
        if attempt + 1 < SSH_TRIES:
            time.sleep(10 * (attempt + 1))
    err = p.stderr.decode("utf-8", "replace") if isinstance(p.stderr, bytes) else (p.stderr or "")
    raise Unreachable(redact(err.strip().splitlines()[-1] if err.strip() else "ssh rc 255")[:200])


# ---------------- actions ----------------

def do_dscheck(a):
    import urllib.request
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    res = {"deepseek_key_present": bool(key)}
    if key:
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                "https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {key}"}), timeout=30)
            d = json.load(r)
            cny = sum(float(b.get("total_balance") or 0) for b in d.get("balance_infos") or [] if b.get("currency") == "CNY")
            usd = sum(float(b.get("total_balance") or 0) for b in d.get("balance_infos") or [] if b.get("currency") == "USD")
            res["deepseek_available"] = bool(d.get("is_available"))
            res["deepseek_balance_ge_60_cny"] = cny >= 60 or usd * 7 >= 60
        except Exception as e:
            res["deepseek_error"] = redact(f"{type(e).__name__}: {e}")
    log("dscheck", **res)
    summary["dscheck"] = res


def candidates_from_sweep():
    rec = {"t": now_iso(), "in_stock": []}
    cands = []
    zones, _ = api.call("DescribeCompShareSupportZone")
    for z in zones.get("ZoneInfo") or []:
        L = {"Region": z["Region"], "Zone": z["Zone"]}
        types, _ = api.call("DescribeAvailableCompShareInstanceTypes", {**L, "InstanceType": "uhost"})
        wanted = {g for g, _ in PREFER}
        tlist = [t for t in types.get("AvailableInstanceTypes") or [] if t.get("Name") in wanted]
        if not tlist:
            continue
        imgs, _ = api.call("DescribeCompShareImages", {**L, "Limit": 100, "Offset": 0})
        avail = [i for i in imgs.get("ImageSet") or [] if i.get("Status") == "Available"]
        image = next((i for p in IMAGE_PREFERENCE for i in avail if (i.get("Name") or "").startswith(p)
                      and "5090" not in i["Name"]), None)
        mine = custom_image(L)
        if mine:                       # 有我们自己的镜像（环境和模型都在里面）就从它起
            image = mine
        if image is None:
            continue
        for t in tlist:
            cap, _ = api.call("CheckCompShareResourceCapacity", {
                **L, "GpuType": t["Name"], "MachineType": "G", "MinimalCpuPlatform": "Auto",
                "CompShareImageId": image["CompShareImageId"], "ChargeType": "Postpay",
                "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": 100}]})
            for s in cap.get("Specs") or []:
                if not s.get("ResourceEnough"):
                    continue
                rank = next((i for i, (g, n) in enumerate(PREFER) if g == t["Name"] and n == s.get("Gpu")), None)
                rec["in_stock"].append([z["Zone"], t["Name"], s.get("Gpu"), s.get("Cpu"), s.get("Mem")])
                if rank is None:
                    continue
                pr, _ = api.call("GetCompShareInstancePrice", {
                    **L, "GpuType": t["Name"], "Gpu": s["Gpu"], "Cpu": s["Cpu"], "Memory": s["Mem"] * 1024,
                    "ChargeType": "Postpay", "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": 100}],
                    "CompShareImageId": image["CompShareImageId"], "Quantity": 1})
                d = (pr.get("PriceDetails") or [{}])[0]
                hourly = round(sum(float(d.get(k) or 0) for k in ("Instance", "Disks", "SystemDisks", "CompShareImage")), 4)
                rec["in_stock"][-1].append(hourly)
                if pr.get("RetCode") == 0 and 0 < hourly <= MAX_HOURLY:
                    cands.append({"rank": rank, "hourly": hourly, "region": z["Region"], "zone": z["Zone"],
                                  "zone_name": z.get("Describe"), "gpu": t["Name"], "count": s["Gpu"], "cpu": s["Cpu"],
                                  "mem": s["Mem"], "vram": (t.get("GraphicsMemory") or {}).get("Value"),
                                  "image": image.get("Name"), "image_id": image["CompShareImageId"],
                                  "custom_image": bool(image.get("custom"))})
    cands.sort(key=lambda c: (c["rank"], c["hourly"], -c["mem"]))
    return rec, cands


def custom_image(L):
    """数据分支 relay/image.json 里记的自定义镜像，在这个可用区可用就返回 {CompShareImageId, Name, custom}。"""
    img = data_json("relay/image.json", {}).get("image_id")
    if not img:
        return None
    try:
        r, _ = api.call("DescribeCompShareCustomImages", {**L, "Limit": 100, "Offset": 0})
    except Exception:
        return None
    for i in r.get("ImageSet") or r.get("CustomImageSet") or []:
        if i.get("CompShareImageId") == img and str(i.get("Status", "")).lower() in ("available", "normal", "usable"):
            return {"CompShareImageId": img, "Name": i.get("Name") or "custom", "custom": True}
    return None


def do_stock(a):
    rec, cands = candidates_from_sweep()
    data_append("relay/stock_log.jsonl", rec)
    log("stock", in_stock=len(rec["in_stock"]), suitable=len(cands))


def do_grab(a):
    if our_machine():
        log("grab_skip_existing")
        return True
    others = [h for h in machines()]
    if others:
        log("grab_abort_other_instances", count=len(others))
        return False
    used = data_json("relay/grab_minutes.json", {"minutes": 0.0, "runs": []})
    budget = min(float(a.get("max_minutes", 300)), GRAB_TOTAL_MINUTES - used["minutes"])
    t0, sweeps, made = time.time(), 0, None
    try:
        while time.time() - t0 < budget * 60:
            ts = time.time()
            try:
                rec, cands = candidates_from_sweep()
            except Exception as e:
                log("sweep_error", error=redact(e))
                time.sleep(SWEEP_EVERY)
                continue
            sweeps += 1
            data_append("relay/stock_log.jsonl", rec)
            if cands:
                c = cands[0]
                name = NAME_PREFIX + datetime.now(timezone.utc).strftime("%m%d%H%M")
                try:
                    r, _ = api.call_write("CreateCompShareInstance", {
                        "Region": c["region"], "Zone": c["zone"], "GpuType": c["gpu"], "GPU": c["count"],
                        "CPU": c["cpu"], "Memory": c["mem"] * 1024, "MachineType": "G", "MinimalCpuPlatform": "Auto",
                        "CompShareImageId": c["image_id"], "Disks": [{"IsBoot": True, "Type": "CLOUD_SSD", "Size": 100}],
                        "ChargeType": "Postpay", "Name": name, "MaxCount": 1})
                except Exception as e:      # 网络错误：不重发，下面查一下实例是否已经建了
                    r = {"RetCode": -1, "Message": f"{type(e).__name__}"}
                if r.get("RetCode") == 0 or our_machine():
                    made = dict(c, name=name, created=now_iso())
                    log("created", **{k: made[k] for k in ("zone", "zone_name", "gpu", "count", "cpu", "mem", "hourly", "image")})
                    break
                log("create_failed", RetCode=r.get("RetCode"), Message=redact(r.get("Message")))
                if r.get("RetCode") != 8357:
                    break
            if sweeps % 20 == 0:
                data_push(f"relay {RUN_ID}: stock log")
            time.sleep(max(0, SWEEP_EVERY - (time.time() - ts)))
    finally:
        used["minutes"] += round((time.time() - t0) / 60, 2)
        used["runs"].append({"run_id": RUN_ID, "minutes": round((time.time() - t0) / 60, 2), "sweeps": sweeps,
                             "got": bool(made)})
        data_write("relay/grab_minutes.json", used)
    summary["grab"] = {"sweeps": sweeps, "minutes": round((time.time() - t0) / 60, 1), "machine": made,
                       "cumulative_grab_minutes": used["minutes"]}
    if made:
        data_write("relay/machine.json", {k: made.get(k) for k in ("name", "created", "zone", "zone_name", "gpu", "count",
                                                                    "cpu", "mem", "vram", "hourly", "image", "custom_image")})
        h = our_machine()
        heartbeat(h)
        t1 = time.time()
        h = wait_running(h)
        log("running", seconds=round(time.time() - t1, 1))
        if h:
            c = login(h)
            t2 = time.time()
            while time.time() - t2 < 600:
                try:
                    p = ssh(c, "echo LOGIN_OK", timeout=60)
                except Unreachable:
                    p = None            # 新机器刚开，sshd 可能还没起来
                if p is not None and "LOGIN_OK" in (p.stdout or ""):
                    break
                time.sleep(5)
            summary["grab"]["create_to_login_s"] = round(time.time() - t1, 1)
            log("login_ok", create_to_login_s=summary["grab"]["create_to_login_s"])
    return bool(made)


def do_start(a):
    h = our_machine()
    if h and h.get("State") == "Stopped":
        r, _ = api.call_write("StartCompShareInstance", loc(h))
        log("start", RetCode=r.get("RetCode"), Message=redact(r.get("Message")))
        h = wait_running(h)
        log("started", ok=bool(h))


def do_reboot(a):
    """SSH 连不上、而实例在接口里仍是 Running 时：重启（同一台机器、同一张卡，不释放）。"""
    h = our_machine()
    if not h or h.get("State") != "Running":
        log("reboot_skipped", state=h.get("State") if h else None)
        return False
    health = data_json("relay/machine_health.json", {})
    rec = health.setdefault(h["UHostId"], {"reboots": [], "unreachable": []})
    if rec["reboots"]:           # 一台连不上的机器只重启一次；再连不上就换机器
        log("reboot_refused", reason="already rebooted once", at=rec["reboots"])
        summary["reboot_refused"] = "already rebooted once: replace the machine"
        return False
    r, _ = api.call_write("RebootCompShareInstance", loc(h))
    rec["reboots"].append(now_iso())
    data_write("relay/machine_health.json", health)
    log("reboot", RetCode=r.get("RetCode"), Message=redact(r.get("Message")))
    time.sleep(30)
    h = wait_running(h, timeout=300)
    if not h:
        raise Unreachable("not Running within 5 minutes after reboot")
    p = ssh(login(h), "uptime; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader", timeout=120)
    log("rebooted", ssh_ok=True, out=(p.stdout or "").strip()[-300:] if p else None)
    return True


CUR = {}


def health_note(**kw):
    """数据分支 relay/machine_health.json：每台机器的重启、连不上、最后一次同步成功、最后一次起任务的时间。"""
    if not CUR.get("id"):
        return
    health = data_json("relay/machine_health.json", {})
    rec = health.setdefault(CUR["id"], {"reboots": [], "unreachable": []})
    rec.update(kw)
    data_write("relay/machine_health.json", health)


def conn():
    h = our_machine()
    if not h:
        raise RuntimeError("no training machine")
    CUR["id"] = h["UHostId"]
    h = wait_running(h, timeout=120) or h
    if h.get("State") != "Running":
        raise RuntimeError(f"machine state {h.get('State')}")
    return login(h)


def do_upload(a):
    c = conn()
    tar = os.path.join(OUT, "code.tgz")
    dd = data_checkout()
    with tarfile.open(tar, "w:gz") as t:
        for p in CODE_PATHS:
            if os.path.exists(os.path.join(REPO, p)):
                t.add(os.path.join(REPO, p), arcname=p, filter=lambda ti: None if "__pycache__" in ti.name else ti)
        for p in DATA_PATHS:      # 来自数据分支：老师轨迹、机器信息（给流水线估算花费）
            if os.path.exists(os.path.join(dd, p)):
                t.add(os.path.join(dd, p), arcname=p)
    with open(tar, "rb") as f:
        p = ssh(c, f"mkdir -p {REMOTE}/repo {REMOTE}/out/logs {REMOTE}/out/pids && tar xzf - -C {REMOTE}/repo && "
                   f"cd {REMOTE}/repo && echo {os.environ.get('GITHUB_SHA', '')} > COMMIT", stdin=f.read(), binary=True)
    log("upload", rc=p.returncode if p else None, bytes=os.path.getsize(tar))
    return bool(p is not None and p.returncode == 0)


def do_run(a):
    c = conn()
    name, cmd = a["name"], a["cmd"]
    inner = f"cd {REMOTE}/repo && {cmd}; echo $? > {REMOTE}/out/pids/{name}.rc"
    if a.get("background", True):
        script = (f"mkdir -p {REMOTE}/out/logs {REMOTE}/out/pids; rm -f {REMOTE}/out/pids/{name}.rc; "
                  f"nohup setsid bash -lc {shlex.quote(inner)} >> {REMOTE}/out/logs/{name}.log 2>&1 < /dev/null & "
                  f"echo $! > {REMOTE}/out/pids/{name}.pid")
        p = ssh(c, script, timeout=60)
        log("run_bg", name=name, rc=p.returncode if p else None)
        if p is not None and p.returncode == 0:
            health_note(last_job_started={"name": name, "at": now_iso()})
        return bool(p is not None and p.returncode == 0)
    else:
        t0 = time.time()
        p = ssh(c, f"mkdir -p {REMOTE}/out/logs {REMOTE}/out/pids; bash -lc {shlex.quote(inner)} 2>&1 | tail -c 20000 "
                   f"| tee -a {REMOTE}/out/logs/{name}.log", timeout=int(a.get("timeout_min", 30)) * 60)
        tail = re.sub(r"(\d{1,3}\.){3}\d{1,3}", "<ip>", (p.stdout or "")[-3000:]) if p else "timeout"
        log("run_fg", name=name, seconds=round(time.time() - t0, 1), tail=tail)
        summary.setdefault("fg_output", {})[name] = tail
        if a.get("gate"):               # 关卡：命令本身失败（退出码非 0）就不往下做
            return bool(p is not None and p.returncode == 0)
        return bool(p is not None and p.returncode != 255)


def status_json():
    p = os.path.join(data_checkout(), DATA_PREFIX, "machine", "status.json")
    return json.load(open(p)) if os.path.exists(p) else None


def job_status(c, names):
    q = " ".join(names)
    script = (f"cd {REMOTE}/out/pids 2>/dev/null; for n in {q}; do "
              f"if [ -f $n.rc ]; then echo $n done $(cat $n.rc); "
              f"elif [ -f $n.pid ] && kill -0 $(cat $n.pid) 2>/dev/null; then echo $n running; "
              f"else echo $n missing; fi; done")
    p = ssh(c, "bash -c " + shlex.quote(script), timeout=60)
    if p is None or not (p.stdout or "").strip():
        log("job_status_empty", rc=p.returncode if p else None, err=(p.stderr or "")[-300:] if p else None)
    st = {}
    for line in (p.stdout if p else "").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            st[parts[0]] = parts[1] + (f" rc={parts[2]}" if len(parts) > 2 else "")
    return st


def do_sync(a=None):
    """结果（不含检查点）-> 数据分支；a = {"ckpt": true} 时另把 ckpt_export 里新的或变过的文件取回，存成 artifact。"""
    c = conn()
    local = os.path.join(data_checkout(), DATA_PREFIX, "machine")
    os.makedirs(local, exist_ok=True)
    p = ssh(c, f"cd {REMOTE}/out && tar czf - --exclude='./ckpt_export' --exclude='*.safetensors' --exclude='*.pt' "
               f"--exclude='*.bin' --exclude='./tmp' .", timeout=900, binary=True)
    if p is not None and p.returncode == 0:
        subprocess.run(["tar", "xzf", "-", "-C", local], input=p.stdout, check=False)
        health_note(last_sync_ok=now_iso())
    pulled = 0
    if a and a.get("ckpt"):
        man = data_json("relay/ckpt_manifest.json", {})
        q = ssh(c, f"cd {REMOTE}/out && [ -d ckpt_export ] && find ckpt_export -type f -printf '%p %s %T@\\n' || true",
                timeout=120)
        cur = {}
        for line in (q.stdout if q else "").splitlines():
            parts = line.split()
            if len(parts) == 3:
                cur[parts[0]] = f"{parts[1]}:{parts[2]}"
        new = [f for f, v in cur.items() if man.get(f, {}).get("v") != v]
        if new:
            ck = os.path.join(OUT, "ckpt")
            os.makedirs(ck, exist_ok=True)
            r = ssh(c, f"cd {REMOTE}/out && tar cf - " + " ".join(shlex.quote(f) for f in new), timeout=3600, binary=True)
            if r is not None and r.returncode == 0:
                subprocess.run(["tar", "xf", "-", "-C", ck], input=r.stdout, check=False)
                pulled = len(r.stdout)
                for f in new:
                    man[f] = {"v": cur[f], "artifact": f"m2pilot-ckpt-{RUN_ID}"}
                data_write("relay/ckpt_manifest.json", man)
    data_push(f"relay {RUN_ID}: sync from training machine")
    log("sync", bytes=len(p.stdout) if p else None, ckpt_bytes=pulled)


def do_wait(a):
    c = conn()
    names = a.get("jobs") or []
    t0 = time.time()
    minutes = min(float(a.get("minutes", 60)), max(1.0, (T0 + ROUND_WAIT_LIMIT - t0) / 60))
    end = t0 + minutes * 60 + SYNC_EVERY          # 预计本轮结束：等待结束 + 收尾同步
    summary["round_expected_end"] = now_iso(end)
    log("wait_plan", minutes=round(minutes, 1), requested=a.get("minutes"), expected_end=now_iso(end))
    last = 0
    st = {}
    while time.time() - t0 < minutes * 60:
        try:
            st = job_status(c, names) if names else {}
            done = bool(names) and len(st) == len(names) and all(not v.startswith("running") for v in st.values())
            if done or time.time() - last >= SYNC_EVERY:
                do_sync({"ckpt": bool(done)})
                last = time.time()
                h = our_machine()
                if h and h.get("State") == "Running":   # 兜底关机始终晚于本轮预计结束
                    heartbeat(h)
                    if time.time() + HEARTBEAT <= end:
                        log("heartbeat_before_round_end", stop_at=summary["stop_at"], expected_end=now_iso(end))
            if done:
                break
        except Unreachable:
            raise
        except Exception as e:      # 接口偶尔出错：记下，下一分钟再试
            log("wait_error", error=redact(e)[:200])
            try:
                c = conn()
            except Exception:
                pass
        time.sleep(min(POLL_EVERY, max(1, t0 + minutes * 60 - time.time())))
    if not (names and len(st) == len(names) and all(not v.startswith("running") for v in st.values())):
        do_sync({"ckpt": True})    # 等待结束时任务还在跑：取回最新的检查点
    summary["jobs"] = st
    summary["status"] = status_json()
    log("wait_end", jobs=st, minutes=round((time.time() - t0) / 60, 1))


def do_kill(a):
    c = conn()
    p = ssh(c, f"pid=$(cat {REMOTE}/out/pids/{a['name']}.pid 2>/dev/null) && kill -TERM -$pid; sleep 5; "
               f"kill -KILL -$pid 2>/dev/null; echo killed", timeout=60)
    log("kill", name=a["name"], out=(p.stdout or "").strip() if p else None)


def do_image(a):
    h = our_machine()
    t0 = time.time()
    r, _ = api.call_write("CreateCompShareCustomImage", {"Region": h["_region"], "Zone": h["Zone"],
                                                          "UHostId": h["UHostId"], "Name": a.get("name", "m2pilot-env")})
    img = r.get("CompShareImageId") or r.get("ImageId")
    log("image_create", RetCode=r.get("RetCode"), Message=redact(r.get("Message")), image=img)
    final, last_hb = None, time.time()
    while img and time.time() - t0 < 3600:
        g, _ = api.call("GetCompShareImageCreateProgress", {"Region": h["_region"], "Zone": h["Zone"],
                                                            "CompShareImageId": img})
        status = str(g.get("Status") or g.get("State") or "")
        if status.lower() in ("available", "success", "succeeded", "done") or g.get("Progress") == 100:
            final = g
            break
        if status.lower() in ("failed", "error"):
            break
        time.sleep(20)
        if time.time() - last_hb > SYNC_EVERY:      # 长等待：兜底关机始终在前面
            heartbeat(h)
            last_hb = time.time()
    info = {"image_id": img, "seconds": round(time.time() - t0, 1), "final": {k: v for k, v in (final or {}).items()
                                                                              if k not in ("request_uuid",)}}
    data_write("relay/image.json", info)
    summary["image"] = info


def do_destroy(a):
    h = our_machine()
    rec = data_json("relay/machine_health.json", {}).get(h["UHostId"], {}) if h else {}
    if rec.get("unreachable"):
        # 连不上的机器：释放前必须说明上面没有仓库里没有的东西（最后一次同步之后没起过任务等），记进数据分支
        if not a.get("confirm_nothing_unsynced"):
            log("destroy_refused", reason="unreachable machine: confirm_nothing_unsynced missing")
            summary["destroy_refused"] = True
            return False
        summary["release_evidence"] = {"machine": h["UHostId"], "health": rec, "confirm": a["confirm_nothing_unsynced"]}
        data_write("relay/released_unreachable.json", {"at": now_iso(), **summary["release_evidence"]})
    if h:     # 释放过的机器另记一份（开机时间、释放时间、单价），流水线估算花费时把它们也算进去
        m = data_json("relay/machine.json", {})
        data_append("relay/machines_past.jsonl", {"name": h.get("Name"), "uhost": h["UHostId"], "gpu": m.get("gpu"),
                                                 "created": m.get("created"), "released": now_iso(),
                                                 "hourly": m.get("hourly")})
    t0 = time.time()
    while h and time.time() - t0 < 900:
        state = h.get("State")
        if state not in ("Stopping", "Starting", "Rebooting"):
            r, _ = api.call_write("TerminateCompShareInstance", {**loc(h), "ReleaseUDisk": True})
            log("terminate", RetCode=r.get("RetCode"), state=state)
            if r.get("RetCode") != 0 and state == "Running":
                r, _ = api.call_write("StopCompShareInstance", loc(h))
                log("stop", RetCode=r.get("RetCode"))
        time.sleep(15)
        h = our_machine()
    left = [{"name": x.get("Name"), "state": x.get("State")} for x in machines()]
    summary["after_destroy_instances"] = left
    data_write("relay/destroyed.json", {"at": now_iso(), "instances_left": left})
    log("destroyed", instances_left=len(left))


def do_fetch_tokenizer(a):
    """Official Qwen3.5-4B tokenizer files at the revision the CPU template check used (for tests off the GPU)."""
    import urllib.request
    rev = a.get("rev", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    d = os.path.join(data_checkout(), DATA_PREFIX, "tokenizer")
    os.makedirs(d, exist_ok=True)
    got = {}
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
              "config.json", "generation_config.json", "special_tokens_map.json"):
        try:
            data = urllib.request.urlopen(f"https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{rev}/{f}", timeout=120).read()
            open(os.path.join(d, f), "wb").write(data)
            got[f] = len(data)
        except Exception as e:
            got[f] = redact(e)
    json.dump({"source": f"https://huggingface.co/Qwen/Qwen3.5-4B/tree/{rev}", "license": "Apache-2.0", "files": got},
              open(os.path.join(d, "SOURCE.json"), "w"), indent=1)
    log("fetch_tokenizer", files=got)


ACTIONS = {"fetch_tokenizer": do_fetch_tokenizer, "dscheck": do_dscheck, "stock": do_stock, "grab": do_grab, "start": do_start, "reboot": do_reboot, "upload": do_upload,
           "run": do_run, "wait": do_wait, "sync": do_sync, "kill": do_kill, "image": do_image, "destroy": do_destroy}


NEEDS_SSH = {"upload", "run", "wait", "sync", "kill"}


def first_ssh_on_this_machine(actions):
    """本轮在换机器（释放、新开、重启、开机）之前，有没有必须连这台机器的动作（可选的诊断不算）。"""
    for x in actions:
        if x["do"] in ("destroy", "grab", "reboot", "start"):
            return False
        if x["do"] in NEEDS_SSH and not x.get("optional"):
            return True
    return False


def record_unreachable(h, action):
    health = data_json("relay/machine_health.json", {})
    health.setdefault(h["UHostId"], {"reboots": [], "unreachable": []})["unreachable"].append(
        {"at": now_iso(), "run_id": RUN_ID, "action": action})
    data_write("relay/machine_health.json", health)


def note_unreachable(e, a):
    summary["unreachable"] = {"failed_action": a["do"] if a else "preflight", "name": (a or {}).get("name"),
                              "at": now_iso(), "error": str(e)[:200],
                              "not_done": [x["do"] for x in REQ.get("actions", [])[len(summary["actions"]):]]}
    log("unreachable", **summary["unreachable"])
    api.TRIES = 2
    try:
        h = our_machine()
        summary["machine_state"] = h.get("State") if h else None
        if h:
            record_unreachable(h, summary["unreachable"]["failed_action"])
            if h.get("State") == "Running":
                heartbeat(h, UNREACHABLE_STOP)      # 连不上就不再按 2 小时续：30 分钟后自动关机
    except Exception as x:
        summary["state_error"] = redact(x)[:200]


def main():
    a = None
    try:
        h = our_machine()
        if h and h.get("State") == "Running" and first_ssh_on_this_machine(REQ.get("actions", [])):
            CUR["id"] = h["UHostId"]
            # 先确认连得上（连不上在两分钟内收尾）；顺手确认网络观察在跑（重启后要重新起）
            ssh(login(h), f"f={REMOTE}/repo/ops/remote/netwatch.sh; [ -f $f ] && ! pgrep -f ops/remote/netwatch.sh >/dev/null "
                          f"&& (nohup setsid bash $f >/dev/null 2>&1 < /dev/null &); true", timeout=60)
        if h and h.get("State") in ("Running", "Starting", "Initializing"):
            heartbeat(h)
        for a in REQ.get("actions", []):
            log("action", **{k: v for k, v in a.items() if k != "cmd"}, cmd=(a.get("cmd") or "")[:200])
            if os.environ.get("RELAY_KEEPALIVE") and not (a["do"] in ("sync", "wait") or
                                                         (a["do"] == "run" and a.get("background") is False)):
                log("skipped_in_keepalive", do=a["do"])     # 和长请求并行时只做查看、同步、心跳
                continue
            try:
                ok = ACTIONS[a["do"]](a)
            except Unreachable as e:
                summary["actions"].append({"do": a["do"], "name": a.get("name"), "ok": False, "unreachable": True})
                if not a.get("optional"):
                    raise
                log("optional_unreachable", do=a["do"], name=a.get("name"), error=str(e)[:200])
                h = our_machine()
                if h:
                    record_unreachable(h, a["do"])
                summary.setdefault("optional_unreachable", []).append({"do": a["do"], "name": a.get("name"),
                                                                       "at": now_iso(), "error": str(e)[:200]})
                continue
            summary["actions"].append({"do": a["do"], "name": a.get("name"), "ok": ok is not False})
            if (a["do"] in ("grab", "upload", "reboot", "destroy") or a.get("gate")) and ok is False:
                log("stop_after_failed", do=a["do"])    # 没传上新代码就不往下跑（免得跑旧代码）
                break
            h = our_machine()
            if h and h.get("State") == "Running" and a["do"] not in ("destroy",):
                heartbeat(h)
    except Unreachable as e:
        note_unreachable(e, a)
    except Exception as e:
        summary["error"] = redact(f"{type(e).__name__}: {e}")[:500]
        log("error", error=summary["error"])
    finally:
        try:
            if summary.get("unreachable"):
                raise StopIteration       # 连不上：不再同步，直接报告
            h = our_machine()
            summary["machine_state"] = h.get("State") if h else None
            if h and h.get("State") == "Running" and REQ.get("actions") and \
                    REQ["actions"][-1]["do"] not in ("sync", "wait", "grab", "destroy") and not summary.get("optional_unreachable"):
                do_sync()
                summary["status"] = status_json()
        except StopIteration:
            pass
        except Unreachable as e:
            note_unreachable(e, {"do": "sync"})
        except Exception as e:
            summary["sync_error"] = redact(e)[:200]
        summary["minutes"] = round((time.time() - T0) / 60, 1)
        data_append("relay/relay_runs.jsonl", {"run_id": RUN_ID, "requested_at": REQ.get("requested_at"),
                                              "finished": now_iso(), **{k: v for k, v in summary.items()
                                                                        if k not in ("grab",)},
                                              "grab": summary.get("grab")})
        try:     # 本轮的逐条日志（已打码）也进数据分支：报告里的时间线由它复算
            src = os.path.join(OUT, "relay_log.jsonl")
            if os.path.exists(src):
                dst = os.path.join(data_checkout(), DATA_PREFIX, "relay", "logs", f"{RUN_ID}.jsonl")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(src) as f, open(dst, "w") as g:
                    g.write(f.read())
        except Exception as e:
            summary["log_copy_error"] = redact(e)[:200]
        data_push(f"relay {RUN_ID}: {','.join(a['do'] for a in REQ.get('actions', []))}")
        head = ""
        if summary.get("unreachable"):
            u = summary["unreachable"]
            head = (f"**连不上训练机器（SSH 失败）**：失败的动作是 `{u['failed_action']}`"
                    f"{'（' + u['name'] + '）' if u.get('name') else ''}，{u['at']}；"
                    f"本轮提前结束，没做的动作：{', '.join(u['not_done']) or '无'}。实例状态：{summary.get('machine_state')}。\n\n")
        elif summary.get("error"):
            head = f"**本轮出错**：{summary['error'][:200]}\n\n"
        body = head + f"train-relay run {RUN_ID} finished {now_iso()} ({summary['minutes']} min).\n\n```json\n" + \
               json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, indent=1)[:6000] + "\n```"
        comment(body)


if __name__ == "__main__":
    main()
