"""模拟 ops/train_relay.py 的几种情况（假的平台接口、假的 ssh、虚拟时钟），由 tests/test_relay.py 调用：
1. 一开始就连不上：10 分钟内收尾，评论开头写"连不上"，兜底关机改成 30 分钟后；
2. 等待中途断开：断开后 10 分钟内收尾，等待这一步报失败；
3. 一直连得上的长等待：等待被截到开轮后 100 分钟内，每次心跳的兜底关机都比本轮预计结束晚；
4. 连不上过的机器：不写明"上面没有仓库里没有的东西"就不释放；写明了才释放。
"""
import base64, importlib, json, os, subprocess, sys, tempfile, types
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ops"))
os.environ.update(COMPSHARE_PUBLIC_KEY="x", COMPSHARE_PRIVATE_KEY="y", GITHUB_RUN_ID="sim")
CLOCK = [1_800_000_000.0]
import time as _time
_time.time = lambda: CLOCK[0]
_time.sleep = lambda s: CLOCK.__setitem__(0, CLOCK[0] + s)

def run_case(actions, ssh_ok_until=None, health=None):
    """ssh_ok_until: virtual time after which ssh fails (None = always fail)."""
    tmp = tempfile.mkdtemp()
    sys.argv = ["train_relay.py", tmp]
    if "train_relay" in sys.modules:
        del sys.modules["train_relay"]
    tr = importlib.import_module("train_relay")
    tr.REQ = {"requested_at": "2027-01-15T08:00:00Z", "actions": actions}
    tr.PR = "9"
    start = CLOCK[0]
    tr.T0 = start
    calls = {"hb": [], "comment": None, "ssh": 0, "write": []}
    data = {"relay/machine_health.json": health or {}}
    tr.data_json = lambda rel, d: json.loads(json.dumps(data.get(rel, d)))
    tr.data_write = lambda rel, obj: data.__setitem__(rel, json.loads(json.dumps(obj)))
    tr.data_append = lambda rel, line: None
    tr.data_push = lambda m: None
    tr.data_checkout = lambda: tmp
    tr.status_json = lambda: None
    tr.comment = lambda body: calls.__setitem__("comment", body)
    host = {"UHostId": "uhost-1", "Name": "gr-train-x", "Zone": "cn-sh2-03", "State": "Running",
            "SshLoginCommand": "ssh -p 2222 root@203.0.113.9", "Password": base64.b64encode(b"pw").decode()}
    def call(action, params=None, timeout=30):
        CLOCK[0] += 1
        if action == "DescribeCompShareInstance":
            return {"UHostSet": [dict(host)]}, 1
        if action == "GetProjectList":
            return {"ProjectSet": [{"ProjectId": "p"}]}, 1
        return {"RetCode": 0}, 1
    def call_write(action, params=None, timeout=60):
        CLOCK[0] += 1
        calls["write"].append(action)
        if action == "UpdateCompShareStopScheduler":
            calls["hb"].append(params["SchedulerStopTime"] - CLOCK[0])
        if action == "TerminateCompShareInstance":
            host["State"] = "Terminated"
        return {"RetCode": 0}, 1
    tr.api.call, tr.api.call_write = call, call_write
    real_run = subprocess.run
    def fake_run(argv, **kw):
        if argv and argv[0] == "sshpass":
            calls["ssh"] += 1
            CLOCK[0] += 15
            if ssh_ok_until is not None and CLOCK[0] < ssh_ok_until:
                cmd = argv[-1]
                out = "pipeline running\n" if "for n in" in cmd else ""
                return subprocess.CompletedProcess(argv, 0, stdout=out if kw.get("text", True) else out.encode(),
                                                   stderr="" if kw.get("text", True) else b"")
            err = "ssh: connect to host 203.0.113.9 port 2222: Connection timed out\n"
            return subprocess.CompletedProcess(argv, 255, stdout="" if kw.get("text", True) else b"",
                                               stderr=err if kw.get("text", True) else err.encode())
        if argv and argv[0] == "tar":
            return subprocess.CompletedProcess(argv, 0)
        return real_run(argv, **kw)
    tr.subprocess.run = fake_run
    tr.main()
    took = (CLOCK[0] - start) / 60
    hb = [round(x / 60) for x in calls["hb"]]
    return tr.summary, calls, took, hb, data

# 1) unreachable from the start: upload, run, wait 90
s, c, took, hb, data = run_case([{"do": "upload"}, {"do": "run", "name": "pipeline", "cmd": "x"},
                                 {"do": "wait", "jobs": ["pipeline"], "minutes": 90}])
print("case1 minutes=%.1f hb_min=%s ssh=%d" % (took, hb, c["ssh"]), "| unreachable:", s.get("unreachable"))
print("  comment head:", c["comment"].split("\n")[0])
print("  actions:", s["actions"], "| health:", data["relay/machine_health.json"])
assert took < 10 and "连不上" in c["comment"] and hb[-1] == 30

# 2) reachable, then drops 40 min into a 90-min wait
CLOCK[0] += 10000
s, c, took, hb, data = run_case([{"do": "wait", "jobs": ["pipeline"], "minutes": 90}], ssh_ok_until=CLOCK[0] + 40 * 60)
print("case2 minutes=%.1f hb_min=%s ssh=%d" % (took, hb, c["ssh"]), "| unreachable:", s.get("unreachable"))
print("  comment head:", c["comment"].split("\n")[0], "| expected_end:", s.get("round_expected_end"))
assert took < 40 + 10 and "连不上" in c["comment"] and s["actions"][0]["ok"] is False

# 3) reachable through a long wait: stop_at always after the round's expected end
CLOCK[0] += 10000
t_start = CLOCK[0]
s, c, took, hb, data = run_case([{"do": "wait", "jobs": ["pipeline"], "minutes": 150}], ssh_ok_until=CLOCK[0] + 10 ** 6)
print("case3 minutes=%.1f hb_min_count=%d min_hb=%s" % (took, len(hb), min(hb)), "| expected_end:", s.get("round_expected_end"),
      "| last stop_at:", s.get("stop_at"))
assert took <= 112 and all(x >= 115 for x in hb)

# 4) optional diag unreachable, destroy needs confirmation
CLOCK[0] += 10000
s, c, took, hb, data = run_case([{"do": "run", "name": "diag", "background": False, "optional": True, "cmd": "x"},
                                 {"do": "destroy"}])
print("case4 destroy without confirm ->", s.get("destroy_refused"), [a for a in s["actions"]], "writes:", c["write"])
assert s.get("destroy_refused") and "TerminateCompShareInstance" not in c["write"]
CLOCK[0] += 10000
s, c, took, hb, data = run_case([{"do": "run", "name": "diag", "background": False, "optional": True, "cmd": "x"},
                                 {"do": "destroy", "confirm_nothing_unsynced": "last sync after last job"}])
print("case5 destroy with confirm -> terminate called:", "TerminateCompShareInstance" in c["write"], "| minutes=%.1f" % took)
assert "TerminateCompShareInstance" in c["write"]
print("ALL OK")
