"""Usage: probe.py NAME REQUEST_JSON_FILE  -> saves runs/m2/siliconflow_probe/NAME.json (no headers, no key)."""
import json, os, sys, time, urllib.request, urllib.error, datetime
name, reqf = sys.argv[1], sys.argv[2]
body = json.load(open(reqf))
url = "https://api.siliconflow.cn/v1/chat/completions"
assert body.get("max_tokens", 0) <= 200
req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
    headers={"Authorization": "Bearer " + os.environ["SILICONFLOW_API_KEY"], "Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        status, raw = r.status, r.read().decode()
except urllib.error.HTTPError as e:
    status, raw = e.code, e.read().decode()
except Exception as e:
    status, raw = None, json.dumps({"_client_error": type(e).__name__ + ": " + str(e)})
dt = time.time() - t0
try:
    resp = json.loads(raw)
except Exception:
    resp = {"_raw_text": raw}
out = {"probe": name, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "url": url,
       "request_body": body, "http_status": status, "latency_s": round(dt, 2), "response": resp}
d = "/home/user/good-roommate/runs/m2/siliconflow_probe"
json.dump(out, open(f"{d}/{name}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"status": status, "latency": round(dt, 2), "response": resp}, ensure_ascii=False)[:3000])
