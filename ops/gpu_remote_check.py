"""Runs on the rented GPU machine (uploaded by gpu_grab.py). No training.

Checks: nvidia-smi / driver / CUDA, BF16 on the GPU, disk space, downloading
Qwen3.5-4B (tokenizer files, then weights) from mirrors in mainland China, and a
rerun of the template round-trip check (token_check.py) with the downloaded
official tokenizer, compared with the committed CPU result.

usage: python3 gpu_remote_check.py BUNDLE_DIR OUT_DIR DEADLINE_UNIX
Writes OUT_DIR/remote_result.json; each step is recorded even if it fails.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BUNDLE, OUT, DEADLINE = sys.argv[1], sys.argv[2], float(sys.argv[3])
MODEL = "Qwen/Qwen3.5-4B"
HF_SHA = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # revision the CPU check used
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "vocab.json", "merges.txt", "config.json", "generation_config.json"]
os.makedirs(OUT, exist_ok=True)
result = {"steps": {}}


def left():
    return DEADLINE - time.time()


def save():
    with open(os.path.join(OUT, "remote_result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


def sh(cmd, timeout=120, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=max(5, min(timeout, left() - 30)), env=env)
        return {"rc": p.returncode, "seconds": round(time.time() - t0, 2),
                "stdout": p.stdout[-6000:], "stderr": p.stderr[-3000:]}
    except subprocess.TimeoutExpired as e:
        return {"rc": "timeout", "seconds": round(time.time() - t0, 2),
                "stdout": (e.stdout or b"")[-3000:].decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")[-3000:]}


def step(name, fn):
    t0 = time.time()
    try:
        result["steps"][name] = fn()
    except Exception as e:  # keep going
        result["steps"][name] = {"error": f"{type(e).__name__}: {e}"}
    result["steps"][name]["wall_seconds"] = round(time.time() - t0, 2)
    save()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def gpu():
    r = {"nvidia_smi": sh("nvidia-smi"),
         "query": sh("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader"),
         "nvcc": sh("nvcc --version || ls /usr/local | grep -i cuda"),
         "os": sh("cat /etc/os-release | head -3; uname -r; nproc; free -g | head -2")}
    head = r["nvidia_smi"]["stdout"]
    for tok in ("Driver Version:", "CUDA Version:"):
        if tok in head:
            r[tok.split()[0].lower() + "_version"] = head.split(tok, 1)[1].split()[0]
    r["torch"] = sh("python3 -c \"import torch,json;a=torch.randn(512,512,device='cuda',dtype=torch.bfloat16);"
                    "b=(a@a).float().abs().mean().item();print(json.dumps({'torch':torch.__version__,"
                    "'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),"
                    "'bf16_supported':torch.cuda.is_bf16_supported(),'bf16_matmul_ok':b==b}))\"", timeout=180)
    return r


def disk():
    return {"df": sh("df -h"), "df_root_bytes": shutil.disk_usage("/")._asdict(),
            "workdir": WORK, "df_workdir_bytes": shutil.disk_usage(WORK)._asdict()}


def pydeps():
    r = {"before": sh("python3 -c \"import importlib.util as u;print({m:bool(u.find_spec(m)) for m in "
                      "['modelscope','jinja2','tokenizers','transformers','huggingface_hub','torch']})\"")}
    r["pip_install"] = sh(f"python3 -m pip install -q -i {PIP_MIRROR} modelscope jinja2 tokenizers 2>&1 | tail -5",
                          timeout=600)
    r["pip_config"] = sh("python3 -m pip config list; cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null")
    return r


def ms_download(patterns, local_dir, timeout):
    code = ("import json,time,sys\nfrom modelscope import snapshot_download\n"
            f"t=time.time();p=snapshot_download({MODEL!r},allow_patterns={patterns!r},local_dir={local_dir!r})\n"
            "print(json.dumps({'path':p,'seconds':round(time.time()-t,2)}))\n")
    return sh(f"python3 - <<'PY'\n{code}PY", timeout=timeout)


def tokenizer_dl():
    r = {}
    d = os.path.join(WORK, "ms")
    r["modelscope"] = ms_download(TOKENIZER_FILES, d, 300)
    r["modelscope_files"] = {f: sha256(os.path.join(d, f)) for f in TOKENIZER_FILES if os.path.exists(os.path.join(d, f))}
    h = os.path.join(WORK, "hfm")
    os.makedirs(h, exist_ok=True)
    r["hf_mirror_pinned"] = {}
    for f in ("tokenizer.json", "chat_template.jinja", "tokenizer_config.json"):
        url = f"https://hf-mirror.com/{MODEL}/resolve/{HF_SHA}/{f}"
        x = sh(f"curl -sSL --fail -m 120 -o {h}/{f} -w '%{{http_code}} %{{size_download}} %{{time_total}}' {url}", 150)
        x["sha256"] = sha256(os.path.join(h, f)) if x["rc"] == 0 else None
        if x["rc"] != 0 and os.path.exists(os.path.join(h, f)):
            os.remove(os.path.join(h, f))  # never feed a partial file to the template check
        r["hf_mirror_pinned"][f] = x
    r["cpu_reference_chat_template_sha256"] = open(os.path.join(BUNDLE, "chat_template.jinja.sha256")).read().split()[0]
    return r


def weights_dl():
    d = os.path.join(WORK, "ms")
    budget = max(60, int(left() - 420))  # leave time for the template check and teardown
    r = {"budget_seconds": budget,
         "modelscope": ms_download(["*.safetensors", "*.json", "*.jinja", "*.txt"], d, budget)}
    r["bytes"] = du(d)
    r["safetensors"] = sorted(f for f in os.listdir(d) if f.endswith(".safetensors")) if os.path.isdir(d) else []
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        r["index_total_size"] = json.load(open(idx)).get("metadata", {}).get("total_size")
    return r


def template_check():
    probes = sorted(os.path.join(BUNDLE, "probes", f) for f in os.listdir(os.path.join(BUNDLE, "probes")))
    cpu = json.load(open(os.path.join(BUNDLE, "template_token_check.json")))
    r = {}
    for label, d in (("modelscope", os.path.join(WORK, "ms")), ("hf_mirror_pinned", os.path.join(WORK, "hfm"))):
        tpl, tok = os.path.join(d, "chat_template.jinja"), os.path.join(d, "tokenizer.json")
        if not (os.path.exists(tpl) and os.path.exists(tok)):
            r[label] = {"skipped": "tokenizer files missing"}
            continue
        x = sh(f"python3 {BUNDLE}/token_check.py {tpl} {tok} " + " ".join(probes), 300)
        try:
            gpu_rows = json.loads(x["stdout"])
            x["identical_to_cpu_result"] = gpu_rows == cpu
            x["rows"] = len(gpu_rows)
            x["mismatches"] = [g for g, c in zip(gpu_rows, cpu) if g != c]
            x["stdout"] = x["stdout"][:200]
        except Exception as e:
            x["parse_error"] = str(e)
        r[label] = x
    return r


for d in ("/root/autodl-tmp", "/data", "/root"):  # prefer a data disk if the image has one
    if os.path.isdir(d) and os.access(d, os.W_OK):
        WORK = os.path.join(d, "gr")
        break
else:
    WORK = "/tmp/gr"
os.makedirs(WORK, exist_ok=True)

result["started_unix"] = time.time()
step("gpu", gpu)
step("disk", disk)
step("pydeps", pydeps)
step("tokenizer_download", tokenizer_dl)
step("template_check", template_check)
step("weights_download", weights_dl)
step("disk_after", disk)
result["finished_unix"] = time.time()
save()
print("REMOTE_DONE")
