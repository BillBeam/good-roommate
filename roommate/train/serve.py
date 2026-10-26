"""启动、停止采样端（vLLM 的 OpenAI 兼容服务）。

- --generation-config vllm：不套用模型自带的默认采样参数，请求里给什么就用什么。
- 只用文字：多模态输入的上限设为 0，不为视觉部分预留显存。
- 开 VLLM_SERVER_DEV_MODE=1 才有 /collective_rpc、/reset_prefix_cache、/sleep 这些接口（只在本机 127.0.0.1 上听）。
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = 8000


def start(model, log_path, gpu_util=0.40, max_model_len=32768, port=PORT, devices=None, sleep_mode=False,
          extra=None, timeout=1200):
    env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", VLLM_ALLOW_INSECURE_SERIALIZATION="1",
               PYTHONPATH=os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
           "--served-model-name", "policy", "--host", "127.0.0.1", "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", str(gpu_util), "--enable-prefix-caching",
           "--generation-config", "vllm", "--seed", "0", "--max-num-seqs", "128",
           "--worker-extension-cls", "roommate.train.vllm_ext.WeightSyncExtension",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if sleep_mode:
        cmd.append("--enable-sleep-mode")
    cmd += list(extra or [])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a")
    f.write(f"\n=== start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {' '.join(cmd)}\n")
    f.flush()
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode}; see {log_path}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return proc, round(time.time() - t0, 1)
        except Exception:
            time.sleep(3)
    proc.kill()
    raise RuntimeError("vLLM did not become healthy")


SERVER_MODULE = "vllm.entrypoints.openai.api_server"


def _ancestors():
    """本进程和它的全部祖先进程。"""
    out, pid = set(), os.getpid()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except OSError:
            break
    return out


def stale_servers(module=SERVER_MODULE):
    """还在跑的采样端进程：按进程自己的参数逐项比（argv 里有 "-m" 紧跟着这个模块名），不按整条命令行做子串匹配，
    并且排除本进程和祖先进程（命令行里恰好写着这个模块名的 shell 不会被当成采样端）。"""
    skip, found = _ancestors(), []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in skip:
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                argv = f.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if any(argv[i] == "-m" and argv[i + 1] == module for i in range(len(argv) - 1)):
            found.append(int(d))
    return found


def kill_stale(module=SERVER_MODULE):
    """结束上一阶段异常退出后留下的采样端（连同它的引擎子进程：它们在同一个进程组里）。返回结束了哪些。"""
    import signal
    killed = stale_servers(module)
    for pid in killed:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return killed


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)
