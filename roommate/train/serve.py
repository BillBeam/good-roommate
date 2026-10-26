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


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)
