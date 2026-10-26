#!/usr/bin/env bash
# 环境确认（做自定义镜像之前的关卡）：setup 正常结束；版本和第一台机器（2026-10-26，runs/m2pilot/machine/setup/env.json）一致；
# GPU 上 BF16 能算；模型文件总字节数、分词器和模板的 sha256 和第一台机器一致；vLLM 能载入模型并生成。任何一项不对就以非 0 退出。
set -euo pipefail
W=/root/work
M=$W/models/Qwen3.5-4B
[ "$(cat $W/out/pids/setup.rc 2>/dev/null)" = "0" ] || { echo "setup not finished or failed"; exit 1; }
echo "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715  $M/chat_template.jinja
5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42  $M/tokenizer.json" | sha256sum -c -
[ "$(du -sb $M | cut -f1)" = "9342820790" ] || { echo "model bytes differ: $(du -sb $M | cut -f1)"; exit 1; }
. $W/venv/bin/activate
mkdir -p $W/out/envcheck
python - <<'PY'
import json, time, torch, importlib
want = {"vllm": "0.31.0", "transformers": "5.17.0", "peft": "0.21.2", "fla": "0.6.0", "safetensors": "0.8.0"}
got = {m: importlib.import_module(m).__version__ for m in want}
assert got == want, (got, want)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
err = float(((x @ x).float() - x.float() @ x.float()).abs().max() / (x.float() @ x.float()).abs().max())
t0 = time.time()
from vllm import LLM, SamplingParams
llm = LLM("/root/work/models/Qwen3.5-4B", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.5,
          generation_config="vllm")
out = llm.generate(["<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"], SamplingParams(temperature=0, max_tokens=16))
res = {"versions": got, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "bf16_matmul_rel_err": err,
       "vllm_load_and_generate_s": round(time.time() - t0, 1), "sample": out[0].outputs[0].text[:80]}
json.dump(res, open("/root/work/out/envcheck/envcheck.json", "w"), ensure_ascii=False, indent=1)
print("ENV_OK", json.dumps(res, ensure_ascii=False))
PY
