#!/usr/bin/env bash
# 在训练机器上装环境、下载模型。由中转以后台任务运行，日志在 /root/work/out/logs/setup.log。
# 计时写进 /root/work/out/setup/timing.json，版本写进 env.json（都会同步回仓库）。
set -euo pipefail
W=/root/work
OUT=$W/out/setup
mkdir -p $W/models/Qwen3.5-4B $OUT
T0=$(date +%s)
mark() { echo "{\"step\": \"$1\", \"seconds\": $(( $(date +%s) - T0 ))}" >> $OUT/timing.jsonl; }
MIRROR=${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
# 版本钉在第一台机器（2026-10-26）实际装上的版本，换机器后环境相同
VLLM=${VLLM_VERSION:-vllm==0.31.0}

# 网络观察（每分钟一行，见 netwatch.sh）和开机时的安全软件、防火墙、sshd 清单
mkdir -p $W/out/diag
bash $W/repo/ops/remote/netwatch_ctl.sh start
bash $W/repo/ops/remote/diag.sh > $W/out/diag/diag_setup.txt 2>&1 || true

nvidia-smi > $OUT/nvidia-smi.txt
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > $OUT/gpu.csv
df -h / > $OUT/df_before.txt
free -g > $OUT/free.txt
nproc > $OUT/nproc.txt

PY=$(command -v python3.12 || command -v python3)
if [ ! -x $W/venv/bin/python ]; then
  $PY -m venv $W/venv 2>/dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv python3-pip >/dev/null && $PY -m venv $W/venv; }
fi
. $W/venv/bin/activate
python -m pip install -q -U pip -i $MIRROR
mark venv
python -m pip install -q -i $MIRROR "$VLLM"
mark vllm
python -m pip install -q -i $MIRROR transformers==5.17.0 peft==0.21.2 flash-linear-attention==0.6.0 safetensors==0.8.0 \
  jinja2 matplotlib
mark extras

# 模型：ModelScope（国内镜像），逐个文件 curl；分片按 index 下载
M=$W/models/Qwen3.5-4B
BASE=https://www.modelscope.cn/models/Qwen/Qwen3.5-4B/resolve/master
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja vocab.json merges.txt \
         model.safetensors.index.json preprocessor_config.json video_preprocessor_config.json generation_config.json; do
  if [ ! -s $M/$f ]; then
    curl -sSL --fail -m 600 -o $M/$f.part "$BASE/$f" && mv $M/$f.part $M/$f || rm -f $M/$f.part
  fi
done
mark model_small_files
for f in $(python -c "import json;print(' '.join(sorted(set(json.load(open('$M/model.safetensors.index.json'))['weight_map'].values()))))"); do
  if [ ! -s $M/$f ]; then curl -sSL --fail -m 3600 -o $M/$f.part "$BASE/$f" && mv $M/$f.part $M/$f; fi
done
mark model_weights
sha256sum $M/chat_template.jinja $M/tokenizer.json > $OUT/model_sha256.txt
du -sb $M > $OUT/model_bytes.txt
python - <<'PY' > $OUT/env.json
import json, importlib, sys, torch
v = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
     "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
     "bf16": torch.cuda.is_bf16_supported()}
for m in ("vllm", "transformers", "peft", "fla", "safetensors"):
    try:
        v[m] = importlib.import_module(m).__version__
    except Exception as e:
        v[m] = f"missing: {e}"
print(json.dumps(v, indent=1))
PY
df -h / > $OUT/df_after.txt
mark done
cat $OUT/env.json
