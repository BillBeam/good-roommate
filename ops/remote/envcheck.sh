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
python $W/repo/ops/remote/envcheck.py
