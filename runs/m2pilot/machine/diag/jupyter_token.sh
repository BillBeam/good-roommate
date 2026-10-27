#!/usr/bin/env bash
# Jupyter 口令从哪来（写死在镜像文件里 / 环境变量 / 每台机器生成）、多长；supervisor 管的其他程序和它们监听的端口。
# 口令本身不打印（只打印来源、长度、sha256 前 8 位，用来和别的机器/镜像比较是不是同一个）。IPv4 打码。
mask() { sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<ip>/g'; }
{
C=/etc/supervisor/conf.d/jupyterlab.conf
echo "== $C (token masked)"; sed -E 's/(token|password)=[^ ]*/\1=<masked>/Ig' $C
JP=$(pgrep -o -f "[j]upyter-lab --")
TOK=$(tr '\0' '\n' < /proc/$JP/cmdline 2>/dev/null | grep -oE 'ServerApp\.token=.*' | head -1 | cut -d= -f2-)
echo "== running token: length=${#TOK} sha8=$(printf %s "$TOK" | sha256sum | cut -c1-8)"
if grep -q "token=%(ENV_" $C; then echo "token source: environment variable via supervisor"; \
elif grep -qE "token=[^ ]+" $C; then echo "token source: literal in $C (same for every machine from this image)"; \
else echo "token source: not in $C"; fi
echo "== env vars mentioning TOKEN/JUPYTER (names only)"; tr '\0' '\n' < /proc/$JP/environ 2>/dev/null | cut -d= -f1 | grep -iE "token|jupyter|pass" 
echo "== supervisor programs"; supervisorctl status 2>&1 | head -20
for f in /etc/supervisor/conf.d/*.conf; do echo "-- $f"; grep -E "^\[program|^command|^autostart|^autorestart" $f | sed -E 's/(token|password|passwd|pwd)[=: ][^ ]*/\1=<masked>/Ig' | cut -c1-220; done
echo "== listening TCP ports and owners"; ss -ltnp 2>/dev/null | awk 'NR>1 {print $4, $6}' | sed -E 's/users:\(\("([^"]+)".*/\1/' | sort -u
} 2>&1 | mask
