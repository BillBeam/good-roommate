#!/usr/bin/env bash
# 每分钟记一行机器自己看到的网络状况到 out/diag/netwatch.jsonl（由 netwatch_ctl.sh 起停；setup 和每轮中转开头都会确保它在跑）：
#   ssh_port：sshd 实际监听的端口（这类机器的 sshd 在容器里，端口不一定是 22）；
#   ssh_sessions：这个端口上已建立的连接数；sshd_procs：sshd 进程数（监听进程 + 每条连接各一个）；
#   fw_drop_pkts：本机防火墙 DROP/REJECT 累计丢弃的包数（没有 iptables 时记 -1）；
#   out_baidu / out_github：机器往外访问是否正常（HTTP 状态码）。
# 只记计数，不记 IP。SSH 连不上时，这份记录能说明：机器是否还活着、连接有没有到达机器、是不是本机防火墙挡的。
OUT=${NETWATCH_W:-/root/work}/out/diag; mkdir -p "$OUT"
PORT=$(awk '/^[[:space:]]*Port[[:space:]]/ {print $2; exit}' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/* 2>/dev/null)
PORT=${PORT:-22}
while true; do
  t=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  ssh_n=$(ss -Htn state established "( sport = :$PORT )" 2>/dev/null | wc -l)
  sshd_n=$(pgrep -c -x sshd 2>/dev/null || echo 0)
  if command -v iptables >/dev/null 2>&1; then
    drops=$(iptables -L -n -v -x 2>/dev/null | awk '/DROP|REJECT/ && $1 ~ /^[0-9]+$/ {s+=$1} END {print s+0}')
  else
    drops=-1
  fi
  b=$(curl -s -m 8 -o /dev/null -w "%{http_code}" https://www.baidu.com || true)
  g=$(curl -s -m 8 -o /dev/null -w "%{http_code}" https://github.com || true)
  load=$(cut -d' ' -f1 /proc/loadavg)
  mem=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  echo "{\"t\":\"$t\",\"ssh_port\":$PORT,\"ssh_sessions\":$ssh_n,\"sshd_procs\":$sshd_n,\"fw_drop_pkts\":$drops,\"out_baidu\":\"${b:-000}\",\"out_github\":\"${g:-000}\",\"load1\":$load,\"mem_avail_mb\":$mem}" >> "$OUT/netwatch.jsonl"
  sleep 60
done
