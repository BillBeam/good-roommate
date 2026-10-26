#!/usr/bin/env bash
# 每分钟记一行机器自己看到的网络状况到 out/diag/netwatch.jsonl（开机自动起，见 setup.sh）：
#   ssh_sessions：已建立的 22 端口连接数；sshd_lines：这一分钟 sshd 的日志行数（有人连进来就会有）；
#   fw_drop_pkts：防火墙 DROP/REJECT 规则累计丢弃的包数；out_baidu / out_github：机器往外访问是否正常（HTTP 状态码）。
# 只记计数，不记 IP。SSH 连不上时，这份记录能说明：机器是否还活着、连接有没有到达机器、是不是本机防火墙挡的。
OUT=/root/work/out/diag; mkdir -p "$OUT"
while true; do
  t=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  ssh_n=$(ss -Htn state established '( sport = :22 )' 2>/dev/null | wc -l)
  sshd_n=$(journalctl -u ssh -u sshd --since "-60s" --no-pager -q 2>/dev/null | wc -l)
  drops=$(iptables -L -n -v -x 2>/dev/null | awk '/DROP|REJECT/ && $1 ~ /^[0-9]+$/ {s+=$1} END {print s+0}')
  b=$(curl -s -m 8 -o /dev/null -w "%{http_code}" https://www.baidu.com || echo 000)
  g=$(curl -s -m 8 -o /dev/null -w "%{http_code}" https://github.com || echo 000)
  load=$(cut -d' ' -f1 /proc/loadavg)
  mem=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  echo "{\"t\":\"$t\",\"ssh_sessions\":$ssh_n,\"sshd_lines\":$sshd_n,\"fw_drop_pkts\":$drops,\"out_baidu\":\"$b\",\"out_github\":\"$g\",\"load1\":$load,\"mem_avail_mb\":$mem}" >> "$OUT/netwatch.jsonl"
  sleep 60
done
