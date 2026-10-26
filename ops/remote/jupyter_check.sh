#!/usr/bin/env bash
# JupyterLab 自查（机器侧）：谁起的、以谁的身份、监听在哪、本机不带口令能不能读写、关掉它会不会连带停掉容器。
# 输出里的 IPv4 一律打码；地址只报类别（回环 / 内网 / 公网）。
mask() { sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<ip>/g'; }
cls() { case "$1" in 127.*) echo loopback;; 10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*|100.6[4-9].*|100.[7-9][0-9].*|100.1[01][0-9].*|100.12[0-7].*) echo private;; 0.0.0.0) echo all-interfaces;; *) echo public;; esac; }
{
echo "== pid 1"; ps -o pid=,user=,args= -p 1 | cut -c1-200
JP=$(pgrep -o -f "[j]upyter-lab")
echo "== jupyter pid=$JP"
[ -n "$JP" ] && ps -o pid=,ppid=,user=,etime=,args= -p "$JP" | sed -E 's/(token|password)=[^ ]*/\1=<value>/Ig' | cut -c1-300
P=$JP; echo "== parent chain"
while [ -n "$P" ] && [ "$P" != "0" ] && [ "$P" != "1" ]; do P=$(ps -o ppid= -p "$P" | tr -d ' '); [ -n "$P" ] && ps -o pid=,args= -p "$P" | cut -c1-160; done
echo "== bind address class"
BIND=$(ps -o args= -p "$JP" 2>/dev/null | grep -oE -- '--(ServerApp\.)?ip=[^ ]+' | head -1 | cut -d= -f2)
echo "arg ip: ${BIND:-<none>} -> $(cls "${BIND:-0.0.0.0}")"
echo "== listening on 8888"; ss -ltnp 2>/dev/null | grep -E ':8888\b' | awk '{print $4}' | while read a; do h=${a%:*}; echo "$a -> $(cls "${h//[\[\]]/}")"; done
echo "== this machine's IPv4 classes"; ip -4 -o addr 2>/dev/null | awk '{print $2, $4}' | while read i a; do echo "$i $(cls "${a%/*}")"; done
echo "== unauthenticated access from inside"
H=${BIND:-127.0.0.1}; [ "$H" = "0.0.0.0" ] && H=127.0.0.1
for path in /api/status /api/contents /lab; do
  code=$(curl -s -m 5 -o /tmp/jc.out -w "%{http_code}" "http://$H:8888$path"); echo "$path -> $code $(head -c 80 /tmp/jc.out | tr '\n' ' ')"
done
echo "== config files"; ls -la /root/.jupyter 2>/dev/null; grep -rhiE "token|password|allow_origin|ip *=" /root/.jupyter /etc/jupyter /usr/local/etc/jupyter 2>/dev/null | sed -E 's/(=|:) *["'"'"']?[^"'"'"' ,]+/\1 <value>/' | head
echo "== who starts it (files on disk mentioning jupyter-lab)"
grep -rlE "jupyter[- ]lab" /etc /usr/local/bin /opt /root/*.sh /init* /entrypoint* /start* 2>/dev/null | grep -v -E "site-packages|share/jupyter" | head -20
} 2>&1 | mask
