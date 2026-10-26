#!/usr/bin/env bash
# 机器侧诊断：安全软件、防火墙、sshd 设置、上一次开机在断连那段时间的日志。
# 用法：diag.sh [开始时间 结束时间]（UTC，如 "2026-10-26 14:20" "2026-10-26 16:30"）
# 输出里的 IPv4 地址一律打码（实例和网关的 IP 不进仓库和 PR 评论）。
mask() { sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<ip>/g'; }
SINCE=${1:-}; UNTIL=${2:-}
{
echo "== uptime"; uptime
echo "== boots"; journalctl --list-boots --no-pager 2>/dev/null | tail -5
echo "== journal persistent"; ls -d /var/log/journal 2>/dev/null || echo "no (volatile)"
echo "== running services"; systemctl list-units --type=service --state=running --no-pager --plain --no-legend 2>/dev/null | awk '{print $1}'
echo "== security-ish processes"
ps -eo comm,args --no-headers | grep -Ei "fail2ban|denyhost|sshguard|guard|secur|hids|aegis|yundun|qcloud|ucloud|wazuh|ossec|crowdsec|firewalld|ufw|agent" | grep -v grep | cut -c1-160
echo "== iptables -S"; (iptables -S 2>/dev/null || echo "no iptables") | head -60
echo "== iptables drops"; (iptables -L -n -v 2>/dev/null | grep -Ei "drop|reject" | head -20)
echo "== nft"; (nft list ruleset 2>/dev/null || echo "no nft") | head -60
echo "== hosts.deny"; tail -5 /etc/hosts.deny 2>/dev/null
echo "== sshd settings"
grep -hEi "^\s*(Port|MaxStartups|MaxSessions|MaxAuthTries|ClientAlive|AllowUsers|DenyUsers|PasswordAuthentication|PerSource)" \
  /etc/ssh/sshd_config /etc/ssh/sshd_config.d/* 2>/dev/null
echo "== ssh version"; ssh -V 2>&1
if [ -n "$SINCE" ]; then
  echo "== previous boot: sshd / network / oom ($SINCE .. $UNTIL)"
  journalctl -b -1 --no-pager -o short-iso --since "$SINCE" --until "$UNTIL" 2>/dev/null \
    | grep -Ei "sshd|oom|killed|link|eth|network|drop|block|penalt|panic|hung|fail2ban|ban" | tail -80
  echo "== previous boot: last lines"; journalctl -b -1 --no-pager -o short-iso 2>/dev/null | tail -15
fi
echo "== auth log tail"; (cat /var/log/auth.log /var/log/secure 2>/dev/null | tail -40)
echo "== netwatch tail"; tail -5 /root/work/out/diag/netwatch.jsonl 2>/dev/null
echo "== mem"; free -m; echo "== disk"; df -h / 2>/dev/null
} 2>&1 | mask
