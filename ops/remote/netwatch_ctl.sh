#!/usr/bin/env bash
# 网络观察（netwatch.sh）的开关：start / stop / restart / status。按 PID 文件管理，
# 不用 pgrep -f / pkill -f（它们会匹配到调用者自己那条命令行，把自己的 ssh 会话杀掉，或误以为已经在跑）。
W=${NETWATCH_W:-/root/work}; PIDF=$W/out/diag/netwatch.pid; SCRIPT=$W/repo/ops/remote/netwatch.sh
mkdir -p $W/out/diag
running() { [ -f $PIDF ] && p=$(cat $PIDF) && [ -n "$p" ] && grep -q netwatch.sh /proc/$p/cmdline 2>/dev/null; }
case "${1:-status}" in
  start)   running || { nohup setsid bash $SCRIPT >/dev/null 2>&1 < /dev/null & echo $! > $PIDF; sleep 1; } ;;
  stop)    running && kill "$(cat $PIDF)"; rm -f $PIDF ;;
  restart) running && kill "$(cat $PIDF)"; rm -f $PIDF; nohup setsid bash $SCRIPT >/dev/null 2>&1 < /dev/null & echo $! > $PIDF; sleep 1 ;;
esac
running && echo "netwatch running" || echo "netwatch NOT running"
