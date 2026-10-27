#!/usr/bin/env bash
# Copy $OUT into $DEST/$SRC on the m2pilot-data branch and push (retries on races).
set -euo pipefail
[ -d "$OUT" ] || { echo "no output dir"; exit 0; }
url="$(git remote get-url origin)"
hdr="$(git config --get http.https://github.com/.extraheader || true)"
for i in 1 2 3 4 5; do
  rm -rf /tmp/pd
  git -c "http.https://github.com/.extraheader=$hdr" clone -q --depth=1 --branch m2pilot-data "$url" /tmp/pd 2>/dev/null \
    || { mkdir -p /tmp/pd && git -C /tmp/pd init -q -b m2pilot-data && git -C /tmp/pd remote add origin "$url"; }
  git -C /tmp/pd config user.name "github-actions[bot]"
  git -C /tmp/pd config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git -C /tmp/pd config http.https://github.com/.extraheader "$hdr"
  mkdir -p "/tmp/pd/$DEST/$SRC" && cp -r "$OUT"/. "/tmp/pd/$DEST/$SRC"/
  git -C /tmp/pd add -A
  git -C /tmp/pd diff --cached --quiet && exit 0
  git -C /tmp/pd commit -q -m "teacher data from run $GITHUB_RUN_ID"
  git -C /tmp/pd push -q origin m2pilot-data && exit 0
  sleep $((i * 5))
done
exit 1
