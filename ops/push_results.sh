#!/usr/bin/env bash
# Copy $OUT into $DEST/$GITHUB_RUN_ID on the chain-results branch and push (retries on races).
set -euo pipefail
[ -d "$OUT" ] || { echo "no output dir"; exit 0; }
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for i in 1 2 3 4; do
  rm -rf /tmp/cr && git clone -q --depth=1 --branch chain-results "$(git remote get-url origin)" /tmp/cr 2>/dev/null \
    || { mkdir -p /tmp/cr && git -C /tmp/cr init -q -b chain-results && git -C /tmp/cr remote add origin "$(git remote get-url origin)"; }
  git -C /tmp/cr config user.name "github-actions[bot]"
  git -C /tmp/cr config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git -C /tmp/cr config http.https://github.com/.extraheader "$(git config --get http.https://github.com/.extraheader)"
  mkdir -p "/tmp/cr/$DEST/$GITHUB_RUN_ID" && cp -r "$OUT"/. "/tmp/cr/$DEST/$GITHUB_RUN_ID"/
  git -C /tmp/cr add "$DEST/$GITHUB_RUN_ID"
  git -C /tmp/cr commit -q -m "results for run $GITHUB_RUN_ID" || exit 0
  git -C /tmp/cr push -q origin chain-results && exit 0
  sleep $((i * 5))
done
exit 1
