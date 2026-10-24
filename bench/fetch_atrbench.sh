#!/bin/sh
# 把 ATRBench 克隆到 external/ATRBench（固定提交），并装它的依赖到 external/venv-atr。
# external/ 在 .gitignore 里：基准的代码和数据都不进本仓库。
set -e
cd "$(dirname "$0")/.."
COMMIT=a2cfefec1832804caf143b230de3a6be86c698b7
mkdir -p external
if [ ! -d external/ATRBench ]; then
  git clone -q https://github.com/BUPT-GAMMA/ATRBench external/ATRBench
fi
git -C external/ATRBench checkout -q $COMMIT
if [ ! -d external/venv-atr ]; then
  python3 -m venv external/venv-atr
  external/venv-atr/bin/pip install -q "openai>=1.0.0" "pydantic>=2.0.0" "pyyaml>=6.0" "tenacity>=8.0.0" \
    "docstring-parser>=0.18.0" "rich>=13.0.0" "socksio>=1.0"
fi
echo "ATRBench @ $COMMIT 就绪"
