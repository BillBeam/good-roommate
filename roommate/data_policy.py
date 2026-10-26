"""数据用途：Claude 的输出只用于评测和分析，不进任何训练数据（SFT、RL、蒸馏都不行）。

- 存放 Claude 输出的目录里放一个 DATA_POLICY.json（for_training = false）：runs/m1v2/claude/ 和每个 llm_claude-* 记录目录。
- 请求缓存 runs/m1v2/llm_cache/claude-*.jsonl 按文件名识别。
- 以后任何读入训练数据（老师轨迹、SFT 样本、RL 回放）的代码，读每个文件之前都要先调 assert_trainable(path)。
"""

import json
import os

MARKER = "DATA_POLICY.json"
POLICY = {"for_training": False,
          "reason": "Claude 的输出只用于评测和分析，不进任何训练数据（SFT、RL、蒸馏都不行）。",
          "applies_to": "本目录及其子目录下的全部文件"}


class NotForTraining(RuntimeError):
    pass


def mark_not_for_training(directory):
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, MARKER)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(POLICY, f, ensure_ascii=False, indent=1)


def trainable(path):
    """path 能不能进训练数据：它自己或任何一层上级目录有 for_training = false 的标记，或者是 Claude 的请求缓存，就不能。"""
    p = os.path.abspath(path)
    if os.path.basename(p).startswith("claude-") or "llm_claude-" in p.replace(os.sep, "/"):
        return False
    d = p if os.path.isdir(p) else os.path.dirname(p)
    while True:
        m = os.path.join(d, MARKER)
        if os.path.exists(m):
            with open(m, encoding="utf-8") as f:
                if not json.load(f).get("for_training", True):
                    return False
        up = os.path.dirname(d)
        if up == d:
            return True
        d = up


def assert_trainable(path):
    if not trainable(path):
        raise NotForTraining(f"{path} 含 Claude 的输出，只用于评测和分析，不能进训练数据")
