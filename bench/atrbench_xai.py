"""在 ATRBench（Apache-2.0 代码，CC BY 4.0 数据）上用 xAI 模型跑官方评测脚本。

不修改 ATRBench 的任何文件：本脚本在导入 ATRBench 之后，把 xAI 模型登记为一个新的
OpenAI 兼容服务商（地址 https://api.x.ai/v1，密钥读环境变量 XAI_API_KEY），再调用它自己的
runner.pipeline（跑 agent）和 evaluator.pipeline（确定性判分，不调用任何模型）。

只跑 oracle_target 变体：该变体跳过学习会话，不需要模拟用户（ATRBench 默认的模拟用户是
GPT-5.4、路由是 Gemini，本项目边界不允许接入），只考 agent 能否把给定的长期规则用到工具调用上。

ATRBench 的轨迹里含基准数据，所以全部输出写到 external/atrbench_outputs（不进仓库）；
bench/summarize_atrbench.py 只把分数和会话编号抽到 runs/m1/bench/atrbench_summary.json。

用法（ATRBench 由 bench/fetch_atrbench.sh 克隆到 external/，该目录不进仓库）：
    python bench/atrbench_xai.py run  --model grok-4.20-0309-non-reasoning --personas anna_strahan
    python bench/atrbench_xai.py eval --model grok-4.20-0309-non-reasoning --personas anna_strahan
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATR_DIR = os.environ.get("ATR_DIR", os.path.join(ROOT, "external", "ATRBench"))
XAI_MODELS = ["grok-4.20-0309-non-reasoning", "grok-4.3", "grok-4.7"]


def register_xai():
    sys.path.insert(0, ATR_DIR)
    import lib.llm as L

    for m in XAI_MODELS:
        L._MODEL_PROVIDER[m] = "xai"
    orig = L._chat_config

    def chat_config(provider):
        if provider == "xai":
            return os.environ["XAI_API_KEY"], "https://api.x.ai/v1"
        return orig(provider)

    L._chat_config = chat_config

    # 两处适配，只作用于 xAI 模型：
    # 1) xAI 要求 seed 为正数，ATRBench 的种子从 0 开始：seed 统一加 1（确定性映射）；
    # 2) 去掉 reasoning_effort，使用模型自身的默认推理设置（非推理模型不接受这个参数）。
    orig_create = L._create_chat_completion

    def create(client, kwargs):
        if kwargs.get("model") in XAI_MODELS:
            kwargs = dict(kwargs)
            kwargs.pop("reasoning_effort", None)
            if kwargs.get("seed") is not None:
                kwargs["seed"] += 1
        return orig_create(client, kwargs)

    L._create_chat_completion = create


def main():
    mode, rest = sys.argv[1], sys.argv[2:]
    os.chdir(ATR_DIR)
    register_xai()
    common = ["--variants", "oracle_target", "--outputs-root",
              os.environ.get("ATR_OUTPUTS_ROOT", os.path.join(ROOT, "external", "atrbench_outputs"))]
    args = []
    i = 0
    while i < len(rest):
        if rest[i] == "--model":
            args += ["--models", rest[i + 1]]
            i += 2
        else:
            args.append(rest[i])
            i += 1
    if mode == "run":
        from runner import pipeline
        sys.argv = ["runner.pipeline", *common, *args, "--reasoning", "off", "--no-progress"]
    else:
        from evaluator import pipeline
        sys.argv = ["evaluator.pipeline", *common, *args]
    pipeline.main()


if __name__ == "__main__":
    main()
