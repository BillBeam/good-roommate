"""M2 试跑：在租来的显卡机器上把 Qwen3.5-4B（开思考）的训练流程跑通一次。

模块：
- teacher：老师轨迹（DeepSeek V4 Pro 在新生成的世界里跑出的对话），在中转的运行器上调接口。
- qwen_chat：Qwen3.5 聊天模板、逐 token 拼接、XML 工具调用的解析。
- shortrange：短程考场（每条轨迹只含连续 2 次见面）。
- metrics：分数、提问数、思考长度、格式错误、两个捷径指标、按用户重抽样的置信区间。
- sampler / agent_local：采样端（vLLM 服务）与本地 agent。
- trainer / sft / grpo：训练端（HF transformers + PEFT LoRA）。
- selfcheck / evaluate：训练前自检、三个时点的评测。
"""
