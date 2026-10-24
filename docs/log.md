# 工作日志

最新的在最上面。每条固定六项：做了什么、一句话结果、关键数字（附日志或文件链接）、没解决的问题、下一步、花费。

---

## 2026-10-24 · M1 造世界，测基线（第一版）

- **做了什么**：搭好第一版模拟环境：一批模拟用户各自过好几个月的"生活"，每隔一两周找 agent 办一件事（订晚餐、找午饭、订酒店，或者算账换算）。用户的真实偏好由程序记着，agent 看不到；用户会顺口提到搬家、健身、添宝宝这类事，偏好随之悄悄改变，但从不明说"我变了"；也会说"这次嗓子疼别吃辣"这种只管一次的话。agent 能读写记忆、能问用户（每问一次扣分）。对错由程序按用户当时的真实状态判，不用任何模型打分。之后用三个 xAI 现成模型测了基线，又在一个外部基准（ATRBench）上跑通了它的官方评测脚本。全部用 `python -m roommate.run` 一条命令跑完，报告见 [reports/m1-baseline.md](../reports/m1-baseline.md)。
- **一句话结果**：环境搭好了、判分自检全部通过，但环境过于简单：最强的现成模型不经训练就拿到满分标准的 81%，超过了 80% 的闸门，M1 还不能算过关。
- **关键数字**：
  - 规模：[24](../runs/m1/world_stats.json) 个模拟用户，每人 [8](../runs/m1/world_stats.json) 次会话；其中一个事件同时改动多项偏好 [29](../runs/m1/world_stats.json) 次，临时情况 [24](../runs/m1/world_stats.json) 次，与偏好无关的会话 [24](../runs/m1/world_stats.json) 次。
  - 判分自检：知道真实状态的标准策略得 [1.000](../runs/m1/policies/oracle/summary.json) 分（满分），什么都不做得 [0.000](../runs/m1/policies/nothing/summary.json) 分；全部 [10 项检查](../runs/m1/selfcheck.json)通过。
  - 两种捷径：每次都把问题问一遍，得 [0.674](../runs/m1/policies/ask_all/summary.json) 分，平均每次问 [2.17](../runs/m1/policies/ask_all/summary.json) 次；不做个性化的保守默认，得 [0.312](../runs/m1/policies/conservative/summary.json) 分。
  - 三个现成模型的得分：grok-4.20 [0.416](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json)，grok-4.3 [0.570](../runs/m1/policies/llm_grok-4.3/summary.json)，grok-4.7 [0.813](../runs/m1/policies/llm_grok-4.7/summary.json)。grok-4.7 的后续任务成功率 [91.7%](../runs/m1/policies/llm_grok-4.7/summary.json)，因用了过时记忆而选错的比例 [6.3%](../runs/m1/policies/llm_grok-4.7/summary.json)。
  - 弱模型输在不写记忆：grok-4.3 在全部会话里写记忆的比例是 [0.0%](../runs/m1/policies/llm_grok-4.3/sessions.jsonl)，grok-4.7 是 [49.5%](../runs/m1/policies/llm_grok-4.7/sessions.jsonl)。
  - 复现：只用缓存重跑，[10 / 10](../runs/m1/repro_check.json) 个策略逐字一致；在会话中途存档再恢复，[3 / 3](../runs/m1/resume_check.json) 个模型的结果与不中断时一致。
  - 外部基准 ATRBench：1 个人设、[11](../runs/m1/bench/atrbench_summary.json) 个测试会话，grok-4.20 通过 [10](../runs/m1/bench/atrbench_summary.json) 个。
- **没解决的问题**：
  - 环境过于简单。强模型只要老老实实记笔记，就能接住大部分变化。
  - 只跑了一个随机种子，各模型之间的差距有多少是运气，还不知道。
  - 模拟用户把"问题问的是哪方面"这一步交给了一个模型判断，人工抽检的准确率是 [96.7%](../runs/m1/classifier_eval.json)，不是百分之百。
  - ATRBench 只能跑不需要模拟用户的那一种变体（它默认的模拟用户来自其他服务商）；HorizonBench 的数据放在 Hugging Face 上，本环境连不上。
- **下一步**：给环境加难，再测一遍基线，直到最强模型回到 80% 以下。候选办法：
  - 变化说得更隐晦。
  - 会话跨度更长、事件更多，让一项偏好被改两次。
  - 限制记忆条数，逼 agent 取舍。
  - 让事件之间相互影响。
  - 补跑多个随机种子。
- **花费**：约 [5.21 美元](../reports/m1-baseline.md#9-花费)（上限 30 美元）。其中三个模型的正式基线 [4.37 美元](../runs/m1/costs.json)，其余是试跑、模拟用户和 ATRBench 评测。

---

## 2026-10-24 · M0 查证

- **做了什么**：两轮文献调研与可行性评估。
- **一句话结果**：问题整体没人做过，但最接近的几篇都在最近两周发出，可抢先的时间在变短。
- **关键数字**：
  - 最接近的 15 项工作里，没有一项同时做到"RL 训练、奖励跨会话回传、偏好隐式变化"（[research/02 第 5 节](../research/02-literature-cross-session.md)）。
  - 最小实验单次训练估算约 150–215 美元，10 组对照约 1.5–2.2 千美元；用量是假设，不是实测（[research/02 第 8 节](../research/02-literature-cross-session.md)）。
- **没解决的问题**：
  - "反事实回放"这一候选训练方法的新颖性还没有专门查证。
  - 训练吞吐（每步耗时）没有公开数字。
- **下一步**：M1 造世界，测基线（不训练）。
- **花费**：0 美元。
