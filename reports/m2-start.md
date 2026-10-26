# M2 起点：未训练的 Qwen3.5-4B 在第二版考场里的得分，以及训练温度下的组内差异

起点低于 0.75，小模型还有东西可学。

日志目录：[runs/m2/](../runs/m2)。数字由 `python -m roommate.m2 --stages stats,report` 从日志生成，链接指向生成该数字的文件。考场（世界、规则、判分）是 M1 第二版原样，种子 1、2、3；分数只算和偏好有关的会话；区间是按用户重抽样（每个世界内分别重抽，5000 次）的 95% 区间。Qwen3.5-4B 经硅基流动调用（免费），DeepSeek 两行和两个脚本策略取自 M1 第二版的日志，没有重跑；DeepSeek 两行开着思考，实际是默认采样、不是温度 0（第 10 节）。

## 中期结论（只用已跑完的子集）

**已完成的范围**（下面每个数字都来自这个子集，不是全量）：

- 思考关、温度 0：种子 1、2、3，[72](../runs/m2/interim/interim_stats.json) 个用户，每个用户的前几次会话（种子 1 每人约 5.2 次、种子 2 每人约 6.2 次、种子 3 每人约 5.2 次），共 [398](../runs/m2/interim/interim_stats.json) 次与偏好有关的会话。
- 思考开、温度 0：种子 1、2、3，[71](../runs/m2/interim/interim_stats.json) 个用户，每个用户的前几次会话（种子 1 每人约 1.0 次、种子 2 每人约 1.1 次、种子 3 每人约 1.2 次），共 [78](../runs/m2/interim/interim_stats.json) 次与偏好有关的会话。
- "知道真实状态"对照组，思考关：种子 1，[24](../runs/m2/interim/interim_stats.json) 次与偏好有关的会话。
- "知道真实状态"对照组，思考开：种子 1，[22](../runs/m2/interim/interim_stats.json) 次与偏好有关的会话。
- 组内差异：思考关、温度 1，种子 1。计划是 24 个用户 × 前 3 次会话 × 4 遍；运行被打断时只跑完了第一次会话（第 1 遍 22、第 2 遍 15、第 3 遍 11、第 4 遍 12 个用户）。至少 2 遍跑完的用户 [21](../runs/m2/interim/interim_stats.json) 个、共 [60](../runs/m2/interim/interim_stats.json) 次第一次会话；4 遍都跑完的只有 [3](../runs/m2/interim/interim_stats.json) 个用户。

子集是"每个用户的前 k 次会话"（全量运行按会话序号一轮一轮推进，从缓存重建时每个用户在第一次缺缓存的会话处截断）。前几次会话要先问，分数天然偏低，所以每个比较都在同一批会话上做：DeepSeek V4 Pro、理想记忆策略、记住但不更新也只取这些会话（M1 第二版的完整记录）。区间按用户重抽样。数据：[interim_stats.json](../runs/m2/interim/interim_stats.json)，生成脚本 `python -m roommate.m2_interim`。

**问题一：起点分数是否低于 0.75？**

| 温度 0 | 子集会话数 | Qwen3.5-4B 偏好会话得分 | 95% 区间 | 同一批会话上 V4 Pro | 理想记忆策略 | 记住但不更新 | 剩余空间比例 | 低于 0.75 的重抽样比例 |
|---|---|---|---|---|---|---|---|---|
| 思考关 | [398](../runs/m2/interim/interim_stats.json) | [-1.065](../runs/m2/interim/interim_stats.json) | [-1.158–-0.969](../runs/m2/interim/interim_stats.json) | [0.831](../runs/m2/interim/interim_stats.json) | [0.924](../runs/m2/interim/interim_stats.json) | [0.417](../runs/m2/interim/interim_stats.json) | [-292.2%](../runs/m2/interim/interim_stats.json)（[-322.4%–-263.9%](../runs/m2/interim/interim_stats.json)） | [100.0%](../runs/m2/interim/interim_stats.json) |
| 思考开 | [78](../runs/m2/interim/interim_stats.json) | [0.519](../runs/m2/interim/interim_stats.json) | [0.420–0.601](../runs/m2/interim/interim_stats.json) | [0.648](../runs/m2/interim/interim_stats.json) | [0.760](../runs/m2/interim/interim_stats.json) | [0.419](../runs/m2/interim/interim_stats.json) | [29.4%](../runs/m2/interim/interim_stats.json)（[-0.6%–53.7%](../runs/m2/interim/interim_stats.json)） | [100.0%](../runs/m2/interim/interim_stats.json) |

丢分去向（同一子集）：

| 温度 0 | 选错的会话 | └ 没选出任何选项 | └ 其中用满 14 轮 | └ 选了错的 | 每次会话提问扣分 | 兜圈子的会话占比 | 每次会话调用 / 输入 / 输出 tokens |
|---|---|---|---|---|---|---|---|
| 思考关 | [322](../runs/m2/interim/interim_stats.json) / [398](../runs/m2/interim/interim_stats.json) | [282](../runs/m2/interim/interim_stats.json) | [282](../runs/m2/interim/interim_stats.json) | [40](../runs/m2/interim/interim_stats.json) | [1.256](../runs/m2/interim/interim_stats.json) | [79.1%](../runs/m2/interim/interim_stats.json)（重复问 [256](../runs/m2/interim/interim_stats.json)、14 轮 [282](../runs/m2/interim/interim_stats.json)、空白填充 [44](../runs/m2/interim/interim_stats.json)） | [10.27](../runs/m2/interim/interim_stats.json) / [21719](../runs/m2/interim/interim_stats.json) / [1414](../runs/m2/interim/interim_stats.json) |
| 思考开 | [9](../runs/m2/interim/interim_stats.json) / [78](../runs/m2/interim/interim_stats.json) | [1](../runs/m2/interim/interim_stats.json) | [1](../runs/m2/interim/interim_stats.json) | [8](../runs/m2/interim/interim_stats.json) | [0.365](../runs/m2/interim/interim_stats.json) | [3.8%](../runs/m2/interim/interim_stats.json)（重复问 [3](../runs/m2/interim/interim_stats.json)、14 轮 [1](../runs/m2/interim/interim_stats.json)、空白填充 [0](../runs/m2/interim/interim_stats.json)） | [3.47](../runs/m2/interim/interim_stats.json) / [6671](../runs/m2/interim/interim_stats.json) / [1479](../runs/m2/interim/interim_stats.json) |
| 对照组·思考关（种子 1） | 选对率 [25.0%](../runs/m2/interim/interim_stats.json) | [17](../runs/m2/interim/interim_stats.json) | [17](../runs/m2/interim/interim_stats.json) | — | — | — | — |
| 对照组·思考开（种子 1） | 选对率 [100.0%](../runs/m2/interim/interim_stats.json) | [0](../runs/m2/interim/interim_stats.json) | [0](../runs/m2/interim/interim_stats.json) | — | — | — | — |

**问题二：温度大于 0 时，同一用户几遍之间的分数有没有差异？**

- **有差异的用户占比**：第一次会话（各遍 prompt 完全相同）在至少 2 遍里跑完的用户 [21](../runs/m2/interim/interim_stats.json) 个，其中几遍分数不全相同的 [17](../runs/m2/interim/interim_stats.json) 个（[81.0%](../runs/m2/interim/interim_stats.json)），选对与否有对有错的 [8](../runs/m2/interim/interim_stats.json) 个；跑完的 [60](../runs/m2/interim/interim_stats.json) 次第一次会话里，用满 14 轮的 [0](../runs/m2/interim/interim_stats.json) 次。
- 注意：组内差异子集只到第一次会话，而且没跑完就被打断（见下面"运行中断"）；跑得慢的会话（兜圈子的）更可能没跑完，所以子集偏向短会话，用满 14 轮的比例会被低估。第一次会话各遍的 prompt 完全相同，比较是干净的。
- 严格口径（4 遍都跑完，只有 [3](../runs/m2/interim/interim_stats.json) 个用户，样本太小，只作参考）：几遍分数不全相同的 [3](../runs/m2/interim/interim_stats.json) / [3](../runs/m2/interim/interim_stats.json)。
- 用户总分的组内标准差：中位数 [0.446](../runs/m2/interim/interim_stats.json)，10% / 90% 分位 [0.393](../runs/m2/interim/interim_stats.json) / [0.503](../runs/m2/interim/interim_stats.json)。
- 按（用户、会话）看：[0](../runs/m2/interim/interim_stats.json) / [3](../runs/m2/interim/interim_stats.json) 组 [4](../runs/m2/interim/interim_stats.json) 遍分数完全相同（[0.0%](../runs/m2/interim/interim_stats.json)；其中每遍都选对且提问数相同 [0](../runs/m2/interim/interim_stats.json) 组），选对与否有对有错的 [3](../runs/m2/interim/interim_stats.json) 组；第一次会话（[4](../runs/m2/interim/interim_stats.json) 遍 prompt 完全相同）分数完全相同的 [0](../runs/m2/interim/interim_stats.json) / [3](../runs/m2/interim/interim_stats.json)。
- 这 [4](../runs/m2/interim/interim_stats.json) 遍的平均得分 [0.246](../runs/m2/interim/interim_stats.json)；用满 14 轮没选 [0](../runs/m2/interim/interim_stats.json) 次。


**结论属于哪一种**：

- 思考关：**起点低，且主要丢在格式和兜圈子**。得分 [-1.065](../runs/m2/interim/interim_stats.json)（区间上界 [-0.969](../runs/m2/interim/interim_stats.json)，低于 0.75）；选错的 [322](../runs/m2/interim/interim_stats.json) 次里 [282](../runs/m2/interim/interim_stats.json) 次没选出任何选项（[87.6%](../runs/m2/interim/interim_stats.json)），全部是用满 14 轮；另外每次会话被重复提问扣 [1.256](../runs/m2/interim/interim_stats.json) 分。
- 思考开：**起点中等**（这一模式在训练温度下的组内差异还没测；思考关上测到有差异）（证据只到前两次会话，跨会话记忆还没被检验）。得分 [0.519](../runs/m2/interim/interim_stats.json)（区间上界 [0.601](../runs/m2/interim/interim_stats.json)，低于 0.75），同一批会话上 V4 Pro [0.648](../runs/m2/interim/interim_stats.json)；丢分主要在提问（每次会话 [0.365](../runs/m2/interim/interim_stats.json)）和选错（[8](../runs/m2/interim/interim_stats.json) 次），没选出选项只有 [1](../runs/m2/interim/interim_stats.json) 次。
- 两种模式都不属于"起点 ≥ 0.75，或组内几乎没有差异"：温度 1 时同一用户几遍之间的分数有差异的占 [81.0%](../runs/m2/interim/interim_stats.json)（测的是思考关；思考开在训练温度下的组内差异还没测）。所以小模型还有东西可学，GRPO 在组内有信号可用；思考关要先解决兜圈子（SFT 的任务），思考开的起点更高、丢分在提问和挑选。训练用哪种模式仍待定（附录第 4 节）。

**运行中断**：2026-10-25 约 14:01（UTC）起，本环境的出口代理按网络策略拒绝连接 `api.siliconflow.cn`（CONNECT 返回 403），之后没有新的请求成功；按规则没有重试、没有换渠道。已完成的请求全部在缓存里，网络恢复后用 `python -m roommate.m2 --stages runs` 补跑，已跑过的部分从缓存逐字重放，结果只更新附录。经过见 [restarts.txt](../runs/m2/logs/restarts.txt)。

---

# 附录：全量运行（后台补跑中，跑完一组更新一组）

以下各节由全量运行的结果生成。中期结论不依赖这里的数字；某一组还没跑完时，相应的行或表是空的。

## 全量结论

- 训练用哪种思考模式待定，这一步不下结论：两种模式的实测对照见第 4 节，第二步在自部署的 BF16 服务上确认模板如何处理思考、两种模式各测一遍后再定；费用估算两种模式各占一行（第 7 节）。

## 1. 得分

| 策略 | 偏好会话得分 | 95% 区间 | 种子 1 / 2 / 3 | 选对率 | 每次会话提问数 | 剩余空间比例 | 剩余空间 95% 区间 | 日志 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4 Pro（思考开，默认采样，top_p 0.95） | [0.833](../runs/m2/stats.json) | [0.804–0.859](../runs/m2/stats.json) | [0.829](../runs/m2/stats.json) / [0.866](../runs/m2/stats.json) / [0.804](../runs/m2/stats.json) | [90.7%](../runs/m2/stats.json) | [0.49](../runs/m2/stats.json) | [79.4%](../runs/m2/stats.json) | [73.6%–84.6%](../runs/m2/stats.json) | [1](../runs/m1v2/seed1/policies/llm_deepseek-v4-pro/sessions.jsonl) [2](../runs/m1v2/seed2/policies/llm_deepseek-v4-pro/sessions.jsonl) [3](../runs/m1v2/seed3/policies/llm_deepseek-v4-pro/sessions.jsonl) |
| DeepSeek Flash（思考开，默认采样，top_p 1.0） | [0.819](../runs/m2/stats.json) | [0.786–0.848](../runs/m2/stats.json) | [0.823](../runs/m2/stats.json) / [0.852](../runs/m2/stats.json) / [0.780](../runs/m2/stats.json) | [89.1%](../runs/m2/stats.json) | [0.48](../runs/m2/stats.json) | [76.5%](../runs/m2/stats.json) | [70.2%–82.4%](../runs/m2/stats.json) | [1](../runs/m1v2/seed1/policies/llm_deepseek-flash/sessions.jsonl) [2](../runs/m1v2/seed2/policies/llm_deepseek-flash/sessions.jsonl) [3](../runs/m1v2/seed3/policies/llm_deepseek-flash/sessions.jsonl) |
| 理想记忆策略 | [0.936](../runs/m2/stats.json) | [0.932–0.940](../runs/m2/stats.json) | [0.938](../runs/m2/stats.json) / [0.937](../runs/m2/stats.json) / [0.935](../runs/m2/stats.json) | [100.0%](../runs/m2/stats.json) | [0.42](../runs/m2/stats.json) | —（参照） | — | [1](../runs/m1v2/seed1/policies/ideal/sessions.jsonl) [2](../runs/m1v2/seed2/policies/ideal/sessions.jsonl) [3](../runs/m1v2/seed3/policies/ideal/sessions.jsonl) |
| 记住但不更新 | [0.435](../runs/m2/stats.json) | [0.405–0.464](../runs/m2/stats.json) | [0.431](../runs/m2/stats.json) / [0.437](../runs/m2/stats.json) / [0.437](../runs/m2/stats.json) | [56.7%](../runs/m2/stats.json) | [0.88](../runs/m2/stats.json) | —（参照） | — | [1](../runs/m1v2/seed1/policies/remember_no_update/sessions.jsonl) [2](../runs/m1v2/seed2/policies/remember_no_update/sessions.jsonl) [3](../runs/m1v2/seed3/policies/remember_no_update/sessions.jsonl) |

剩余空间比例 =（得分 − 记住但不更新）/（理想记忆策略 − 记住但不更新），和 M1 第二版闸门④同一口径。温度与 M1 一致（0）；思考开时 `max_tokens` 8192、关时 2048，硅基流动的 `max_tokens` 不限制思考长度（[config.json](../runs/m2/config.json)）。


## 2. 小模型特有的失败，以及兜圈子

每次与偏好有关的会话算一次（主表三个种子各 168 次，共 504 次；"知道真实状态"对照组只跑了种子 1，168 次）。格式错误不含"只说话、没调工具"：那种情况环境会提醒，模型可以接着调工具。"兜圈子"指一次会话里出现下面任意一种：同一问题重复问（声明的属性相同或问题原文相同）、跑到 14 轮上限、空白填充（被长度上限截断、正文八成以上是空白的一轮）。

## 3. 丢分是记忆问题，还是挑店或格式问题

"知道真实状态"对照组只跑了种子 1（限速下按优先级缩减），下表 Qwen 两行的失分拆解只算种子 1 的 168 次会话。对照组：每次会话开始时，把用户当前的真实情况（含本次的临时情况）直接写进记忆，其余不变。一次选错的会话，对照组选对了 → 记作"没记住或没察觉变化"；对照组也错 → 记作"挑店或格式问题"，再分成"没选出任何选项"和"选了错的"。

| 策略 | 每次会话失分 | 其中提问 | 没记住或没察觉变化 | 挑店或格式问题 | └ 没选出选项 / 选错 | 对照组选对率 | 对照组得分 | 对照组每次提问 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4 Pro（M1，对照只跑了种子 1） | [0.171](../runs/m1v2/gates.json) | [0.076](../runs/m1v2/gates.json) | [0.083](../runs/m1v2/gates.json) | [0.012](../runs/m1v2/gates.json) | — | [98.8%](../runs/m1v2/gates.json) | — | [0.00](../runs/m1v2/gates.json) |
| DeepSeek Flash（M1） | [0.181](../runs/m1v2/gates.json) | [0.072](../runs/m1v2/gates.json) | [0.105](../runs/m1v2/gates.json) | [0.004](../runs/m1v2/gates.json) | — | [99.0%](../runs/m1v2/gates.json) | — | [0.00](../runs/m1v2/gates.json) |

## 4. 两种思考模式的对照，以及训练温度下的组内差异

**训练用哪种思考模式：待定，这一步不下结论。** 下表是两种模式在种子 1–3、温度 0 上的实测对照。训练用哪种模式，第二步在自部署的 BF16 服务上确认模板如何处理思考（官方模板会去掉早于最后一条用户消息的思考内容，影响多轮前缀是否一致）、两种模式各测一遍后再定。

组内差异只跑了思考关：这是上一张卡按"将来训练用的模式"做的取舍（当时按思考关），在限速下排在基线和对照组之后。现在训练模式改为待定，思考开在训练温度下的组内差异还没有测，留到第二步和 BF16 重测一起做。

采样参数 {"temperature": 1.0, "top_p": 1.0}（GRPO 采样不截断分布）。种子 1 的 24 个用户，每人 8 次会话整段重复跑 K 遍；每一遍的请求 seed 不同（硅基流动的 seed 本来也不保证复现）。一个用户的 K 遍就是 GRPO 的一组。

（还没有结果）

## 5. token 用量与接口吞吐

| 组 | 每次会话调用数 | 每次会话 prompt tokens | 每次会话输出 tokens | 其中思考 tokens |
|---|---|---|---|---|

本轮实际发出 [7158](../runs/m2/spend_calls.jsonl) 次请求，prompt [14906407](../runs/m2/spend_calls.jsonl) tokens，输出 [1370815](../runs/m2/spend_calls.jsonl) tokens；单次请求耗时中位数 [53.84](../runs/m2/spend_calls.jsonl) 秒、90% 分位 [222.97](../runs/m2/spend_calls.jsonl) 秒（2026-10-25T09:36:38+00:00 到 2026-10-25T14:01:19+00:00）。

按分钟统计，有请求的 [257](../runs/m2/spend_calls.jsonl) 分钟里，每分钟 tokens 的中位数 [56751](../runs/m2/spend_calls.jsonl)、90% 分位 [113935](../runs/m2/spend_calls.jsonl)、最大 [604452](../runs/m2/spend_calls.jsonl)。最大值来自刚开始的一小段突发；之后稳定在定价页上最低一级的 TPM 80000 附近（[价目快照](../runs/m2/pricing)），所以本账号实际受最低一级的限流。同时发出的请求多到上百个时，多出来的只在对方排队、单次耗时升到几分钟，所以正式运行把同时发出的请求限制在 64 个以内（[运行日志](../runs/m2/logs)）。

## 6. 硅基流动上的模型和 Hugging Face 公开权重是不是同一份

详见 [硅基流动与 HF 权重对照](m2-siliconflow-vs-hf.md)。要点：

- **能间接确认的**：分词器和聊天模板是官方的。13 次请求的 prompt token 数，与用 HF 官方模板（版本 `851bf6e`）在本地渲染后逐个相等；工具调用是官方的 XML 格式（`qwen3_coder`），由服务端解析成 JSON；默认开思考；多轮里的思考内容按官方规则保留。模型名、上下文长度（262144）、上线日期都和 HF 仓库一致。
- **确认不了的**：权重是否逐位相同、数值精度。国内站没标精度，国际站把同系列 9B 标为 FP8，4B 很可能也是 FP8。接口不返回 logprobs，seed 也不保证复现，没法直接比较输出分布。
- **对分数的影响（判断，不是测量）**：只算量化，估计偏 0–2 分，多半偏低；调用方式不一致（思考开关、`max_tokens` 是否包含思考、`thinking_budget`、思考内容回传、重试）每一项都可能带来 5–20 分以上，本轮都显式设定了。工具解析的差异估计 0–3 分，方向不定（本地 vLLM 不一定更好）。要测出来，需要在本地用 BF16 权重和同一个考场重跑，或者用强制解码打分逐 token 比较，都要一张 GPU。
- **部署精度与将来训练的权重**：硅基流动的部署精度未知（很可能 FP8），将来训练用的是 HF 上的 BF16 权重。两者之间的分数差估计在 0–2 分（判断，依据是 FP8 在 8B 以上模型上接近无损的文献，4B 和多步任务可能更敏感），比本表的区间宽度小，但不能忽略。**正式训练前，会在训练平台上用 BF16 权重、本地 vLLM 和同一个考场重测一遍起点**（思考开关、温度、`max_tokens` 口径按 [对照报告](m2-siliconflow-vs-hf.md) 的设置对齐），以那次的分数作为训练的起点和对照，本表只作参考。

## 7. 训练平台与框架选型，以及试跑和正式训练的估算

选型详见 [平台与框架选型](m2-platform-survey.md)。推荐：

- **平台**：优云智算 Compshare。支付宝、微信支付在官方充值文档里写明；定价页当天看到的价格是 4090 24G 按量 1.92 元/时（抢占 1.31），A800 80G 按量 7.36（抢占 3.02）；有 OpenAPI 和命令行工具能开机、关机。备选 AutoDL（支付宝、微信已确认；搜索摘录的单价更低，但普通实例没有开关机接口）。
- **三条路线**：比较了三条路线，见选型报告 Part C。
  - **租卡 + 自部署兼容 Tinker 接口的服务**（魔搭 Twinkle、SkyRL 的 Tinker 后端）：训练循环由我们的代码驱动。8 次会话由我们照常按顺序跑完，记忆由考场自己保存，每次会话切成一条训练样本，优势按会话写 return-to-go。
  - **租卡 + 开源训练框架**（TRL、verl）：循环由框架驱动，"一条 prompt 对应一条序列"。只能按会话做样本、数据里放旧策略的记忆快照，跨会话回报要近似或靠实验性 API。
  - **阿里云 PAI**：Model Gallery 的 GRPO 只写了 Qwen3；百炼托管 RL 最小 9B、要商务开通；自建的单价是租卡平台的 2–5 倍。
- **推荐**：走路线一。服务端先试 Twinkle（官方示例默认就是 Qwen3.5-4B，需 2 卡），备选 SkyRL（单卡 A800，要求驱动 ≥ r580）；路线二的 TRL 作保底，不用 PAI。理由是这个结构在路线一里是普通的 Python，现有考场代码只需换一个记录 token 的 client。
- **把本考场接进去的主要风险**：两家 Tinker 服务端都没有"Qwen3.5-4B + 多轮工具"的公开验证；它们把 LoRA 动态加载进 vLLM，线性注意力层上可能出错；SDK 版本锁定落后于最新 cookbook；老师轨迹、SFT、rollout、评测四处的模板和工具解析器必须一致；线性注意力层在训练端和 vLLM 端的 logprob 偏差；环境版本要求新。

**估算**。训练用哪种思考模式待定（第 4 节），两种模式各占一行，实测量的来源见下表第二列（全量没跑完时取中期子集），已经包含兜圈子的开销。SFT 之后兜圈子会变少，rollout 会更便宜，所以 GRPO 一栏偏保守。老师 DeepSeek V4 Pro 每次会话 [0.026](../runs/m1v2/spend_ledger.json) 元，来自 M1 第二版账本。吞吐、步数、组大小是假设，写在 stats.json 的 `estimate_by_mode.<模式>.assumptions` 里。

| 模式 | 实测量来自 | 每次会话调用 / 输入 / 输出 tokens（实测） | 训练序列（估） |
|---|---|---|---|
| 思考关 | 中期子集、温度 0（每个用户的前几次会话） | [10.27](../runs/m2/stats.json) / [21719](../runs/m2/stats.json) / [1414](../runs/m2/stats.json) | [3067](../runs/m2/stats.json) |
| 思考开 | 中期子集、温度 0（每个用户的前几次会话） | [3.47](../runs/m2/stats.json) / [6671](../runs/m2/stats.json) / [1479](../runs/m2/stats.json) | [2972](../runs/m2/stats.json) |

| 方案 | 模式 | 卡 | 老师轨迹 | 装环境 | SFT | GRPO | 上机合计 | 显卡费（按量） | 显卡费（抢占） | 合计（老师 + 按量显卡） |
|---|---|---|---|---|---|---|---|---|---|---|
| 试跑：4 个新世界的老师轨迹 + SFT + 30 步 GRPO，单卡 | 思考关 | 1 × RTX 4090 24G | 768 次会话，[19.92](../runs/m2/stats.json) 元 | 3.0–6.0 小时 | 0.4–0.9 小时 | 4.7–9.4 小时 | 8.2–16.3 小时 | 16–31 元 | 11–21 元 | 36–51 元 |
| 试跑：4 个新世界的老师轨迹 + SFT + 30 步 GRPO，单卡 | 思考开 | 1 × RTX 4090 24G | 768 次会话，[19.92](../runs/m2/stats.json) 元 | 3.0–6.0 小时 | 0.4–0.9 小时 | 4.1–8.2 小时 | 7.5–15.1 小时 | 14–29 元 | 10–20 元 | 34–49 元 |
| 试跑，2 卡 4090（Twinkle 官方示例的配置，时长按单卡保守估） | 思考关 | 2 × RTX 4090 24G | 768 次会话，[19.92](../runs/m2/stats.json) 元 | 3.0–6.0 小时 | 0.4–0.9 小时 | 4.7–9.4 小时 | 8.2–16.3 小时 | 31–63 元 | 21–43 元 | 51–83 元 |
| 试跑，2 卡 4090（Twinkle 官方示例的配置，时长按单卡保守估） | 思考开 | 2 × RTX 4090 24G | 768 次会话，[19.92](../runs/m2/stats.json) 元 | 3.0–6.0 小时 | 0.4–0.9 小时 | 4.1–8.2 小时 | 7.5–15.1 小时 | 29–58 元 | 20–40 元 | 49–78 元 |
| 试跑，换 A800 | 思考关 | 1 × A800 80G | 768 次会话，[19.92](../runs/m2/stats.json) 元 | 3.0–6.0 小时 | 0.2–0.4 小时 | 2.3–4.6 小时 | 5.5–11.0 小时 | 40–81 元 | 17–33 元 | 60–101 元 |
| 试跑，换 A800 | 思考开 | 1 × A800 80G | 768 次会话，[19.92](../runs/m2/stats.json) 元 | 3.0–6.0 小时 | 0.2–0.4 小时 | 2.0–4.1 小时 | 5.2–10.5 小时 | 39–77 元 | 16–32 元 | 59–97 元 |
| 正式：20 个新世界 + SFT + 300 步 GRPO | 思考关 | 1 × A800 80G | 3840 次会话，[99.60](../runs/m2/stats.json) 元 | 0.0–0.0 小时 | 1.1–2.1 小时 | 45.6–91.2 小时 | 46.7–93.4 小时 | 344–687 元 | 141–282 元 | 443–787 元 |
| 正式：20 个新世界 + SFT + 300 步 GRPO | 思考开 | 1 × A800 80G | 3840 次会话，[99.60](../runs/m2/stats.json) 元 | 0.0–0.0 小时 | 1.1–2.1 小时 | 40.7–81.4 小时 | 41.8–83.5 小时 | 307–615 元 | 126–252 元 | 407–714 元 |

GRPO 每步 = 4（试跑）或 8（正式）个用户 × 每组 8 遍 × 8 次会话；rollout 时长按实测的输出 tokens 和未命中前缀缓存的输入 tokens 除以假设吞吐计算，训练按估计的序列长度、前向加反向、另加一次前向算旧策略 logprob 计算；SFT 按每次会话 3000 tokens、2 遍。这些吞吐没有实测，误差可能到 2 倍；试跑的头 10 步就是用来把它们换成实测值的。思考开的输出里思考 tokens 占大头，解码吞吐的假设对它影响更大。

## 8. 花费与 token 用量

- 硅基流动（Qwen3.5-4B，免费）：累计 [0.00](../runs/m2/spend_ledger.json) 元，上限 [50.00](../runs/m2/spend_ledger.json) 元。逐次调用明细在 [spend_calls.jsonl](../runs/m2/spend_calls.jsonl)。
- 账本记下的调用 [7158](../runs/m2/spend_ledger.json) 次，prompt [14906407](../runs/m2/spend_ledger.json) tokens，输出 [1370815](../runs/m2/spend_ledger.json) tokens（按用途的明细见账本）。
- 另有：硅基流动对照调研时手动发了 17 次请求（免费，记录在 [siliconflow_probe/](../runs/m2/siliconflow_probe)，不在账本里，均为 0 元）；选型只做网页调研，没有注册、充值、租卡。
- 本卡总花费：[0.00](../runs/m2/spend_ledger.json) 元（上限 50 元）。

## 9. 复现

单元测试：[OK](../runs/m2/tests.txt)。

## 10. 没解决的问题

- 硅基流动上的权重和精度只能间接确认；本地 BF16 与硅基流动之间的分数差没有实测。
- 评测用温度 0，请求参数和 M1 相同；但 M1 的 DeepSeek 开着思考，DeepSeek 官方文档写明思考模式下 temperature 不生效、设了也不报错，所以 M1 的 DeepSeek 实际是默认采样（V4 Pro 带工具时 top_p 固定 0.95，Flash 没传 top_p、取默认值 1；[文档原文与核对](../runs/m1v2/docs/deepseek_sampling_2026-10-26.json)）。和"同一批会话上的 V4 Pro"比时，两边的解码方式并不相同：Qwen 按温度 0 请求，V4 Pro 是默认采样。另外官方不建议 Qwen 的思考模式用贪心解码；思考开这一组可能因此被低估。
- 估算里的显卡吞吐是假设，没有实测。

