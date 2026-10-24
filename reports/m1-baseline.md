# M1 基线报告：跨会话个人 agent 环境第一版

生成方式：`python -m roommate.run`（随机种子 0）；本文件由 `roommate/report.py` 从 `runs/m1/` 下的日志自动生成，每个数字都链接到生成它的日志文件。生成日期：2026-10-24。

## 一句话结果

**环境过于简单**：最强基线 `grok-4.7` 不经训练就拿到知道真实状态的标准策略得分的 [81.3%](../runs/m1/policies/llm_grok-4.7/summary.json)，超过 80% 的闸门；但它在有过时记忆陷阱的任务上仍有 [6.3%](../runs/m1/policies/llm_grok-4.7/summary.json) 选了按过时偏好看是对的选项。

> **环境过于简单**：最强基线得分达到标准策略的 [81.3%](../runs/m1/policies/llm_grok-4.7/summary.json)（≥ 80%）。

## 1. 环境是什么样的

每个模拟用户有一份由程序维护的**真实状态**（11 项偏好：过敏、饮食方式、爱吃的菜系、吃不吃辣、预算档次、住处、公司位置、出行方式、有没有小孩、有没有宠物、晚上要不要早回家），agent 看不到。用户跨好几个月与 agent 进行多次会话，每次提出一个任务（周末晚餐、工作日午饭、出游订酒店，或与偏好无关的换算、算账），agent 要从 6 个候选里选一个。

会话分四类：

- **普通任务**：按用户当时的真实状态判对错。
- **生活事件**：用户在提任务时顺口提到一件事（搬家、开始健身、家里添了宝宝、养了狗、升职调岗、怀孕、换工作降薪等），只描述发生了什么，从不说"我的偏好变了"；事件会同时改动几项相关偏好，agent 要自己推断。
- **临时情况**：用户提到只针对这一次的情况（嗓子发炎、公司报销、这顿想省点、这次换个区聚）。本次按临时情况判，之后恢复长期偏好；agent 若把它写成长期记忆，后面就会选错。
- **与偏好无关的会话**：换算、算账、算日期，有客观答案，不需要个性化，提问纯属浪费。

agent 有四个工具：读记忆、写记忆、删记忆、向用户提问（每问一次扣 0.15 分，一个问题用户最多回答 2 个方面），最后调用选择工具交卷。每次会话 agent 都不记得以前的对话，跨会话信息只有记忆。

**判分**全部由代码完成：所选选项满足用户当时真实状态的每一项要求得 1 分，否则 0 分；再减去提问成本。生成器保证每个任务恰好有一个正确选项。若某项偏好此前被事件改过，生成器会放入一个"过时记忆陷阱"选项（按旧偏好看完全合适、按新偏好不合适）；若之前有过临时情况，会放入"临时情况陷阱"选项。

指标定义：

- **后续任务成功率**：第 2 次会话起、需要个性化的任务的成功率。
- **重复提问率**：所提问题中，用户给出的回答与此前对同一方面的回答完全相同（即信息没变、白问了）的比例。
- **过时记忆错误率**：在有过时记忆陷阱的任务里，选错且所选选项按过时的偏好看是合适的比例。
- **临时情况污染错误率**：同理，针对临时情况陷阱。

**规模与各类情况的出现次数**

| 项目 | 数量 |
|---|---|
| 模拟用户 | [24](../runs/m1/world_stats.json) |
| 每人会话数 | [8](../runs/m1/world_stats.json) |
| 会话总数 | [192](../runs/m1/world_stats.json) |
| 一个事件改动多项相关偏好 | [29](../runs/m1/world_stats.json) |
| 一个事件只改动一项偏好（对照） | [4](../runs/m1/world_stats.json) |
| 不应改写长期记忆的临时情况 | [24](../runs/m1/world_stats.json) |
| 与已存偏好无关的会话 | [24](../runs/m1/world_stats.json) |
| 含过时记忆陷阱的任务 | [111](../runs/m1/world_stats.json) |
| 含临时情况陷阱的任务 | [23](../runs/m1/world_stats.json) |

事件类型分布：体检血糖偏高 [4](../runs/m1/world_stats.json)，养了狗 [5](../runs/m1/world_stats.json)，升职调岗 [6](../runs/m1/world_stats.json)，家里添了宝宝 [2](../runs/m1/world_stats.json)，开始健身 [3](../runs/m1/world_stats.json)，怀孕 [3](../runs/m1/world_stats.json)，换工作降薪 [3](../runs/m1/world_stats.json)，搬家 [7](../runs/m1/world_stats.json)。
临时情况分布：公司报销 [4](../runs/m1/world_stats.json)，嗓子发炎 [6](../runs/m1/world_stats.json)，这次换个区聚 [6](../runs/m1/world_stats.json)，这顿想省点 [8](../runs/m1/world_stats.json)。

完整的隐藏世界（每个用户每次会话的真实状态、候选选项、正确答案、陷阱）见 [world.json](../runs/m1/world.json)。

## 2. 判分自检

全部通过。明细见 [selfcheck.json](../runs/m1/selfcheck.json)。

| 检查 | 结果 | 依据 |
|---|---|---|
| 每个任务恰好一个正确选项 | 通过 | [192/192](../runs/m1/selfcheck.json) |
| 陷阱选项满足过时/被污染的状态、违反当前真实状态 | 通过 | [134/134](../runs/m1/selfcheck.json) |
| 标准策略（知道真实状态）得满分 | 通过 | [平均分 1.0，成功率 1.0，提问 0 次](../runs/m1/selfcheck.json) |
| 什么都不做得零分 | 通过 | [平均分 0.0，成功率 0.0](../runs/m1/selfcheck.json) |
| 每次都问一遍：只靠回答即可全部做对（回答信息充分） | 通过 | [成功率 1.0，平均每次会话提问 2.172 次](../runs/m1/selfcheck.json) |
| 只记不更新的策略会掉进过时记忆陷阱（陷阱有效） | 通过 | [过时记忆错误率 0.685](../runs/m1/selfcheck.json) |
| 随大流选项不明显优于随机（选项本身不泄露答案） | 通过 | [随大流 0.236，随机 0.181](../runs/m1/selfcheck.json) |
| 规模：至少 20 个用户、每人至少 6 次会话 | 通过 | [24 个用户 × 8 次会话](../runs/m1/selfcheck.json) |
| 同时存在多项偏好事件、临时情况、无关会话 | 通过 | [多项偏好事件 29，临时情况 24，无关会话 24](../runs/m1/selfcheck.json) |
| 同一种子重新生成的世界完全相同 | 通过 | [726f4d7fb0a1ff1b](../runs/m1/selfcheck.json) |

## 3. 参照策略

| 策略 | 平均每次会话得分 | 占标准策略 | 任务成功率 | 后续任务成功率 | 平均每次会话提问 | 重复提问率 | 过时记忆错误率 | 临时情况污染错误率 |
|---|---|---|---|---|---|---|---|---|
| 标准策略（知道真实状态） | [1.000](../runs/m1/policies/oracle/summary.json) | [100.0%](../runs/m1/policies/oracle/summary.json) | [100.0%](../runs/m1/policies/oracle/summary.json) | [100.0%](../runs/m1/policies/oracle/summary.json) | [0.00](../runs/m1/policies/oracle/summary.json) | — | [0.0%](../runs/m1/policies/oracle/summary.json) | [0.0%](../runs/m1/policies/oracle/summary.json) |
| 什么都不做 | [0.000](../runs/m1/policies/nothing/summary.json) | [0.0%](../runs/m1/policies/nothing/summary.json) | [0.0%](../runs/m1/policies/nothing/summary.json) | [0.0%](../runs/m1/policies/nothing/summary.json) | [0.00](../runs/m1/policies/nothing/summary.json) | — | [0.0%](../runs/m1/policies/nothing/summary.json) | [0.0%](../runs/m1/policies/nothing/summary.json) |
| 每次都把问题问一遍 | [0.674](../runs/m1/policies/ask_all/summary.json) | [67.4%](../runs/m1/policies/ask_all/summary.json) | [100.0%](../runs/m1/policies/ask_all/summary.json) | [100.0%](../runs/m1/policies/ask_all/summary.json) | [2.17](../runs/m1/policies/ask_all/summary.json) | [58.5%](../runs/m1/policies/ask_all/summary.json) | [0.0%](../runs/m1/policies/ask_all/summary.json) | [0.0%](../runs/m1/policies/ask_all/summary.json) |
| 保守默认（不做个性化） | [0.312](../runs/m1/policies/conservative/summary.json) | [31.2%](../runs/m1/policies/conservative/summary.json) | [31.2%](../runs/m1/policies/conservative/summary.json) | [19.4%](../runs/m1/policies/conservative/summary.json) | [0.00](../runs/m1/policies/conservative/summary.json) | — | [25.2%](../runs/m1/policies/conservative/summary.json) | [39.1%](../runs/m1/policies/conservative/summary.json) |
| 随机选 | [0.167](../runs/m1/policies/random/summary.json) | [16.7%](../runs/m1/policies/random/summary.json) | [16.7%](../runs/m1/policies/random/summary.json) | [18.1%](../runs/m1/policies/random/summary.json) | [0.00](../runs/m1/policies/random/summary.json) | — | [16.2%](../runs/m1/policies/random/summary.json) | [34.8%](../runs/m1/policies/random/summary.json) |
| 只记不更新（问过就记，从不察觉变化） | [0.493](../runs/m1/policies/remember_no_update/summary.json) | [49.3%](../runs/m1/policies/remember_no_update/summary.json) | [56.8%](../runs/m1/policies/remember_no_update/summary.json) | [42.4%](../runs/m1/policies/remember_no_update/summary.json) | [0.49](../runs/m1/policies/remember_no_update/summary.json) | [0.0%](../runs/m1/policies/remember_no_update/summary.json) | [68.5%](../runs/m1/policies/remember_no_update/summary.json) | [0.0%](../runs/m1/policies/remember_no_update/summary.json) |
| 随大流（选属性最常见的选项） | [0.229](../runs/m1/policies/majority/summary.json) | [22.9%](../runs/m1/policies/majority/summary.json) | [22.9%](../runs/m1/policies/majority/summary.json) | [23.6%](../runs/m1/policies/majority/summary.json) | [0.00](../runs/m1/policies/majority/summary.json) | — | [18.9%](../runs/m1/policies/majority/summary.json) | [34.8%](../runs/m1/policies/majority/summary.json) |

- **每次都把问题问一遍**：任务全部做对（成功率 [100.0%](../runs/m1/policies/ask_all/summary.json)），但平均每次会话提问 [2.17](../runs/m1/policies/ask_all/summary.json) 次，提问成本吃掉 [0.326](../runs/m1/policies/ask_all/summary.json) 分，得分 [0.674](../runs/m1/policies/ask_all/summary.json)；其中 [58.5%](../runs/m1/policies/ask_all/summary.json) 的问题是重复提问。
- **保守默认**：不提问（提问成本 [0.000](../runs/m1/policies/conservative/summary.json)），得分 [0.312](../runs/m1/policies/conservative/summary.json)，后续任务成功率只有 [19.4%](../runs/m1/policies/conservative/summary.json)。说明在这个环境里不个性化必然吃亏。
- 随机选和随大流的成功率接近，说明候选选项本身没有泄露答案；只记不更新的策略在过时记忆陷阱上大量出错，说明陷阱有效。

## 4. 现成模型的基线

| 策略 | 平均每次会话得分 | 占标准策略 | 任务成功率 | 后续任务成功率 | 平均每次会话提问 | 重复提问率 | 过时记忆错误率 | 临时情况污染错误率 |
|---|---|---|---|---|---|---|---|---|
| 标准策略（知道真实状态） | [1.000](../runs/m1/policies/oracle/summary.json) | [100.0%](../runs/m1/policies/oracle/summary.json) | [100.0%](../runs/m1/policies/oracle/summary.json) | [100.0%](../runs/m1/policies/oracle/summary.json) | [0.00](../runs/m1/policies/oracle/summary.json) | — | [0.0%](../runs/m1/policies/oracle/summary.json) | [0.0%](../runs/m1/policies/oracle/summary.json) |
| `grok-4.20-0309-non-reasoning` | [0.416](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [41.6%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [57.8%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [55.6%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [1.08](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [28.0%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [21.6%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [30.4%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) |
| `grok-4.3` | [0.570](../runs/m1/policies/llm_grok-4.3/summary.json) | [57.0%](../runs/m1/policies/llm_grok-4.3/summary.json) | [68.2%](../runs/m1/policies/llm_grok-4.3/summary.json) | [61.8%](../runs/m1/policies/llm_grok-4.3/summary.json) | [0.75](../runs/m1/policies/llm_grok-4.3/summary.json) | [27.8%](../runs/m1/policies/llm_grok-4.3/summary.json) | [14.4%](../runs/m1/policies/llm_grok-4.3/summary.json) | [26.1%](../runs/m1/policies/llm_grok-4.3/summary.json) |
| `grok-4.7` | [0.813](../runs/m1/policies/llm_grok-4.7/summary.json) | [81.3%](../runs/m1/policies/llm_grok-4.7/summary.json) | [91.1%](../runs/m1/policies/llm_grok-4.7/summary.json) | [91.7%](../runs/m1/policies/llm_grok-4.7/summary.json) | [0.66](../runs/m1/policies/llm_grok-4.7/summary.json) | [16.7%](../runs/m1/policies/llm_grok-4.7/summary.json) | [6.3%](../runs/m1/policies/llm_grok-4.7/summary.json) | [8.7%](../runs/m1/policies/llm_grok-4.7/summary.json) |

最强基线 `grok-4.7` 的得分占标准策略得分的 [81.3%](../runs/m1/policies/llm_grok-4.7/summary.json)，达到 80%：**环境过于简单**。
它与满分的差距来自两块：选错少拿 [0.089](../runs/m1/policies/llm_grok-4.7/summary.json) 分，提问成本 [0.098](../runs/m1/policies/llm_grok-4.7/summary.json) 分（平均每次会话 [0.66](../runs/m1/policies/llm_grok-4.7/summary.json) 问 × 0.15）。也就是说，一个会写记忆的强模型在这一版环境里不经训练就能把大部分偏好记住、把大部分变化接住；下一轮需要加难（见 docs/log.md 的"下一步"）。

**按会话类型的成功率**

| 模型 | 普通任务 | 生活事件当次 | 临时情况当次 | 与偏好无关 | 无关会话里平均提问 | 未作答 |
|---|---|---|---|---|---|---|
| `grok-4.20-0309-non-reasoning` | [49.5%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [78.8%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [37.5%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [87.5%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [1.42](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) | [1](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/summary.json) |
| `grok-4.3` | [65.8%](../runs/m1/policies/llm_grok-4.3/summary.json) | [69.7%](../runs/m1/policies/llm_grok-4.3/summary.json) | [45.8%](../runs/m1/policies/llm_grok-4.3/summary.json) | [100.0%](../runs/m1/policies/llm_grok-4.3/summary.json) | [0.00](../runs/m1/policies/llm_grok-4.3/summary.json) | [0](../runs/m1/policies/llm_grok-4.3/summary.json) |
| `grok-4.7` | [88.3%](../runs/m1/policies/llm_grok-4.7/summary.json) | [93.9%](../runs/m1/policies/llm_grok-4.7/summary.json) | [91.7%](../runs/m1/policies/llm_grok-4.7/summary.json) | [100.0%](../runs/m1/policies/llm_grok-4.7/summary.json) | [0.00](../runs/m1/policies/llm_grok-4.7/summary.json) | [0](../runs/m1/policies/llm_grok-4.7/summary.json) |

**记忆工具的使用**（来自每次会话的记录）

| 模型 | 读过记忆的会话占比 | 写过记忆的会话占比 | 最后一次会话后平均记忆条数 |
|---|---|---|---|
| `grok-4.20-0309-non-reasoning` | [94.3%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/sessions.jsonl) | [10.4%](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/sessions.jsonl) | [1.29](../runs/m1/policies/llm_grok-4.20-0309-non-reasoning/sessions.jsonl) |
| `grok-4.3` | [57.3%](../runs/m1/policies/llm_grok-4.3/sessions.jsonl) | [0.0%](../runs/m1/policies/llm_grok-4.3/sessions.jsonl) | [0.00](../runs/m1/policies/llm_grok-4.3/sessions.jsonl) |
| `grok-4.7` | [98.4%](../runs/m1/policies/llm_grok-4.7/sessions.jsonl) | [49.5%](../runs/m1/policies/llm_grok-4.7/sessions.jsonl) | [7.88](../runs/m1/policies/llm_grok-4.7/sessions.jsonl) |

每次会话的完整对话（含工具调用）见各模型目录下的 `transcripts.jsonl`，例如 [grok-4.7](../runs/m1/policies/llm_grok-4.7/transcripts.jsonl)。

所有模型使用同一份系统提示（见 `roommate/agent_llm.py`），温度 0，固定 seed，推理设置用模型默认值；每次会话最多 12 轮模型调用，未调用选择工具即记为未作答。

**用到大模型的地方（单独标注）**：判分不用任何大模型。模拟用户里只有一步用了大模型——把 agent 的自由文本问题归到"饮食、口味、预算……"等话题上（`grok-4.20-0309-non-reasoning`，温度 0），回答内容再由代码按真实状态从模板生成，因此归类模型看不到真实状态，也不影响判分。在本项目自写的 [30](../runs/m1/classifier_eval.json) 条人工标注问题上，归类完全正确的比例为 [96.7%](../runs/m1/classifier_eval.json)。

## 5. 复现与存档恢复

- **复现**：同一种子重新生成世界、重跑全部策略（大模型基线只用缓存重跑），[10](../runs/m1/repro_check.json) / [10](../runs/m1/repro_check.json) 个策略的逐条会话记录与首次运行逐字一致（比较的是记录的哈希，见 [repro_check.json](../runs/m1/repro_check.json)）。大模型的每个请求都按内容哈希缓存在 `runs/m1/llm_cache/`，所以克隆本仓库后执行 `python -m roommate.run --offline` 不联网也能得到完全相同的结果。
- **中断—恢复**：每个大模型基线都在第 [4](../runs/m1/resume_check.json) 个会话边界存档（存档里有全部用户的真实状态、agent 记忆和已有判分记录），丢掉内存中的环境、从存档恢复后跑完，[3](../runs/m1/resume_check.json) / [3](../runs/m1/resume_check.json) 个结果与不中断时逐字一致（[resume_check.json](../runs/m1/resume_check.json)）。
- **单元测试**（`python -m unittest discover -s tests -t .`，输出见 [tests.txt](../runs/m1/tests.txt)）：在全部 9 个会话边界（第 0 到第 8 次会话前后）分别存档、恢复、续跑，结果都与不中断时一致；用一个只记不更新的脚本策略和一个确定性的假模型 agent 各测一遍。另有反例：恢复时故意丢掉记忆，结果就不一致，说明这项检验确实有效。

## 6. 迁移测试用的外部基准

拟用的外部基准（许可证以仓库页面为准；ATRBench、HorizonBench、PAHF、MemPABench 的许可证文件已在克隆后核对）：

| 基准 | 测什么 | 许可证 | 本轮状态 |
|---|---|---|---|
| [ATRBench](https://github.com/BUPT-GAMMA/ATRBench) | 先问后用：学习会话里问到的长期规则能否用到后续工具调用 | 代码 Apache-2.0；数据 CC BY 4.0（派生自 NVIDIA Nemotron-Personas-USA，CC BY 4.0） | **已跑通评测脚本**（见下） |
| [HorizonBench](https://github.com/stellalisy/HorizonBench) | 生活事件驱动的偏好演化，旧偏好作为干扰项 | 代码 Apache-2.0；数据 CC-BY-4.0 | 数据在 Hugging Face，本环境网络不通，未跑 |
| [MemPABench](https://github.com/mempabench/mempabench) | 99 次会话、事件驱动变化、澄清/确认/自主行动 | 代码 MIT；数据 CC-BY-4.0（据 README） | 未跑（默认需其他服务商的模型） |
| [PAHF](https://github.com/facebookresearch/PAHF) | 人设变化、行动前澄清与行动后反馈 | MIT | 未跑 |
| [PERMA](https://github.com/PolarisLiu1/PERMA) | 事件驱动偏好的多会话基准 | Apache-2.0（据徽章） | 未跑 |
| [MultiSessionCollab](https://github.com/Shuhaibm/multisessioncollab) | 多会话协作 | MIT | 未跑 |
| [τ²-bench](https://github.com/sierra-research/tau2-bench) | 带用户模拟器的工具任务 | MIT | 未跑 |

**ATRBench 试跑**：用 ATRBench 自己的 runner 和确定性 evaluator（不改它的任何文件，只在外面登记 xAI 为一个 OpenAI 兼容服务商，见 `bench/atrbench_xai.py`），跑了 1 个人设（`cassandra_tovar`）的 [11](../runs/m1/bench/atrbench_summary.json) 个测试会话，模型 `grok-4.20-0309-non-reasoning`，变体 `oracle_target`：[10](../runs/m1/bench/atrbench_summary.json) 个通过，准确率 [90.9%](../runs/m1/bench/atrbench_summary.json)，花费约 [0.1042 美元](../runs/m1/bench/atrbench_summary.json)。
只跑 `oracle_target` 的原因：这个变体跳过学习会话、不需要模拟用户；ATRBench 默认的模拟用户和路由分别是 GPT-5.4 和 Gemini，本轮边界只允许用 xAI。两处适配只作用于 xAI 模型：xAI 要求 seed 为正，故 seed 加 1；去掉 `reasoning_effort` 参数，用模型默认推理设置。

**怎样保证训练不接触这些基准的数据**：

1. 训练数据只来自本项目的生成器 `roommate/world.py`，它是随机种子的纯函数，不读任何文件；单元测试在生成世界时把文件读写全部禁掉来验证这一点（`tests/test_isolation.py`）。
2. 外部基准的代码、数据、轨迹只放在 `external/` 下，该目录写在 `.gitignore` 里，不进仓库；仓库里只保留只含分数和会话编号的摘要（`runs/m1/bench/atrbench_summary.json`）。
3. 环境与训练代码（`roommate/`）不引用 `external/`、`bench/` 或任何基准名称，由单元测试逐个文件检查。
4. 评测入口单独放在 `bench/`，只在评测时由人手动运行；今后的训练脚本只从 `roommate` 取数据。
5. 选题上也避开重叠：本环境是中文、自写模板、自定的属性体系，与上述英文基准没有共用的文本或人设。

## 7. 复用的公开资源及许可证

| 资源 | 怎么用的 | 许可证 |
|---|---|---|
| [ATRBench](https://github.com/BUPT-GAMMA/ATRBench)（提交 a2cfefe） | 运行时克隆到 `external/`，调用它的评测脚本；不复制、不分发其代码和数据 | 代码 Apache-2.0；数据 CC BY 4.0 |
| [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)（经由 ATRBench 的数据间接使用） | 同上，仅评测时读取 | CC BY 4.0 |
| ATRBench 评测的 Python 依赖（openai、pydantic、PyYAML、tenacity、docstring-parser、rich、socksio 及其间接依赖） | 只装在 `external/venv-atr`，不分发；逐包许可证见 [atrbench_deps_licenses.txt](../runs/m1/bench/atrbench_deps_licenses.txt) | Apache-2.0 / MIT / BSD / PSF |
| Python 标准库 | 环境本身只用标准库，无第三方依赖 | PSF-2.0 |

借鉴了设计思路、但没有复制任何代码或数据的工作：HorizonBench（事件同时改动多项相关偏好、旧偏好作为干扰项）、Memora（惩罚使用过时记忆）、PAHF（记住后不再问而错过变化）、CAPTURE（区分真变化与临时情况）、ATRBench 与 PPP（"每次都问"作为捷径对照、提问计成本）。出处见 [research/02](../research/02-literature-cross-session.md)。环境里的人名、店名、台词、回答模板都是本项目自写。

## 8. 用到的模型与官方单价

单价来自 xAI 官方接口 `GET https://api.x.ai/v1/language-models`，查价日期 2026-10-24，原始响应见 [xai_language_models_2026-10-24.json](../runs/m1/pricing/xai_language_models_2026-10-24.json)（价格字段单位为 1e-4 美元 / 百万 tokens）。xAI 文档站 docs.x.ai 在本环境的网络策略下无法访问，所以以接口返回为准。

| 模型 | 用途 | 输入（美元/百万 tokens） | 缓存命中的输入 | 输出 |
|---|---|---|---|---|
| `grok-4.20-0309-non-reasoning` | 基线 agent、模拟用户的提问归类、ATRBench 试跑 | [1.25](../runs/m1/pricing/xai_language_models_2026-10-24.json) | [0.20](../runs/m1/pricing/xai_language_models_2026-10-24.json) | [2.50](../runs/m1/pricing/xai_language_models_2026-10-24.json) |
| `grok-4.3` | 基线 agent | [1.25](../runs/m1/pricing/xai_language_models_2026-10-24.json) | [0.20](../runs/m1/pricing/xai_language_models_2026-10-24.json) | [2.50](../runs/m1/pricing/xai_language_models_2026-10-24.json) |
| `grok-4.7` | 基线 agent | [2.00](../runs/m1/pricing/xai_language_models_2026-10-24.json) | [0.50](../runs/m1/pricing/xai_language_models_2026-10-24.json) | [6.00](../runs/m1/pricing/xai_language_models_2026-10-24.json) |

## 9. 花费

花费按 xAI 每个响应里的 `usage.cost_in_usd_ticks` 逐条累计（[spend_ledger.json](../runs/m1/spend_ledger.json)，[costs.json](../runs/m1/costs.json)）；用缓存重跑不产生费用。

| 模型 | 用途 | 实际请求次数 | 花费 |
|---|---|---|---|
| `grok-4.20-0309-non-reasoning` | 基线 agent | [739](../runs/m1/costs.json) | [0.4180 美元](../runs/m1/costs.json) |
| `grok-4.20-0309-non-reasoning` | 试跑（调提示词、估成本）· agent | [213](../runs/m1/costs.json) | [0.1648 美元](../runs/m1/costs.json) |
| `grok-4.20-0309-non-reasoning` | 试跑 · 提问归类 | [93](../runs/m1/costs.json) | [0.0241 美元](../runs/m1/costs.json) |
| `grok-4.20-0309-non-reasoning` | 模拟用户的提问归类 | [470](../runs/m1/costs.json) | [0.0926 美元](../runs/m1/costs.json) |
| `grok-4.3` | 基线 agent | [423](../runs/m1/costs.json) | [0.7811 美元](../runs/m1/costs.json) |
| `grok-4.3` | 试跑（调提示词、估成本）· agent | [36](../runs/m1/costs.json) | [0.0753 美元](../runs/m1/costs.json) |
| `grok-4.3` | 试跑 · 提问归类 | [25](../runs/m1/costs.json) | [0.0370 美元](../runs/m1/costs.json) |
| `grok-4.7` | 基线 agent | [562](../runs/m1/costs.json) | [3.1742 美元](../runs/m1/costs.json) |
| `grok-4.7` | 试跑（调提示词、估成本）· agent | [47](../runs/m1/costs.json) | [0.3425 美元](../runs/m1/costs.json) |
| `grok-4.20-0309-non-reasoning` | ATRBench 试跑（按 token 估算） | — | [0.1042 美元](../runs/m1/bench/atrbench_summary.json) |

合计约 **5.21 美元**（上表各项之和），上限 30 美元。

## 10. 已知局限

- 只有一个随机种子（0）；各基线之间的差距没有给出种子间波动，M4 之前需要补多种子。
- 任务是多选一，判分只看选没选对，不评价对话质量；模拟用户的回答来自模板，比真人整齐。
- 提问归类用了大模型，归错时用户会答非所问，对 agent 略不公平（准确率见第 4 节）。
- 大模型首次调用本身不是逐字确定的；复现依赖请求缓存，换模型版本或改提示词后需要重新付费调用。
- 系统提示里提醒了"偏好可能随生活变化、有些情况只针对这一次"，这是对基线较友好的设定。
