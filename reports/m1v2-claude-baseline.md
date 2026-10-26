# M1 第二版：Claude 系列基线（Haiku 5.5、Sonnet 5.5、Opus 5.5、Fable 5.1），与 DeepSeek 同考场并排

Claude 做穿考场，但未到理想。

日志目录：[runs/m1v2/claude/](../runs/m1v2/claude)（试跑、估算、计划、账本、逐次调用、复现检查）与 `runs/m1v2/seed{1,2,3}/policies/llm_claude-*`（逐次会话记录与对话）。数字由 `python -m roommate.claude_run --stages stats,report` 从日志生成，链接指向 [claude_stats.json](../runs/m1v2/claude/claude_stats.json)，它再由逐次会话记录算出。分数只算和偏好有关的会话；区间都是按用户重抽样（每个世界内分别重抽，5000 次）的 95% 区间。Claude 的输出只用于评测和分析，不进任何训练数据（[DATA_POLICY.json](../runs/m1v2/claude/DATA_POLICY.json)）。

## 1. 三个问题

**最强的模型离理想记忆策略还差多少。** 三个闸门世界上分最高的是 Claude Opus 5.5：每次会话 [0.911](../runs/m1v2/claude/claude_stats.json) 分（区间 [0.900–0.921](../runs/m1v2/claude/claude_stats.json)），理想记忆策略 [0.936](../runs/m1v2/claude/claude_stats.json)，差 [0.025](../runs/m1v2/claude/claude_stats.json)（区间 [0.014–0.038](../runs/m1v2/claude/claude_stats.json)）；剩余空间比例 [94.9%](../runs/m1v2/claude/claude_stats.json)（区间 [92.4%–97.1%](../runs/m1v2/claude/claude_stats.json)）。Claude Fable 5.1 得 [0.898](../runs/m1v2/claude/claude_stats.json)（剩余空间比例 [92.3%](../runs/m1v2/claude/claude_stats.json)），比它低 [0.013](../runs/m1v2/claude/claude_stats.json)，配对差的区间 [-0.006–0.032](../runs/m1v2/claude/claude_stats.json) 含 0，两者分不出高低。比 DeepSeek V4 Pro 高 [0.078](../runs/m1v2/claude/claude_stats.json)（区间 [0.049–0.110](../runs/m1v2/claude/claude_stats.json)）。

**丢分落在哪几类。** Claude Opus 5.5（种子 1，有对照组）每次会话失分 [0.090](../runs/m1v2/claude/claude_stats.json)：提问 [0.072](../runs/m1v2/claude/claude_stats.json)、没记住或没察觉变化 [0.018](../runs/m1v2/claude/claude_stats.json)、挑店或格式问题 [0.000](../runs/m1v2/claude/claude_stats.json)；DeepSeek V4 Pro（种子 1，有对照组）每次会话失分 [0.171](../runs/m1v2/claude/claude_stats.json)：提问 [0.076](../runs/m1v2/claude/claude_stats.json)、没记住或没察觉变化 [0.083](../runs/m1v2/claude/claude_stats.json)、挑店或格式问题 [0.012](../runs/m1v2/claude/claude_stats.json)。理想记忆策略只在提问上失 [0.064](../runs/m1v2/claude/claude_stats.json)。所以 Opus 5.5 比理想记忆策略多失的 [0.026](../runs/m1v2/claude/claude_stats.json) 里，提问多花 [0.009](../runs/m1v2/claude/claude_stats.json)，没记住或没察觉变化 [0.018](../runs/m1v2/claude/claude_stats.json)，挑店或格式问题 [0.000](../runs/m1v2/claude/claude_stats.json)；知道真实状态时它 [100.0%](../runs/m1v2/claude/claude_stats.json) 选对，挑店本身不是问题。其余模型没有对照组，第 5 节给出按选错选项种类的粗分。

**差距随模型变大怎么变。** 三个世界上从小到大：Claude Haiku 5.5 [0.684](../runs/m1v2/claude/claude_stats.json)（剩余空间 [49.7%](../runs/m1v2/claude/claude_stats.json)），Claude Sonnet 5.5 [0.867](../runs/m1v2/claude/claude_stats.json)（剩余空间 [86.2%](../runs/m1v2/claude/claude_stats.json)），Claude Opus 5.5 [0.911](../runs/m1v2/claude/claude_stats.json)（剩余空间 [94.9%](../runs/m1v2/claude/claude_stats.json)），Claude Fable 5.1 [0.898](../runs/m1v2/claude/claude_stats.json)（剩余空间 [92.3%](../runs/m1v2/claude/claude_stats.json)）。相邻两档的分差（大减小）：Haiku 5.5 → Sonnet 5.5 [0.183](../runs/m1v2/claude/claude_stats.json)（区间 [0.141–0.226](../runs/m1v2/claude/claude_stats.json)）；Sonnet 5.5 → Opus 5.5 [0.044](../runs/m1v2/claude/claude_stats.json)（区间 [0.026–0.062](../runs/m1v2/claude/claude_stats.json)）；Opus 5.5 → Fable 5.1 [-0.013](../runs/m1v2/claude/claude_stats.json)（区间 [-0.032–0.006](../runs/m1v2/claude/claude_stats.json)，区间含 0，分不出高低）。提问花掉的分：Haiku 5.5 [0.046](../runs/m1v2/claude/claude_stats.json)，Sonnet 5.5 [0.077](../runs/m1v2/claude/claude_stats.json)，Opus 5.5 [0.075](../runs/m1v2/claude/claude_stats.json)，Fable 5.1 [0.068](../runs/m1v2/claude/claude_stats.json)；选到近似项（多半是没记住或没察觉变化）的比例：Haiku 5.5 [13.9%](../runs/m1v2/claude/claude_stats.json)，Sonnet 5.5 [3.2%](../runs/m1v2/claude/claude_stats.json)，Opus 5.5 [0.8%](../runs/m1v2/claude/claude_stats.json)，Fable 5.1 [2.4%](../runs/m1v2/claude/claude_stats.json)。具体看第 6 节。

## 2. 设置：和 DeepSeek 基线一致的地方与对不齐的地方

一致：同一个考场（`runs/m1v2/seed{1,2,3}/world.json`，种子 1、2、3 各 24 个用户 × 8 次会话，没有重新生成）、同一段系统提示和同一组工具（[agent_claude.claude_tools](https://github.com/BillBeam/good-roommate/blob/955e7427f807b7e9900d4b1abcbfb2e124879fd2/roommate/agent_claude.py#L27-L29) 把 [agent_llm.tools](https://github.com/BillBeam/good-roommate/blob/955e7427f807b7e9900d4b1abcbfb2e124879fd2/roommate/agent_llm.py#L43) 的名字、说明、参数原样搬过来，系统提示直接用 [agent_llm.system_prompt](https://github.com/BillBeam/good-roommate/blob/955e7427f807b7e9900d4b1abcbfb2e124879fd2/roommate/agent_llm.py#L23)）、同样的 14 轮上限和"只输出文字时提醒调用工具"、同一套判分（roommate/grading.py，没有改）、同样是随机采样：DeepSeek 的请求里带了 temperature 0，但 DeepSeek 官方文档写明思考模式不支持 temperature，设了不报错、也不生效，所以 DeepSeek 基线实际是默认采样（[文档原文与核对](../runs/m1v2/docs/deepseek_sampling_2026-10-26.json)）；Claude 这几个模型不收 temperature，也是默认采样。温度因此不算两边的差异；两边也都没有 seed，复现都只靠缓存。每次会话从零开始、跨会话只靠记忆工具。"知道真实状态"对照组同样是把真实状态写进记忆，只给最强档（Opus 5.5）跑种子 1，和 DeepSeek 只给 V4 Pro 跑种子 1 的口径一致。

对不齐的设置（逐条）：

| 设置 | DeepSeek 基线 | Claude 基线 | 可能的影响 |
|---|---|---|---|
| top_p | V4 Pro：思考模式下传的 top_p 被忽略，带工具时固定 0.95，agent 的每个请求都带工具，所以是 0.95。Flash：思考模式下 top_p 生效、下限 0.95，请求没传，取默认值 1（[DeepSeek 文档原文与核对](../runs/m1v2/docs/deepseek_sampling_2026-10-26.json)） | 不能设：这几个模型不收 top_p（发了就 400），用接口默认 | V4 Pro 每一步都截掉概率最低的 5% 尾部，偶然挑到冷门词的机会比 Flash 和 Claude 少一点；最关心的 Opus 5.5 对 V4 Pro 正好隔着这一条。两边都是随机采样，这一条影响的是波动大小，不是"贪心对采样" |
| 思考强度 | 思考开（默认）；没传 effort，按 /models 接口取默认的 high（可选 low、high、max） | 四个模型都开自适应思考（[agent_claude](https://github.com/BillBeam/good-roommate/blob/955e7427f807b7e9900d4b1abcbfb2e124879fd2/roommate/agent_claude.py#L80)），effort 统一显式设成 high（Opus 5.5、Haiku 5.5 默认是 medium，Sonnet 5.5、Fable 5.1 默认是 high）；Opus 5.5、Fable 5.1 关不掉思考 | 两家的 effort 都叫 high，但刻度是各自定的，不能说是同等的思考量；每次会话实际输出多少（含思考）见第 8 节。四个 Claude 用同一个 effort，档位之间可比；换成各自的默认值，Opus 5.5 和 Haiku 5.5 会想得更少、更便宜 |
| 思考内容可见 | reasoning_content 是原文，工具调用循环里原样带回 | 只返回摘要（display = summarized），思考块连签名原样带回 | 不影响模型行为和花费（思考照常计费）；对话记录里能看到的推理只有摘要，分析失分时证据少一些 |
| 缓存键 | 按请求内容（加用途）缓存 | 另带"种子|用户|会话"（[llm.LLMClient.chat](https://github.com/BillBeam/good-roommate/blob/955e7427f807b7e9900d4b1abcbfb2e124879fd2/roommate/llm.py#L262)） | 两边都是随机采样时，DeepSeek 那边两个用户的请求要是逐字相同，会共用同一个回答；每次的提示里都有这个用户自己的选项，实际几乎碰不上。只影响复现时用哪一个缓存，不影响模型行为 |
| 请求格式 | OpenAI 兼容的 chat/completions：tool 角色的消息、function 工具 | Messages API：工具结果放在同一条 user 消息的 tool_result 块里，工具定义是 input_schema；文字逐字相同 | 模型看到的信息相同；两家各自在服务端把工具定义转成自己的提示，这部分没法对齐 |
| 最大输出 | 没设 max_tokens（用接口默认） | max_tokens = 32000，流式取最终消息 | 截断的轮数见第 7 节；没有截断就没有影响 |
| 提示缓存 | DeepSeek 自动缓存 | 开顶层自动缓存（cache_control，5 分钟） | 只影响花费，不影响模型看到的内容 |
| 拒答 | 没有这一类 | 安全分类器可能拒答（stop_reason = refusal）；不开换模型重试（会把别的模型的回答混进来），拒答这次会话按没作答计 | 拒答次数见第 7 节 |
| 对照组 | V4 Pro 种子 1（Flash 另有种子 1–3） | Opus 5.5 种子 1 | V4 Pro 和 Opus 5.5 的失分拆解都在种子 1 的 168 次偏好会话上，同口径可比 |
| Fable 5.1 的覆盖 | — | 跑完的世界：种子 1、2、3（见第 3 节） | 三个世界都跑完就和别的模型同表比；否则看第 4 节种子 1 的表 |
| tokenizer | DeepSeek 自己的 | Claude 自己的 | token 数不能跨厂商比，花费可以比 |

## 3. 全量之前：试跑实测与估算

每个模型先只跑种子 1 的前 2 个用户的全部会话（每人 8 次），Opus 5.5 的对照组同样试跑。试跑的请求照常进缓存，全量时这两个用户直接命中缓存，不重复花钱。

| 试跑 | 会话数 | 每次会话花费（美元） | 两个用户各自的每次会话花费 | 每次会话调用数 | 每次会话 tokens：未缓存输入 / 缓存写入 / 缓存命中 / 输出 | 试跑得分 |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 16 | [0.000679](../runs/m1v2/claude/pilot.json) | 0.0008 / 0.0006 | 2.2 | 6 / 2317 / 2369 / 729 | [0.814](../runs/m1v2/claude/pilot.json) |
| Claude Sonnet 5.5 | 16 | [0.00885](../runs/m1v2/claude/pilot.json) | 0.0088 / 0.0089 | 2.2 | 6 / 2128 / 2334 / 328 | [0.864](../runs/m1v2/claude/pilot.json) |
| Claude Opus 5.5 | 16 | [0.0215](../runs/m1v2/claude/pilot.json) | 0.0237 / 0.0192 | 2.4 | 7 / 2349 / 2900 / 457 | [0.925](../runs/m1v2/claude/pilot.json) |
| Claude Fable 5.1 | 16 | [0.0695](../runs/m1v2/claude/pilot.json) | 0.0712 / 0.0678 | 3.3 | 9 / 2810 / 5108 / 660 | [0.864](../runs/m1v2/claude/pilot.json) |
| Claude Opus 5.5（知道真实状态） | 16 | [0.0148](../runs/m1v2/claude/pilot.json) | 0.0148 / 0.0149 | 1.9 | 6 / 2291 / 1690 / 150 | [1.000](../runs/m1v2/claude/pilot.json) |

按实测重新估算的全卡总价（四个模型各三个世界 + Opus 对照组）：[60.75 美元](../runs/m1v2/claude/estimate.json)（按两个用户里较贵的那个算 [63.05 美元](../runs/m1v2/claude/estimate.json)），上限 150 美元。估算没超过上限，不用裁剪。

| 组 | 会话数（其中要新跑的） | 估算（美元） | 计划 |
|---|---|---|---|
| Claude Haiku 5.5 · 种子 1 | 192（176） | [0.12](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Haiku 5.5 · 种子 2 | 192（192） | [0.13](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Haiku 5.5 · 种子 3 | 192（192） | [0.13](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Sonnet 5.5 · 种子 1 | 192（176） | [1.56](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Sonnet 5.5 · 种子 2 | 192（192） | [1.70](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Sonnet 5.5 · 种子 3 | 192（192） | [1.70](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Opus 5.5 · 种子 1 | 192（176） | [3.78](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Opus 5.5 · 种子 2 | 192（192） | [4.13](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Opus 5.5 · 种子 3 | 192（192） | [4.13](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Fable 5.1 · 种子 1 | 192（176） | [12.23](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Fable 5.1 · 种子 2 | 192（192） | [13.34](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Fable 5.1 · 种子 3 | 192（192） | [13.34](../runs/m1v2/claude/estimate.json) | 跑 |
| Claude Opus 5.5（知道真实状态） · 种子 1 | 192（176） | [2.61](../runs/m1v2/claude/estimate.json) | 跑 |

计划花费（含试跑）：[60.75 美元](../runs/m1v2/claude/estimate.json)。账本另留 5 美元给并发中还没记账的请求：累计加余量到 150 美元就不再发新请求（[llm.Ledger.check](https://github.com/BillBeam/good-roommate/blob/955e7427f807b7e9900d4b1abcbfb2e124879fd2/roommate/llm.py#L133)）。全量时每组开跑前再按已花的钱核一遍（[claude_run.stage_full](https://github.com/BillBeam/good-roommate/blob/d7bcdc6be2aa7cf43c30904868179306c161a0ea/roommate/claude_run.py#L186)）。

全量运行记录（[full_log.json](../runs/m1v2/claude/full_log.json)；Haiku、Sonnet、Opus 是三个进程同时跑的，"累计"是三个进程共用的账本当时的总额，所以三列交错上涨；Fable 在它们跑完之后单独跑）：

- claude-sonnet-5-5@1：跑完，累计 6.68 美元
- claude-sonnet-5-5@2：跑完，累计 11.85 美元
- claude-sonnet-5-5@3：跑完，累计 17.04 美元
- claude-haiku-5-5@1：跑完，累计 7.05 美元
- claude-haiku-5-5@2：跑完，累计 12.29 美元
- claude-haiku-5-5@3：跑完，累计 17.75 美元
- claude-opus-5-5@1：跑完，累计 7.18 美元
- claude-opus-5-5@2：跑完，累计 13.00 美元
- claude-opus-5-5@3：跑完，累计 18.45 美元
- claude-opus-5-5@1@ctrl：跑完，累计 21.06 美元
- claude-fable-5-1@1：跑完，累计 32.51 美元
- claude-fable-5-1@2：跑完，累计 44.51 美元
- claude-fable-5-1@3：跑完，累计 57.02 美元

## 4. 结果：DeepSeek 与 Claude 同表

三个闸门世界（每个模型 504 次偏好会话）：

| 模型 / 策略 | 思考与采样（[出处](../runs/m1v2/docs/deepseek_sampling_2026-10-26.json)） | 每次会话得分 | 区间 | 选对率 | 每次会话提问数 | 剩余空间比例 | 区间 | 离理想记忆还差 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek Flash（V4.1） | 思考开（effort 默认 high）；默认采样（温度不生效）；top_p 1.0（没传，取默认值） | [0.819](../runs/m1v2/claude/claude_stats.json) | [0.786–0.848](../runs/m1v2/claude/claude_stats.json) | [89.1%](../runs/m1v2/claude/claude_stats.json) | [0.48](../runs/m1v2/claude/claude_stats.json) | [76.5%](../runs/m1v2/claude/claude_stats.json) | [70.2%–82.4%](../runs/m1v2/claude/claude_stats.json) | [0.118](../runs/m1v2/claude/claude_stats.json) |
| DeepSeek V4 Pro | 思考开（effort 默认 high）；默认采样（温度不生效）；top_p 0.95（带工具时固定） | [0.833](../runs/m1v2/claude/claude_stats.json) | [0.804–0.859](../runs/m1v2/claude/claude_stats.json) | [90.7%](../runs/m1v2/claude/claude_stats.json) | [0.49](../runs/m1v2/claude/claude_stats.json) | [79.4%](../runs/m1v2/claude/claude_stats.json) | [73.6%–84.6%](../runs/m1v2/claude/claude_stats.json) | [0.103](../runs/m1v2/claude/claude_stats.json) |
| Claude Haiku 5.5 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.684](../runs/m1v2/claude/claude_stats.json) | [0.641–0.725](../runs/m1v2/claude/claude_stats.json) | [73.0%](../runs/m1v2/claude/claude_stats.json) | [0.31](../runs/m1v2/claude/claude_stats.json) | [49.7%](../runs/m1v2/claude/claude_stats.json) | [40.7%–58.3%](../runs/m1v2/claude/claude_stats.json) | [0.252](../runs/m1v2/claude/claude_stats.json) |
| Claude Sonnet 5.5 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.867](../runs/m1v2/claude/claude_stats.json) | [0.849–0.885](../runs/m1v2/claude/claude_stats.json) | [94.4%](../runs/m1v2/claude/claude_stats.json) | [0.52](../runs/m1v2/claude/claude_stats.json) | [86.2%](../runs/m1v2/claude/claude_stats.json) | [82.5%–89.6%](../runs/m1v2/claude/claude_stats.json) | [0.069](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.911](../runs/m1v2/claude/claude_stats.json) | [0.900–0.921](../runs/m1v2/claude/claude_stats.json) | [98.6%](../runs/m1v2/claude/claude_stats.json) | [0.50](../runs/m1v2/claude/claude_stats.json) | [94.9%](../runs/m1v2/claude/claude_stats.json) | [92.4%–97.1%](../runs/m1v2/claude/claude_stats.json) | [0.025](../runs/m1v2/claude/claude_stats.json) |
| Claude Fable 5.1 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.898](../runs/m1v2/claude/claude_stats.json) | [0.882–0.912](../runs/m1v2/claude/claude_stats.json) | [96.6%](../runs/m1v2/claude/claude_stats.json) | [0.46](../runs/m1v2/claude/claude_stats.json) | [92.3%](../runs/m1v2/claude/claude_stats.json) | [89.3%–95.1%](../runs/m1v2/claude/claude_stats.json) | [0.038](../runs/m1v2/claude/claude_stats.json) |
| 理想记忆策略（满分参照） | — | [0.936](../runs/m1v2/claude/claude_stats.json) | [0.932–0.940](../runs/m1v2/claude/claude_stats.json) | [100.0%](../runs/m1v2/claude/claude_stats.json) | [0.42](../runs/m1v2/claude/claude_stats.json) | [100.0%](../runs/m1v2/claude/claude_stats.json) | — | [0.000](../runs/m1v2/claude/claude_stats.json) |
| 问过就记、不读台词（从不察觉变化） | — | [0.435](../runs/m1v2/claude/claude_stats.json) | [0.405–0.464](../runs/m1v2/claude/claude_stats.json) | [56.7%](../runs/m1v2/claude/claude_stats.json) | [0.88](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) | — | [0.502](../runs/m1v2/claude/claude_stats.json) |
| 每次都问、不写记忆 | — | [-0.230](../runs/m1v2/claude/claude_stats.json) | [-0.262–-0.198](../runs/m1v2/claude/claude_stats.json) | [38.9%](../runs/m1v2/claude/claude_stats.json) | [4.13](../runs/m1v2/claude/claude_stats.json) | [-132.6%](../runs/m1v2/claude/claude_stats.json) | — | [1.166](../runs/m1v2/claude/claude_stats.json) |

剩余空间比例 = (模型分 − 最简单策略里的最高分 [0.435](../runs/m1v2/claude/claude_stats.json)，即"问过就记、不读台词（从不察觉变化）") / (理想记忆策略 [0.936](../runs/m1v2/claude/claude_stats.json) − 最简单策略里的最高分)。M1 第二版报告第 7 节的基线表已把 Claude 一并列进去（reports/m1v2-baseline.md）；闸门④的定义没有改，仍按约定只看 DeepSeek V4 Pro。

种子 1 单独看（有对照组的世界；每个模型 168 次偏好会话，区间只在种子 1 内重抽用户）：

| 模型 / 策略 | 思考与采样（[出处](../runs/m1v2/docs/deepseek_sampling_2026-10-26.json)） | 每次会话得分 | 区间 | 选对率 | 每次会话提问数 | 剩余空间比例 | 区间 | 离理想记忆还差 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek Flash（V4.1） | 思考开（effort 默认 high）；默认采样（温度不生效）；top_p 1.0（没传，取默认值） | [0.823](../runs/m1v2/claude/claude_stats.json) | [0.769–0.873](../runs/m1v2/claude/claude_stats.json) | [89.3%](../runs/m1v2/claude/claude_stats.json) | [0.46](../runs/m1v2/claude/claude_stats.json) | [77.4%](../runs/m1v2/claude/claude_stats.json) | [66.5%–87.1%](../runs/m1v2/claude/claude_stats.json) | [0.114](../runs/m1v2/claude/claude_stats.json) |
| DeepSeek V4 Pro | 思考开（effort 默认 high）；默认采样（温度不生效）；top_p 0.95（带工具时固定） | [0.829](../runs/m1v2/claude/claude_stats.json) | [0.768–0.880](../runs/m1v2/claude/claude_stats.json) | [90.5%](../runs/m1v2/claude/claude_stats.json) | [0.51](../runs/m1v2/claude/claude_stats.json) | [78.6%](../runs/m1v2/claude/claude_stats.json) | [65.7%–88.5%](../runs/m1v2/claude/claude_stats.json) | [0.109](../runs/m1v2/claude/claude_stats.json) |
| Claude Haiku 5.5 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.749](../runs/m1v2/claude/claude_stats.json) | [0.693–0.802](../runs/m1v2/claude/claude_stats.json) | [79.2%](../runs/m1v2/claude/claude_stats.json) | [0.29](../runs/m1v2/claude/claude_stats.json) | [62.7%](../runs/m1v2/claude/claude_stats.json) | [50.6%–73.8%](../runs/m1v2/claude/claude_stats.json) | [0.189](../runs/m1v2/claude/claude_stats.json) |
| Claude Sonnet 5.5 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.871](../runs/m1v2/claude/claude_stats.json) | [0.838–0.899](../runs/m1v2/claude/claude_stats.json) | [94.6%](../runs/m1v2/claude/claude_stats.json) | [0.51](../runs/m1v2/claude/claude_stats.json) | [86.8%](../runs/m1v2/claude/claude_stats.json) | [80.4%–92.4%](../runs/m1v2/claude/claude_stats.json) | [0.067](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.910](../runs/m1v2/claude/claude_stats.json) | [0.888–0.928](../runs/m1v2/claude/claude_stats.json) | [98.2%](../runs/m1v2/claude/claude_stats.json) | [0.48](../runs/m1v2/claude/claude_stats.json) | [94.5%](../runs/m1v2/claude/claude_stats.json) | [89.3%–98.7%](../runs/m1v2/claude/claude_stats.json) | [0.028](../runs/m1v2/claude/claude_stats.json) |
| Claude Fable 5.1 | 自适应思考（effort high）；默认采样（不收 temperature / top_p） | [0.901](../runs/m1v2/claude/claude_stats.json) | [0.875–0.924](../runs/m1v2/claude/claude_stats.json) | [97.0%](../runs/m1v2/claude/claude_stats.json) | [0.46](../runs/m1v2/claude/claude_stats.json) | [92.9%](../runs/m1v2/claude/claude_stats.json) | [88.0%–97.1%](../runs/m1v2/claude/claude_stats.json) | [0.036](../runs/m1v2/claude/claude_stats.json) |
| 理想记忆策略（满分参照） | — | [0.938](../runs/m1v2/claude/claude_stats.json) | [0.930–0.945](../runs/m1v2/claude/claude_stats.json) | [100.0%](../runs/m1v2/claude/claude_stats.json) | [0.42](../runs/m1v2/claude/claude_stats.json) | [100.0%](../runs/m1v2/claude/claude_stats.json) | — | [0.000](../runs/m1v2/claude/claude_stats.json) |
| 问过就记、不读台词（从不察觉变化） | — | [0.431](../runs/m1v2/claude/claude_stats.json) | [0.373–0.485](../runs/m1v2/claude/claude_stats.json) | [56.0%](../runs/m1v2/claude/claude_stats.json) | [0.86](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) | — | [0.507](../runs/m1v2/claude/claude_stats.json) |
| 每次都问、不写记忆 | — | [-0.269](../runs/m1v2/claude/claude_stats.json) | [-0.322–-0.213](../runs/m1v2/claude/claude_stats.json) | [36.3%](../runs/m1v2/claude/claude_stats.json) | [4.21](../runs/m1v2/claude/claude_stats.json) | [-138.2%](../runs/m1v2/claude/claude_stats.json) | — | [1.207](../runs/m1v2/claude/claude_stats.json) |

## 5. 丢分落在哪几类

**失分拆解**（每次偏好会话的平均失分；口径和 DeepSeek 相同：选错的会话按同一会话在"知道真实状态"对照组里选没选对来分，对照组选对了算"没记住或没察觉变化"，对照组也选错算"挑店或格式问题"；所以只算有对照组记录的会话）：

| 策略 | 算了哪些会话 | 总失分 | 提问 | 没记住或没察觉变化 | 挑店或格式问题 | 对照组成功率 | 对照组平均提问 |
|---|---|---|---|---|---|---|---|
| DeepSeek Flash（V4.1） | 种子 1、2、3，504 次 | [0.181](../runs/m1v2/claude/claude_stats.json) | [0.072](../runs/m1v2/claude/claude_stats.json) | [0.105](../runs/m1v2/claude/claude_stats.json) | [0.004](../runs/m1v2/claude/claude_stats.json) | [99.0%](../runs/m1v2/claude/claude_stats.json) | [0.00](../runs/m1v2/claude/claude_stats.json) |
| DeepSeek V4 Pro | 种子 1，168 次 | [0.171](../runs/m1v2/claude/claude_stats.json) | [0.076](../runs/m1v2/claude/claude_stats.json) | [0.083](../runs/m1v2/claude/claude_stats.json) | [0.012](../runs/m1v2/claude/claude_stats.json) | [98.8%](../runs/m1v2/claude/claude_stats.json) | [0.00](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5 | 种子 1，168 次 | [0.090](../runs/m1v2/claude/claude_stats.json) | [0.072](../runs/m1v2/claude/claude_stats.json) | [0.018](../runs/m1v2/claude/claude_stats.json) | [0.000](../runs/m1v2/claude/claude_stats.json) | [100.0%](../runs/m1v2/claude/claude_stats.json) | [0.00](../runs/m1v2/claude/claude_stats.json) |
| 理想记忆策略（满分参照） | 三个闸门世界 | [0.064](../runs/m1v2/claude/claude_stats.json) | [0.064](../runs/m1v2/claude/claude_stats.json) | [0.000](../runs/m1v2/claude/claude_stats.json) | [0.000](../runs/m1v2/claude/claude_stats.json) | （对照用标准策略） | — |
| 问过就记、不读台词（从不察觉变化） | 三个闸门世界 | [0.565](../runs/m1v2/claude/claude_stats.json) | [0.133](../runs/m1v2/claude/claude_stats.json) | [0.433](../runs/m1v2/claude/claude_stats.json) | [0.000](../runs/m1v2/claude/claude_stats.json) | （对照用标准策略） | — |

"挑店或格式问题"再细分（对照组也选错的会话里）：

- DeepSeek Flash（V4.1）：选错 [55](../runs/m1v2/claude/claude_stats.json) 次，其中对照组也选错 [2](../runs/m1v2/claude/claude_stats.json) 次；这些里没作答（格式或轮数用满）[0](../runs/m1v2/claude/claude_stats.json) 次，和对照组选了同一个错项 [2](../runs/m1v2/claude/claude_stats.json) 次。对照组选对、普通运行选错的那些会话选到的选项：选到近似项（用户自己的状态差一点） 31 次，选到别的用户的选项 22 次
- DeepSeek V4 Pro：选错 [16](../runs/m1v2/claude/claude_stats.json) 次，其中对照组也选错 [2](../runs/m1v2/claude/claude_stats.json) 次；这些里没作答（格式或轮数用满）[0](../runs/m1v2/claude/claude_stats.json) 次，和对照组选了同一个错项 [2](../runs/m1v2/claude/claude_stats.json) 次。对照组选对、普通运行选错的那些会话选到的选项：选到近似项（用户自己的状态差一点） 7 次，选到别的用户的选项 7 次
- Claude Opus 5.5：选错 [3](../runs/m1v2/claude/claude_stats.json) 次，其中对照组也选错 [0](../runs/m1v2/claude/claude_stats.json) 次；这些里没作答（格式或轮数用满）[0](../runs/m1v2/claude/claude_stats.json) 次，和对照组选了同一个错项 [0](../runs/m1v2/claude/claude_stats.json) 次。对照组选对、普通运行选错的那些会话选到的选项：选到别的用户的选项 2 次，选到近似项（用户自己的状态差一点） 1 次

**不用对照组的粗分**（每个模型都有）：提问花掉的分，加上选错时选到的是哪类选项。近似项是用户自己的状态差一点（事件之前的旧状态、忽略或误用临时情况、多一个没发生的变化），选到它多半是没记住或没察觉变化；选到别的用户的选项、或没作答，多半是挑店或格式问题。

| 模型 | 世界 | 提问花掉的分 | 选对 | 选到近似项 | 选到别的用户的选项 | 没作答 |
|---|---|---|---|---|---|---|
| DeepSeek Flash（V4.1） | 1、2、3 | [0.072](../runs/m1v2/claude/claude_stats.json) | [89.1%](../runs/m1v2/claude/claude_stats.json) | [6.5%](../runs/m1v2/claude/claude_stats.json) | [4.4%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| DeepSeek V4 Pro | 1、2、3 | [0.074](../runs/m1v2/claude/claude_stats.json) | [90.7%](../runs/m1v2/claude/claude_stats.json) | [5.6%](../runs/m1v2/claude/claude_stats.json) | [3.8%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| Claude Haiku 5.5 | 1、2、3 | [0.046](../runs/m1v2/claude/claude_stats.json) | [73.0%](../runs/m1v2/claude/claude_stats.json) | [13.9%](../runs/m1v2/claude/claude_stats.json) | [13.1%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| Claude Sonnet 5.5 | 1、2、3 | [0.077](../runs/m1v2/claude/claude_stats.json) | [94.4%](../runs/m1v2/claude/claude_stats.json) | [3.2%](../runs/m1v2/claude/claude_stats.json) | [2.4%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5 | 1、2、3 | [0.075](../runs/m1v2/claude/claude_stats.json) | [98.6%](../runs/m1v2/claude/claude_stats.json) | [0.8%](../runs/m1v2/claude/claude_stats.json) | [0.6%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| Claude Fable 5.1 | 1、2、3 | [0.068](../runs/m1v2/claude/claude_stats.json) | [96.6%](../runs/m1v2/claude/claude_stats.json) | [2.4%](../runs/m1v2/claude/claude_stats.json) | [1.0%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |

按近似项种类看选对率（每个模型在它跑完的世界上合计；括号里是会话数）：

| 模型 | 事件之前的旧状态 | 忽略本次临时情况 | 把上次的临时情况当长期 | 多一个没发生的事件 | 多一个没发生的临时情况 | 像是某事件之前的状态（假想） | 像是某临时情况之前的状态（假想） |
|---|---|---|---|---|---|---|---|
| DeepSeek Flash（V4.1） | [89.7%](../runs/m1v2/claude/claude_stats.json)（117） | [91.7%](../runs/m1v2/claude/claude_stats.json)（72） | [89.5%](../runs/m1v2/claude/claude_stats.json)（57） | [86.5%](../runs/m1v2/claude/claude_stats.json)（141） | [89.1%](../runs/m1v2/claude/claude_stats.json)（55） | [90.0%](../runs/m1v2/claude/claude_stats.json)（20） | [90.5%](../runs/m1v2/claude/claude_stats.json)（42） |
| DeepSeek V4 Pro | [91.5%](../runs/m1v2/claude/claude_stats.json)（117） | [98.6%](../runs/m1v2/claude/claude_stats.json)（72） | [94.7%](../runs/m1v2/claude/claude_stats.json)（57） | [83.7%](../runs/m1v2/claude/claude_stats.json)（141） | [87.3%](../runs/m1v2/claude/claude_stats.json)（55） | [90.0%](../runs/m1v2/claude/claude_stats.json)（20） | [97.6%](../runs/m1v2/claude/claude_stats.json)（42） |
| Claude Haiku 5.5 | [86.3%](../runs/m1v2/claude/claude_stats.json)（117） | [90.3%](../runs/m1v2/claude/claude_stats.json)（72） | [70.2%](../runs/m1v2/claude/claude_stats.json)（57） | [63.1%](../runs/m1v2/claude/claude_stats.json)（141） | [60.0%](../runs/m1v2/claude/claude_stats.json)（55） | [75.0%](../runs/m1v2/claude/claude_stats.json)（20） | [59.5%](../runs/m1v2/claude/claude_stats.json)（42） |
| Claude Sonnet 5.5 | [99.1%](../runs/m1v2/claude/claude_stats.json)（117） | [95.8%](../runs/m1v2/claude/claude_stats.json)（72） | [96.5%](../runs/m1v2/claude/claude_stats.json)（57） | [88.7%](../runs/m1v2/claude/claude_stats.json)（141） | [98.2%](../runs/m1v2/claude/claude_stats.json)（55） | [95.0%](../runs/m1v2/claude/claude_stats.json)（20） | [90.5%](../runs/m1v2/claude/claude_stats.json)（42） |
| Claude Opus 5.5 | [100.0%](../runs/m1v2/claude/claude_stats.json)（117） | [100.0%](../runs/m1v2/claude/claude_stats.json)（72） | [96.5%](../runs/m1v2/claude/claude_stats.json)（57） | [97.2%](../runs/m1v2/claude/claude_stats.json)（141） | [98.2%](../runs/m1v2/claude/claude_stats.json)（55） | [100.0%](../runs/m1v2/claude/claude_stats.json)（20） | [100.0%](../runs/m1v2/claude/claude_stats.json)（42） |
| Claude Fable 5.1 | [98.3%](../runs/m1v2/claude/claude_stats.json)（117） | [98.6%](../runs/m1v2/claude/claude_stats.json)（72） | [93.0%](../runs/m1v2/claude/claude_stats.json)（57） | [95.0%](../runs/m1v2/claude/claude_stats.json)（141） | [98.2%](../runs/m1v2/claude/claude_stats.json)（55） | [95.0%](../runs/m1v2/claude/claude_stats.json)（20） | [97.6%](../runs/m1v2/claude/claude_stats.json)（42） |

## 6. 随模型变大

Claude 四档从小到大（Haiku 5.5 → Sonnet 5.5 → Opus 5.5 → Fable 5.1）。四档都在三个闸门世界上跑完，下表都是三个世界的数。

| 模型 | 世界 | 得分 | 剩余空间比例 | 离理想记忆还差 | 每次提问 | 提问里被答"说过了" | 写记忆的会话 | 提问花掉的分 | 选到近似项 | 没作答 |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 1、2、3 | [0.684](../runs/m1v2/claude/claude_stats.json) | [49.7%](../runs/m1v2/claude/claude_stats.json) | [0.252](../runs/m1v2/claude/claude_stats.json) | [0.31](../runs/m1v2/claude/claude_stats.json) | [5.8%](../runs/m1v2/claude/claude_stats.json) | [49.6%](../runs/m1v2/claude/claude_stats.json) | [0.046](../runs/m1v2/claude/claude_stats.json) | [13.9%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| Claude Sonnet 5.5 | 1、2、3 | [0.867](../runs/m1v2/claude/claude_stats.json) | [86.2%](../runs/m1v2/claude/claude_stats.json) | [0.069](../runs/m1v2/claude/claude_stats.json) | [0.52](../runs/m1v2/claude/claude_stats.json) | [3.1%](../runs/m1v2/claude/claude_stats.json) | [52.6%](../runs/m1v2/claude/claude_stats.json) | [0.077](../runs/m1v2/claude/claude_stats.json) | [3.2%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5 | 1、2、3 | [0.911](../runs/m1v2/claude/claude_stats.json) | [94.9%](../runs/m1v2/claude/claude_stats.json) | [0.025](../runs/m1v2/claude/claude_stats.json) | [0.50](../runs/m1v2/claude/claude_stats.json) | [3.6%](../runs/m1v2/claude/claude_stats.json) | [73.0%](../runs/m1v2/claude/claude_stats.json) | [0.075](../runs/m1v2/claude/claude_stats.json) | [0.8%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |
| Claude Fable 5.1 | 1、2、3 | [0.898](../runs/m1v2/claude/claude_stats.json) | [92.3%](../runs/m1v2/claude/claude_stats.json) | [0.038](../runs/m1v2/claude/claude_stats.json) | [0.46](../runs/m1v2/claude/claude_stats.json) | [2.2%](../runs/m1v2/claude/claude_stats.json) | [95.4%](../runs/m1v2/claude/claude_stats.json) | [0.068](../runs/m1v2/claude/claude_stats.json) | [2.4%](../runs/m1v2/claude/claude_stats.json) | [0.0%](../runs/m1v2/claude/claude_stats.json) |

## 7. 工具调用与格式

| 记录 | 会话数 | 平均轮数 | 没作答的偏好会话 | 用满 14 轮 | 拒答（安全分类器） | 只输出文字、没调工具的轮 | 输出被截断的轮 | 接口出错 |
|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 576 | [2.22](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) |
| Claude Sonnet 5.5 | 576 | [2.34](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5 | 576 | [2.44](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5（知道真实状态） | 192 | [1.89](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) |
| Claude Fable 5.1 | 576 | [3.29](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) | [0](../runs/m1v2/claude/claude_stats.json) |

## 8. token 用量与实际花费

Claude 累计实际花费 [57.02 美元](../runs/m1v2/claude/spend_ledger.json)（上限 150 美元），由账本按官方价目快照（[anthropic_pricing_2026-10-26.json](../runs/m1v2/pricing/anthropic_pricing_2026-10-26.json)）和每次响应里的 usage 实时累计，逐次调用见 [spend_calls.jsonl](../runs/m1v2/claude/spend_calls.jsonl)。缓存命中的请求不再花钱，也不计入"实际请求"。

| 模型 | 用途 | 实际请求 | 输入 tokens（其中缓存命中 / 缓存写入） | 输出 tokens（含思考） | 实际花费（美元） |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | 普通运行（含试跑） | 1281 | 2719856（1389136 / 1327006） | 422916 | [0.39 美元](../runs/m1v2/claude/spend_ledger.json) |
| Claude Sonnet 5.5 | 普通运行（含试跑） | 1347 | 2784574（1528705 / 1252023） | 195571 | [5.25 美元](../runs/m1v2/claude/spend_ledger.json) |
| Claude Opus 5.5 | 普通运行（含试跑） | 1408 | 2972509（1669860 / 1298681） | 230806 | [11.46 美元](../runs/m1v2/claude/spend_ledger.json) |
| Claude Opus 5.5 | 对照组（知道真实状态） | 363 | 770035（327586 / 441339） | 28803 | [2.85 美元](../runs/m1v2/claude/spend_ledger.json) |
| Claude Fable 5.1 | 普通运行（含试跑） | 1895 | 4496538（2911016 / 1580580） | 330773 | [37.07 美元](../runs/m1v2/claude/spend_ledger.json) |
| **合计** | | | | | [57.02 美元](../runs/m1v2/claude/spend_ledger.json) |

每次会话的平均用量（按记录里每次会话原响应的 usage 算，缓存命中的会话也照原样计入，所以等于"按价目每次会话要花多少"）：

| 记录 | 会话数 | 调用数 | 未缓存输入 | 缓存写入 | 缓存命中 | 输出（含思考） | 花费（美元） |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 576 | 2.22 | 6 | 2304 | 2412 | 734 | [0.00068](../runs/m1v2/claude/claude_stats.json) |
| Claude Sonnet 5.5 | 576 | 2.34 | 7 | 2174 | 2654 | 340 | [0.00911](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5 | 576 | 2.44 | 7 | 2255 | 2899 | 401 | [0.0199](../runs/m1v2/claude/claude_stats.json) |
| Claude Opus 5.5（知道真实状态） | 192 | 1.89 | 6 | 2299 | 1706 | 150 | [0.0149](../runs/m1v2/claude/claude_stats.json) |
| Claude Fable 5.1 | 576 | 3.29 | 9 | 2744 | 5054 | 574 | [0.0644](../runs/m1v2/claude/claude_stats.json) |

对照：DeepSeek 基线的花费（[M1 第二版账本](../runs/m1v2/spend_ledger.json)，闸门世界三个种子，含重跑；tokenizer 不同，token 数不能直接和 Claude 比）：

| 模型 | 用途 | 实际请求 | 输入 tokens（其中缓存命中） | 输出 tokens（含思考） | 花费（美元） | 每次会话（美元） |
|---|---|---|---|---|---|---|
| DeepSeek Flash（V4.1） | 普通运行 | 1649 | 2539714（2102272） | 338642 | [0.28 美元](../runs/m1v2/spend_ledger.json) | 0.0005（576 次） |
| DeepSeek V4 Pro | 普通运行 | 1501 | 2826819（2436096） | 949383 | [2.19 美元](../runs/m1v2/spend_ledger.json) | 0.0038（576 次） |
| DeepSeek Flash（V4.1） | 对照组 | 575 | 904978（687360） | 161905 | [0.13 美元](../runs/m1v2/spend_ledger.json) | 0.0002（576 次） |
| DeepSeek V4 Pro | 对照组 | 172 | 313227（254976） | 187954 | [0.42 美元](../runs/m1v2/spend_ledger.json) | 0.0022（192 次） |

## 9. 中途被打断怎么办

全量运行每跑完一个模型的一个种子，就做一次 checkpoint（[claude_run.stage_checkpoint](https://github.com/BillBeam/good-roommate/blob/d7bcdc6be2aa7cf43c30904868179306c161a0ea/roommate/claude_run.py#L239)）：把这个模型的请求缓存并入 `runs/m1v2/llm_cache/<模型>.jsonl.gz`（不删还在写的 .jsonl，只收完整的行），连同逐次会话记录、账本、逐次调用和全量运行记录（`runs/m1v2/claude/full_log.json`，逐组追加）一起提交并推到分支。所以中途被打断，最多丢掉正在跑的那一组里还没拿到回复的请求；已经花钱拿到的回复都在分支上。接着跑：

1. 同一个容器还在：直接重跑同一条命令，例如 `ROOMMATE_CLAUDE_ONLY=claude-fable-5-1@2,claude-fable-5-1@3 python -m roommate.claude_run --stages full`。
2. 换了容器：先 `git pull` 拿到分支上的 .jsonl.gz、账本和记录，`pip install anthropic==2.3.0`，环境变量里要有 CLAUDE_EVAL_API_KEY，再重跑同一条命令。
3. 重跑时已经拿到的回复全部命中缓存、不再花钱（缓存键是请求内容加"种子|用户|会话"，和第一次完全一样）；账本从分支上那一份接着累计，150 美元的上限照样有效；被打断时写了一半的缓存行会被跳过，那次请求重新发。
4. 都跑完、没有进程再写缓存之后，跑 `python -m roommate.claude_run --stages compact,repro` 把 .jsonl 并进 .jsonl.gz 并检查只用缓存能逐字复现；想在任何时候手动推一次，跑 `--stages checkpoint`。

这次全量开跑时还没有逐组推送：它是运行中途加的，已经在跑的进程用的是旧代码。所以这次的缓存和记录是全部跑完之后（Fable 5.1 最后一组 2026-10-26T11:08:23+00:00 跑完）才一次推上去的；在那之前，回复只存在运行容器的本地缓存里。全量记录按进程一批一条（[full_log.json](../runs/m1v2/claude/full_log.json)）。以后的全量运行按上面的做法逐组推送。

## 10. 复现、数据用途与局限

- 只用缓存重跑全部 Claude 记录：[13](../runs/m1v2/claude/repro_check.json) / 13 组与保存的记录逐字一致。两边都是默认采样，重新向接口发同样的请求不保证得到同样的回答；复现只靠缓存。
- Claude 的输出只用于评测和分析：存放它们的目录都有 DATA_POLICY.json（for_training = false），以后读训练数据的代码要先过 [data_policy.assert_trainable](https://github.com/BillBeam/good-roommate/blob/bfc7434430d0f8c0d53ca510e20a886b1b278d07/roommate/data_policy.py#L47)。
- 密钥只从环境变量 CLAUDE_EVAL_API_KEY 读取（[llm.LLMClient._post_anthropic](https://github.com/BillBeam/good-roommate/blob/955e7427f807b7e9900d4b1abcbfb2e124879fd2/roommate/llm.py#L335)），不写进文件、日志和提交；请求显式发到 https://api.anthropic.com，不用环境里给别的工具配的 ANTHROPIC_BASE_URL。
- 每个模型只在同一个考场上跑一遍；两边都是默认采样，同一个用户重跑的波动没有单独量（M2 起点体检里的重复采样方法可以照搬，这次没做）。
- 失分拆解里的"没记住或没察觉变化 / 挑店或格式问题"只有带对照组的模型有（DeepSeek Flash（V4.1）、DeepSeek V4 Pro、Claude Opus 5.5），其他模型只有按选错选项种类的粗分。
- 试跑只用了 2 个用户，估算有偏差的余地；花费以账本里的实际累计为准。
