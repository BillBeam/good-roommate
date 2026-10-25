# 硅基流动上的 Qwen/Qwen3.5-4B 和 Hugging Face 公开权重是不是同一份

调研日期 2026-10-25。探针日志在 [runs/m2/siliconflow_probe/](../runs/m2/siliconflow_probe)，每个文件都记了请求体和响应体，不含请求头和密钥。共发了 17 次请求：1 次 `GET /v1/models`，16 次 `POST /v1/chat/completions`，其中 1 次返回 503、1 次客户端超时、1 次返回 400。所有请求的 `max_tokens` 都不超过 200，但 p13 实际生成了 708 个 token，原因见 3.6。下文“探针 pXX”指该目录下同名的 JSON 文件。

## 本考场怎么调用（和下文各项风险的对应）

起点测量（[M2 起点报告](m2-start.md)）的调用设置见 [runs/m2/config.json](../runs/m2/config.json) 和 [roommate/m2.py](../roommate/m2.py)：

- **思考开关**：每个请求都显式传 `enable_thinking`（true 或 false），不依赖默认值（对应第 5 节 #1）。
- **思考内容回传**：思考模式下，工具循环里把 `reasoning_content` 原样带回，和官方模板的保留规则一致（#5）。
- **输出长度**：
  - `max_tokens` 非思考 2048、思考 8192。在硅基流动上它只限制最终回答，不限制思考（#2）。
  - 思考开另设 `thinking_budget` 4096（#3）。第一次运行没设：温度 0 时有的请求思考停不下来，600 秒超时、重试也一样，三组思考开的会话卡了约 1.5 小时。已完成的请求思考最长 1999 tokens，所以 4096 只截断失控的请求。被截断时，思考草稿会混进回答（3.6），这类轮次单独记为格式问题 `think_runaway`。本地 vLLM 用"思考加回答共 32K"的 `max_tokens`，作用相同，截断后的表现可能不同。
  - 被截断的轮次记为格式问题 `truncated`。
- **重试**：503、429、超时最多重试 20 次，退避最长 60 秒。重试仍失败就整组停下，不记成零分（#4）。
- **并发**：同时发出的请求不超过 64 个。超过服务商的处理能力时，请求只会在对方排队；实测吞吐见起点报告第 6 节。
- **采样**：
  - 评测用温度 0，与 M1 一致。官方不建议思考模式用贪心解码（容易重复），这里为了和 M1 可比照旧用 0，影响见起点报告。
  - 组内差异用温度 1、top_p 1。
  - `seed` 照传，但服务端不保证复现（3.3），所以复现靠请求缓存。

上面的设置在本地 vLLM 上全部可以照搬，唯一要改的是 `max_tokens` 的口径：本地要把思考算进去，设到 32K 左右。

## 一句话结论

能间接确认的部分：SiliconFlow 用的是官方的聊天模板和分词器，默认的思考模式、工具调用的 XML 格式和服务端解析方式也都和官方一致。具体证据是 13 次成功请求的 `prompt_tokens`，和用官方模板在本地渲染、分词后的 token 数逐个相等（见 3.1）。

确认不了的部分：权重是否逐位相同，部署用的是什么数值精度。国内站没有标精度；国际站把同系列的 Qwen3.5-9B 标为 FP8，所以 4B 很可能也是 FP8，但这一点没有证实。

分数偏差最大的来源不在权重，而在调用方式：
- `max_tokens` 不限制思考长度。
- `thinking_budget` 截断后，思考内容会混进最终回答。
- 不支持 `seed`，也不支持 `logprobs`。
- 有 503 和超时，需要重试。
- 多轮对话里必须把 `reasoning_content` 回传。

这些设置任何一项和本地 vLLM 不一致，都可能带来 5 分以上的差距。只看 FP8 量化本身，我估计（本报告判断）在 agent 基准上偏差在 0–2 个百分点，通常比任务数有限带来的抽样误差还小。

---

## 1. Hugging Face 模型卡（Qwen/Qwen3.5-4B）

来源：模型 API <https://huggingface.co/api/models/Qwen/Qwen3.5-4B>，查到的版本 sha 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，lastModified 2026-03-02。以下文件都按这个版本下载：[README](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md)、[config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)、[chat_template.jinja](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja)、[tokenizer_config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer_config.json)。模板副本和 sha256 存在 [hf_ref/](../runs/m2/siliconflow_probe/hf_ref)。

| 项目 | 内容 | 出处 |
|---|---|---|
| 类型 | 后训练模型，不是基座。README 写 "This repository contains model weights … for the post-trained model"，"Training Stage: Pre-training & Post-training"。标签 `base_model:Qwen/Qwen3.5-4B-Base` | README 第 17、46 行；模型 API 的 tags |
| 架构 | `Qwen3_5ForConditionalGeneration`，带视觉编码器。语言部分 32 层，排布为 8 × (3 × Gated DeltaNet 线性注意力 + 1 × Gated Attention)。hidden 2560，FFN 9216，词表 248320，词嵌入与输出层共享权重，带 MTP 层，`dtype: bfloat16` | config.json；README 中的 Model Overview |
| 权重大小 | `total_size` 9,319,737,856 字节，即 BF16 约 4.66B 参数（含视觉部分） | model.safetensors.index.json |
| 许可 | Apache-2.0 | 模型 API 的 cardData |
| 上下文 | 原生 262,144 token，用 YaRN 可扩展到 1,010,000。官方建议上下文不低于 128K，以保留思考能力 | README 第 64、730–732 行 |
| 思考开关 | 默认开启思考。README："Qwen3.5 models operate in thinking mode by default"。模板在 `enable_thinking` 为 false 时输出 `<think>\n\n</think>\n\n`，否则输出 `<think>\n`。用 vLLM/SGLang 时通过 `chat_template_kwargs: {"enable_thinking": false}` 关闭，阿里云百炼直接用顶层 `enable_thinking` 参数 | README 第 712、994、1001 行；chat_template.jinja 末尾 |
| 工具调用格式 | Qwen3-Coder 风格的 XML，不是 Hermes 风格的 JSON：`<tool_call>\n<function=名>\n<parameter=参数名>\n值\n</parameter>…</function>\n</tool_call>`。工具定义以 JSON 形式放进 system 中的 `<tools>…</tools>`；工具结果以 user 角色加 `<tool_response>` 回传 | chat_template.jinja |
| 推荐解析器 | vLLM：`--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`；SGLang：`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`。vLLM 配方页要求 vLLM ≥ 0.17.0 | README 第 754、785 行；[vLLM recipes](https://recipes.vllm.ai/Qwen/Qwen3.5-4B) |
| 历史中的思考内容 | 只保留“最后一条真实用户消息之后”的 assistant 轮的 `reasoning_content`，也就是当前这一轮工具循环里的思考；更早的轮次只保留最终回答。Best Practices 第 4 条："No Thinking Content in History"，由模板实现 | chat_template.jinja（`loop.index0 > ns.last_query_index`）；README 第 1153 行 |
| 推荐输出长度 | 一般 32,768；数学或编程竞赛题 81,920 | README 第 1147 行 |

**推荐采样参数**（README 第 835–838 行和第 1136–1145 行）：

| 场景 | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|---|
| 思考，通用 | 1.0 | 0.95 | 20 | 0 | 1.5 | 1.0 |
| 思考，精确编程 | 0.6 | 0.95 | 20 | 0 | 0.0 | 1.0 |
| 非思考，通用 | 0.7 | 0.8 | 20 | 0 | 1.5 | 1.0 |
| 非思考，推理 | 1.0 | **0.95 / 1.0** | **20 / 40** | 0 | **1.5 / 2.0** | 1.0 |

最后一行在模型卡里前后不一致：Quickstart 写 0.95 / 20 / 1.5，Best Practices 写 1.0 / 40 / 2.0。做基准时要明确选一组并写进配置。

**官方 agent 分数**：BFCL-V4 50.3，TAU2-Bench 79.9（README 第 205–218 行）。另有注释："TAU2-Bench: we follow the official setup except for the airline domain … applying the fixes proposed in the Claude Opus 4.5 system card"（第 318 行）。4B 的 TAU2 分数（79.9）高于 9B（79.1），这是原表的数，没有核实。

## 2. SiliconFlow（国内站 siliconflow.cn）的文档和模型页

| 项目 | 内容 | 出处 |
|---|---|---|
| 模型 ID | `/v1/models` 返回的 98 个模型里有 `Qwen/Qwen3.5-4B`，`owned_by` 和 `created` 都为空 | 探针 [p01](../runs/m2/siliconflow_probe/p01_models.json) |
| 价格 | 输入 ¥0、输出 ¥0（每百万 token），即免费。同系列 9B 为输入 ¥1.5、输出 ¥12 | [siliconflow.cn/pricing](https://siliconflow.cn/pricing) 页面内嵌的 JSON：`"modelName":"Qwen/Qwen3.5-4B" … "pricing":[{"price":"0","specification":"completion"},{"price":"0","specification":"prompt"}]` |
| 上下文 | `contextLen: 262144` | 同上 |
| 功能标记 | `functionCallSupport: true`，`jsonModeSupport: true`，`vlm: true`，`supportFT: false`；标签为 Tools、视觉、4B、256K、推理模型；描述中写“模型默认启用思考模式（Thinking Mode），支持工具调用” | 同上 |
| 上线时间 | `publishTime` 1772467200，即 2026-03-02，和 HF 版本的 lastModified 是同一天 | 同上 |
| 速率限制 | L0：RPM 1000，TPM 80,000；L1：1200 / 120k；L2：2000 / 160k；L3：4000 / 320k；L4：8000 / 1M；L5：10000 / 5M | 同上（`modelInfo` 字段） |
| 数值精度或量化 | **国内站没有标注**。定价页和 API 文档里都没有 FP8、INT8、INT4 或“量化”的字样 | 同上；对定价页 HTML 做了 grep |
| 精度的旁证 | 国际站 siliconflow.com 的 Qwen3.5-9B 页面标的是 "Precision FP8 · Context length 262K"。国际站没有 4B 的页面。SiliconFlow 博客也说 "Each SiliconFlow model page provides deployment information such as architecture, precision" | [siliconflow.com/models/qwen3-5-9b](https://www.siliconflow.com/models/qwen3-5-9b)；[博客：FP8 Inference](https://www.siliconflow.com/blog/fp8-inference-api-cost-pricing) |
| `enable_thinking` | 布尔值，“在推理模式与非推理模式之间切换。该字段适用于大多数推理模型”。文档没有列出具体模型 | [API 文档：chat-completions](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions) |
| `thinking_budget` | “思维链输出的最大 token 数量”，范围 128–32768 | 同上 |
| `max_tokens` | 文档说明不包含思维链（实测也是这样，见 3.6） | 同上 |
| `min_p` | “该字段仅适用于 Qwen3”。对 Qwen3.5 是否生效，文档没有说 | 同上 |
| `top_k` | 上限 100 | 同上 |
| 文档未列出的参数 | `presence_penalty`、`seed`、`logprobs`、`top_logprobs` | 同上 |

第三方聚合站的信息不可靠。modelcompare.dev 把这个条目标成“community fine-tune or derivative”（[链接](https://modelcompare.dev/models/siliconflow-cn/qwen3-5-4b)），和 SiliconFlow 自己的描述矛盾，本报告不采用。

## 3. 探针结果

### 3.1 模板指纹：服务端的 prompt token 数和官方模板逐个相等

方法：用 HF 版本 `851bf6e` 的 `chat_template.jinja` 和 `tokenizer.json` 在本地渲染每次请求，按 HF 的 jinja 环境，`tojson` 不排序键、不转义非 ASCII；然后分词计数，和 SiliconFlow 返回的 `usage.prompt_tokens` 比较。脚本在 [tools/token_check.py](../runs/m2/siliconflow_probe/tools/token_check.py)，结果在 [template_token_check.json](../runs/m2/siliconflow_probe/template_token_check.json)。

| 情形 | API 返回 | 官方模板渲染 | 说明 |
|---|---|---|---|
| 非思考，单轮（p02、p04、p05、p06、p08、p10） | 23 | 23 | 完全相等。开启思考时会是 21 |
| 思考，单轮（p13 不传参数，p14 传 true） | 18 | 18 | 不传参数时默认开启思考，关闭时会是 20 |
| 带工具（p11、p12、p16、p17） | 341 / 406 / 329 / 339 | 336 / 401 / 324 / 334 | 每次都恰好多 5 个 token。给每个工具的 `function` 补上 `"strict": …` 字段后，四次都完全相等。推测是服务端框架序列化工具时加了这个字段，未证实 |
| 多轮，历史中带 `reasoning_content`（p15） | 473 | 保留思考时 473，去掉思考时 433 | 服务端把请求里的 `reasoning_content` 交给了模板 |

结论：分词器相同，模板相同，包括工具说明、XML 工具格式、默认思考和历史思考的保留规则。如果服务端换成 Hermes 风格的 JSON 模板，工具说明的文字完全不同，token 数不可能和官方模板逐个相等。

这个方法查不了权重和精度：两个不同精度的部署，prompt token 数也一样。

### 3.2 (a) temperature=0 是否可重复

- 同一请求（非思考，`temperature: 0`，max_tokens 80）共发 4 次：3 次成功，返回的文字和 token 数（26）完全相同，见 [p02](../runs/m2/siliconflow_probe/p02_temp0_nothink_run1.json)、[p04](../runs/m2/siliconflow_probe/p04_temp0_nothink_run2_retry.json)、[p05](../runs/m2/siliconflow_probe/p05_temp0_nothink_run3.json)。
- 另 1 次在 28.9 秒后返回 503：`{"code":50508,"message":"System is too busy now. Please try again later."}`，见 [p03](../runs/m2/siliconflow_probe/p03_temp0_nothink_run2_http503.json)。
- 只测了 26 个 token 的短输出。长输出或批处理负载不同的时候，贪心解码不保证逐位一致，这一点没有测。

### 3.3 (b) 采样参数和 seed

- 同时传 `top_k: 20, top_p: 0.8, min_p: 0, presence_penalty: 1.5, frequency_penalty: 0, seed: 12345, temperature: 0.7`，返回 200，没有报错，见 [p06](../runs/m2/siliconflow_probe/p06_sampling_params_seed12345_run1.json)。
- 返回 200 只说明参数被接受，不说明参数生效；API 不返回 logprobs，所以无法验证。
- 同一请求、同一个 seed 再发一次，输出不同：p06 是“这只猫……\n\n三个形容词：\n1. 慵懒\n2. 毛茸茸\n3. 可爱”，[p08](../runs/m2/siliconflow_probe/p08_sampling_params_seed12345_run2.json) 是“那只猫……\n形容词：慵懒、温暖、惬意”。可见 **seed 不能保证可复现**。
- 两次之间还有 1 次请求 120 秒没有响应，客户端超时，见 [p07](../runs/m2/siliconflow_probe/p07_sampling_params_seed12345_run2_client_timeout.json)。

### 3.4 (c) logprobs

- 传 `logprobs: true, top_logprobs: 5`，返回 400：``"Value error, `top_logprobs` is not allowed."``，见 [p09](../runs/m2/siliconflow_probe/p09_logprobs_with_top_logprobs_http400.json)。
- 只传 `logprobs: true`，返回 200，但 `choices[0].logprobs` 为 `null`，**参数被静默忽略**，见 [p10](../runs/m2/siliconflow_probe/p10_logprobs_only.json)。
- 所以没法直接比较 SiliconFlow 和本地的 token 分布，只能用第 5 节的“强制解码打分”间接比较。

### 3.5 (d)(e) 工具调用解析

- **简单调用**（[p11](../runs/m2/siliconflow_probe/p11_toolcall_simple.json)）：返回 `finish_reason: "tool_calls"`，`content: ""`，`arguments` 是合法 JSON：`{"city": "北京", "days": 3, "unit": "celsius"}`，其中 `days` 是整数 3。
  - 输出 50 个 token。按官方 XML 格式分词，同样的调用是 51 个 token（含 `<|im_end|>`）；换成 Hermes JSON 格式是 34 个。所以模型输出的是 XML，由服务端解析。
  - 服务端按 schema 把字符串转成了整数，这和 vLLM `qwen3_coder` 解析器的行为一致：它按参数类型做 `int()`、`json.loads` 等转换（[vLLM 源码文档](https://docs.vllm.ai/en/v0.21.0/api/vllm/tool_parsers/qwen3coder_tool_parser/)）。但无法据此确认服务端用的就是这个解析器。
- **特殊字符参数**（[p12](../runs/m2/siliconflow_probe/p12_toolcall_odd_args.json)）：要求把三行文本原样写入 `content`，第 1 行含双引号和 `&`，第 2 行含反斜杠 `C:\temp\new`，第 3 行是 `</parameter> {"k": [1, 2]}`。
  - 返回的 JSON 合法；引号、反斜杠和换行都正确转义；`line_count` 是整数 3，`overwrite` 是布尔值 true。
  - **第 3 行整行丢了。** 输出 77 个 token；本地按“有第 3 行”计算是 91 个，按“没有第 3 行”计算是 78 个。所以更可能是模型自己没写这一行，而不是解析器截断。
  - 这说明 XML 格式有一个固有缺陷：参数值里没法安全地出现 `</parameter>`。在非流式模式下，vLLM 的解析器用正则匹配到第一个 `</parameter>` 就结束这个值（同一份源码文档）。即使模型写出了这一行，本地 vLLM 也会截断它。这个问题和服务商无关，本地一样存在。
- **强制指定函数**（[p16](../runs/m2/siliconflow_probe/p16_tool_choice_forced_named.json)）：`tool_choice` 指定 `get_weather`，用户只说“你好，今天心情不错”，模型仍然调用了 get_weather，参数是编出来的 `{"city": "Beijing", "days": 1, ...}`。
  - 输出 55 个 token。同一调用的 XML 格式约 52 个 token，纯 JSON 参数约 20 个。所以服务端可能不是“用结构化输出直接生成 JSON 参数”的实现，而 vLLM 的指定函数调用是这样做的（推断，未核实）。
  - 用到强制调用的基准，两边的行为可能不同。
- **思考 + 工具**（[p17](../runs/m2/siliconflow_probe/p17_toolcall_thinking_default.json)）：不传 `enable_thinking`，同时给工具。返回了 `reasoning_content`（62 个 token），`tool_calls` 也解析正确。`content` 是 `"\n\n"`：思考结束后留下的换行没有去掉，见 3.6。

### 3.6 (f) 思考开关和 `thinking_budget`

- **不传 `enable_thinking`**（[p13](../runs/m2/siliconflow_probe/p13_thinking_default_unset.json)）：默认思考，和 HF 模板的默认一致。
  - `max_tokens: 200`，实际 `completion_tokens: 708`，其中 `reasoning_tokens: 651`，耗时 55 秒。**`max_tokens` 只限制最终回答，不限制思考**，和 SiliconFlow 文档一致。本地 vLLM 的 `max_tokens` 限制的是思考加回答的总长度，这是两边语义的差别（vLLM 的语义按 OpenAI 惯例推断，未逐版本核对）。
  - `content` 以 `"\n\n"` 开头。
- **`enable_thinking: true, thinking_budget: 128`**（[p14](../runs/m2/siliconflow_probe/p14_thinking_true_budget128.json)）：
  - `reasoning_tokens: 129`，思考在句子中间被截断，停在“34   (1”。
  - 后面的 `content` 不是答案，而是接着思考往下写的草稿（“\n        ----\n         391\n        ```\n    *   Method 2: …”），最后 `finish_reason: "length"`。
  - 结论：**预算用完后思考被硬截断，模型没有收尾，最终回答被思考内容污染。** 做基准时不要用 `thinking_budget`。如果要限制思考长度，应在本地和 SiliconFlow 上用同一种做法，比如“截断后插入 `</think>` 再续写”。
- **多轮回传 `reasoning_content`**（[p15](../runs/m2/siliconflow_probe/p15_multiturn_reasoning_in_history.json)）：见 3.1。服务端会把请求里的 `reasoning_content` 按官方规则放进 prompt：最后一条用户消息之后的工具循环里保留。不回传时 prompt 少 40 个 token。

## 4. 能不能验证权重相同

**不能直接验证。** 能拿到的都是间接证据：

1. 模型 ID 相同（`Qwen/Qwen3.5-4B`）；描述（4B、256K、默认思考、201 种语言）、上下文长度（262144）、上线日期（2026-03-02）都和 HF 版本一致（见第 2 节）。
2. 分词器和聊天模板逐 token 一致（见 3.1）。这能排除“换了模板”和“换了分词器”。
3. 工具调用是 XML 格式，按 schema 做类型转换，思考内容单独分离。这些行为和官方推荐的 `qwen3_coder` 加 `qwen3` 解析器组合一致（见 3.5、3.6）。
4. temperature=0 时短输出可以重复（见 3.2）。

验证不了的：

- **精度**：国内站没有标注。国际站把 9B 标为 FP8，所以 4B 很可能也是 FP8（推断）。也可能是 BF16，或者别的格式，比如 INT8 或 FP8 KV cache，都不能排除。
- **是否微调过、是否换过版本**：没有版本号；`system_fingerprint` 为空；`/v1/models` 不给创建时间。服务商可以随时静默替换部署。
- **输出分布**：不支持 `logprobs`，`seed` 也不生效，没法直接比较 token 分布。

在没有服务商配合的情况下，最强的间接检验是第 6 节 B 的强制解码打分：拿 SiliconFlow 的贪心输出，在本地 BF16 权重上逐 token 检查它是不是 argmax。

## 5. 和“本地 vLLM + HF 权重 + 官方模板”相比，分数可能因为哪些原因不同

按我估计的影响大小排列。“影响大小”一列是本报告的判断，不是测量结果。

| # | 机制 | 证据 | 影响方向和大小（判断） |
|---|---|---|---|
| 1 | **思考开关不一致**：SiliconFlow 不传参数时默认思考；本地 vLLM 用官方模板，默认也思考。只要有一边显式关了，另一边没关，两边比的就是两种模式 | 3.1、3.6；README 第 712 行 | 很大，可能 5–20+ 分。必须两边都显式设置 |
| 2 | **`max_tokens` 语义不同**：SiliconFlow 不限制思考长度，本地限制思考加回答的总长度。同样的 `max_tokens`，本地更容易在思考中途被截断，任务因此失败 | p13 | 本地分数偏低，偏多少取决于设置。按官方建议给到 32K，可以基本消除 |
| 3 | **`thinking_budget` 截断污染**：最终回答里混入思考草稿 | p14 | 用了这个参数，分数会大幅偏低。不要用 |
| 4 | **网络错误和限流**：一共 16 次 chat 请求，出现 1 次 503（28.9 秒）和 1 次超过 120 秒没有响应；L0 级别 TPM 只有 80k，agent 任务的 prompt 动辄上万 token | p03、p07；第 2 节 | 如果把出错的回合直接算失败，分数偏低，偏多少取决于出错率。必须重试，并单独统计出错率 |
| 5 | **多轮思考的回传**：服务端遵守官方规则，但 OpenAI 风格的客户端常常不把 `reasoning_content` 发回去。另外，本地 vLLM 的新旧版本，响应和请求里这个字段的名字可能不同（`reasoning_content` 或 `reasoning`，需要按所用版本核对） | p15 | 两边用同一个 harness 时，误差相互抵消；两边字段名不一致时，工具循环的 prompt 不一样。估计 0–5 分 |
| 6 | **工具调用解析的边界情况**：XML 不规范时，调用会被丢掉或当成普通文本返回；`<tool_call>` 写在思考里时，会被分进 `reasoning_content`，因为解析器按 `</think>` 切分；强制指定函数的实现也可能不同。参考：Kimi 官方的 K2 Vendor Verifier 测了 K2-thinking，各家服务商的 schema 准确率在 83.05%–100% 之间，其中 SiliconFlow 98.96%，原版 vLLM 87.22%，原版 SGLang 95.52% | p12、p16；[K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier)（测试时间 2025-11-15，模型是 Kimi，不是 Qwen）；[vLLM qwen3_coder 源码](https://docs.vllm.ai/en/v0.21.0/api/vllm/tool_parsers/qwen3coder_tool_parser/)；类似故障见 [Ollama #14493](https://github.com/ollama/ollama/issues/14493)（Ollama 把 Qwen3.5 错配成 Hermes 管线） | 方向不定，本地不一定更好。估计 0–3 分；强制调用多的基准影响更大 |
| 7 | **数值精度（FP8 的可能性大）**：权重或激活量化后，接近平局的 token 会翻转；多步 agent 里，一次偏离会一路累积下去 | 第 2 节的旁证；关于量化的文献："FP8 (W8A8-FP) is effectively lossless across all model scales"（[Kurtic et al., arXiv:2411.02355](https://arxiv.org/abs/2411.02355)，测的是 Llama 3.1 8B–405B，没有测 4B 或 agent 任务） | 通常略偏低。估计 0–2 分。4B 小模型和长程任务可能更敏感；如果其实是 INT4，可能偏低 2–5 分 |
| 8 | **采样参数不可控或没生效**：`seed` 不生效；不支持 `logprobs`；`min_p` 按文档只对 Qwen3 生效；不传参数时的默认值文档没写清（视觉模型一节的例子是 temperature 0.7、top_k 50） | p06、p08、p09、p10；API 文档 | 两边都显式传入全部参数，可以基本消除；否则 1–5 分 |
| 9 | **输出里的空白**：`content` 开头多出 `"\n\n"` | p13、p17 | 按字符串严格比较的评分器会误判，模板回传时会被 trim 掉。修好评分器就是 0 |
| 10 | **部署被静默更换**：没有版本号 | 第 4 节 | 跨时间比较时，基线会漂移；每次评测都要记日期，并跑一组固定探针 |
| 11 | **批处理和内核导致的不确定性**：本地 vLLM 在 temperature=0 时也不保证逐位一致 | 3.2 | 两边都存在，体现为方差，不是系统偏差 |

### 偏差量级估计（本报告判断）

- **只算 FP8 量化本身**：对 TAU2 或 BFCL 这类工具调用基准，系统偏差估计在 0 到 2 个百分点，多半是负的。依据有两条。第一，FP8 W8A8 在 8B 以上模型的常规基准上接近无损（上面的文献）。第二，4B 更小，加上多步累积，所以把上限放宽到大约 2 分。这个数没有直接证据。
- **抽样误差的量级**：通过率 p=0.5 时，二项分布的标准误在 n=100 个任务时约 5 分，n=300 时约 2.9 分。所以 FP8 带来的偏差，靠单次运行基本看不出来；要用多次重复加按任务配对的检验。
- **调用方式的偏差**（第 5 节表中 #1–#5、#8）：任何一项设置不一致，都可能带来 5–20+ 分，远大于量化的影响。只要两边都显式设置并处理错误，这部分可以压到接近 0。
- **工具解析的偏差**（#6）：两边都按官方推荐跑 `qwen3_coder` 时，估计在 0–3 分；但方向不确定。K2VV 的数据显示，原版 vLLM 的 schema 准确率反而低于好几家服务商。

## 6. 要测出偏差，需要做什么

A. **本地 BF16 参照**（要 GPU，按要求本次没有租）：用 HF 版本 `851bf6e` 的权重，vLLM ≥ 0.17，参数 `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`，上下文 ≥128K。用同一个 harness，两边显式设置相同的采样参数和 `enable_thinking`，`max_tokens` 设为 32K 并按“思考加回答”统一口径，不用 `thinking_budget`，回传 `reasoning_content`。每个任务两边各跑 k≥3 次，用按任务配对的检验比较通过率（配对 bootstrap 或 McNemar 检验）。503 和超时重试到成功为止，出错率单独报告。

B. **不需要 GPU 的权重和精度检验（强制解码打分）**：
1. 用约 50 个 prompt，在 SiliconFlow 上以 temperature=0、非思考模式取贪心输出（单条约 100–200 token）。
2. 在本地用 BF16 权重对这些输出做强制解码，每一步检查 SiliconFlow 选的 token 是不是本地的 argmax，并记录 top-1 和 top-2 的概率差。
3. 如果部署和本地逐位相同，SiliconFlow 的输出应该几乎全是本地的 argmax。如果 SiliconFlow 用的是量化版本，应该会有少量不一致，而且集中在 top-1 和 top-2 很接近的位置；如果在概率差很大的位置也大量不一致，就说明权重本身不同，或者被微调过。这些是预期，没有实测。

本机有 4 核 CPU、15 GB 内存，BF16 权重 9.3 GB，在 CPU 上做这件事勉强可行但会很慢；GPU 上很快。这个检验大约需要 50 次 API 调用，超出了本次 20 次的预算。

C. **工具调用一致性**：参照 K2VV 的做法，取基准里几百个工具调用点，比较两边“是否触发调用”的一致程度（F1）和参数是否符合 schema。重点看强制调用（p16 这一类）和参数里有特殊字符的情况（p12 这一类）。

D. **固定监测**：每次评测前，先跑 3.1 的模板指纹和 3.2 的贪心探针，记下日期和输出的哈希。哈希变了，就说明部署可能换了。

## 7. 没有确认的事项

- SiliconFlow 国内站 Qwen3.5-4B 的数值精度和 KV cache 精度。可以问客服，或者用 6B 间接判断。
- 服务端的推理框架和解析器到底是什么。目前只知道行为和 vLLM/SGLang 的 `qwen3_coder` 一致。
- tools 情形下多出的 5 个 token 是不是来自 `strict` 字段。这只是一个能让 token 数完全对上的假设。
- `min_p`、`top_k`、`presence_penalty` 在服务端是否真的生效。
- 强制指定函数在服务端怎么实现。
- 免费模型在高峰期的出错率。本次只有 16 次请求，样本太小。
