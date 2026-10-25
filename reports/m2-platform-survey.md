# M2 选型：国内 GPU 租用平台与 Qwen3.5-4B 的 LoRA-GRPO 多轮工具调用训练框架

> 调研日期 2026-10-25。只做了网页调研，没有注册、充值或租用任何服务。结论和推荐见文末；按本轮实测 token 用量做的时长与费用估算在 [M2 起点报告](m2-start.md) 第 7 节。
> 标注约定：
> - 【官方·见于 2026-10-25】：本人当天直接抓取官方页面看到的数字或文字。
> - 【官方·搜索摘录】：搜索引擎缓存的官方页面文字。官网是 JS 渲染，本人未能在渲染后的页面上复核。
> - 【第三方】：博客、知乎、CSDN、导购站等，日期见括注。
> - "未查到"：没有找到可靠来源。
> 价格单位为人民币元，均为单卡每小时价格，除非另有说明。

---

## 0. 目标模型的关键事实（决定了框架选型）

- **架构是混合线性注意力，而且是 VLM 封装。** `config.json` 中 `architectures=["Qwen3_5ForConditionalGeneration"]`、`model_type="qwen3_5"`。文本部分共 32 层，`layer_types` 按"3 层 `linear_attention`（Gated DeltaNet）+ 1 层 `full_attention`"循环（`full_attention_interval: 4`）。另有 `mtp_num_hidden_layers: 1`、`vocab_size: 248320`、`mamba_ssm_dtype: float32`，并带 `vision_config`。来源：https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/config.json 【官方·见于 2026-10-25】
- **默认开启思考模式，工具调用解析器是 `qwen3_coder`。** 模型卡写明默认输出 `<think>...</think>`，可用 `enable_thinking: False` 关闭。vLLM 推荐参数为 `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`，SGLang 推荐 `--tool-call-parser qwen3_coder`。原生上下文 262,144。模型卡称发布时需要 vLLM、SGLang、transformers 的 main 或最新版本。来源：https://huggingface.co/Qwen/Qwen3.5-4B 【官方】
- **依赖的版本下限（以 ms-swift 的 Qwen3.5 最佳实践为准）：**
  - `transformers>=5.9`
  - `flash-linear-attention>=0.4.2`
  - `causal-conv1d`（从 git 安装）
  - `flash-attn==2.8.3`
  - `vllm>=0.17.0`

  该文档在 RL 实验中统一设置 `enable_thinking false`。来源：https://swift.readthedocs.io/en/v4.4/BestPractices/Qwen3_5-Best-Practice.html 。vLLM 官方 recipe 同样要求 `vLLM >= 0.17.0`，并称单张 24GB 卡可以跑 Qwen3.5-9B：https://recipes.vllm.ai/Qwen/Qwen3.5-9B
- **当前 PyPI 最新版本**（查询 https://pypi.org/pypi/<pkg>/json ，2026-10-25）：

  | 包 | 版本 | 发布日期 |
  |---|---|---|
  | vllm | 0.32.0 | 2026-10-18 |
  | sglang | 0.5.22 | 2026-10-16 |
  | transformers | 5.20.2 | 2026-10-19 |
  | peft | 0.21.2 | 2026-10-01 |

- **国内下载渠道：**
  - ModelScope 有 `Qwen/Qwen3.5-4B`（API 返回 200，架构同上）：https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B
  - hf-mirror.com 可以访问。
  - AutoDL 文档列出了 hf-mirror 和内置的 `source /etc/network_turbo` 学术加速，加速范围包括 huggingface.co：https://www.autodl.com/docs/network_turbo/
- **显存量级（第三方）：** Twinkle 文档称 Qwen3.5-4B 的 bf16 权重约 8GB，单卡 LoRA 训练约 18GB：https://twinkle-kit.readthedocs.io/en/latest/Usage%20Guide/Introduction-with-Qwen3.5.html 。ms-swift 的 Qwen3.5-4B LoRA SFT 示例为 "4 × 20GiB"（`max_length 2048`，ZeRO-2），来源同上面的 ms-swift 文档。

---

## Part A：GPU 租用平台

### A.1 汇总表

| 平台 | 支付宝/微信 | RTX 4090 24G | RTX 5090 32G | A800/A100 80G | H800/H20/L20 等 | 竞价/抢占 | 计费粒度 | API/CLI 开关机 | 实名 |
|---|---|---|---|---|---|---|---|---|---|
| AutoDL | **均支持**【官方】 | 1.88【官方·搜索摘录】；2.19（2026-07 区间中位数）【第三方】 | 2.78【官方·搜索摘录】 | A800 5.59【官方·搜索摘录】；A800 6.29（2026-07）【第三方】 | H800 9.98【官方·搜索摘录】 | 未查到（只有包日/周/月） | 按秒，整点扣费，最低 0.01 元【官方】 | 普通容器实例**没有** API；"容器实例Pro"有创建/开机/关机/释放 API，需要个人实名【官方】 | Pro API 需实名；开放端口需企业认证【官方】 |
| 优云智算 Compshare（UCloud） | **均支持**【官方·见于 2026-10-25】 | 按量 **1.92**，抢占 **1.31**（14C32G）【官方·见于 2026-10-25】 | 按量 3.36，抢占 2.10【官方·见于 2026-10-25】 | A800 按量 7.36，抢占 3.02；A100 80G 按量 10.75，抢占 7.52【官方·见于 2026-10-25】 | H20 96G 按量 7.50（无抢占价）【官方·见于 2026-10-25】 | **有抢占式** | 秒级计费，按小时出账【官方】 | **有** OpenAPI（Start/StopCompShareInstance）+ `pip install compshare-cli`【官方】 | 注册实名送体验时长【第三方】；是否强制实名未查到 |
| 矩池云 matpool | "扫码支付"，官方未写明支付宝/微信【官方】 | 2.20（当日全部售罄）【官方·见于 2026-10-25】 | 2.99（售罄）【官方·见于 2026-10-25】 | 只有 A100 40G 2.80（售罄）【官方·见于 2026-10-25】 | 未见 | 未查到 | 进入运行中开始计费，释放后停止；没有"关机保留"，数据靠网盘【官方】 | 只有 Beta 版 "PaaS API"（任务式，http，Token 需找客服申请），不是实例开关机 API【官方】 | **充值前必须实名**【官方】 |
| 恒源云 gpushare | **均支持**，充值无手续费，退款扣 1%【官方】 | 1.32（2026-04/05 汇总）【第三方】 | 未查到 | 未查到 | 未查到 | 未查到 | 按秒计费，整点结算，开机冻结 1 小时费用【官方】 | 未查到 | 未查到 |
| Featurize | 未查到 | "低至 1.87"【官方·见于 2026-10-25】 | 未查到 | 未见；有 PRO 6000 96G "低至 6.19"【官方】 | 未见 | 未查到 | 按量/包月【官方】 | 未查到 | 未查到 |
| 智星云 | 未查到 | 1.65【官方·见于 2026-10-25】；1.50（2026-05）【第三方】 | 未见 | A100 80G 8.0【官方·见于 2026-10-25】 | 未见 | 未查到 | 未查到 | 未查到（有第三方提及其 OpenAPI，未核实） | 未查到 |
| 仙宫云 | 未查到 | 2.37（2025-03 推广文）【第三方】 | 未查到 | 未查到 | 未查到 | 未查到 | 按分钟（同一推广文）【第三方】 | 未查到 | 未查到 |
| 阿里云 PAI-DSW / ECS | **支付宝**、网银；官方页未列微信【官方】 | 不提供 4090（未见） | — | 未查到官方单价 | gn8is L20 48G：按量 30–40，抢占 3–4（2026-07）【第三方】；gn7i A10 24G：按量 8–12，抢占 0.8–1.2【第三方】 | ECS 有抢占式 | DSW 按分钟计费，每小时出账，有 2–3 小时延迟【官方】 | 有（阿里云 OpenAPI，本次未细查） | **充值前必须实名**【官方】 |
| 腾讯云 HAI | **微信支付**、QQ 钱包、网银；官方未列支付宝【官方】 | 未见 | 未见 | 未见 | 基础型 16GB（T4 级）约 1.2；进阶型 32GB（V100 级）约 3.6【第三方】 | 未查到 | 按秒计费，整点结算【官方】 | 未细查 | 未查到 |
| SiliconFlow 硅基流动 | — | 未发现 GPU 租用 | — | — | — | — | — | — | — |

### A.2 分平台说明与来源

**AutoDL**
- **充值方式**：微信、支付宝、对公汇款；PayPal "即将上线"。【官方·见于 2026-10-25】https://www.autodl.com/docs/price/ （也镜像在 https://api.autodl.com/docs/price/ ）
- **按量计费规则**："开机开始计费，关机结束计费……精确到秒，最低计费0.01元"；"整点扣费一次和关机时扣费一次"。另有包日/周/月。来源同上。
- **价格**：`docs/price` 页写"请以网站显示的现价为准"，没有具体数字。首页价格由 JS 渲染，本人抓取不到。搜索引擎缓存的首页文字为（会员 95 折）：
  - "NVIDIA RTX 4090 / 24GB … ￥1.88 /时"
  - RTX 5090 2.78
  - H800 9.98
  - A800-80GB 5.59

  这些属于【官方·搜索摘录】，未能复核，下单前需要再确认。https://www.autodl.com/
- **第三方实测价**（晨涧云，2026-07-02，各价格区间的中位数）：
  - 4090：2.19/时，月 1196
  - A800 80G：6.29/时，月 3500
  - A100 40G：3.45/时

  https://www.mornai.cn/news/gpu/mornai-and-autodl-gpu-rent-price-comparison-2026-07/
- **API 和实名**："容器实例Pro" API 提供创建、开机、关机、释放、状态查询，Host 为 `https://api.autodl.com`。文档写明"使用容器实例Pro API需进行个人实名认证或者企业认证"。附录的规格 ID 有 `h800`、`5090-p`、`pro6000-p`、`v-48g`（4090-48G）、`4090D` 等，没有列 24G 版 4090 和 A800。https://www.autodl.com/docs/instance_pro_api/
- **普通容器实例与 Pro 的区别**：普通容器实例"不支持API创建管理实例"；Pro 是存算分离，系统盘为网络存储，IO 比本地 NVMe 低。https://www.autodl.com/docs/instance_pro/
- **存储**：
  - 普通实例默认送 50GB 数据盘，扩容另计：https://www.autodl.com/docs/local_disk/
  - 文件存储 20GB 内免费，超出部分 0.01 元/GB/日：https://www.autodl.com/docs/fs/
  - Pro 系统盘 30GB 基础部分 0.10 元/日（见 price 页）。
- **其他限制**：开放端口需要企业认证，个人用户只能走 SSH 隧道。https://www.autodl.com/docs/port/
- **镜像**：API 附录的公共基础镜像多为 CUDA 11.x、torch ≤2.0（见 instance_pro_api 附录）。Qwen3.5 需要 transformers v5、vLLM ≥0.17（对应 torch 2.10），预计要自己装环境或用社区镜像（推断）。

**优云智算 Compshare（UCloud 优刻得）**
- **价格**（官方定价页，【官方·见于 2026-10-25】）：https://www.compshare.cn/price-list

  | 规格 | 按量 | 包日 | 包月 | 抢占式 |
  |---|---|---|---|---|
  | RTX4090 24G，14C32G | 1.92 | 42.33 | 1154.64 | 1.31 |
  | RTX4090 48G | 3.45–3.80 | — | — | 无 |
  | RTX5090 32G，14C48G | 3.36 | — | — | 2.10 |
  | A800 80G，1 卡 16C240G | 7.36 | — | 4420.32 | 3.02 |
  | H20 96G，1 卡 | 7.50 | — | — | 无 |
  | A100 80G，1 卡 | 10.75 | — | — | 7.52 |

  注意：首页另写"4090 仅需 2.15、5090 3.32、A800 6.23"，与定价页不一致，以下单页为准。https://www.compshare.cn/
- **存储**：
  - 云硬盘 100GB 免费，扩容 0.01 元/GB/天
  - 云存储 0.004 元/GB/天
  - 镜像存储 30GB 免费
  - 无卡模式 0.15 元/小时
  - 30TB "免费公共模型库"

  来源：定价页和首页。
- **计费**："按照小时计费的后付费模式，秒级计费"；按量实例"关机后CPU、GPU和内存会被回收并停止收费，云盘和镜像资源会保留并继续收费"。https://www.compshare.cn/docs/operation/charge/billdescribe （页面标注 Last updated 2026-09-23）
- **抢占式实例规则**：https://www.compshare.cn/docs/operation/gpuspot/gpuspot
  - 可能随时被回收，回收后数据保留 7 天。
  - 创建后 1 小时为保护期，期内被系统回收免单。
  - 不能关机，只能删除。
  - 不能转为独占式。
- **API 和 CLI**：
  - StartCompShareInstance 支持 `WithoutGpu` 无卡开机；StopCompShareInstance 关闭抢占式实例需要 `Force=true`。https://www.compshare.cn/docs/gpus/instance/startcompshareinstance 、https://www.compshare.cn/docs/gpus/instance/stopcompshareinstance
  - CLI：`pip install compshare-cli`，支持批量生命周期操作、JSON 输出、可断线恢复的远程任务。https://www.compshare.cn/docs/mcp-guide/cli
- **网络加速**：支持 `.huggingface.co`、`.github.com`、`.pythonhosted.org`、`.wandb.ai` 等域名，容器实例自动配置。https://www.compshare.cn/docs/operation/gpu/uaaa
- **支付**：账号充值文档写明快捷充值"支持渠道：支付宝、微信支付"，另可线下汇款。【官方·见于 2026-10-25】https://www.compshare.cn/docs/uaccount/bill/charge
- **实名**：是否强制实名未查到。第三方称"注册实名后可免费体验10小时RTX40系"：https://ai-bot.cn/compshare/

**矩池云**
- **价格**（主机市场，【官方·见于 2026-10-25】，抓取当时以下卡型全部显示"售罄"）：https://matpool.com/host-market/gpu
  - RTX 4090 24G：2.20
  - RTX 5090 32G：2.99
  - A100 40G：2.80
  - A40/A6000 48G：2.85
  - RTX 3090：1.85（有货）
- **实名与充值**："需要在充值前完成实名认证手续。点击充值后会弹出扫码支付页面"。https://matpool.com/supports/doc-quick-start/
- **计费**："进入运行中状态时开始计费，直到用户停止并释放主机后计费终止"，另有包日/周/月。https://matpool.com/supports/reference/faqs/
- **网盘**：欠费后清空（保留 5G 免费空间），扩容按月计费（见 quick-start）。
- **API**："PaaS API"为任务式接口（创建/查询/取消任务），处于 Beta，"仅支持 http 调用"，Token 需通过微信客服申请。https://apidoc.matpool.com/

**恒源云**
- **计费**：
  - 按量计费"计费时间粒度精确到秒……每小时整点进行一次结算"，创建时冻结 1 小时费用。
  - 未充值只能开 1 个实例；充值不满 100 元只能单卡。
  - 关机后 GPU 不保留，10 天后删除实例。

  https://gpushare.com/docs/product/price/
- **存储**：`/hy-tmp` 本地盘关机 24 小时后清除；"个人数据"云盘 0.0004 元/GB/小时（约 0.0096 元/GB/日）。https://gpushare.com/docs/data/storage/
- **支付**："微信支付、支付宝充值需要手续费吗？不需要交手续费"；退款扣 1%。https://gpushare.com/docs/faq/deposit/
- **GPU 单价**：官方文档不列单价。第三方 2026 年汇总称 4090 约 1.32/时：https://gitcode.csdn.net/6a02e16154b52172bc73777e.html
- **API/CLI 开关机**：未查到。

**Featurize**：首页写 RTX 4090"按量使用低至 ¥1.87/小时"（16 核 EPYC 9354、60GB 内存、750GB 免费磁盘），RTX 3090 1.62，PRO 6000 96G 6.19，另送 30G 免费云盘。【官方·见于 2026-10-25】https://featurize.cn/ 。支付方式、API、实名均未查到。

**智星云**：首页列出【官方·见于 2026-10-25】，其他信息未查到。https://www.ai-galaxy.cn/

| 卡型 | 单价 |
|---|---|
| A100 80G | 8.0 |
| V100 32G | 1.32 |
| A5000 | 1.62 |
| A6000 48G | 2.40 |
| A40 48G | 2.20 |
| 4090 24G | 1.65 |
| 3090 | 1.10 |

**仙宫云**：官网对本人的抓取请求直接断开连接（connection reset），未能核实。第三方推广文（2025-03）称"按分钟计费，4090 一小时 2.37 元"，时效性差：https://www.datalearner.com/blog/1051701266330292 。API 未查到。

**阿里云**
- **充值**：在线充值支持支付宝、银联、个人/企业网银、对公汇款，页面未列微信；"充值前请确保您的阿里云账号已完成实名认证"。https://help.aliyun.com/zh/document_detail/37107.html
- **PAI-DSW 计费**："账单金额=（单价/60）×服务时长（分钟）"，账单有 2–3 小时延迟；系统盘扩容部分在实例停止后仍计费；NAT 网关和 EIP 单独计费；有新用户免费试用和资源包。页面没有 GPU 单价。https://help.aliyun.com/zh/pai/product-overview/dsw-billing-description
- **ECS GPU 价格**（第三方，2026-07-10，社区用户投稿，非官方）：
  - gn7i A10 24G：按量 8–12，抢占 0.8–1.2
  - gn8is L20 48G：按量 30–40，抢占 3–4

  https://developer.aliyun.com/article/1746672 。官方价格页未能抓取到单价，记为未查到。

**腾讯云 HAI**
- **计费**：算力套餐按量"时间粒度精确到秒……每个整点结算"；免费附带 80GB 云盘和每月 500GB 流量，超出部分 0.8 元/GB；关机后云盘 15 天内免费。https://cloud.tencent.com/document/product/1721/101029
- **规格与价格**（第三方）：基础型 16GB（T4 级）、进阶型 32GB（V100 级）；按 7 天日常价 201.6 和 604.8 元折算，约 1.2 和 3.6 元/时。https://zhuanlan.zhihu.com/p/1933113947512345866 、https://www.foreignserver.com/gpu/txyhaigpu.html
- **适用性判断**：V100/T4 不支持 bf16，FlashAttention 等 kernel 支持也差（属于常识推断），不适合本任务。
- **充值**：微信、QQ 钱包、网银，文档未列支付宝。https://cloud.tencent.com/document/product/555/7425

**SiliconFlow 硅基流动**：官方微调指南只提供托管 SFT，可微调模型只列了 Qwen2.5-7B/14B/32B/72B-Instruct，数据格式只有 system/user/assistant 三种角色。没有 RL/GRPO，也没有工具消息格式，未发现 GPU 租用产品。https://docs.siliconflow.cn/cn/userguide/guides/fine-tune 。结论：不能用来训练，最多当作 teacher 轨迹的推理 API（推理价格本次未查）。

### A.3 预算换算（按上表单价，仅作量级参考）

300 元大约能买到：

| 方案 | 单价 | 可用时长 |
|---|---|---|
| AutoDL 4090 | 1.88 | 约 160 小时 |
| AutoDL 5090 | 2.78 | 约 108 小时 |
| AutoDL A800 | 5.59 | 约 54 小时 |
| Compshare 4090 抢占 | 1.31 | 约 229 小时 |
| Compshare A800 抢占 | 3.02 | 约 99 小时 |

另需加上数据盘和镜像存储费用。

---

## Part B：开源 GRPO 框架

### B.1 汇总表

| 框架 | (1) Qwen3.5 支持 | (2) LoRA + vLLM/SGLang rollout | (3) 多轮工具调用、自定义环境与奖励 | (4) 1–2 卡跑 4B | (5) 成熟度/最新版本 |
|---|---|---|---|---|---|
| **verl** | README 列出 Qwen3.5；仓库有 `verl/models/transformers/qwen3_5.py` 补丁 | FSDP+PEFT，支持 vLLM 和 SGLang；SGLang 目前需要 `merge=True` | `AgentLoopBase.run` 可完全自定义；token-in-token-out；`response_mask`；内置 `qwen3_coder` 工具解析器（注释写明"for qwen3_coder/qwen3.5"） | quickstart 为单卡 24GB（0.5B 模型）；第三方 Qwen3.5-4B GRPO 用了 4 卡 | PyPI 0.9.1（2026-09-20）；Agent Loop 标注 alpha |
| **TRL GRPOTrainer** | 工具/agent 训练的"Tested with"列表含 Qwen3.5；内置 Qwen3.5 训练用模板 | `use_peft` + vLLM colocate/server；同步权重前先 `merge_adapter()` | `tools=[...]` 与 `environment_factory`（每个 rollout 一个环境实例，公共方法自动暴露为工具，`reset`、`get_reward`）；`max_tool_calling_iterations`；**environment_factory 标注为 experimental** | colocate 模式单卡可跑（推断，未实测） | 1.16.0（2026-10-15），发版很频繁 |
| **ms-swift** | 有专门的 Qwen3.5 最佳实践（含 GRPO 示例） | 支持 `--tuner_type lora` 的 GRPO；vLLM colocate/server | `MultiTurnScheduler`（`step`/`check_finished`/`run`）、`--max_turns`、`gym_scheduler`；自定义 `run` 时**每轮可作为独立记录**，同一轨迹统一分配奖励 | Qwen3.5-4B LoRA SFT 示例为 4×20GiB；单卡 GRPO 未见示例 | 4.6.0（2026-10-13） |
| **OpenRLHF** | 0.10 起支持 VLM，"train VLMs like Qwen3.5" | README 列出 LoRA/QLoRA（`--ds.lora.rank`）；RL 中 LoRA 怎样同步到 vLLM，README 未说明 | `AgentInstanceBase` 的 `reset`/`step`；token-in-token-out；`--train.agent_func_path`；有异步模式 | Hybrid Engine 让 vLLM 与训练共享 GPU | 0.11.2（2026-09-14） |
| **ROLL**（阿里） | 2026-03-06 起支持 Qwen3.5 Dense 和 MoE（Megatron 示例，80GB） | 支持 LoRA（含 Megatron LoRA） | Agentic pipeline，对齐 GEM 环境接口，有 Tool Use 文档，支持 GiGPO 等 step 级算法 | 有单机 quick start；定位是大规模集群 | GitHub release 日期未查到（PyPI 上的 `roll` 是无关的同名包） |
| **AReaL** | 支持表中 Qwen3.5 在 Megatron/FSDP 两个后端都打勾 | LoRA：Megatron（搭配 vLLM 推理）✅，FSDP ✅；默认推理后端 SGLang | agentic RL 教程；"black-box agent"只需替换 `base_url` | README 有单节点示例 | PyPI 1.0.4（2026-06-10）；README 宣布 AReaL 2.0（2026-07-01） |
| **slime** | README 列出 Qwen3.5 | README 未提 LoRA（未查到） | `--custom-generate-function-path` 可写多轮/工具/环境逻辑 | 基于 Megatron+SGLang，偏重型 | 没有 PyPI 发布；README 有 Qwen3.8 新示例，说明在活跃开发 |
| **SkyRL** | release notes 称 Qwen3.5 支持 FSDP/Megatron/Jax 后端；有 `run_qwen3.5_0.8b.sh` 示例；Qwen3.5 sample packing 曾有 bug（issue #1525） | 支持 LoRA（新闻提到 LoRA recipe 和 multi-LoRA）；Qwen3.5 + LoRA + vLLM 未核实 | skyrl-gym（Gymnasium API）；SkyRLGymGenerator 多轮 token-in-token-out | 未查到单卡示例 | skyrl 0.4.0（2026-10-01） |
| **rLLM** | 取决于后端（verl），未查到明确说明 | 通过 verl 后端 | 包装任意 agent，网关捕获 token ids 和 logprobs；Episode→Trajectory→Step 结构 | verl 后端是多卡分布式；`tinker` 后端是远程服务（国内是否可用未核实） | 不在 PyPI，从 git 安装 |
| **ART**（OpenPipe） | 称"大多数 vLLM/HF 兼容模型"都能用；Qwen3.5 未查到明确说明 | 只做 LoRA；服务端在 vLLM 里加载最新 LoRA，用 Unsloth 训练 | OpenAI 兼容客户端，代码里自己写多轮循环，按 Trajectory 赋奖励 | 适合单卡；训练时推理会阻塞 | openpipe-art 0.5.21（2026-10-20） |
| **Unsloth** | 有 Qwen3.5-4B GRPO/GSPO notebook | notebook 设置 `fast_inference=False`，**没有用 vLLM**；文档不建议对 Qwen3.5 做 4-bit QLoRA | 底层是 TRL GRPOTrainer；多轮工具调用未见官方示例 | 单卡（Colab） | 2026.10.8（2026-10-23） |

### B.2 各框架要点与来源

**verl**（仓库已迁到 verl-project/verl）
- **Qwen3.5 支持**：
  - README："Compatible with Hugging Face Transformers and Modelscope Hub: Qwen3.5, Qwen3, …"；同时列出"Multi-turn with tool calling"和 LoRA RL。https://raw.githubusercontent.com/volcengine/verl/main/README.md
  - Qwen3.5 补丁文件：https://raw.githubusercontent.com/verl-project/verl/main/verl/models/transformers/qwen3_5.py
- **工具解析器**：注册了 `hermes`、`gpt-oss`、`qwen3_coder`（"Tool parser for qwen3_coder/qwen3.5 model"）、`glm`、`seed`、`minimax`。https://raw.githubusercontent.com/verl-project/verl/main/verl/experimental/agent_loop/tool_parser.py
- **ToolAgentLoop 配置项**：`multi_turn.max_user_turns`、`max_assistant_turns`、`max_parallel_calls`、`max_tool_response_length`、`format`（选择解析器）。https://raw.githubusercontent.com/verl-project/verl/main/verl/experimental/agent_loop/tool_agent_loop.py
- **Agent Loop**：
  - 只需实现 `run(sampling_params, **kwargs) -> AgentLoopOutput`，输出 `prompt_ids`、`response_ids`、`response_mask`（工具返回的 token 置 0）。
  - 文档推荐 token 级 `generate`，理由是重新 tokenize 会产生偏差，而这"is critical to RL training"。
  - 每个 prompt **只返回一条轨迹**；整个模块标注 alpha。

  https://verl.readthedocs.io/en/latest/advance/agent_loop.html
- **多轮 tokenization 风险**：文档指出 Qwen3/QwQ 的模板渲染时会剥离历史推理内容，导致 delta tokenization 出错；verl 提供 `tokenization_sanity_check_mode`。https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
- **LoRA**：
  - 关键配置：`lora_rank`、`lora_alpha`、`target_modules`，以及 `rollout.load_format=safetensors`、`layered_summon`。
  - SGLang 需要 `merge=True`。
  - 建议 rank ≥32，学习率提高约一个数量级。

  https://verl.readthedocs.io/en/latest/advance/ppo_lora.html
- **单卡**：quickstart 要求"a GPU with at least 24 GB HBM"，示例 `trainer.n_gpus_per_node=1`。https://raw.githubusercontent.com/verl-project/verl/main/docs/start/quickstart.rst
- **第三方佐证**：有模型卡称用 verl 0.9.0.dev、4 卡 FSDP2 + vLLM 对 Qwen3.5-4B 做了 GRPO。https://friendli.ai/models/ZetaRRR/Qwen3.5-4B-VerIH-step200

**TRL GRPOTrainer**
- **Agent 训练**：`tools`（带类型注解和 Google 风格 docstring 的函数）、`environment_factory`（需 `transformers>=5.2.0`）、`get_reward`、`max_tool_calling_iterations`。"Tested with" 列表含 Qwen3.5。模板必须是"prefix-preserving"的，已知模型族会自动换成训练用模板。另有训练/推理不一致的重要性采样校正（TIS/MIS，`vllm_importance_sampling_mode`）。https://raw.githubusercontent.com/huggingface/trl/main/docs/source/grpo_trainer.md
- **Qwen3.5 训练模板**：`qwen3_5_think_training.jinja` 和 `qwen3_5_nothink_training.jinja`。前者"used by Qwen3.5-4B and larger"，即 **4B 默认开启思考**。改动包括：始终输出 thinking 块以保证前缀不变，并加上 `{% generation %}` 标记。https://raw.githubusercontent.com/huggingface/trl/main/docs/source/chat_templates.md
- **experimental 标注**：源码对 `environment_factory`（和 `rollout_func`）发出 "experimental feature. This API may change" 警告。https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_trainer.py
- **LoRA 同步**：同步 vLLM 前调用 `model.merge_adapter()`，之后 `unmerge_adapter()`。因此 vLLM 端看到的是合并后的权重，**不依赖 vLLM 的 LoRA 模块命名**。https://raw.githubusercontent.com/huggingface/trl/main/trl/generation/vllm_generation.py

**ms-swift**
- **Qwen3.5 最佳实践**：
  - 依赖版本见第 0 节。
  - GRPO 示例：Qwen3.5-2B 全参，vLLM colocate；Qwen3.5-35B-A3B 用 Megatron 做 LoRA GRPO。
  - "RL: All experiments set enable_thinking false"。
  - SFT 用 `--add_non_thinking_prefix true --loss_scale ignore_empty_think`。
  - 已知问题：FLA 训练慢（FLA issue 758）。

  https://swift.readthedocs.io/en/v4.4/BestPractices/Qwen3_5-Best-Practice.html
- **多轮**：
  - `MultiTurnScheduler`：`step` 可返回 `response_token_ids`、`response_loss_mask`、`rollout_logprobs`、`rollout_infos`。
  - 奖励函数通过 kwargs 读取 `rollout_infos` 和 `trajectory_inputs`。
  - 自定义 `run` 时每轮可以是独立记录。
  - 异步引擎只能用于 server 模式。
  - `gym_scheduler` 封装带 `reset`/`step` 的 Env。
  - 文档没有说明 LoRA 与多轮的组合。

  https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/multi_turn.html

**OpenRLHF**：README 的要点（https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/main/README.md ）：
- "token-in-token-out agent execution … zero text-level mismatches"
- 多轮接口：`AgentInstanceBase` 的 `reset`/`step`
- 0.10 支持 Qwen3.5 VLM
- LoRA 合并工具：`openrlhf.cli.lora_combiner`

**ROLL**：README 新闻的要点（https://raw.githubusercontent.com/alibaba/ROLL/main/README.md ）：
- 03/06/2026 支持 Qwen3.5
- 09/23/2025 对齐 GEM 环境，支持 Tool Use
- 07/31/2025 支持 LoRA
- 02/03/2026 支持 Megatron+LoRA 和 FSDP2

**AReaL**：README 的模型支持表和后端表（Qwen3.5 在 Megatron/FSDP 打勾；LoRA 在 Megatron 下需搭配 vLLM 推理，FSDP 下 ✅）。另有 `gsm8k_grpo_lora.yaml` 示例。https://raw.githubusercontent.com/inclusionAI/AReaL/main/README.md

**slime**：README 写明支持 Qwen3.5，以及 Megatron+SGLang、`--custom-generate-function-path`。https://raw.githubusercontent.com/THUDM/slime/main/README.md

**SkyRL**：
- 新闻页：https://raw.githubusercontent.com/NovaSky-AI/SkyRL/main/README.md
- Qwen3.5 后端支持见 release notes：https://github.com/NovaSky-AI/SkyRL/releases
- 示例脚本与 packing bug：https://github.com/NovaSky-AI/SkyRL/issues/1525
- MTP + prefix caching 问题：https://github.com/NovaSky-AI/SkyRL/issues/1981
- 多轮 token-in-token-out：https://docs.skyrl.ai/docs/tutorials/skyrl_gym_generator

**rLLM**：后端为 verl、tinker、fireworks，网关自动捕获 token ids 和 logprobs。https://raw.githubusercontent.com/rllm-org/rllm/main/README.md

**ART**：客户端/服务端架构，"server … runs the model's latest LoRA in vLLM"，"Inference is blocked while training executes"。https://raw.githubusercontent.com/OpenPipe/ART/main/README.md

**Unsloth**：
- README 列出 Qwen3.5 (4B) GSPO notebook：https://raw.githubusercontent.com/unslothai/unsloth/main/README.md
- notebook 代码为 `FastVisionModel.from_pretrained("unsloth/Qwen3.5-4B", fast_inference=False, load_in_4bit=False)`，然后 TRL `GRPOConfig(num_generations=2, max_completion_length=1024, ...)`：https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Qwen3_5_(4B)_Vision_GRPO.ipynb
- Unsloth 文档称 Qwen3.5 需要 transformers v5，不建议 4-bit QLoRA：https://unsloth.ai/docs/models/qwen3.5/fine-tune

### B.3 接入本环境的主要风险

本环境的结构：每个 session 最多 14 轮，可调用工具 `memory_read`/`memory_write`/`memory_delete`/`ask_user`/`choose_option`；同一模拟用户连续 8 个 session 组成一个 episode，记忆在 session 之间延续；奖励按 session 用代码计算。

1. **跨 session 的信用分配。**
   - **问题**：如果把一个 session 当作一条 GRPO 样本，第 k 个 session 里 `memory_write` 的价值要到后续 session 才体现。只用本 session 的奖励会系统性低估"写记忆"的价值，甚至惩罚它。
   - **框架层面的限制**：主流框架都是"一个 prompt 对应一条轨迹"的组内比较（verl 文档写明每个 prompt 返回一条轨迹）。8 个 session 各有独立上下文，无法自然地拼成一条轨迹。
   - **可选做法**：
     - (a) session 级 GRPO：组内样本共享同一个"用户 × session 序号 × 记忆快照"，奖励 = 本 session 奖励 + 对记忆质量的 shaping 项。
     - (b) 用冻结策略把剩余 session 跑完，计算 return-to-go 作为奖励，成本按剩余 session 数线性增加。
     - (c) 用 ms-swift 自定义 `run` 时"每轮/每段作为独立记录、按轨迹统一分配奖励"的机制。
     - (d) 参考 ROLL 等框架支持的 GiGPO 等 step 级优势估计。
   - 以上都是设计层面的推断，需要实验验证。
2. **GRPO 分组的可比性。** 同组样本必须共享同一个模拟用户、随机种子和初始记忆状态，否则组内优势是噪声。TRL `environment_factory` 下，`reset(**kwargs)` 要从数据行读入种子和记忆快照。属于设计推断。
3. **对话模板与工具调用解析不一致。**
   - Qwen3.5 官方工具解析器是 `qwen3_coder`（XML 风格），不是 Hermes JSON。
   - verl 要设置 `multi_turn.format=qwen3_coder`；TRL 依赖自身的 Qwen3.5 训练模板。
   - teacher 轨迹做 SFT 时必须用**同一个模板**渲染，否则 SFT 学到的格式与 rollout 解析器对不上。
   - 用 vLLM 加载 LoRA 适配器做推理时，可能拿到与训练时不同的对话模板（第三方模型卡提示）：https://friendli.ai/models/art-dsit/qwen3.5-4b-no-robots-lora
4. **思考模式产生的 token。**
   - Qwen3.5-4B 默认开启思考（模型卡；TRL 模板说明"think … used by Qwen3.5-4B and larger"）。
   - Qwen3 系列模板会剥离历史 `<think>`，破坏前缀一致性（TRL 和 verl 文档都提到）。
   - 14 轮 × 思考长度会迅速撑长上下文。ms-swift 的 RL 实验统一关闭思考。
   - 建议 teacher 数据、SFT、rollout、评测四处统一设置 `enable_thinking=False`，或者四处统一保留思考。
5. **rollout 引擎与训练引擎的 per-token logprob 不一致。**
   - TRL 文档：vLLM 与训练引擎"can produce different outputs due to precision effects"，提供 TIS/MIS 校正。
   - verl 和 OpenRLHF 强调用 token-in-token-out 避免重新 tokenize 带来的偏差。
   - Qwen3.5 的 Gated DeltaNet 层：训练端走 FLA/causal-conv1d kernel，vLLM 端走自己的 GDN kernel，状态精度为 `mamba_ssm_dtype: float32`。不一致程度可能比纯 attention 模型更大（推断，未核实），应监控 `logprob diff` 和 IS 比率。
6. **LoRA 目标模块与 vLLM 动态加载适配器。**
   - 线性注意力层的投影是 `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj` 等。vLLM Gaudi 后端代码按"有 `in_proj_qkv` 就走 LoRA 拆分路径，否则走融合的 `in_proj_qkvz`"区分：https://docs.vllm.ai/projects/gaudi/en/latest/api/vllm_gaudi/models/qwen3_5.html
   - vLLM 论坛有用户报告：对这些模块训练 LoRA 后动态加载适配器会"一直输出"，先合并再加载则正常（回复是自动答复，未经维护者确认）：https://discuss.vllm.ai/t/vllm-0-23-0-lora/2976
   - 规避方法：用"合并后同步"的路径（TRL 默认合并；verl 可设 `lora.merge`），或在 SFT/RL 中统一目标模块。
   - Qwen3.5 是 VLM 封装，LoRA 应冻结视觉塔（ms-swift 示例为 `--freeze_vit true`）。
7. **上下文长度。**
   - 单个 session 14 轮加上工具返回，`memory_read` 的返回可能很长，需要截断（verl 有 `max_tool_response_length`）。
   - 如果把 8 个 session 拼进一个上下文，长度会成倍增长。
   - vLLM 对混合架构的 prefix caching 支持情况未核实；SkyRL issue #1981 记录了"MTP + prefix caching"在 Qwen3.5/3.6 上的精度问题，建议 rollout 时**不开 MTP 投机解码**。
8. **环境与版本的工程风险。**
   - 需要 transformers v5、vLLM ≥0.17、FLA ≥0.4.2，causal-conv1d 要从源码编译。
   - 租用平台的公共镜像多为旧版 CUDA/torch（AutoDL API 附录），要预留装环境的时间，并保存为镜像。
   - TRL `environment_factory` 和 verl Agent Loop 都标注为 experimental 或 alpha，需要锁定版本。
   - SkyRL 记录过 Qwen3.5 sample packing 的 bug，需要固定 transformers 版本。
9. **训练速度。** ms-swift 文档把"训练慢"列为 Qwen3.5 的已知问题（FLA issue 758）。Unsloth 的 Qwen3.5 GRPO notebook 不用 vLLM，单卡生成会很慢。

---

## Part C：三条训练路线

三条路线：租卡并自己部署兼容 Tinker 接口的开源服务（魔搭 Twinkle、SkyRL 的 Tinker 后端 skyrl-tx）；租卡并用开源训练框架（TRL、verl）；阿里云 PAI（含百炼托管训练）。每条都回答两个问题：

- "一个用户 8 次会话按顺序进行、记忆跨会话保留"的结构怎么放进去；
- 多轮带工具调用的训练循环由我们的代码驱动，还是由框架驱动。

调研方式：读官方文档、README，并对公开仓库做只读浅克隆后看源码，没有运行。三处关键事实另外复核过：
- tinker-cookbook 的 `trajectory_to_data` 文档字符串（"if we get a sequence that's not an extension of the previous sequence…"）；
- Twinkle 的 Tinker 客户端 GRPO 示例默认 `BASE_MODEL='Qwen/Qwen3.5-4B'`，服务端示例配置 `supported_models` 是 `Qwen/Qwen3.5-4B`；
- 百炼强化学习训练支持的模型只有 qwen3.5-9b 和两个 flash 模型，没有 4B，要联系商务经理开通，后付费 41 元/小时/实例。

### C.0 本考场的代码结构对选路线的影响（读自本仓库）

下面是读代码得到的事实，文件为 `roommate/env.py`、`roommate/agent_llm.py`：

1. **一次会话的多轮循环由 `LLMAgent.run_session` 驱动。**
   - 消息格式是 OpenAI 风格：`messages = [system, user]`。
   - 每轮调用 `self.client.chat(body, role=...)`，`body` 里带 `tools`。
   - 返回的 `tool_calls` 由 `_exec` 分发到 `SessionContext.memory_read / memory_write / memory_delete / ask / choose`。
   - 最多 `MAX_TURNS = 14` 轮；`choose_option` 之后 `ctx.done=True`，循环结束。
   - 整段对话写进 `ctx.notes["transcript"]`。
2. **跨会话的状态都在 `Episode` 上。**
   - `Episode.memory` 是 `MemoryStore`，`Episode.user` 是 `SimUser`，策略本身不保存跨会话状态（模块文档字符串原话："策略本身不保存跨会话状态"）。
   - `Episode.run_session` 每调用一次推进一个会话，结束时用 `grade_session(...)` 记一条带分数的 `rec`。
3. **会话边界可以存档和恢复。** `Episode.state_dict() / load_state_dict()` 包含用户状态、记忆、判分记录；`Env.save / Env.load` 存取整个世界。
4. **会话 k 的 prompt 只取决于世界和会话本身，与记忆无关。** prompt 由 `render_session(sess)` 生成，只用会话的周次、用户原话、候选项。记忆只能通过 `memory_read` 工具的返回进入上下文。

由此得出的三点**推断**：

- **(i) 接入点只有 `client.chat`。** 写一个"记录型 client"实现 `chat(body)`：把 `messages + tools` 渲染成 token，调用训练服务的 `sample()`（或 vLLM 的 token 接口），解析工具调用，返回 OpenAI 格式的响应，同时在旁边记下 prompt token、采样 token 和 logprob。这样 `env.py` 和 `agent_llm.py` 基本不用改。
- **(ii) 同一用户的 G 条 episode，第 k 次会话的 prompt token 完全相同。** 因此"按会话序号 k 分组"做 GRPO 是可行的。组内差异只来自策略在这次及以前的行为，即记忆和对话的差异。
- **(iii) 可以按会话做训练样本。** 用 `state_dict()` 在会话边界做记忆快照，就能把"第 k 次会话 + 快照"当作独立的训练样本，这是路线二最自然的做法。不过快照来自旧策略，状态分布是 off-policy 的。

---

### C.1 路线一：租 GPU，自建 Tinker 兼容服务，训练循环写在客户端

#### C.1.1 Tinker API 与 tinker-cookbook 的相关事实

- **四个原语都在客户端调用。**
  - 原语是 `forward_backward(data, loss_fn)`、`optim_step(AdamParams)`、`sample(prompt, sampling_params, num_samples)`、`save_weights_and_get_sampling_client()`。
  - 内置损失有 `cross_entropy`（SFT）和 `importance_sampling`（RL）等。训练数据是一组 `Datum(model_input=token 列表, loss_fn_inputs={target_tokens, weights 或 logprobs/advantages/mask})`。
  - 来源：SkyRL Tinker 概览 https://docs.skyrl.ai/docs/tinker/overview ；skyrl-tx README 示例 https://raw.githubusercontent.com/NovaSky-AI/SkyRL/main/skyrl-tx/README.md
- **tinker-cookbook 的 RL 抽象天然允许"一条轨迹 = 多段独立上下文"。** 这一条对本考场最关键。
  - `Env.step()` 返回的 `StepResult.next_observation` 可以是任意 token 序列，不要求是上一步的延续。
  - `trajectory_to_data()` 的文档字符串原话："if we get a sequence that's not an extension of the previous sequence, then that results in a new Datum"。即观测不再是"前缀 + 新内容"时，自动切出一条新的 Datum。
  - 每条轨迹只有一个标量 advantage（`traj_advantage`），默认是组内中心化的总回报：`compute_advantages` 用的是 `rewards_G - rewards_G.mean()`。
  - 来源：
    - https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py
    - https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/types.py
- **cookbook 有 Qwen3.5 渲染器。**
  - `renderers/qwen3_5.py` 的注释写明 Qwen3.5 的工具调用是 XML 格式 `<function=name><parameter=...>`，并且 HF 模板会"Always adds <think>…</think> blocks to assistant messages after the last user message"。
  - `model_info.py` 里 `Qwen3.5-4B` 对应的渲染器有 `qwen3_5`、`qwen3_5_disable_thinking`、`qwen3_5_preserve_thinking` 三个。
  - 来源：
    - https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers/qwen3_5.py
    - https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/model_info.py
- **版本（PyPI，2026-10-25 查询）：**
  - `tinker` SDK 最新 0.35.0（2026-10-21），Apache-2.0。https://pypi.org/pypi/tinker/json
  - `tinker-cookbook` 0.5.8（2026-10-21），其 `pyproject.toml` 要求 `tinker>=0.34.0`。https://pypi.org/pypi/tinker-cookbook/json
  - 这与下面两个服务端锁定的 SDK 版本冲突，见 1.2、1.3。

#### C.1.2 ModelScope Twinkle（魔搭）

| 项目 | 事实 | 来源 |
|---|---|---|
| 定位 | "client-server training framework"，接口是 Tinker API 的超集，可以用 Tinker 客户端或 Twinkle 原生客户端访问；后端支持 Transformers 和 Megatron；多租户 LoRA 共享一份基座 | https://raw.githubusercontent.com/modelscope/twinkle/main/README.md |
| Qwen3.5 支持 | 支持模型表列出 "qwen3.5 series 2B ~ 27B, transformers>=5.2.0, Megatron ✔"。服务端示例配置 `cookbook/client/server/transformer/server_config.yaml` 和 `megatron/server_config_4b.yaml` 的 `supported_models` 都是 `Qwen/Qwen3.5-4B`。Tinker 客户端 GRPO 示例 `cookbook/client/tinker/short_math_grpo.py` 默认 `BASE_MODEL='Qwen/Qwen3.5-4B'` | README 同上；https://github.com/modelscope/twinkle/tree/main/cookbook/client |
| LoRA | Tinker 客户端只演示 `create_lora_training_client(base_model, rank)`。服务端 Tinker 处理器固定使用 `LoraConfig(r=rank, target_modules='all-linear')`，见 `src/twinkle/server/model/tinker_handlers.py`。在 Tinker 路径上不能自选目标模块（读源码所得） | https://twinkle-kit.readthedocs.io/en/latest/Usage%20Guide/Server%20and%20Client/Tinker-Compatible-Client.html ；https://github.com/modelscope/twinkle/blob/main/src/twinkle/server/model/tinker_handlers.py |
| 损失 | Tinker 桥只接受 `cross_entropy` 和 `importance_sampling`；`ppo`、`cispo`、`dro` 返回 HTTP 400。`importance_sampling` 实际走 Twinkle 的 GRPO 损失（PPO 式 clip、可选 KL、序列平均），文档自称"not an exact match to Tinker's IS objective" | Tinker-Compatible-Client 页（同上） |
| 采样 | Sampler 服务用 vLLM（`sampler_type: vllm`，`engine_args.enable_lora: true`，`logprobs_mode: processed_logprobs`）。客户端-服务端模式下按 URI 动态加载各租户的 LoRA；多轮 RL 示例同步权重时用 `merge_and_sync=False` | server_config.yaml；`cookbook/rl/multi_turn/multi_turn_grpo.py`；`src/twinkle/sampler/vllm_sampler/vllm_engine.py` |
| 硬件 | 服务端文档的示例都是多卡：Transformers 后端 model 1 卡 + sampler 1 卡，Megatron 4B 示例 model 2 卡 + sampler 1 卡。文档写"each component needs to occupy a separate Node"。vLLM 引擎代码里有"Colocated with training (Ray)"和 sleep 模式，但**单卡共置的示例或文档未查到**。Qwen3.5 最佳实践称"LoRA 模式下单卡显存占用大约 18GB"，同页代码注释却写"~8G * 8"，两处矛盾 | https://twinkle-kit.readthedocs.io/en/latest/Usage%20Guide/Server%20and%20Client/Server.html ；https://twinkle-kit.readthedocs.io/en/latest/Usage%20Guide/Introduction-with-Qwen3.5.html |
| 工具调用格式 | 有 `hermes_qwen` 解析器，能解析 `<tool_call><function=…><parameter=…>`（即 qwen3_coder 风格的 XML），也能解析 JSON | https://github.com/modelscope/twinkle/blob/main/src/twinkle/template/tools/qwen.py |
| 多轮 | 原生 `twinkle_agentic.rollout.MultiTurnRollout`：逐轮编码、采样、解析工具、插入桥接 token（`labels=-100`），输出 `input_ids/labels/completion_mask/logprobs`；"Each trajectory runs independently"。另有 API 教师模式 `api_appended_as='demonstration'`，可把外部 API 生成的轮次作为 SFT 演示 | https://twinkle-kit.readthedocs.io/en/latest/Components/Agentic/Rollout.html |
| 成熟度 | 首版 2026-02-13；PyPI `twinkle-kit` 最新 0.5.1（2026-08-14）；main 分支最近提交 2026-10-23。依赖锁 `tinker==0.16.1`（2026-03），落后于 cookbook 要求的 ≥0.34，所以最新 cookbook 不能原样使用（**推断**：要么用旧版 cookbook，要么自写循环）。官方 docker 镜像在阿里云杭州仓库：`modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:twinkle-0.3.0` | README；https://pypi.org/pypi/twinkle-kit/json ；`pyproject.toml` |
| 托管服务（仅供参考，不属于本路线） | ModelScope 免费托管的 Twinkle 训练服务 `base_url=https://www.modelscope.cn/twinkle`，走 Tinker 兼容 API，"backed by one training base at a time"，当前为 Qwen3.8-27B，**不是我们的模型**；需要 ModelScope 账号 | README；https://twinkle-kit.readthedocs.io/en/latest/Usage%20Guide/Train-as-a-Service.html |

**单卡可行性（推断）**：
- **4090 24G**：按文档示例，Twinkle 的 Tinker 服务需要 2 张卡（训练 1、采样 1）。单张 4090 不可行，除非自己改成共置，这条路没有文档。
- **2 × 4090**：按"LoRA 约 18GB"的说法，训练一张、vLLM 一张勉强可行，但 14 轮会话的上下文可能达到数千 token，余量很小。
- **1 × A800 80G**：理论上可以给 Ray 多报一张卡，比如 `ray start --num-gpus=2`，让两个组件挤在同一张物理卡上，各设 `gpu_memory_utilization≈0.4`。这是 hack，未见文档。
- **建议**：用 2 卡机（2×4090，或 A800 加一张小卡）最贴近官方示例。平台是否有 2 卡 4090 机型需要下单前确认，上文 里 Compshare 只列了单卡价格。

#### C.1.3 SkyRL 的 Tinker 后端（skyrl-tx，现为 `skyrl.tinker` + `skyrl.backends`）

| 项目 | 事实 | 来源 |
|---|---|---|
| 定位 | "implements a backend for the Tinker API … your own Tinker-like service running on your own hardware"。仓库重组后，原 `skyrl-tx` 目录的代码移到 `skyrl/{backends, tinker, tx, utils}` | https://raw.githubusercontent.com/NovaSky-AI/SkyRL/main/skyrl-tx/README.md |
| 两种后端 | (1) **skyrl-train 后端**（`--backend fsdp` 或 `megatron`）：FastAPI 接收请求写入 SQLite，后台 engine 批处理后分发给 FSDP/Megatron worker，采样走 vLLM（经 vLLM Router）；(2) 原 tx 的 **JAX 后端**（`skyrl/tx/models/` 下有 `qwen3_5.py`，`tests/tx/models/` 下有 `test_qwen3_5_lora_training.py`） | https://docs.skyrl.ai/docs/tinker/architecture ；仓库源码 |
| Qwen3.5 支持 | 支持模型页称 Megatron 后端经 Megatron-Bridge + vLLM "fully supported" Qwen-3.5 Dense 0.8B/2B/**4B**/9B/27B。FSDP 后端用 `trainer.policy.language_model_only=true` 只加载语言模型部分；示例 `run_qwen3.5_0.8b.sh` 注明 GDN 层加 flash-attn 时 sample packing 有问题，暂时关闭 packing。**但 Tinker 文档和 cookbook 页面的示例都用 `Qwen3-4B-Instruct-2507`，Qwen3.5 走 Tinker 路径的验证记录未查到** | https://docs.skyrl.ai/docs/getting-started/supported_models ；https://docs.skyrl.ai/docs/examples/lora ；https://github.com/NovaSky-AI/SkyRL/blob/main/examples/train/models/run_qwen3.5_0.8b.sh ；https://docs.skyrl.ai/docs/tinker/cookbook |
| LoRA | 由客户端 `lora_rank` 决定，服务端无需额外配置。多租户 LoRA 只有 Megatron 支持。FSDP 的 LoRA 同步方式是把适配器"hot-load"进 vLLM（`_save_lora_adapters_and_sync` → `load_lora_adapter`），不先合并；Megatron 默认 `merge_lora=true`。Tinker 的 `train_attn/train_mlp/train_unembed` 开关尚未接通（源码注释指向 issue #1632），目标模块要在服务端用 `--backend-config` 设置 | https://docs.skyrl.ai/docs/tinker/configuration ；`skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py`；`skyrl/backends/skyrl_train_backend.py` |
| 损失与限制 | 只接通了 `cross_entropy` 和 `importance_sampling`；`sample()` 不返回 prompt logprobs；不支持 KL 惩罚 | https://docs.skyrl.ai/docs/tinker/limitations |
| 硬件 | 默认 `trainer.placement.colocate_all=True`（训练与推理共用 GPU）、`policy_num_gpus_per_node=1`、`inference_engine.num_engines=1`、vLLM `gpu_memory_utilization=0.8` → **默认就是单卡共置**（读 `skyrl/train/config/config.py` 和配置文档所得）。环境要求 "CUDA 13.0 (NVIDIA driver r580 or newer)"；依赖锁 `torch==2.13.0`、`vllm==0.31.0`、`transformers>=5.6.1,<=5.16.1`、`peft==0.18.1`；`tinker` extra 锁 `tinker>=0.25.0,<0.28.0`，同样落后于最新 cookbook | https://docs.skyrl.ai/docs/tinker/quickstart ；`pyproject.toml` |
| 成熟度 | PyPI `skyrl` 0.4.0（2026-10-01）；README 新闻：2025-10-06 发布 SkyRL tx，2026-02-13 宣布实现 Tinker API；Tinker 文档称 cookbook 脚本 "zero code changes" 即可运行 | https://pypi.org/pypi/skyrl/json ；https://raw.githubusercontent.com/NovaSky-AI/SkyRL/main/README.md |

**单卡可行性（推断）**：
- **A800 80G**：默认配置就是单卡共置，4B LoRA 余量充足。
- **4090 24G**：vLLM 默认占 0.8 的显存，要调低，并开训练端 offload，加上 Qwen3.5 248K 词表的 logits 显存，很紧，不建议。
- **前提**：租用平台宿主机的驱动必须 ≥ r580。驱动在宿主机上，容器内无法升级。上文 没有查到各平台的驱动版本，**这是下单前必须问清的硬条件**。

#### C.1.4 其他开源的 Tinker 兼容实现

- **OpenRL（gke-labs/open-rl）**：
  - 定位："implements Tinker compatible API … run on your own infrastructure (machine or a kubernetes cluster)"。来源：https://raw.githubusercontent.com/gke-labs/open-rl/main/README.md
  - 硬件：本地部署"At least 2 GPUs are recommended (one for the vLLM sampler, one for the trainer)"，每卡 ≥23GB。来源：https://github.com/gke-labs/open-rl/blob/main/docs/setup/local-setup.md
  - 采样：单进程模式可以用 torch 在训练进程内采样（`SAMPLING_BACKEND=torch`）。来源：https://github.com/gke-labs/open-rl/blob/main/docs/configuration.md
  - Qwen3.5：源码有专门处理（`automodel_worker.py` 提到 Qwen3.5 的线性注意力投影、MTP 头），显存估算表只列了 9B/27B；4B 未查到。
  - 版本：git tag 最新 v0.0.7，main 分支 2026-10-23 仍有提交；Google 博客介绍发布于 2026-06。
  - 兼容度：自测的 Tinker 方法兼容表为 48 个支持、47 个不支持（基于 `tinker==0.29.0`）。来源：https://github.com/gke-labs/open-rl/blob/main/docs/tinker-client-compatibility.md
- **tiny-open-rl（ShubyM/tiny-open-rl）**：用 Redis Streams 和 PyTorch 实现 Tinker 训练 API，约 370 行，README 明说采样"Not here yet"。只适合阅读学习，不能直接用。来源：https://raw.githubusercontent.com/ShubyM/tiny-open-rl/main/README.md
- **rLLM 的 `tinker` 后端**：见 上文 B.2。它能否指向自建的 Tinker 兼容服务，未查到。

#### C.1.5 路线一回答 (a)(b)(c)

**(a) 8 次会话按顺序进行、记忆跨会话延续：在本路线里是"自然"的。**

- 一条 episode 由我们自己的代码完整跑完：照常调用 `Env/Episode.run_session` 8 次。记忆延续由 `Episode` 自己负责，与训练服务无关。
- 记录型 client 把每次会话的每一轮记成 (prompt_tokens, sampled_tokens, logprobs)，提交方式有两种：
  1. **直接用 tinker-cookbook 的 `Env`**：把整条 episode 包成一个 `Env`。会话边界处 `next_observation` 换成新会话的 prompt，`trajectory_to_data` 会自动按"非前缀延续"切成多条 Datum，大致每次会话一条（来源见 1.1）。缺点是 cookbook 默认整条 episode 只有一个 advantage。
  2. **自己写 advantage**（推荐，**推断**）：Datum 是客户端自己拼的，`advantages` 是逐 token 的张量，可以给会话 k 的 token 填会话级的 return-to-go 优势。做法是组内 G 条 episode 共享用户和世界种子，对每个 k 计算 $R_k=\sum_{j\ge k} r_j$，在同一个 k 的 G 个值之间做中心化或标准化。由 §0 推断 (ii)，同一 k 的 prompt 相同，这样分组是合理的。
- **代价**：一个组要跑 G×8 次会话，episode 内部只能串行；不同用户、不同组之间可以并行（`Env.run` 已经有线程池）。

**(b) 谁驱动多轮循环：我们的代码。**

- **模板和 token 由客户端控制。** 渲染用 cookbook 的 `qwen3_5_disable_thinking` 渲染器，或我们自己用 HF 模板加 `enable_thinking=False`。`sample()` 收到的是我们给的 token，返回采样 token 和 logprob。训练时用的正是采样时用的 token，不需要重新 tokenize。Qwen3.5 模板会剥离历史思考内容（B.3 风险 3、4），即使前缀不连续，cookbook 的切分逻辑也只是多出几条 Datum，不会训练到模型没见过的上下文（**推断**）。
- **logprob 一致性。** `sample()` 返回的是 vLLM 的 logprob（Twinkle 设为 `processed_logprobs`），训练端重算后用 `importance_sampling` 损失校正。Twinkle 实际用的是 PPO-clip 版本（1.2）。Gated DeltaNet 在训练端和推理端走不同 kernel 带来的偏差仍然存在（B.3 风险 5），要监控。
- **复用现有代码。** `env.py` 和 `agent_llm.py` 不用改，只需新增一个实现 `chat(body)` 的 client，约 150–300 行（**推断**）。工具调用的解析可以用 cookbook 的 Qwen3.5 渲染器，或 Twinkle 的 `hermes_qwen` 解析器。另外两点要注意：
  - `LLMAgent` 请求里的 `temperature: 0` 和 `seed` 要由 client 改成 RL 采样参数。
  - 解析失败的轮次可以像 cookbook 那样设 `mask=0`（`PARSE_ERROR_MASKED_METRIC_KEY`）。

**(c) 主要风险与工作量。**

- **风险：**
  1. **两家服务端都没有"Qwen3.5-4B + Tinker 路径 + 多轮工具"的公开端到端验证。** Twinkle 的 Qwen3.5-4B 证据最强，但只有单轮 GSM8K 示例；SkyRL 的 Tinker 路径只验证过 Qwen3。
  2. **SDK 版本锁定。** Twinkle 锁 `tinker==0.16.1`，SkyRL 锁 `<0.28`，最新 cookbook 要求 ≥0.34，只能用旧版 cookbook 或自写循环。
  3. **LoRA 同步走动态加载。** Twinkle 和 SkyRL-FSDP 都是把适配器动态加载进 vLLM，而 B.3 风险 6 记录了线性注意力层 LoRA 动态加载可能出错。Twinkle 的 Tinker 路径还把目标模块固定为 `all-linear`。
  4. **驱动和硬件门槛。** SkyRL 需要 CUDA 13 / 驱动 r580；Twinkle 按官方示例需要 2 卡。
  5. **项目年轻。** Twinkle 首版 2026-02，SkyRL Tinker 2026-02；接口仍在变动。
- **工作量（推断）**：
  - 环境与镜像 1–2 天；
  - 记录型 client、episode 采集、advantage、`forward_backward` 循环 3–5 天；
  - SFT（把老师轨迹的 transcript 渲染成 `cross_entropy` Datum）1 天；
  - 合计约 1–1.5 周。
  - 客户端代码只依赖 Tinker API，以后换服务端（Twinkle ↔ SkyRL ↔ OpenRL）的成本低。

---

### C.2 路线二：租 GPU，用开源训练框架（TRL GRPOTrainer / verl）

框架的基础事实见 上文 Part B（Qwen3.5 支持、LoRA 同步、版本等）。这里只补充与 8 次会话结构有关的内容。

#### C.2.1 TRL GRPOTrainer（PyPI 1.16.0，2026-10-15）

- **`environment_factory` 的形态。**
  - 每次 rollout 新建一个环境实例，公共方法暴露为工具。
  - 给了 `train_dataset` 时，数据行的其他列作为关键字参数传给 `reset(**kwargs)`。
  - `get_reward()` "is called once per completed rollout"。
  - 文档提醒组内初始状态要一致："have `reset()` derive the state deterministically from a shared key"。
  - 来源：https://raw.githubusercontent.com/huggingface/trl/main/docs/source/grpo_trainer.md
- **`rollout_func`（experimental）。**
  - 接收"the list of prompts allocated to the current process and the trainer instance"。
  - 必须返回 `prompt_ids/completion_ids/logprobs`，可选 `env_mask` 屏蔽环境 token。
  - "it is responsible for returning the correct number of completions per prompt"，即每个 prompt 正好 `num_generations` 条，每条是一个序列。
  - 来源：https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_trainer.py
  - TRL 文档把它定位为 `environment_factory` 不适用时的退路："You write the entire generation and environment interaction loop yourself"。https://raw.githubusercontent.com/huggingface/trl/main/docs/source/openenv.md
- **"黑盒"异步路径。** 还有一条实验性的黑盒路径：`AsyncGRPOTrainer` + `HarnessRolloutWorker`，用代理捕获 agent 自己循环里的 token。它绑定 OpenEnv/Harbor 生态（同一 openenv.md）。能否把一次 episode 中的多段调用拆成多行训练数据，未查到。

**(a) 8 次会话：只能"弯折"。**

- **自然做法：按会话做样本。**
  - 一次 rollout 是一个会话。数据行写入 (世界种子, uid, 会话序号 k, 第 k 次会话开始前的 `Episode.state_dict()` 快照)，prompt 列直接放 `render_session` 的文本。
  - `reset(**kwargs)` 用快照恢复 `Episode`；工具方法转调 `SessionContext`；`get_reward()` 返回 `grade_session` 的分数。
  - 问题一：快照要另外生成，比如用老师或上一轮策略跑 episode 存档，属于 off-policy 状态分布，需要定期重新生成。TRL 不提供这个外循环。
  - 问题二：本会话的分数不包含"写记忆"对后续会话的价值（B.3 风险 1）。
- **想要 return-to-go：**
  - 一种做法是在 `get_reward()` 里用冻结策略把剩下的会话跑完。但 colocate 模式下，环境拿不到推理引擎（**推断**），需要另起一个 vLLM 服务，或调用外部 API。
  - 另一种做法是用 `rollout_func` 自己跑完整 episode。可以把"同一用户的 8 次会话"当作 8 个 prompt 放进同一批，`rollout_func` 内部串行跑 G 条 episode，再对第 k 个 prompt 返回 G 条第 k 次会话的序列。由 §0 推断 (ii)，prompt_ids 一致，满足 TRL 的形状约定（**推断**，未见先例）。这样做要求一个进程拿到同一用户的全部 8 个 prompt：单卡时可以控制，多卡要自己处理切片。生成和工具解析都要自己写，相当于在 TRL 里重写路线一的客户端循环，还要依赖实验性 API。
- **结论：** 按会话做样本是自然的；跨会话的回报要么用 shaping 近似，要么依赖实验性的 `rollout_func`。

**(b) 谁驱动多轮循环：框架（`environment_factory` 模式）。**

- TRL 负责采样、解析、执行工具、拼 token。
- 工具调用循环要求模板 "prefix-preserving"，已知模型族会自动换成训练用模板：Qwen3.5 有 `qwen3_5_think_training.jinja` 和 `qwen3_5_nothink_training.jinja`（上文 B.2）。这意味着训练和 rollout 用的模板与部署时的官方模板不同（**推断**：评测时也要用同一个模板，否则存在分布偏移）。
- logprob 一致性靠 TIS/MIS 校正（上文 B.2）。
- **复用现有代码**：`SessionContext` 的工具可以原样转调，但 `LLMAgent` 的循环、NUDGE 提示（不调工具时插入的提醒）、14 轮上限要改用 TRL 的机制重新实现：`max_tool_calling_iterations`，以及 TRL 默认"不调工具就结束"。语义与现有代码不同：现有代码不调工具时会插入 NUDGE 继续下一轮。

**(c) 风险与工作量。**

- **风险：**
  - `environment_factory` 和 `rollout_func` 都是 experimental；
  - 单会话样本带来信用分配偏差；
  - "不调工具就结束"的行为与考场不一致；
  - 快照分布偏移。
- **优点**：单卡 colocate、合并后再同步 vLLM（避开 LoRA 动态加载问题，上文 B.2）、Qwen3.5 在 "Tested with" 列表里、社区最大。
- **工作量（推断）**：按会话做样本的版本 3–5 天（环境类、快照数据集、奖励）；加上 return-to-go，再加 1 周左右（`rollout_func` 或外部推理服务）。

#### C.2.2 verl（PyPI 0.9.1，Agent Loop 标注 alpha）

- **事实。**
  - `AgentLoopBase.run(sampling_params, **kwargs) -> AgentLoopOutput`。
  - `AgentLoopOutput` 只有一组 `prompt_ids / response_ids / response_mask / response_logprobs`，外加一个 `reward_score`。
  - 来源：https://raw.githubusercontent.com/verl-project/verl/main/verl/experimental/agent_loop/agent_loop.py ；文档写明每个 prompt 返回一条轨迹（上文 B.2）。

**(a) 8 次会话：** 与 TRL 相同，自然的做法是按会话做样本，快照通过数据集的 `extra_info` 或 kwargs 传入 `run()`。如果在 `run()` 里跑完整 episode，就只能把 8 次会话拼成一条 response。这样训练时第 k 次会话会"看到"前面会话的文本，与 rollout 时的真实上下文不符，**不可取**（**推断**）。要一个 episode 产出多段样本，需要 rLLM 这类上层封装（Episode→Trajectory→Step，上文 B.2），或改 verl 的数据流。

**(b) 谁驱动循环：** 框架调度，但 `run()` 内部完全由我们写，token 进 token 出，是 verl 推荐的做法。

**(b) 对现有代码的影响：** 要把 `LLMAgent` 的循环改写成异步 token 级循环，调用 `server_manager.generate`，用内置的 `qwen3_coder` 解析器。`SessionContext` 可以复用。

**(c) 风险与工作量：** 配置复杂；多卡和 Ray 栈重；跨会话回报同样要弯折。工作量（**推断**）约 1–2 周。上文 已建议把 verl 留作放量备选。

（ms-swift 自定义 `run` 时"每轮可作为独立记录、按轨迹统一分配奖励"，是开源框架里离"一个 episode 多段样本"最近的机制，见 上文 B.2。本文没有进一步核实。）

---

### C.3 路线三：阿里云 PAI（以及百炼）

#### C.3.1 产品与能力事实

- **Model Gallery 一键微调。**
  - 官方 Qwen3 快速开始写明"Qwen3-32B/14B/8B/4B/1.7B/0.6B 模型已支持SFT（全参/LoRA/QLoRA微调）和GRPO训练"。
  - GRPO 的数据格式、能否自定义奖励代码、是否支持多轮工具，该页都未说明；第三方转载的 PAI 系列文章同样没有这些细节。
  - **Qwen3.5 在 Model Gallery 的微调或 GRPO：未查到**。搜到的 Qwen3.5 资料只涉及一键部署。
  - 来源：https://help.aliyun.com/zh/pai/use-cases/deploy-fine-tune-and-evaluate-qwen3-in-quickstart/ ；https://segmentfault.com/a/1190000046878263
- **PAI-ChatLearn（开源，PAI 团队）。**
  - 支持 GRPO/GSPO，FSDP/Megatron + vLLM/SGLang，2025/9 起支持 Agentic RL。
  - README 最新新闻停在 2025/10；main 分支最近提交 2025-10-23；PyPI `pai-chatlearn` 最新 1.2.0（2025-08-25）。
  - 浅克隆后全仓 grep 不到 Qwen3.5；LoRA 只在 Megatron 配置里出现。
  - **结论：已停更约一年，不支持 Qwen3.5，不适用。**
  - 来源：https://raw.githubusercontent.com/alibaba/ChatLearn/main/README.md ；https://pypi.org/pypi/pai-chatlearn/json ；PAI 官方教程（Qwen3-8B，DSW 调试 → DLC 多机）https://help.aliyun.com/zh/pai/use-cases/qwen3-reinforcement-learning-best-practices
- **ROLL（阿里开源）**：README 称 2026-03 支持 Qwen3.5，有 Agentic pipeline 和 GiGPO 等 step 级算法（上文 B.2）。PAI 上 ROLL 的官方镜像或教程未查到；在 DLC 上只能当作自定义镜像任务来跑（**推断**）。
- **百炼（Model Studio）的托管训练**，离"托管多轮 RL"最近，但不满足我们的条件：
  - **SFT**：Qwen3.5-4B 只支持 SFT（全参和高效），¥0.015/千 Token，不支持 DPO。https://help.aliyun.com/zh/model-studio/model-training-overview
  - **RL 支持的模型**：只列了 qwen3.5-9b、千问3.6-flash、千问3.5-flash，**没有 4B**；需要"联系商务经理开启强化学习训练功能"。
  - **RL 的接入方式**：Rollout 和 Reward 是用户写的 Python 类，提交时注册到函数计算（FC）。示例超参 `"algorithm": "gspo"`；示例 agent 用 LangGraph 编排，通过 MCP 调用工具。
  - **RL 计费**：只能按训练单元（MTU）计费，后付费"41.00 元/小时/实例"，9B 推荐 24 个，demo 最低 24 个。**推断**：约 984 元/小时起。
  - 来源：https://help.aliyun.com/zh/model-studio/rl-training-overview/
  - **Rollout 接口**："Rollout 通过 OpenAI 兼容 API 调用训练中的模型"，每个样本调用一次 `process`，输出 `messages + reward_score`。页面没有提到 token 或 logprob，也没有提到 LoRA。https://help.aliyun.com/zh/model-studio/rl-function-development-guide
- **价格（官方定价页，华北2北京）：**
  - PAI-DSW / PAI-DLC 公共资源按量：`ecs.gn7i-c8g1.2xlarge`（A10 24G ×1）13.9812 元/时；`ecs.gn7e-c16g1.4xlarge`（GU100 ×1）38.2162 元/时。GU100 是否就是 A100/A800 80G 级，页面没写，属于**推断**。
  - PAI-EAS（华东2上海）：A10 10.5 元/时。
  - 来源：https://cn.aliyun.com/page-source/price/detail/machinelearning_price?from_alibabacloud=
  - 对比 上文：Compshare A800 按量 7.36、抢占 3.02；4090 1.92。
  - DLC 计费页附录还列了 L20（gn8is）、H20（gn8v）等规格，但没有单价。https://help.aliyun.com/zh/pai/product-overview/billing-of-dlc
- **计费规则**：DSW 和 DLC 按分钟计费，账单延迟 2–3 小时；DSW 有新用户免费试用（额度未写）和抵扣资源包（只适用 gn7i、gn6v 等）；DSW 是否有抢占式实例，页面未提及。来源：https://help.aliyun.com/zh/pai/dsw-billing-description
- **支付与实名**：在线充值支持支付宝；"充值前请确保您的阿里云账号已完成实名认证"（上文 A.2）。

#### C.3.2 路线三回答 (a)(b)(c)

- **(a) 8 次会话。**
  - **托管的部分（Model Gallery GRPO、百炼 RL）**："每个样本调用一次 rollout、返回一组 messages 加一个分数"，与 TRL/verl 一样是"一条 prompt 对应一条轨迹"。要处理 8 次会话，只能按会话做样本、把快照放进数据。在百炼 RL 的 rollout 函数里跑完整 episode 再只回传一段 messages，问题与 verl 拼接相同。是否允许一次 rollout 回传多段独立对话，未查到。
  - **非托管的部分（DSW/DLC 上自己装 TRL、verl 或 Tinker 兼容服务）**：与路线一、二完全相同，只是换成阿里云的卡。
- **(b) 谁驱动循环。**
  - 百炼 RL：rollout 代码是我们写的，但模型通过 OpenAI 兼容 API 调用，tokenization 和 logprob 由平台处理，我们看不到也控制不了（**推断**，依据是页面没有提 token 和 logprob）。
  - Model Gallery GRPO：是否支持多轮工具，未查到。
  - DSW/DLC 自建：同路线一、二。
- **(c) 风险与工作量。**
  - 没有可用的托管方案：Qwen3.5-4B 的托管 RL 未查到；百炼 RL 不支持 4B，需要商务开通，最低约 984 元/小时（**推断**）。
  - 自建的话，同等能力的卡比 上文 里的租用平台贵 2–5 倍（A10 13.98 对比 4090 1.92；GU100 38.2 对比 A800 7.36），而且要实名。
  - 工作量等于路线一或二，再加上熟悉 PAI 控制台和镜像的时间。

---

### C.4 对比表

| 维度 | 路线一：自建 Tinker 兼容服务 | 路线二：TRL / verl | 路线三：阿里云 PAI / 百炼 |
|---|---|---|---|
| Qwen3.5-4B 证据 | Twinkle：官方示例默认就是 Qwen3.5-4B（单轮 GRPO）；SkyRL：后端支持 Qwen3.5-4B，Tinker 路径未见验证 | TRL "Tested with" 含 Qwen3.5；verl 有 Qwen3.5 补丁和 `qwen3_coder` 解析器 | Model Gallery 只写了 Qwen3；百炼 RL 无 4B |
| 8 次会话结构 | **自然**：我们的代码跑整条 episode；cookbook 自动按非前缀切成多条 Datum；可以逐 token 写会话级 return-to-go 优势 | 按会话做样本自然；跨会话回报要弯折（shaping、外部推理，或实验性 `rollout_func`） | 托管部分同路线二且更受限；自建部分同路线一、二 |
| 谁驱动多轮循环 | 我们（客户端调用 `sample` → 拼 Datum → `forward_backward`） | 框架（`environment_factory` / AgentLoop） | 百炼：我们写 rollout，平台管 token |
| 模板与 logprob 控制 | 完全自控，训练的 token 就是采样的 token | TRL 换成训练模板，用 TIS/MIS 校正；verl token 级 | 百炼不可见 |
| 复用 env.py / agent_llm.py | **几乎原样**，新增一个 `chat()` client | 工具可复用，循环要重写，NUDGE 语义不同 | 同路线二，或要搬到 FC |
| LoRA → vLLM | Twinkle、SkyRL-FSDP 都是动态加载适配器（B.3 风险 6） | TRL 合并后同步；verl 可设合并 | — |
| 单卡 | SkyRL 默认单卡共置（需驱动 r580）；Twinkle 示例需 2 卡 | TRL colocate 单卡可跑；verl quickstart 单卡 | DSW/DLC 单卡可开，但贵 |
| 成熟度 | 年轻（2026-02 起），SDK 版本锁定冲突 | TRL、verl 最成熟，但相关 API 是 experimental/alpha | ChatLearn 停更；百炼 RL 要商务开通 |
| 主要成本 | 2×4090 或 1×A800，按 上文 单价 | 1×4090 或 1×A800 | A10 13.98 元/时、GU100 38.2 元/时；百炼 RL 约 984 元/时起（推断） |
| 工作量（推断） | 1–1.5 周 | 按会话做样本 3–5 天；加 return-to-go 约 +1 周 | 不划算 |

---

## 推荐（平台 × 框架）

**推荐：**
- **路线**：走路线一。训练循环由我们的代码驱动，按 Tinker 接口写（采样 → 拼训练样本 → forward_backward）。服务端先试魔搭 Twinkle，SkyRL 的 Tinker 后端作备选。
- **保底**：路线二的 TRL GRPOTrainer，按会话做样本。
- **不用**：阿里云 PAI。
- **平台**：优云智算 Compshare，备选 AutoDL。

### 平台：优云智算 Compshare

满足卡片的硬条件且证据最全的一家：

- **付款**：账号充值文档写明快捷充值支持支付宝、微信支付（https://www.compshare.cn/docs/uaccount/bill/charge ）。
- **价格是当天在官方定价页上看到的**，不是搜索摘录（https://www.compshare.cn/price-list ）：
  - 4090 24G 按量 1.92，抢占 1.31；
  - A800 80G 按量 7.36，抢占 3.02；
  - 5090 32G 按量 3.36，抢占 2.10。
- **能用接口开机关机**：OpenAPI 有 StartCompShareInstance / StopCompShareInstance，另有 `pip install compshare-cli`（链接见 A.2）。可以写成"开机 → 训练 → 存检查点 → 关机"的脚本，不用盯着控制台。
- **其他**：秒级计费；关机后不收 GPU 费，云盘照收；有 huggingface.co、github.com、pythonhosted.org 加速，有 30TB 公共模型库；无卡模式 0.15 元/小时，可以先在无卡模式里装环境。
- **风险**：
  - 抢占实例随时可能被回收，不能关机只能删除，必须频繁存检查点。
  - 首页和定价页的价格不一致，下单前以下单页为准。
  - 是否强制实名未查到。

**备选 AutoDL**：

- 支付宝、微信已在官方文档确认，按秒计费，有学术加速。
- 搜索摘录到的单价更低（4090 1.88、A800 5.59），但没能在官方页面复核。
- 普通实例没有开关机接口，只有"容器实例 Pro"有，而且要实名；Pro 规格表里没有 24G 的 4090 和 A800。

**不推荐**：

- 矩池云：当天 4090、5090、A100 全部售罄，充值前要实名。
- 腾讯云 HAI：只有 T4、V100 级别的卡，不支持 bf16。
- 阿里云：没有官方单价可查，充值前要实名。
- 恒源云、智星云、Featurize、仙宫云：单价或开关机接口查不到官方来源。

**卡型**：

- 路线一走 Twinkle 时，试跑按官方示例租 2 卡机：2 × 4090 24G，训练一张、vLLM 一张。Compshare 只列了单卡价格，2 卡机型要下单前确认。
- 走 SkyRL 或 TRL 时，单卡就能跑（vLLM 和训练放在同一张卡上）。4B 的 bf16 权重约 9.3GB，24G 卡会很紧，A800 80G 更稳。
- 正式训练用 A800 80G，能开更大的组和更长的上下文。

### 路线：一（自建 Tinker 兼容服务，循环在我们这边）

决定性的是"8 次会话"这个结构（C.0、C.4）：

- **路线一里它是普通的 Python。**
  - 我们照常用 `Env` 把一个用户的 8 次会话按顺序跑完，记忆由 `Episode` 自己保存，和训练服务无关。
  - 每次会话切成一条训练样本。tinker-cookbook 的 `trajectory_to_data` 遇到"新观测不是上一段的延续"就自动切出新样本。
  - 优势按会话写 return-to-go：组内 G 遍共享同一用户，第 k 次会话的 prompt 完全相同，按 k 分组做中心化。
  - 本卡测的"组内差异"就是在回答这样分组有没有信号（[起点报告](m2-start.md) 第 4 节）。
- **路线二、三里它要弯折。**
  - TRL 的 `rollout_func`、verl 的 `AgentLoopOutput`、百炼的 rollout 都是"一条 prompt 对应一条序列"。
  - 自然的做法只有"按会话做样本、数据里放旧策略跑出来的记忆快照"，状态分布是 off-policy 的。
  - 跨会话的回报要么用 shaping 近似，要么靠实验性 API 自己重写循环。
- **谁驱动循环**：
  - 路线一：我们的代码。只需给 `LLMAgent` 换一个能记录 token 和 logprob 的 `client.chat`，`env.py`、`agent_llm.py` 不用改；训练用的 token 就是采样时的 token。
  - 路线二：框架。TRL 会换成训练专用模板；模型不调工具时直接结束对话，和考场"插提醒、继续下一轮"的规则不同（C.2.1）。

服务端怎么选：

- **先试 Twinkle**：Qwen3.5-4B 的证据最直接（官方示例默认就是它），镜像在阿里云杭州仓库，模型从魔搭下载。
  - 官方示例训练、采样各一张卡，所以试跑要租 2 卡机。
  - Tinker 路径上 LoRA 目标模块固定为 `all-linear`，并动态加载进 vLLM，正撞上 B.3 风险 6，冒烟测试第一项就查它。
- **备选 SkyRL**：默认单卡共置，1 张 A800 能跑；但要求宿主机驱动 ≥ r580，Tinker 路径上没有 Qwen3.5 的验证记录。
- **共同风险**：两家锁定的 `tinker` SDK 版本都落后于最新 tinker-cookbook 要求的 ≥0.34，要用旧版 cookbook 或自己写循环（本来也打算自己写 advantage）；项目都很年轻（2026-02 起）。

**保底 TRL**：两家服务端在 Qwen3.5 上都不稳定时，退到 TRL 的 `environment_factory`，按会话做样本。好处是单卡 colocate、合并后同步 vLLM、Qwen3.5 在测试列表里。记录型 client、快照、奖励函数的代码大部分可以复用。verl 留作放量。

**不用 PAI**：
- Model Gallery 的 GRPO 只写了 Qwen3；PAI-ChatLearn 已停更、不支持 Qwen3.5。
- 百炼托管 RL 最小 9B，要商务开通，41 元/小时/单元，推荐 24 个单元起。
- 自建的话，单价是上面租卡平台的 2–5 倍（A10 13.98 元/时、GU100 38.2 元/时），而且要实名。

**工作量（推断）**：路线一约 1–1.5 周；TRL 按会话做样本 3–5 天，加跨会话回报再加约 1 周。

### 接入本考场的主要风险（按影响排序）

1. **跨会话的回报**：
   - 第 k 次会话写的记忆，要到后面的会话才见效。所有框架都是"一条 prompt 对应一条轨迹"，没有现成的"一次经历 = 8 条独立上下文"。
   - 路线一的做法：同组 G 遍共享同一用户，按会话序号 k 分组，第 k 次会话的 prompt 完全相同；优势用本次会话的分，和"本次及以后各次之和"（return-to-go）两个版本对照。
   - 路线二的做法：按会话做样本、数据里放记忆快照，跨会话回报只能近似。
   - 本轮测的组内差异（[M2 起点报告](m2-start.md) 第 5 节）就是为了看这两种分组下有没有可学的信号。
2. **模板与工具解析**：
   - Qwen3.5 的工具调用是 XML 格式（`qwen3_coder`），不是 Hermes JSON。
   - 老师轨迹、SFT、rollout、评测四处必须用同一个模板渲染、同一个解析器。
   - 本轮在硅基流动上实测的工具调用格式错误见起点报告第 3 节；本地 vLLM 的解析器可能更差，也可能更好，见 [硅基流动与 HF 权重对照](m2-siliconflow-vs-hf.md) 第 5 节。
3. **思考模式**：
   - Qwen3.5-4B 默认开思考。官方模板会去掉早于最后一条用户消息的思考内容，破坏前缀一致。
   - 14 轮加上思考，会迅速撑长上下文。
   - 建议四处统一关思考（ms-swift 的强化学习实验就是这样）。本轮思考开、关两组的对比见起点报告第 2 节。
4. **rollout 引擎与训练引擎的 logprob 不一致**：Gated DeltaNet 层在训练端用 FLA kernel，在 vLLM 端用另一套 kernel，偏差可能比纯注意力模型大（推断）。要盯住重要性采样比，TRL 有 TIS/MIS 校正。
5. **LoRA 目标模块**：
   - 冻结视觉塔。
   - 线性注意力层的投影层要不要挂 LoRA，SFT 和 RL 两处要一致。
   - 同步权重走合并路径。
6. **版本与环境**：需要 transformers v5、vLLM ≥0.17、FLA ≥0.4.2，causal-conv1d 要从源码编译。平台镜像普遍偏旧，先在无卡模式里装好、存成镜像，再开卡。
7. **rollout 不开 MTP 投机解码**：SkyRL 记录过 MTP 加前缀缓存在 Qwen3.5/3.6 上的精度问题。

### 试跑前先做的验证（花费很少）

1. 下单前问清：宿主机驱动版本（SkyRL 要 ≥ r580），以及有没有 2 卡 4090 机型（Twinkle）。
2. 在 Compshare 无卡模式下装好环境、拉好 Twinkle 镜像，存成镜像。
3. 开 2 卡机，用官方 `server_config.yaml`（Qwen3.5-4B）起 Twinkle，跑几步 `short_math_grpo.py`。确认三件事：`sample()` 带 logprob；LoRA 能同步到 vLLM；动态加载适配器后输出不异常（B.3 风险 6）。
4. 写记录型 client，用思考关的 Qwen3.5 渲染器跑 1 个用户的 8 次会话，和硅基流动的结果对比格式错误率、兜圈子比例（起点报告第 2 节）。
5. **在训练平台上用 BF16 权重重测起点**：本考场种子 1–3、思考关、温度 0 全跑一遍，作为训练的起点和对照（硅基流动的部署精度未知，见 [对照报告](m2-siliconflow-vs-hf.md)）。
6. 跑 10 步 RL：记录每次会话切出的样本数、采样端和训练端的 logprob 差、显存峰值、每步耗时，用它们替换起点报告第 7 节估算里的吞吐假设。
7. Twinkle 不通就换 SkyRL（单卡 A800，`--backend fsdp`），重复第 3–6 步；都不通就退到 TRL。
