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

## 推荐（平台 × 框架）

**推荐：平台用优云智算 Compshare，备选 AutoDL；框架用 TRL GRPOTrainer 跑试跑，verl 作为放量备选，ms-swift 作为国内文档最全的替补。**

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

- 试跑先用 1 张 4090 24G。4B 的 bf16 权重约 9.3GB，vLLM 和训练放在同一张卡上会比较紧。如果显存不够，换 5090 32G（3.36 元/小时）或 A800。
- 正式训练用 1 张 A800 80G，能开更大的组和更长的上下文。

### 框架：TRL GRPOTrainer（试跑）

- **Qwen3.5 有官方测过的记录**：GRPO 文档的工具/agent 训练 "Tested with" 列表里有 Qwen3.5（已复核：https://raw.githubusercontent.com/huggingface/trl/main/docs/source/grpo_trainer.md ）。仓库里还带了 Qwen3.5 训练用的前缀一致模板，思考开、关各一份。
- **形态和本考场一一对应**：`environment_factory` 每次 rollout 建一个环境实例，环境的公开方法就是工具。本考场的 `memory_read`、`memory_write`、`memory_delete`、`ask_user`、`choose_option` 原样就是工具；`reset(**kwargs)` 从数据行读入"世界种子、用户、第几次会话、记忆快照"；`get_reward` 返回代码判的分。文档写明，组内要同一个初始状态，就让 `reset()` 从共享的键确定性地生成状态。这正是 GRPO 分组需要的。
- **LoRA 先合并再同步到 vLLM**：避开了"线性注意力层的 LoRA 适配器在 vLLM 里动态加载会出错"的风险（见风险 6）。
- **单卡**：vLLM 与训练可以放在同一张卡上（colocate），一个进程就能跑，对单人最省事。
- **风险**：
  - `environment_factory` 标注为 experimental，要锁定版本。
  - 一次 rollout 只对应一次会话。跨会话的回报怎么算，要自己在数据和奖励里设计，见下面"接入本考场的主要风险"第 1 条。

### 放量备选：verl

- 社区和成熟度最好。
- 内置 Qwen3.5 用的 `qwen3_coder` 工具解析器。
- `AgentLoopBase.run` 可以完全自定义，直接处理 token、带 `response_mask`。
- LoRA 支持 vLLM 和 SGLang。
- 代价是配置复杂，Agent Loop 还是 alpha，每个 prompt 只返回一条轨迹。

### 国内替补：ms-swift

- 中文文档，有 Qwen3.5 专门的最佳实践，强化学习一律关思考。
- 自定义多轮 `run` 时，每轮可以作为单独一条记录、按整条轨迹统一给奖励。对"8 次会话组成一次经历"可能更顺手，待验证。

### 接入本考场的主要风险（按影响排序）

1. **跨会话的回报**：
   - 第 k 次会话写的记忆，要到后面的会话才见效。所有框架都是"一条 prompt 对应一条轨迹"，没有现成的"一次经历 = 8 条独立上下文"。
   - 试跑建议按会话做 GRPO：同组样本共享"用户、第几次会话、记忆快照"，prompt 完全相同。
   - 奖励先只用本次会话的分，再加一个 return-to-go 版本做对照：本次之后，用冻结策略把剩下几次会话跑完，把分数加进来。
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

1. 在 Compshare 无卡模式下装好环境、存成镜像。
2. 开 1 张 4090，用 vLLM ≥0.17 起 Qwen3.5-4B（`--tool-call-parser qwen3_coder`，`enable_thinking=False`），用本考场跑 1 个用户的 8 次会话，和硅基流动上的结果对一对。
3. 用 TRL 的 Qwen3.5 训练模板渲染老师轨迹，检查前缀一致性。
4. 跑 10 步 LoRA GRPO，记录 vLLM 与训练端的 logprob 差异、显存峰值和每步耗时，用它替换起点报告第 7 节估算里的吞吐假设。
