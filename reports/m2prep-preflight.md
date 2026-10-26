# M2 租卡前的三项前提检查（不花钱）

2026-10-25。全程没有创建实例，花费 0 元。查询客户端只放行 Describe/Get/Check/List 开头的接口（[tools/compshare_ro.py](../tools/compshare_ro.py)），密钥只从环境变量读取。两个变量都存在；它们的值没有打印，也没有写进文件或提交。

## 一句话结果

| 前提 | 结论 |
|---|---|
| ① 优云智算接口能调通 | **能**。签名正确，查库存、查价、查余额都能返回。但**此刻 4090_48G 和 A800 在所有可用区的库存都是 0**，账户余额是 0.00 元 |
| ② 本会话能 SSH 到外部主机 | **不能**。改用 GitHub Actions 做中转：开机、远程启动任务、取回结果、销毁实例都由它完成。还缺一样凭据：**把优云智算的 API 密钥对填进仓库的 Actions secrets** |
| ③ 聊天模板逐轮拼接与整段套模板一致 | **不一致，会删掉前面轮次的思考**。模型文件被拦，token 层面先用非官方分词器做了，**官方分词器的逐 token 比较挪到第二步** |

## ① 优云智算：库存、单价、未充值能调哪些接口

接口地址是 `https://api.compshare.cn/`，可用区为 cn-wlcb-01（华北二A），计费方式是按量（Dynamic），单位元/小时。原始返回见 [compshare_readonly.json](../runs/m2prep/compshare_readonly.json)。

| | 4090_48G ×1（16 核 / 96G） | A800 80G ×1（16 核 / 240G） |
|---|---|---|
| 本账号价格（`GetCompShareInstanceUserPrice`） | **3.80** | **7.36** |
| 通用查价接口（`GetCompShareInstancePrice`）此刻返回 | 3.61（打九五折） | 6.99（打九五折） |
| 系统盘 100G | 0 | 0 |
| 系统盘每多 100G | +0.05 | +0.05 |
| 数据盘 500G（SSD） | +0.25 | +0.25 |
| **合计：系统盘 100G + 数据盘 500G** | **4.05** | **7.61** |
| 8 卡整机（不含数据盘） | 32.50 | 57.54 |

- **库存**（快照时间 2026-10-25 14:01 UTC，即北京时间周日 22:01）：4090_48G 在华北二A和上海二B都是 0，A800 是 0，A100 有 2 张，V100S 有 6 张。抢占式库存全部是 0，而且 4090_48G 本身不支持抢占式。只看了这一个时间点，**下一步应定时轮询库存，看一天里什么时候有卡**。
- **不充值能调用的接口**：查可用区、查库存、查机型、查镜像（含社区镜像和自定义镜像）、查实例（目前 0 台）、查价（两个接口都能用）、查余额（`GetBalance`，结果是 0.00）、查项目、查用户信息、查已加入的团队，都返回 RetCode 0。团队相关的接口返回参数错误或"无团队权限"，和余额无关。
- **没有验证的事**：创建实例的接口按约定没有调用，所以"余额为 0 时能不能开机"没有实测，大概率不能。按 SDK 文档，Dynamic 是"按小时预付费"，关机不收费的只有 Postpay，而且只在部分可用区提供。所以跑完要直接**销毁**实例，只关机不能保证停止计费。

## ② 本会话能否 SSH 出去

- 所有主机的 22 端口都连不上：github.com:22 和任意公网 IP 的 22 端口都直接拒绝，经代理隧道连 22 端口会超时。
- HTTPS 只放行白名单里的域名：example.com、1.1.1.1 都返回 403，ssh.github.com:443 也返回 403。租来的机器拿到的是任意公网 IP，本会话连 HTTPS 都打不过去。
- 能用的出口只有：`api.compshare.cn`、PyPI，以及 GitHub（通过 MCP 工具，可以触发 Actions、读写本仓库）。

**可行方案：用 GitHub Actions 中转。本会话只负责触发任务和读取结果**

1. 本会话通过 MCP 触发 `workflow_dispatch`，参数是任务名、提交号、时长上限 `MAX_HOURS`。
2. Actions 运行器（可以 SSH 出去）完成以下步骤：用 secrets 里的密钥调 `CreateCompShareInstance`；登录方式只能选密码（接口没有 UserData 或开机脚本参数），密码在运行器内存里随机生成；用 `UpdateCompShareStopScheduler` 设置服务端定时关机，作为兜底；轮询 `DescribeCompShareInstance` 拿到公网 IP。
3. 运行器 SSH 进机器：克隆仓库到指定提交，从 ModelScope 下载模型，用 `nohup` 启动任务。任务脚本最后执行 `shutdown -h now`，另外还设了一个 `shutdown -h +MAX` 硬超时。
4. 回收工作流（每 30 分钟定时跑一次）：列出全部实例，对已经关机或超过时限的实例先用 scp 取回 `results/`，用 Actions 自带的 `GITHUB_TOKEN` 提交到 `results/<run_id>` 分支，然后调 `TerminateCompShareInstance` 销毁。这样机器上不放任何 GitHub 凭据。
5. 本会话从结果分支读取结果。止损线有三道：任务正常结束后自行关机、服务端定时关机、回收工作流销毁实例。

**还缺的那一样凭据**：仓库 Actions secrets 里的优云智算 API 密钥对（`COMPSHARE_PUBLIC_KEY` / `COMPSHARE_PRIVATE_KEY`）。本会话没有写入 secrets 的权限，也不应该把环境变量里的值转写出去，只能由你在仓库的 Settings → Secrets and variables → Actions 里填写。建议另建一个子账号密钥，只开优云智算权限，并设消费上限。

除此之外开跑前还要两样东西，但都不是凭据：充值，否则余额 0 开不了机；确认 GitHub 运行器能访问 `api.compshare.cn` 和国内 IP 的 22 端口，这一点目前未验证，第一次运行时就能看出来。

## ③ Qwen3.5 聊天模板往返测试（CPU）

- **模型文件被拦截的域名**：`huggingface.co`、`hf-mirror.com`、`www.modelscope.cn`，代理都返回 CONNECT 403。所以官方的 `tokenizer_config.json` 和 `tokenizer.json` 拿不到。
- **替代来源**（都来自 PyPI）：
  - 模板取自 xinference 4.1.0 内置的 `qwen3.5` 模板，存为 [qwen35_chat_template.jinja](../runs/m2prep/qwen35_chat_template.jinja)。
  - 分词器用的是第三方 tiktoken 移植 `qwen-tokenizer` 0.3.0，它的特殊 token 编号和 xinference 记录的停止符（248044、248046）对得上。两者都不是官方文件。
  - 渲染器和 transformers 5.20.2 自带的 `render_jinja_template` 做了对照，24 种渲染逐字一致。
- **测试会话**：两个用户问题，6 个助手轮次，每轮都带思考；第一个问题里有 ask_user → write_memory → choose_option 的工具循环。脚本是 [tools/qwen35_template_roundtrip.py](../tools/qwen35_template_roundtrip.py)，结果在 [template_roundtrip.json](../runs/m2prep/template_roundtrip.json)。

| 比较 | 结果 |
|---|---|
| 逐轮拼接 vs 整段套模板，文本 | 不一致：分别是 2861 和 2719 个字符，从第 1675 个字符开始不同 |
| 逐轮拼接 vs 整段套模板，token（非官方分词器） | 不一致：分别是 763 和 699 个 token，从第 423 个 token 开始不同 |
| 分块编码后拼接 vs 同一段文本整体重新编码 | 一致（763 个 token），说明分块的边界不会改变切词 |
| 在最后一个用户问题处重新套一次模板，之后逐轮累加 vs 整段套模板 | **逐 token 一致** |
| 第 t+1 轮的提示词是否以"第 t 轮提示词 + 第 t 轮生成"开头 | 工具循环里都是；新用户消息进来的那一轮不是 |

**会不会删掉前面轮次的思考：会。** 模板会找出最后一条真正的用户消息（以 `<tool_response>` 包着的工具返回不算），在它之前的所有助手轮次都只渲染正文和工具调用，`<think>…</think>` 被整段删掉。这一点不看 `reasoning_content` 字段还是正文里的 `<think>`，两种写法都会删。最后一个用户问题之后的工具循环保留思考。本例中第 1–4 轮的思考被删，第 5–6 轮保留。

**对 M2 的含义**

- 训练样本要按"每条真正的用户消息"切段：每段开头按模板重新渲染，段内逐轮累加 token。这样和推理时看到的完全一致（上表第 4 行）。如果整段会话一路累加不重新渲染，训练时看到的前文比推理时多出前面轮次的思考。
- 现在的环境里，用户对 ask_user 的回答走的是 `tool` 消息，不会触发删除。但模型只说话、没调工具时，[agent_llm.py](../roommate/agent_llm.py) 会把提醒（NUDGE）作为 `user` 消息塞进去，**这会把本次会话前面的思考全部删掉**。M2 里应把它改成工具返回，或者在切段时把它当作新的一段。
- 模板要求工具调用的 `arguments` 是字典。现在的代码存的是 JSON 字符串，直接套模板会报错 `Can only get item pairs from a mapping`，要先解析成字典再套。

**挪到第二步的事**：网络放行 Hugging Face 或 ModelScope 之后，或者在租来的机器上，用官方 `Qwen/Qwen3.5-4B` 的分词器和模板重跑同一个脚本，确认逐 token 结果和这里一致。另外还要对比模型实际采样出的 token 和同一段文本重新编码后的 token，这一项只有在 GPU 上才能做。
