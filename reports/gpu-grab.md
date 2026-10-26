# 真机测试：自动抢卡、开机自检、销毁

真机测试通过，第二步所需条件都具备。

做了两次真机测试，都在第 1 次查库存时抢到 4090，测完都已销毁。

- **第 1 次**：脚本有问题。镜像自带的 Python 没有 pip，hf-mirror 返回了 429，所以下载和模板测试没跑成，2.4 分钟后销毁。
- **第 2 次**：修好后经你同意再开一台，四项都通过，17.9 分钟后销毁。

| 检查项 | 结果（第 2 次） |
| --- | --- |
| 连得进去 | 开机后 104 秒能用密码登录 |
| 驱动和 CUDA | 驱动 595.80，CUDA 13.2，显卡 RTX 4090 24G |
| 从国内镜像下载 | 分词器从 ModelScope 下载 4 秒；权重 9.32 GB 从 ModelScope 下载 560 秒 |
| 模板往返测试 | 用官方分词器重跑，13 条结果和 CPU 上的逐条一致 |

- **余额**：余额够。
- **实际扣费**：不写入公开分支，已在会话里单独报告。
- **上限**：单价 ≤11 元/时、单次运行 ≤45 分钟、本卡总花费 ≤10 元，都没超过。

结果文件都在 `chain-results` 分支：

| 运行 | 结果 | 机器上的检查 | 库存记录 |
| --- | --- | --- | --- |
| [40296742134](https://github.com/BillBeam/good-roommate/actions/runs/40296742134)（第 1 次） | [result.json](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/grab/40296742134/result.json) | [remote_result.json](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/grab/40296742134/remote_result.json) | [stock_log.jsonl](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/grab/40296742134/stock_log.jsonl) |
| [40297931688](https://github.com/BillBeam/good-roommate/actions/runs/40297931688)（第 2 次） | [result.json](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/grab/40297931688/result.json) | [remote_result.json](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/grab/40297931688/remote_result.json) | [stock_log.jsonl](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/grab/40297931688/stock_log.jsonl) |

下文的数字，没有特别说明的都来自第 2 次运行。

## 1. 怎么抢卡

### 触发

会话往 `chain-request` 分支推送请求文件，内容是 `mode: grab`，再加 5 小时窗口。工作流 [chain-probe.yml](../.github/workflows/chain-probe.yml) 收到后：

- 先由 plan 作业读出模式，再交给 grab 作业。
- grab 作业跑完后，不管成败，cleanup 作业都会运行一次，销毁所有名为 `gr-probe-*` 的实例。

### 抢卡规则

规则写在 [ops/gpu_grab.py](../ops/gpu_grab.py)。

- **查库存的频率**：每 30 秒最多查一轮。一轮查遍 5 个可用区、所有合适的卡型。
- **合适的卡**：
  - 显存 ≥24 GB，且原生支持 BF16。可选卡型有 3090、4090、4090 48G、5090、A100、A800、H20 等。
  - 按量计费，单价 ≤11 元/时。
  - 单卡、100 GB 系统盘。
  - 多个可选时，选最便宜的。
- **只开 1 台**：
  - 开机前如果账号里已有任何实例，就放弃。
  - `MaxCount` 设为 1，每次运行只发一次创建请求。只有平台返回"售罄"时才重试，最多 3 次。
- **到点销毁**：
  - 开机后 38 分钟内必须测完。
  - 开机后 40 分钟开始销毁。
  - 同时设了平台的定时关机，在开机后 44 分钟兜底。
- **账户信息不公开**：PR 评论和 `chain-results` 分支里只写"余额够"或"余额不够"。平台的报错信息里数字一律打码。
- **窗口内没抢到**：5 小时后停下，并报告库存记录。这次两次运行都在第 1 轮就抢到，没有走到这一步。

### 两次开机时的库存

两次都只查了 1 轮。下表是查到的有货规格。

| 时间（UTC） | 有货（可用区 卡型 核数 内存 单价 元/时） |
| --- | --- |
| 09:04:29（第 1 次） | 华北二A 4090 16 核 64G 2.05 / 16 核 94G 2.24；上海二A 5090 14 核 48G 3.84 / 14 核 64G 3.87 |
| 09:14:02（第 2 次） | 华北二A 4090 16 核 64G 2.05 / 16 核 94G 2.24；上海二B 4090 16 核 64G 2.05；华北一C 4090 14 核 32G 2.13 / 48G 2.16 / 64G 2.19；上海二C 4090 48G 20 核 64G 3.78 / 90G 3.84 |

除上表外，其他卡型都无货：华北二A 的 5090、4090 48G、3090、A800、H20、A100，以及上海二B 的 5090、4090 48G。

## 2. 开机与连接

两次开的都是同一种机器：

- 卡型：RTX 4090 24G
- 可用区：华北二A（cn-wlcb-01）
- 规格：16 核，64G 内存
- 单价：**2.05 元/时**（按量）
- 镜像：vLLM v0.25.1，100 GB 系统盘

| | 第 1 次 | 第 2 次 |
| --- | --- | --- |
| 创建接口返回 | 18.4 秒 | 11.5 秒 |
| 开机到运行中 | 76.2 秒 | 73.9 秒 |
| **开机到能登录** | **107.9 秒** | **103.6 秒** |
| 实例存活时长 | 2.38 分钟 | 17.86 分钟 |
| 销毁（Terminate 一次成功） | 19.7 秒 | 18.9 秒 |

- 登录方式：平台给的是映射到 23 端口的 SSH，用户 root，密码登录。
- 密码和 IP 在 Actions 日志里都被遮盖，没有写进任何文件。

### SSH 稳定性

登录前先从运行器测了 SSH 连接。运行器在美国的 Azure 机房。

| | 第 1 次 | 第 2 次 |
| --- | --- | --- |
| TCP 连接 20 次 | 19 次成功 | 19 次成功 |
| 建连中位数 | 228.7 毫秒 | 207.1 毫秒 |
| 建连最大值 | 242.4 毫秒 | 224.0 毫秒 |
| ssh-keyscan 握手 3 次 | 全部成功，每次约 3.2 秒 | 全部成功，每次约 2.9 秒 |

两次都在 20 次里失败了 1 次，原因没有查清，可能是刚变成"运行中"时 sshd 还没就绪。之后的登录、上传、15 分钟的远程命令，都没有断过。

## 3. 机器上的检查结果

检查脚本是 [ops/gpu_remote_check.py](../ops/gpu_remote_check.py)，没有训练。各步骤用时如下（第 2 次）：

| 步骤 | 用时 |
| --- | --- |
| 找 Python 环境（含 apt 装 pip） | 45.8 秒 |
| 显卡检查 | 0.2 秒 |
| pip 装 modelscope、jinja2、tokenizers | 276.5 秒 |
| 下载分词器 | 50.5 秒 |
| 模板往返测试 | 1.6 秒 |
| 下载权重 | 560.5 秒 |
| 查平台模型库 | 1.7 秒 |

### 显卡、驱动和 CUDA

- `nvidia-smi`：`NVIDIA GeForce RTX 4090, 595.80, 24564 MiB, 8.9`，即驱动 595.80、显存 24564 MiB、算力 8.9（Ada）。
- 驱动支持的最高 CUDA 版本是 13.2。
- 机器上装了 CUDA 12.8、13.0、13.2 三套工具包（`/usr/local/cuda-*`）。
- 系统 Ubuntu 22.04.5，16 核，内存 62G。
- 这套驱动和 CUDA 够第二步用：Qwen3.5-4B 的 vLLM 配方要求 vLLM ≥0.17，vLLM 0.17 以后的版本按 CUDA 12.x/13 构建。
- **BF16 没在 GPU 上实测**。算力 8.9 的卡在硬件上原生支持 BF16，但这个镜像里没有 torch，计划中的 BF16 矩阵乘没跑成。
- **镜像名和内容不符**：镜像叫 "vLLM v0.25.1"，但它的 `/usr/bin/python3`（3.10.12）里没有 torch，没有 vLLM，连 pip 都没有。脚本只好用 apt 装 pip。第二步要么换带 torch 的镜像（例如 `cuda132_torch2130_py312`），要么开机后自己装 vLLM。

完整输出见 [remote_result.json](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/grab/40297931688/remote_result.json) 里的 `steps.gpu`。

### 磁盘空间

| | 大小 | 可用 |
| --- | --- | --- |
| 根目录 `/`（overlay），开机时 | 79G | 76G |
| 根目录，下载完权重后 | 79G | 67G |
| `/model`（平台共享模型库，UPFS） | 40T | 13T |

- 平台模型库 `/model` 里有 HuggingFace、ModelScope、comfyui、ollama 等目录。
- 往下找了 4 层，**没有 Qwen3.5-4B**，只有一个 ComfyUI 用的 `qwen3.5_4b_bf16.safetensors` 文本编码器，不能直接当权重用。

### 下载

**分词器**

| 来源 | 用时 | 结果 |
| --- | --- | --- |
| ModelScope（curl） | 6 个文件共 **4.0 秒** | 全部成功；`tokenizer.json` 12.8 MB 用 1.1 秒。`generation_config.json` 在 ModelScope 上不存在，返回 404 |
| hf-mirror（固定到 CPU 测试用的版本 `851bf6e`） | 44.0 秒 | 成功，但触发了 429 限流，`tokenizer.json` 重试第 3 次才拿到 |
| ModelScope SDK | 1.8 秒 | 成功 |

**文件校验**

- `chat_template.jinja` 的 sha256 是 `a4aee8af…f715`，和仓库里 CPU 测试的参照 [hf_ref](../runs/m2/siliconflow_probe/hf_ref/chat_template.jinja.sha256) 相同。
- 两个来源的 `tokenizer.json` 都是 `5f9e4d49…cb42`。

**权重**

| | 数值 |
| --- | --- |
| 来源 | ModelScope（curl），2 个分片 |
| 总大小 | 9,319,828,096 字节 |
| 总用时 | **560 秒** |
| 平均速度 | 16.6 MB/s |
| 分片 1 | 5.33 GB，318 秒 |
| 分片 2 | 3.99 GB，235 秒 |
| 结果 | 两个分片都下载成功 |

用的是单连接 curl。换成 ModelScope 的多线程下载器应该更快，这一点未测。

**pip**

- 镜像自带的 pip 源是清华源。脚本另外指定了阿里云的源。
- 装 modelscope、jinja2、tokenizers 用了 276 秒，偏慢。

### 模板往返测试

- **方法**：用下载下来的官方 `chat_template.jinja` 和 `tokenizer.json`，在机器上重跑 [token_check.py](../runs/m2/siliconflow_probe/tools/token_check.py)。13 次探针请求逐个渲染、分词、计数，再和仓库里 CPU 上的结果 [template_token_check.json](../runs/m2/siliconflow_probe/template_token_check.json) 比较。
- **结果**：从 ModelScope 和 hf-mirror 两个来源下载的分词器，结果都和 CPU 上的**逐条一致**，13 条里 0 条不一致。所以 [M2 报告](m2-siliconflow-vs-hf.md)的结论在官方分词器上成立：SiliconFlow 用的就是官方模板和分词器。

## 4. 销毁与花费

- **销毁**：
  - 两次都是 Terminate 一次成功，约 19 秒后实例消失。
  - cleanup 作业随后又查了一遍，没有剩余实例。
  - 本会话最后直接查了 3 个地域，实例列表都为空。
- **余额**：余额够。
- **扣费**：本卡实际扣费不写入公开分支，已在会话里单独报告，远低于 10 元上限。
- **单次运行上限**：两次分别存活 2.4 分钟和 17.9 分钟，都没到 45 分钟。

## 5. 对第二步的提示

- **镜像**："vLLM v0.25.1" 镜像里实际没有 vLLM 和 torch。第二步换带 torch 的镜像，或者开机后自己装，并把安装时间算进预算。
- **权重下载**：单连接约 9 分钟。可以改用多线程下载，或者第一次下载后存到云盘或自制镜像里，下次不用重下。
- **pip**：装包要 4–5 分钟，可以考虑做一个装好环境的自制镜像。
- **hf-mirror**：有限流，主要用 ModelScope。
- **SSH**：开机后前几秒的连接可能失败，要带重试。之后的连接一直稳定，延迟约 210–230 毫秒。
- **库存**：这次两次都在第 1 轮就有 4090，而 06:10 UTC 前后查的时候各卡型几乎全部无货（见[链路探针报告](chain-probe.md)第 4 节），库存随时段变化很大。
