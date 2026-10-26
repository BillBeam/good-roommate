# 链路探针：云端会话 → GitHub Actions → 优云智算

链路已通，开机测试最少充 3 元。

不开机能验的环节都验过了，花费 0 元，没有创建任何实例。

- 会话推送请求后约 1 分钟拿到结果，会话被 PR 评论唤醒。
- 运行器能和国内主机的 22 端口完成 SSH 握手。
- 优云智算的只读接口全部调通，开机参数合法。

两处不通：

- 会话不能直接用 `workflow_dispatch` 触发工作流（403）。
- 定时触发晚了 7 分钟以上还没跑，已删掉。

最少充值：

- 按计划的 4090 规格做 30 分钟测试，实际花约 1.03 元。
- 余额至少要够一小时（2.05 元），所以最少充 3 元。
- 4090 当时全部无货，换 4090 48G 要充 4 元。

结果文件都在 `chain-results` 分支：

| 运行 | 触发 | 内容 |
| --- | --- | --- |
| [40276121645](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40276121645/result.json) | 推送 | 主结果：余额、查价、开机参数校验、SSH |
| [40277985533](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40277985533/result.json) | 推送 | 同上，加各卡型库存 |
| [40278185947](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/result.json) | 推送 | 同上，加计费文档原文（[docs/](https://github.com/BillBeam/good-roommate/tree/chain-results/runs/chain/40278185947/docs)） |

## 1. 触发与回传

工作流 [chain-probe.yml](../.github/workflows/chain-probe.yml) 有两种触发方式：

- 手动触发（`workflow_dispatch`）。
- 推送到专用分支 `chain-request`，只认这一个分支。

它不响应 `pull_request`、`pull_request_target`、`issue_comment` 等别人能引发的事件。运行时读触发提交里的 `ops/probe_request.json`，拿到请求时间和要评论的 PR 号。结果推到 `chain-results` 分支，再在 PR 上留一条摘要评论。

### 推送触发：实测三次

| | 第 1 次 | 第 2 次 | 第 3 次 |
| --- | --- | --- | --- |
| 内容 | 基本探针 | 加库存扫描 | 加文档抓取 |
| 会话推送请求 | 05:47:19 | 06:07:19 | 06:09:30 |
| 运行创建 | 05:47:22（+3 秒） | — | — |
| 探针开始 | 05:47:28（+9 秒） | 06:07:28（+9 秒） | 06:09:38（+8 秒） |
| 结果推到 chain-results 并评论 PR | 05:48:11（+52 秒） | 06:08:58（+99 秒） | 06:12:25（+175 秒） |
| 会话被唤醒 | 05:48:15（**+56 秒**） | 06:09:01（+102 秒） | 06:12:29（+179 秒） |
| 会话读到完整 result.json | 05:48:25（**+66 秒**） | 06:09:09（+110 秒） | — |

时间都是 UTC。"会话推送"和"会话读到"取自本会话的时钟；运行创建和各步骤时间取自 GitHub 的运行记录。

- **固定开销约 15 秒**：推送到探针开始约 9 秒，评论到会话被唤醒 3–4 秒，拉结果约 10 秒。其余是探针本身的耗时。
- **唤醒**：三次都由 PR 上 github-actions[bot] 的评论唤醒，没有一次漏掉。

### 定时触发：没跑成

- 请求在 05:43:13 提交，cron 定在 05:58。
- 06:05:22 查运行列表，仍然没有这次运行，已晚 7 分钟以上。
- 按新的边界，06:07 把定时触发删掉了，这次运行最终没有发生。
- 对比：推送触发 1 分钟内拿到结果；定时触发的延迟不可控，GitHub 的 cron 晚几分钟到几十分钟都常见。

### 手动触发：被拒

- MCP 工具和 `gh`（会话里的 GitHub 令牌）都返回 `403 Resource not accessible by integration`。
- 原因：会话的令牌是 Claude GitHub App 的安装令牌，没有 Actions 写权限。
- 怎么才能通，二选一：
  - 给 Claude GitHub App 加上 "Actions: Read and write" 权限（GitHub 设置 → Applications → Claude → 权限）。
  - 在会话环境变量里放一个有 `actions:write` 的细粒度个人令牌。
- 另外，工作流文件必须先在默认分支 main 上，GitHub 才会登记它。只放在 PR 分支上时，触发返回 404。
- 推送触发不受这条限制：用 App 令牌推送（不是用工作流里的 `GITHUB_TOKEN`）会正常触发工作流。

## 2. 优云智算只读接口

客户端是 [ops/compshare_api.py](../ops/compshare_api.py)，只用标准库实现签名，带只读白名单：创建、开机、关机这类动作都调不了。

| 接口 | 结果 |
| --- | --- |
| GetBalance | 通。总额、可用、赠金、冻结、信用都是 [0.00](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40276121645/result.json) |
| DescribeCompShareSupportZone | 通。5 个可用区：华北二A、上海二A、上海二B、上海二C、华北一C |
| DescribeAvailableCompShareInstanceTypes | 通 |
| DescribeCompShareImages | 通。每页最多 100 条，超过返回 RetCode 230，需要分页 |
| CheckCompShareResourceCapacity（库存） | 通 |
| GetCompShareInstancePrice（查价） | 通 |

### 开机参数

开机参数在 [ops/launch_config.json](../ops/launch_config.json)，查价和以后真正开机用的是同一组：

- 可用区：华北二A（cn-wlcb-01）
- 卡型：4090 × 1，16 核，64G 内存
- 镜像：vLLM v0.25.1（`compshareImage-1t4y5aeosple`）
- 系统盘：100G CLOUD_SSD
- 计费：按量（Postpay）

### 参数校验

| 检查 | 结果 |
| --- | --- |
| 规格在机型的合法尺寸里 | 是 |
| 规格在库存接口的规格表里 | 是 |
| 镜像可用 | 是 |
| 查价 | RetCode 0，**2.05 元/时**；系统盘 0 元，镜像 0 元，原价 2.15 |
| 同规格抢占式 | 1.31 元/时 |
| 当前有货 | **否** |

查价接口不校验规格。把 CPU 改成不合法的 17 核，照样返回 2.05（[对照](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40276121645/result.json)）。所以"参数合法"不能只看查价成功，要同时看前两行。

## 3. SSH 握手：GitHub 运行器 → 国内主机 22 端口

- 运行器在 Azure westcentralus（美国中西部）。
- 方法：TCP 连 22 端口 5 次，读 SSH 版本串；再用 `ssh-keyscan` 做完整的密钥交换，拿到主机公钥就算握手成功。不登录。
- 下表的延迟是 TCP 建连耗时，约等于一次往返。

| 主机 | IP | 第 1 次运行 | 第 2 次运行 | 第 3 次运行 | 握手 |
| --- | --- | --- | --- | --- | --- |
| **codeup.aliyun.com**（阿里云） | 118.31.165.50 | 243 毫秒 | 224 毫秒 | 225 毫秒 | 成功，keyscan 1.3 秒，`SSH-2.0-Go` |
| git.code.tencent.com（腾讯） | 106.52.160.162 | 616 毫秒 | 431 毫秒 | 469 毫秒 | 成功，keyscan 1.3–1.5 秒 |
| gitee.com | 180.76.198.x / 199.x | 1132 毫秒 | 超时 | 超时 | 第 1 次成功，后两次 10 秒内连不上 |

- 结论用 codeup.aliyun.com：三次都握手成功，延迟中位数 **约 225 毫秒**，三次之间相差不到 20 毫秒。
- 优云智算的实例在华北和上海，运行器到实例的延迟应该与此相近，这一点未实测。
- gitee 不稳定，说明到国内主机的连通性因主机而异。真正开机后要直接测到实例的连接，并给 SSH 加重试。

## 4. 各合适卡型的库存

扫描范围：所有可用区里，适合跑 Qwen3.5-4B 的卡型，包括 4090、4090 48G、5090、3090、A800、A100、H20，每种查按量和抢占两种计费。

- 镜像用 vLLM v0.25.1，系统盘 100G。
- 价格是单卡最小规格的每小时价格（元）。
- 第 2 次运行在 06:07–06:08 UTC，第 3 次在 06:09–06:12 UTC。

| 可用区 | 卡型（显存） | 单卡规格 | 按量 | 抢占 | 第 2 次运行有货 | 第 3 次运行有货 |
| --- | --- | --- | --- | --- | --- | --- |
| 华北二A | 4090（24G） | 16 核 64G | 2.05 | 1.31 | 无 | 无 |
| 上海二B | 4090（24G） | 16 核 64G | 2.05 | 1.31 | 无 | 无 |
| 华北一C | 4090（24G） | 14 核 32G | 2.13 | 不支持 | 无 | 无 |
| 华北二A | 4090 48G | 16 核 96G | 3.61 | 不支持 | 无 | 无 |
| 上海二B | 4090 48G | 16 核 96G | 3.61 | 不支持 | 无 | 无 |
| 上海二C | 4090 48G | 20 核 64G | 3.78 | 不支持 | **有**（按量：20 核 64G、20 核 90G） | 无 |
| 华北二A | 5090（32G） | 16 核 96G | 3.79 | 2.09 | 无 | 无 |
| 上海二B | 5090（32G） | 16 核 96G | 3.79 | 2.09 | 无 | 无 |
| 上海二A | 5090（32G） | 14 核 48G | 3.84 | 不支持 | 无 | 无 |
| 华北二A | 3090（24G） | 16 核 64G | 1.13 | 0.79 | 无 | 无 |
| 华北二A | A800（80G） | 16 核 240G | 6.99 | 2.87 | 无 | 无 |
| 华北二A | A100（80G） | 16 核 64G | 10.21 | 7.14 | **有**（按量） | 无 |
| 华北二A | H20（96G） | 16 核 240G | 7.12 | 不支持 | 无 | 无 |

来源：[第 2 次运行](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40277985533/result.json)、[第 3 次运行](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/result.json)。

- 库存变化很快：两次运行只隔两三分钟，第 2 次有货的两项，第 3 次就没了。
- 开机前要先查库存。真正开机的脚本应该在几个可用区和卡型之间轮询，有货就开。
- 抢占式不支持的组合，查价接口有时返回 0 元，有时报 230。表里统一记为"不支持"，与[创建接口文档](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/gpus_instance_createcompshareinstance.txt)一致。

## 5. 做一次 30 分钟按量测试，最少要充多少

### 官方规则

规则摘自运行器抓回的官方文档原文，见 [docs/](https://github.com/BillBeam/good-roommate/tree/chain-results/runs/chain/40278185947/docs)。

- **计费方式**："按照小时计费的后付费模式，秒级计费……在使用至整点时，生成后付费账单……在当前账户余额充足的情况下，系统将自动为您支付订单费用。"按量没有预付和冻结；"不足一小时补足一小时"只适用于预付费的包时。（[计费方式说明](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/operation_charge_billdescribe.txt)）
- **余额不足**："若您的账户可用余额不足支持扣费，将产生欠费订单。"欠费 30 分钟发停服告警，1 小时自动关机，24 小时回收。（[回收规则](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/operation_charge_recycle.txt)）
- **有欠费单就不能开机**："每个账号下若存在≥1条欠费订单则无法使用和创建新资源。"（[常见问题](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/operation_gpufaq.txt)）
- **余额门槛**：个人账户的文档里没写开机要多少余额。团队文档写的是"当余额不足以抵扣所选卡型单个小时的费用时无法创建资源"（[团队子账号](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/uaccount_team_teammember.txt)）。**推断**个人账户也按一小时检查，所以按一小时准备余额。
- **配额与赠金**：没充过值（LV0）也能开 1 台。完成个人实名认证、历史充值为 0 时是 LV1，一次性赠金 **5 元**，配额 1 台；首次充值后配额变为 5 台。（[会员等级](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/overview_member_membership.txt)）
- **单笔最低充值**：充值文档没写下限。（[账户充值](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/uaccount_bill_charge.txt)）
- **余额能退**：支付宝充值 90 天内、微信充值 1 年内可快捷提现，用不完的可以退回。（[提现规则](https://github.com/BillBeam/good-roommate/blob/chain-results/runs/chain/40278185947/docs/uaccount_bill_withdrawalrules.txt)）

### 算账

| 方案 | 每小时 | 30 分钟实际花费 | 开机前余额至少 | 建议充 |
| --- | --- | --- | --- | --- |
| **计划规格：4090 24G，华北二A** | 2.05 | 约 1.03 | 2.05（一小时） | **3 元** |
| 4090 无货时换 4090 48G（上海二C 第 2 次运行有货） | 3.78 | 约 1.89 | 3.78 | 4 元 |
| 3090 24G（显存同 4090，便宜，当时无货） | 1.13 | 约 0.57 | 1.13 | 2 元 |

- 系统盘 100G 在免费额度内，镜像 0 元。
- 测完要删除实例，不能只关机。关机后云盘和镜像继续计费，关机满 7 天才自动回收。

### 可能不用充钱

如果做个人实名认证拿到 5 元赠金，并且赠金可以付 GPU 按量费用，那么一次 30 分钟测试可能不需要充值。这两点都没查实：

- 现在余额里赠金一项是 0.00，说明还没有实名，或者赠金还没发。
- 文档没写赠金能不能用于 GPU 按量。

实名后再跑一次探针，看 GetBalance 里赠金一项（AmountFree）是不是 5.00，就能确认。

## 6. 密钥与边界

| 检查 | 结果 |
| --- | --- |
| 密钥存放位置 | 只放在 Actions secrets 和本会话的环境变量里 |
| 仓库各分支（main、PR 分支、chain-request、chain-results）的文件和全部提交历史 | 搜两把密钥原文，命中 0 次 |
| 运行日志 | 只出现为 `***` |
| 输出文件 | 写盘前检查不含密钥；ProjectId 和 request_uuid 已打码 |
| 实例与费用 | 没有创建任何实例，没有产生费用 |
| 考场规则和世界的代码 | 没有改动 |
| 仓库可见性 | 公开。余额（0.00）出现在 PR 评论和 chain-results 分支里，以后有余额时要考虑是否继续公开 |
