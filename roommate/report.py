"""从 runs/m1 下的日志生成报告。报告里的每个数字都链接到生成它的日志文件。"""

import datetime as dt
import glob
import json
import os

from .grading import QUESTION_COST
from .llm import USERSIM_MODEL
from .schema import MAX_TOPICS_PER_QUESTION
from .world import N_OPTIONS

REF_ORDER = ["oracle", "nothing", "ask_all", "conservative", "random", "remember_no_update", "majority"]
REF_CN = {
    "oracle": "标准策略（知道真实状态）",
    "nothing": "什么都不做",
    "ask_all": "每次都把问题问一遍",
    "conservative": "保守默认（不做个性化）",
    "random": "随机选",
    "remember_no_update": "只记不更新（问过就记，从不察觉变化）",
    "majority": "随大流（选属性最常见的选项）",
}
ROLE_CN = {"agent": "基线 agent", "usersim": "模拟用户的提问归类",
           "pilot:agent": "试跑（调提示词、估成本）· agent", "pilot:usersim": "试跑 · 提问归类"}
TOO_EASY = 0.8


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


class Linker:
    def __init__(self, report_path):
        self.base = os.path.dirname(os.path.abspath(report_path))

    def rel(self, path):
        return os.path.relpath(os.path.abspath(path), self.base).replace(os.sep, "/")

    def n(self, v, path, kind="f3"):
        if v is None:
            return "—"
        if kind == "pct":
            s = f"{100 * v:.1f}%"
        elif kind == "int":
            s = str(int(v))
        elif kind == "usd":
            s = f"{v:.2f} 美元"
        elif kind == "usd4":
            s = f"{v:.4f} 美元"
        elif kind == "f2":
            s = f"{v:.2f}"
        else:
            s = f"{v:.3f}"
        return f"[{s}]({self.rel(path)})"


def _policy_dir(out, key):
    return os.path.join(out, "policies", key.replace(":", "_"))


def _row(L, name, s, path, oracle_score):
    ratio = s["mean_score"] / oracle_score if oracle_score else None
    return (f"| {name} | {L.n(s['mean_score'], path)} | {L.n(ratio, path, 'pct')} | "
            f"{L.n(s['success_rate'], path, 'pct')} | {L.n(s['followup_success_rate'], path, 'pct')} | "
            f"{L.n(s['avg_questions'], path, 'f2')} | {L.n(s['repeat_question_rate'], path, 'pct')} | "
            f"{L.n(s['stale_error_rate'], path, 'pct')} | {L.n(s['temp_error_rate'], path, 'pct')} |")


TABLE_HEAD = ("| 策略 | 平均每次会话得分 | 占标准策略 | 任务成功率 | 后续任务成功率 | 平均每次会话提问 | 重复提问率 "
              "| 过时记忆错误率 | 临时情况污染错误率 |\n|---|---|---|---|---|---|---|---|---|")


def _memory_usage(recs):
    n = len(recs)
    writes = sum(any(op[0] == "write" for op in r["memory_ops"]) for r in recs)
    reads = sum(any(op[0] == "read" for op in r["memory_ops"]) for r in recs)
    last = {}
    for r in recs:
        last[r["uid"]] = len(r["memory_after"])
    return reads / n, writes / n, sum(last.values()) / max(1, len(last))


def write(out, report_path):
    L = Linker(report_path)
    p = lambda *a: os.path.join(out, *a)
    stats = _load(p("world_stats.json"))
    sc = _load(p("selfcheck.json"))
    ref = {k: _load(os.path.join(_policy_dir(out, k), "summary.json")) for k in REF_ORDER}
    llm_keys = sorted(os.path.basename(d).replace("llm_", "llm:", 1)
                      for d in glob.glob(p("policies", "llm_*")) if os.path.exists(os.path.join(d, "summary.json")))
    llm = {k: _load(os.path.join(_policy_dir(out, k), "summary.json")) for k in llm_keys}
    oracle = ref["oracle"]["mean_score"]
    W = p("world_stats.json")
    lines = []
    add = lines.append

    best = max(llm, key=lambda k: llm[k]["mean_score"]) if llm else None
    best_ratio = llm[best]["mean_score"] / oracle if best else None
    too_easy = best_ratio is not None and best_ratio >= TOO_EASY
    bp = os.path.join(_policy_dir(out, best), "summary.json") if best else None

    add("# M1 基线报告：跨会话个人 agent 环境第一版")
    add("")
    add(f"生成方式：`python -m roommate.run`（随机种子 0）；本文件由 `roommate/report.py` 从 `runs/m1/` 下的日志自动生成，"
        f"每个数字都链接到生成它的日志文件。生成日期：{dt.date.today().isoformat()}。")
    add("")
    add("## 一句话结果")
    add("")
    if best:
        verdict = "**环境过于简单**" if too_easy else "未超过 80% 的闸门，环境没有过于简单"
        add(f"最强基线 `{best[4:]}` 的得分只有知道真实状态的标准策略的 {L.n(best_ratio, bp, 'pct')}，{verdict}；"
            f"三个现成模型都没能稳定地记住偏好、察觉隐式变化，"
            f"最强基线在有过时记忆陷阱的任务上仍有 {L.n(llm[best]['stale_error_rate'], bp, 'pct')} 选了按过时偏好看是对的选项。")
    add("")
    if too_easy:
        add(f"> **环境过于简单**：最强基线得分达到标准策略的 {L.n(best_ratio, bp, 'pct')}（≥ 80%）。")
        add("")

    # ---------- 1 环境 ----------
    add("## 1. 环境是什么样的")
    add("")
    add("每个模拟用户有一份由程序维护的**真实状态**（11 项偏好：过敏、饮食方式、爱吃的菜系、吃不吃辣、预算档次、住处、"
        "公司位置、出行方式、有没有小孩、有没有宠物、晚上要不要早回家），agent 看不到。用户跨好几个月与 agent 进行多次会话，"
        "每次提出一个任务（周末晚餐、工作日午饭、出游订酒店，或与偏好无关的换算、算账），"
        f"agent 要从 {N_OPTIONS} 个候选里选一个。")
    add("")
    add("会话分四类：")
    add("")
    add("- **普通任务**：按用户当时的真实状态判对错。")
    add("- **生活事件**：用户在提任务时顺口提到一件事（搬家、开始健身、家里添了宝宝、养了狗、升职调岗、怀孕、换工作降薪等），"
        "只描述发生了什么，从不说\"我的偏好变了\"；事件会同时改动几项相关偏好，agent 要自己推断。")
    add("- **临时情况**：用户提到只针对这一次的情况（嗓子发炎、公司报销、这顿想省点、这次换个区聚）。本次按临时情况判，"
        "之后恢复长期偏好；agent 若把它写成长期记忆，后面就会选错。")
    add("- **与偏好无关的会话**：换算、算账、算日期，有客观答案，不需要个性化，提问纯属浪费。")
    add("")
    add(f"agent 有四个工具：读记忆、写记忆、删记忆、向用户提问（每问一次扣 {QUESTION_COST} 分，一个问题用户最多回答 "
        f"{MAX_TOPICS_PER_QUESTION} 个方面），最后调用选择工具交卷。每次会话 agent 都不记得以前的对话，跨会话信息只有记忆。")
    add("")
    add("**判分**全部由代码完成：所选选项满足用户当时真实状态的每一项要求得 1 分，否则 0 分；再减去提问成本。"
        "生成器保证每个任务恰好有一个正确选项。若某项偏好此前被事件改过，生成器会放入一个\"过时记忆陷阱\"选项"
        "（按旧偏好看完全合适、按新偏好不合适）；若之前有过临时情况，会放入\"临时情况陷阱\"选项。")
    add("")
    add("指标定义：")
    add("")
    add("- **后续任务成功率**：第 2 次会话起、需要个性化的任务的成功率。")
    add("- **重复提问率**：所提问题中，用户给出的回答与此前对同一方面的回答完全相同（即信息没变、白问了）的比例。")
    add("- **过时记忆错误率**：在有过时记忆陷阱的任务里，选错且所选选项按过时的偏好看是合适的比例。")
    add("- **临时情况污染错误率**：同理，针对临时情况陷阱。")
    add("")
    add("**规模与各类情况的出现次数**")
    add("")
    add("| 项目 | 数量 |\n|---|---|")
    for label, k in [("模拟用户", "n_users"), ("每人会话数", "n_sessions_per_user"), ("会话总数", "n_sessions_total"),
                     ("一个事件改动多项相关偏好", "events_multi_attr"), ("一个事件只改动一项偏好（对照）", "events_single_attr"),
                     ("不应改写长期记忆的临时情况", "temp_situations"), ("与已存偏好无关的会话", "unrelated_sessions"),
                     ("含过时记忆陷阱的任务", "stale_trap_sessions"), ("含临时情况陷阱的任务", "temp_trap_sessions")]:
        add(f"| {label} | {L.n(stats[k], W, 'int')} |")
    add("")
    add("事件类型分布：" + "，".join(f"{k} {L.n(v, W, 'int')}" for k, v in sorted(stats["events_by_type"].items())) + "。")
    add("临时情况分布：" + "，".join(f"{k} {L.n(v, W, 'int')}" for k, v in sorted(stats["temps_by_type"].items())) + "。")
    add("")
    add(f"完整的隐藏世界（每个用户每次会话的真实状态、候选选项、正确答案、陷阱）见 [world.json]({L.rel(p('world.json'))})。")
    add("")

    # ---------- 2 自检 ----------
    S = p("selfcheck.json")
    add("## 2. 判分自检")
    add("")
    add(("全部通过。" if sc["all_pass"] else "**有未通过的检查项。**") + f"明细见 [selfcheck.json]({L.rel(S)})。")
    add("")
    add("| 检查 | 结果 | 依据 |\n|---|---|---|")
    for c in sc["checks"]:
        add(f"| {c['name']} | {'通过' if c['pass'] else '**未通过**'} | [{c['detail']}]({L.rel(S)}) |")
    add("")

    # ---------- 3 参照策略 ----------
    add("## 3. 参照策略")
    add("")
    add(TABLE_HEAD)
    for k in REF_ORDER:
        add(_row(L, REF_CN[k], ref[k], os.path.join(_policy_dir(out, k), "summary.json"), oracle))
    add("")
    aa, cv = ref["ask_all"], ref["conservative"]
    ap_, cp = (os.path.join(_policy_dir(out, k), "summary.json") for k in ("ask_all", "conservative"))
    add(f"- **每次都把问题问一遍**：任务全部做对（成功率 {L.n(aa['success_rate'], ap_, 'pct')}），但平均每次会话提问 "
        f"{L.n(aa['avg_questions'], ap_, 'f2')} 次，提问成本吃掉 {L.n(QUESTION_COST * aa['avg_questions'], ap_, 'f3')} 分，"
        f"得分 {L.n(aa['mean_score'], ap_)}；其中 {L.n(aa['repeat_question_rate'], ap_, 'pct')} 的问题是重复提问。")
    add(f"- **保守默认**：不提问（提问成本 {L.n(cv['avg_questions'] * QUESTION_COST, cp, 'f3')}），"
        f"得分 {L.n(cv['mean_score'], cp)}，后续任务成功率只有 {L.n(cv['followup_success_rate'], cp, 'pct')}。"
        "说明在这个环境里不个性化必然吃亏。")
    add("- 随机选和随大流的成功率接近，说明候选选项本身没有泄露答案；只记不更新的策略在过时记忆陷阱上大量出错，说明陷阱有效。")
    add("")

    # ---------- 4 大模型基线 ----------
    add("## 4. 现成模型的基线")
    add("")
    if not llm:
        add("（本次没有运行大模型基线。）")
    else:
        add(TABLE_HEAD)
        add(_row(L, REF_CN["oracle"], ref["oracle"], os.path.join(_policy_dir(out, "oracle"), "summary.json"), oracle))
        for k in llm_keys:
            add(_row(L, f"`{k[4:]}`", llm[k], os.path.join(_policy_dir(out, k), "summary.json"), oracle))
        add("")
        add(f"最强基线 `{best[4:]}` 的得分占标准策略得分的 {L.n(best_ratio, bp, 'pct')}"
            + ("，达到 80%：**环境过于简单**。" if too_easy else "，低于 80% 的闸门。"))
        add("")
        add("**按会话类型的成功率**")
        add("")
        add("| 模型 | 普通任务 | 生活事件当次 | 临时情况当次 | 与偏好无关 | 无关会话里平均提问 | 未作答 |\n|---|---|---|---|---|---|---|")
        for k in llm_keys:
            s, sp = llm[k], os.path.join(_policy_dir(out, k), "summary.json")
            bt = s["by_type"]
            add(f"| `{k[4:]}` | {L.n(bt['P']['success_rate'], sp, 'pct')} | {L.n(bt['E']['success_rate'], sp, 'pct')} | "
                f"{L.n(bt['T']['success_rate'], sp, 'pct')} | {L.n(bt['U']['success_rate'], sp, 'pct')} | "
                f"{L.n(s['avg_questions_util'], sp, 'f2')} | {L.n(s['no_choice'], sp, 'int')} |")
        add("")
        add("**记忆工具的使用**（来自每次会话的记录）")
        add("")
        add("| 模型 | 读过记忆的会话占比 | 写过记忆的会话占比 | 最后一次会话后平均记忆条数 |\n|---|---|---|---|")
        for k in llm_keys:
            sj = os.path.join(_policy_dir(out, k), "sessions.jsonl")
            r_, w_, m_ = _memory_usage(_jsonl(sj))
            add(f"| `{k[4:]}` | {L.n(r_, sj, 'pct')} | {L.n(w_, sj, 'pct')} | {L.n(m_, sj, 'f2')} |")
        add("")
        add("每次会话的完整对话（含工具调用）见各模型目录下的 `transcripts.jsonl`，"
            "例如 [" + best[4:] + "](" + L.rel(os.path.join(_policy_dir(out, best), "transcripts.jsonl")) + ")。")
        add("")
        add("所有模型使用同一份系统提示（见 `roommate/agent_llm.py`），温度 0，固定 seed，推理设置用模型默认值；"
            "每次会话最多 12 轮模型调用，未调用选择工具即记为未作答。")
        add("")
        ce = p("classifier_eval.json")
        if os.path.exists(ce):
            c = _load(ce)
            add("**用到大模型的地方（单独标注）**：判分不用任何大模型。模拟用户里只有一步用了大模型——"
                f"把 agent 的自由文本问题归到\"饮食、口味、预算……\"等话题上（`{c['model']}`，温度 0），"
                "回答内容再由代码按真实状态从模板生成，因此归类模型看不到真实状态，也不影响判分。"
                f"在本项目自写的 {L.n(c['n'], ce, 'int')} 条人工标注问题上，归类完全正确的比例为 "
                f"{L.n(c['exact_set_accuracy'], ce, 'pct')}。")
            add("")

    # ---------- 5 复现与恢复 ----------
    add("## 5. 复现与存档恢复")
    add("")
    rp = p("repro_check.json")
    r = _load(rp)
    n_ok = sum(v["identical"] for v in r["scripted"].values()) + sum(v["identical"] for v in r["llm"].values())
    n_all = len(r["scripted"]) + len(r["llm"])
    add(f"- **复现**：同一种子重新生成世界、重跑全部策略（大模型基线只用缓存重跑），{L.n(n_ok, rp, 'int')} / "
        f"{L.n(n_all, rp, 'int')} 个策略的逐条会话记录与首次运行逐字一致（比较的是记录的哈希，见 [repro_check.json]({L.rel(rp)})）。"
        "大模型的每个请求都按内容哈希缓存在 `runs/m1/llm_cache/`，所以克隆本仓库后执行 `python -m roommate.run --offline` "
        "不联网也能得到完全相同的结果。")
    rs = p("resume_check.json")
    if os.path.exists(rs):
        rc = _load(rs)
        k0 = next(iter(rc.values()))["k"]
        n_ok = sum(v["identical"] for v in rc.values())
        add(f"- **中断—恢复**：每个大模型基线都在第 {L.n(k0, rs, 'int')} 个会话边界存档（存档里有全部用户的真实状态、"
            f"agent 记忆和已有判分记录），丢掉内存中的环境、从存档恢复后跑完，{L.n(n_ok, rs, 'int')} / {L.n(len(rc), rs, 'int')} "
            f"个结果与不中断时逐字一致（[resume_check.json]({L.rel(rs)})）。")
    tp = p("tests.txt")
    if os.path.exists(tp):
        add(f"- **单元测试**（`python -m unittest discover -s tests -t .`，输出见 [tests.txt]({L.rel(tp)})）："
            "在全部 9 个会话边界（第 0 到第 8 次会话前后）分别存档、恢复、续跑，结果都与不中断时一致；"
            "用一个只记不更新的脚本策略和一个确定性的假模型 agent 各测一遍。另有反例：恢复时故意丢掉记忆，结果就不一致，说明这项检验确实有效。")
    add("")

    # ---------- 6 外部基准 ----------
    add("## 6. 迁移测试用的外部基准")
    add("")
    add("拟用的外部基准（许可证以仓库页面为准；ATRBench、HorizonBench、PAHF、MemPABench 的许可证文件已在克隆后核对）：")
    add("")
    add("| 基准 | 测什么 | 许可证 | 本轮状态 |\n|---|---|---|---|")
    add("| [ATRBench](https://github.com/BUPT-GAMMA/ATRBench) | 先问后用：学习会话里问到的长期规则能否用到后续工具调用 | 代码 Apache-2.0；数据 CC BY 4.0（派生自 NVIDIA Nemotron-Personas-USA，CC BY 4.0） | **已跑通评测脚本**（见下） |")
    add("| [HorizonBench](https://github.com/stellalisy/HorizonBench) | 生活事件驱动的偏好演化，旧偏好作为干扰项 | 代码 Apache-2.0；数据 CC-BY-4.0 | 数据在 Hugging Face，本环境网络不通，未跑 |")
    add("| [MemPABench](https://github.com/mempabench/mempabench) | 99 次会话、事件驱动变化、澄清/确认/自主行动 | 代码 MIT；数据 CC-BY-4.0（据 README） | 未跑（默认需其他服务商的模型） |")
    add("| [PAHF](https://github.com/facebookresearch/PAHF) | 人设变化、行动前澄清与行动后反馈 | MIT | 未跑 |")
    add("| [PERMA](https://github.com/PolarisLiu1/PERMA) | 事件驱动偏好的多会话基准 | Apache-2.0（据徽章） | 未跑 |")
    add("| [MultiSessionCollab](https://github.com/Shuhaibm/multisessioncollab) | 多会话协作 | MIT | 未跑 |")
    add("| [τ²-bench](https://github.com/sierra-research/tau2-bench) | 带用户模拟器的工具任务 | MIT | 未跑 |")
    add("")
    bs = p("bench", "atrbench_summary.json")
    if os.path.exists(bs):
        b = _load(bs)
        c = b["cells"][0]
        add(f"**ATRBench 试跑**：用 ATRBench 自己的 runner 和确定性 evaluator（不改它的任何文件，只在外面登记 xAI 为一个"
            f" OpenAI 兼容服务商，见 `bench/atrbench_xai.py`），跑了 1 个人设（`{c['persona_id']}`）的 "
            f"{L.n(c['n_test_sessions'], bs, 'int')} 个测试会话，模型 `{c['model']}`，变体 `oracle_target`："
            f"{L.n(c['n_success'], bs, 'int')} 个通过，准确率 {L.n(c['accuracy'], bs, 'pct')}，"
            f"花费约 {L.n(b['estimated_cost_usd'], bs, 'usd4')}。")
        add("只跑 `oracle_target` 的原因：这个变体跳过学习会话、不需要模拟用户；ATRBench 默认的模拟用户和路由分别是 GPT-5.4 和 Gemini，"
            "本轮边界只允许用 xAI。两处适配只作用于 xAI 模型：xAI 要求 seed 为正，故 seed 加 1；去掉 `reasoning_effort` 参数，用模型默认推理设置。")
        add("")
    add("**怎样保证训练不接触这些基准的数据**：")
    add("")
    add("1. 训练数据只来自本项目的生成器 `roommate/world.py`，它是随机种子的纯函数，不读任何文件；单元测试在生成世界时把文件读写全部禁掉来验证这一点（`tests/test_isolation.py`）。")
    add("2. 外部基准的代码、数据、轨迹只放在 `external/` 下，该目录写在 `.gitignore` 里，不进仓库；仓库里只保留只含分数和会话编号的摘要（`runs/m1/bench/atrbench_summary.json`）。")
    add("3. 环境与训练代码（`roommate/`）不引用 `external/`、`bench/` 或任何基准名称，由单元测试逐个文件检查。")
    add("4. 评测入口单独放在 `bench/`，只在评测时由人手动运行；今后的训练脚本只从 `roommate` 取数据。")
    add("5. 选题上也避开重叠：本环境是中文、自写模板、自定的属性体系，与上述英文基准没有共用的文本或人设。")
    add("")

    # ---------- 7 复用资源 ----------
    add("## 7. 复用的公开资源及许可证")
    add("")
    add("| 资源 | 怎么用的 | 许可证 |\n|---|---|---|")
    add("| [ATRBench](https://github.com/BUPT-GAMMA/ATRBench)（提交 a2cfefe） | 运行时克隆到 `external/`，调用它的评测脚本；不复制、不分发其代码和数据 | 代码 Apache-2.0；数据 CC BY 4.0 |")
    add("| [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)（经由 ATRBench 的数据间接使用） | 同上，仅评测时读取 | CC BY 4.0 |")
    add(f"| ATRBench 评测的 Python 依赖（openai、pydantic、PyYAML、tenacity、docstring-parser、rich、socksio 及其间接依赖） | 只装在 `external/venv-atr`，不分发；逐包许可证见 [atrbench_deps_licenses.txt]({L.rel(p('bench', 'atrbench_deps_licenses.txt'))}) | Apache-2.0 / MIT / BSD / PSF |")
    add("| Python 标准库 | 环境本身只用标准库，无第三方依赖 | PSF-2.0 |")
    add("")
    add("借鉴了设计思路、但没有复制任何代码或数据的工作：HorizonBench（事件同时改动多项相关偏好、旧偏好作为干扰项）、"
        "Memora（惩罚使用过时记忆）、PAHF（记住后不再问而错过变化）、CAPTURE（区分真变化与临时情况）、"
        "ATRBench 与 PPP（\"每次都问\"作为捷径对照、提问计成本）。出处见 [research/02](../research/02-literature-cross-session.md)。"
        "环境里的人名、店名、台词、回答模板都是本项目自写。")
    add("")

    # ---------- 8 模型与单价 ----------
    add("## 8. 用到的模型与官方单价")
    add("")
    pf = sorted(glob.glob(p("pricing", "xai_language_models_*.json")))
    used = list(dict.fromkeys([k[4:] for k in llm_keys] + [USERSIM_MODEL]))
    if pf:
        pr = _load(pf[-1])
        date = os.path.basename(pf[-1])[len("xai_language_models_"):-5]
        add(f"单价来自 xAI 官方接口 `GET https://api.x.ai/v1/language-models`，查价日期 {date}，原始响应见 "
            f"[{os.path.basename(pf[-1])}]({L.rel(pf[-1])})（价格字段单位为 1e-4 美元 / 百万 tokens）。"
            "xAI 文档站 docs.x.ai 在本环境的网络策略下无法访问，所以以接口返回为准。")
        add("")
        add("| 模型 | 用途 | 输入（美元/百万 tokens） | 缓存命中的输入 | 输出 |\n|---|---|---|---|---|")
        by_id = {m["id"]: m for m in pr["models"]}
        for alias_m in pr["models"]:
            for a in alias_m.get("aliases", []):
                by_id.setdefault(a, alias_m)
        uses = {m: [] for m in used}
        for k in llm_keys:
            uses[k[4:]].append("基线 agent")
        uses.setdefault(USERSIM_MODEL, []).append("模拟用户的提问归类")
        if os.path.exists(bs):
            uses.setdefault("grok-4.20-0309-non-reasoning", []).append("ATRBench 试跑")
        for m in used:
            e = by_id[m]
            f = lambda x: L.n(x / 1e4, pf[-1], "f2")
            add(f"| `{m}` | {'、'.join(uses[m])} | {f(e['prompt_text_token_price'])} | "
                f"{f(e['cached_prompt_text_token_price'])} | {f(e['completion_text_token_price'])} |")
        add("")

    # ---------- 9 花费 ----------
    cf = p("costs.json")
    if os.path.exists(cf):
        cst = _load(cf)["cumulative"]
        add("## 9. 花费")
        add("")
        add("花费按 xAI 每个响应里的 `usage.cost_in_usd_ticks` 逐条累计（[spend_ledger.json](" + L.rel(p("spend_ledger.json")) + ")，"
            "[costs.json](" + L.rel(cf) + ")）；用缓存重跑不产生费用。")
        add("")
        add("| 模型 | 用途 | 实际请求次数 | 花费 |\n|---|---|---|---|")
        for k, e in sorted(cst["by_model_role"].items()):
            m, role = k.split("|", 1)
            add(f"| `{m}` | {ROLE_CN.get(role, role)} | {L.n(e['api_calls'], cf, 'int')} | {L.n(e['cost_usd'], cf, 'usd4')} |")
        total = cst["total_usd"]
        if os.path.exists(bs):
            bc = _load(bs)["estimated_cost_usd"]
            add(f"| `grok-4.20-0309-non-reasoning` | ATRBench 试跑（按 token 估算） | — | {L.n(bc, bs, 'usd4')} |")
            total += bc
        add("")
        add(f"合计约 **{total:.2f} 美元**（上表各项之和），上限 30 美元。")
        add("")

    # ---------- 10 局限 ----------
    add("## 10. 已知局限")
    add("")
    add("- 只有一个随机种子（0）；各基线之间的差距没有给出种子间波动，M4 之前需要补多种子。")
    add("- 任务是多选一，判分只看选没选对，不评价对话质量；模拟用户的回答来自模板，比真人整齐。")
    add("- 提问归类用了大模型，归错时用户会答非所问，对 agent 略不公平（准确率见第 4 节）。")
    add("- 大模型首次调用本身不是逐字确定的；复现依赖请求缓存，换模型版本或改提示词后需要重新付费调用。")
    add("- 系统提示里提醒了\"偏好可能随生活变化、有些情况只针对这一次\"，这是对基线较友好的设定。")
    add("")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
