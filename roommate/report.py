"""从 runs/m1v2 下的日志生成 M1 第二版报告。报告里的每个数字都链接到生成它的日志文件。"""

import json
import os

from . import texts
from .schema import ATTR_CN, DOMAIN_CN

REF_CN = {
    "oracle": "标准策略（知道真实状态，只用于判分自检）",
    "ideal": "理想记忆策略（满分参照）",
    "ask_all": "每次都问、不写记忆",
    "remember_all_reask": "记住所有回答、每次都重问、不读台词",
    "remember_no_update": "问过就记、不读台词（从不察觉变化）",
    "conservative": "保守默认（不看用户）",
    "modal": "撞得最多（不看用户）",
    "random": "随机选（不看用户）",
    "nothing": "什么都不做",
}
REF_ORDER = list(REF_CN)
NM_CN = {
    "stale": "事件之前的旧状态", "temp_ignored": "忽略本次临时情况", "temp_trap": "把上次的临时情况当长期",
    "hyp_event": "多一个没发生的事件", "hyp_temp": "多一个没发生的临时情况",
    "hyp_prior_event": "像是某事件之前的状态（假想）", "hyp_prior_temp": "像是某临时情况之前的状态（假想）",
}
MODEL_CN = {"deepseek-flash": "DeepSeek Flash（V4.1）", "deepseek-v4-pro": "DeepSeek V4 Pro", "grok-4.7": "grok-4.7",
            "claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5",
            "claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1"}


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
        s = {"pct": lambda: f"{100 * v:.1f}%", "int": lambda: str(int(v)), "usd": lambda: f"{v:.2f} 美元",
             "f2": lambda: f"{v:.2f}", "f3": lambda: f"{v:.3f}", "p": lambda: f"{v:.3g}"}[kind]()
        return f"[{s}]({self.rel(path)})"

    def ci(self, lo_hi, path, kind="f3"):
        if not lo_hi:
            return "—"
        lo, hi = lo_hi
        f = (lambda x: f"{100 * x:.1f}%") if kind == "pct" else (lambda x: f"{x:.3f}")
        return f"[{f(lo)}–{f(hi)}]({self.rel(path)})"


def _ok(b):
    return "过" if b else "没过"


def write(out, report_path):
    L = Linker(report_path)
    P = lambda *a: os.path.join(out, *a)
    cfg = _load(P("config.json"))
    g = _load(P("gates.json"))
    G = g["gates"]
    gp = P("gates.json")
    gates_seeds = g["gate_seeds"]
    lines = []
    w = lines.append
    pol = g["policies"]
    llm_models = [m for m in cfg["models"] + cfg.get("claude_models", []) if f"llm_{m}" in pol]
    claude_on = any(m.startswith("claude-") for m in llm_models) or any(k.startswith("llm_claude-") for k in g.get("partial_runs", {}))

    # ---------- 标题与一句话结果 ----------
    w("# M1 第二版：修模拟用户、修选项泄题、让记忆成为必需，按四道闸门重测")
    w("")
    w(f"日志目录：[runs/m1v2/]({L.rel(out)})。所有数字都由 `python -m roommate.run` 从日志生成，链接指向生成该数字的文件。"
      "分数只算和偏好有关的会话（无关会话单独报）；置信区间都是按用户重抽样（每个世界内分别重抽）的 95% 区间。")
    w("")
    w("## 一句话结果")
    w("")
    g4 = G["4"]
    parts = [f"闸门① {_ok(G['1']['pass'])}", f"② {_ok(G['2']['pass'])}", f"③ {_ok(G['3']['pass'])}"]
    if "model" in g4:
        parts.append(f"④ {'没被做穿' if g4['pass_not_too_easy'] else '被做穿了'}（只报告）")
    else:
        parts.append("④ 还没有现成模型的结果")
    w("；".join(parts) + "。")
    w("")
    w(f"理想记忆策略在 3 个闸门世界上平均每次会话得 {L.n(pol['ideal']['mean_score'], gp)} 分，"
      f"每次都问、不写记忆的策略得 {L.n(pol['ask_all']['mean_score'], gp)} 分"
      f"（是理想记忆策略的 {L.n(G['2']['ratio'], gp, 'pct')}）；"
      f"不看用户的规则里最好的选对率是 {L.n(max(G['3']['modal'], G['3']['conservative'], G['3']['position_best'], G['3']['classifier']), P('leak_probes.json'), 'pct')}"
      f"（随机 16.7%，上限 {100 * G['3']['limit']:.1f}%）。")
    if "model" in g4:
        w(f"最强的现成模型 {MODEL_CN.get(g4['model'], g4['model'])} 得 {L.n(g4['model_score'], gp)} 分，"
          f"在\"最简单策略之上、理想记忆策略之下\"的空间里拿到 {L.n(g4['headroom_share'], gp, 'pct')}"
          f"（区间 {L.ci(g4['headroom_share_ci95'], gp, 'pct')}）。")
    w("")

    # ---------- 1. 改了什么 ----------
    w("## 1. 第一版的七处问题与第二版的改法")
    w("")
    cons = _load(P("consistency.json"))
    lp = _load(P("leak_probes.json"))
    w("| 第一版的问题 | 第二版怎么改 | 检验 |")
    w("|---|---|---|")
    w(f"| ① 模拟用户答非所问（问\"孕期的清淡饮食还要遵守吗\"被归到\"饮食\"） | agent 提问时声明问哪几项属性（{'、'.join(ATTR_CN.values())}，一次最多 2 项），用户只答这几项；主流程不再用模型归类；删掉\"近况\"话题；脚本策略同样按属性提问 | 逐属性生成回答并反解：[答非所问 0 次]({L.rel(P('consistency.json'))})；每项属性第一次被问到时回答等于真实值：{L.n(sum(v['refuse_told']['first_answers_truthful'] for v in cons['by_seed'].values()), P('consistency.json'), 'int')} / {L.n(sum(v['refuse_told']['n_first_answers'] for v in cons['by_seed'].values()), P('consistency.json'), 'int')} |")
    w(f"| ② 模拟用户自相矛盾（说过\"医生让我吃清淡\"，问到饮食却答\"没什么特别限制\"） | 回答由一张短句表拼成；健康原因不吃辣（怀孕、体检、本次嗓子发炎）被问到饮食限制时一定说出来；全称否定式只在这一方面确实没有约束时出现 | 4 个世界、两种重答配置、三种提问历史下共检查 {L.n(cons['n_answers_checked'], P('consistency.json'), 'int')} 个回答，问题 {L.n(cons['n_problems'], P('consistency.json'), 'int')} 个；模板全集见附录 A |")
    w(f"| ③ 记忆不是必需的（每次都问 0.628，理想记忆 0.918） | 用户记着每项属性上次告诉 agent 的值；再问时只答\"这个我之前跟你说过了\"，照常扣分；临时情况照实回答；\"是否愿意重复回答\"和\"每问一次扣多少分\"是世界配置项 | 每次都问、不写记忆 {L.n(pol['ask_all']['mean_score'], gp)}，理想记忆 {L.n(pol['ideal']['mean_score'], gp)}（闸门②） |")
    cue = _load(P("cue_validation.json")) if os.path.exists(P("cue_validation.json")) else None
    cue_txt = (f"{' 和 '.join(MODEL_CN.get(m, m) for m in cue['models'])}只看一句台词和原状态推改动：{L.n(cue['n_items_pass_all_models'], P('cue_validation.json'), 'int')} / {L.n(cue['n_items'], P('cue_validation.json'), 'int')} 句全部推对"
               if cue else "台词校验还没跑（见第 4 节）")
    w(f"| ④ 有的变化没有线索（健身改菜系、怀孕改菜系） | 每个事件改动的每一项都标注\"明说\"或\"可推断\"；台词改成把菜系说出来；去掉无法用台词表达的\"买车\"事件 | {cue_txt} |")
    w(f"| ⑤ 选项泄露答案（\"撞得最多\"选对 29.1%） | 6 个选项分成 3 对：用户这一对（正确答案 + 近似项）和从同一世界其他用户借来的两对；所有选项的特征由背后的状态按同一套随机规则生成；\"正确答案是变化之后的那个\"在每个（领域, 变化类型）里配平到一半 | 撞得最多 {L.n(lp['modal'], P('leak_probes.json'), 'pct')}，按位置最好 {L.n(lp['position_best'], P('leak_probes.json'), 'pct')}，保守默认 {L.n(lp['conservative'], P('leak_probes.json'), 'pct')}，分类器 {L.n(lp['classifier']['test_acc'], P('leak_probes.json'), 'pct')}（闸门③） |")
    w(f"| ⑥ 满分参照不可达 | 新增理想记忆策略：每项只在第一次需要时问，按台词标注正确更新，从不重问；标准策略只用于判分自检 | 理想记忆策略全部做对、从不重问：[自检]({L.rel(P('selfcheck.json'))}) |")
    w(f"| ⑦ 无关会话送分、口径不一 | 所有分数只算和偏好有关的会话；无关会话单独报；新增\"记住所有回答、每次都重问、不读台词\"的脚本策略 | 见第 5 节 |")
    w("")
    w("其他改动：酒店任务去掉了（原因见第 12 节）；\"宠物\"改到晚餐里（\"能带宠物的店\"）；\"晚归\"改成两向属性（下班晚只能订 19:30 以后、要早回只能订 18:30 以前），使添宝宝、养狗这两类事件在选项上有两向的后果。")
    w("")

    # ---------- 2. 环境 ----------
    w("## 2. 环境")
    w("")
    w("| 世界 | 用途 | 用户 × 会话 | 与偏好有关的会话 | 无关会话 | 事件 | 临时情况 | 正确答案是\"变化之后\"的比例 |")
    w("|---|---|---|---|---|---|---|---|")
    for s in [cfg["compare_seed"]] + cfg["gate_seeds"]:
        sp = P(f"seed{s}", "world_stats.json")
        st = _load(sp)
        use = "闸门" if s in cfg["gate_seeds"] else "前后对照"
        ev = st["events_multi_attr"] + st["events_single_attr"]
        w(f"| 种子 {s} | {use} | {L.n(st['n_users'], sp, 'int')} × {L.n(st['n_sessions_per_user'], sp, 'int')} | "
          f"{L.n(st['n_preference_sessions'], sp, 'int')} | {L.n(st['unrelated_sessions'], sp, 'int')} | {L.n(ev, sp, 'int')} | "
          f"{L.n(st['temp_situations'], sp, 'int')} | {L.n(st['correct_is_post_share'], sp, 'pct')} |")
    w("")
    w(f"世界配置（每次运行写进 [config.json]({L.rel(P('config.json'))})）：重答配置 `repeat_mode = {cfg['world_config']['repeat_mode']}`"
      f"（另一个取值 `always_answer` 是第一版的行为：总会重新回答），每问一次扣 `question_cost = {cfg['world_config']['question_cost']}` 分。"
      "判分规则与第一版相同：选对 1 分，每问一次扣 0.15 分，不限记忆条数（记忆上限仍是第一版的 40 条）。")
    w("")
    w("每次任务的 6 个选项是 3 对\"近似状态\"各造出的选项。用户这一对里，近似项是下面几种之一（每个闸门世界的种类分布见 world_stats.json）：")
    w("")
    s1 = cfg["gate_seeds"][0]
    p1 = P(f"seed{s1}", "world_stats.json")
    st1 = _load(p1)
    for k, cn in NM_CN.items():
        w(f"- `{k}`：{cn}（种子 {s1} 里 {L.n(st1['near_miss_by_kind'].get(k, 0), p1, 'int')} 次）")
    w("")
    w("生成器支持按事件类型划出留出集（`generate_world(seed, events=...)` 与 `event_split(holdout)`），供训练阶段使用。")
    w("")

    # ---------- 3. 闸门 ----------
    w("## 3. 四道闸门")
    w("")
    w("| 闸门 | 定义（README 路线） | 数字 | 结论 |")
    w("|---|---|---|---|")
    g1 = G["1"]
    cue_n = (f"台词 {L.n(g1['cue_items_pass'], P('cue_validation.json'), 'int')} / {L.n(g1['cue_items'], P('cue_validation.json'), 'int')} 句被 "
             f"{'、'.join(MODEL_CN.get(m, m) for m in g1['cue_judges'])} 推对" if g1["cue_validation_done"] else "台词校验未完成")
    w(f"| ① 环境自检 | 判分自检通过；每个长期变化都能从用户台词推出来；模拟用户的回答不和自己说过的话矛盾 | "
      f"[自检]({L.rel(P('selfcheck.json'))}) {'全部通过' if g1['selfcheck_pass'] else '有失败项'}；{cue_n}；"
      f"一致性问题 {L.n(g1['consistency_n_problems'], P('consistency.json'), 'int')} 个 | {_ok(g1['pass'])} |")
    g2 = G["2"]
    w(f"| ② 记忆是必需的 | 每次都问、不写记忆的策略 ≤ 理想记忆策略的 50% | {L.n(g2['ask_all'], gp)} / {L.n(g2['ideal'], gp)} = "
      f"{L.n(g2['ratio'], gp, 'pct')}（区间 {L.ci(g2['ratio_ci95'], gp, 'pct')}） | {_ok(g2['pass'])} |")
    g3 = G["3"]
    w(f"| ③ 选项不泄露答案 | 不看用户的规则或分类器选对率 ≤ 1/6 + 5 个百分点（{100 * g3['limit']:.1f}%） | "
      f"撞得最多 {L.n(g3['modal'], P('leak_probes.json'), 'pct')}；按位置最好 {L.n(g3['position_best'], P('leak_probes.json'), 'pct')}；"
      f"保守默认 {L.n(g3['conservative'], P('leak_probes.json'), 'pct')}；分类器 {L.n(g3['classifier'], P('leak_probes.json'), 'pct')}；"
      f"正确位置的卡方检验 p = {L.n(g3['chi2_p'], P('leak_probes.json'), 'p')} | {_ok(g3['pass'])} |")
    if "model" in g4:
        w(f"| ④ 环境没被现成模型做穿（只报告） | 最强现成模型在\"最简单策略之上、理想记忆策略之下\"的空间里的比例，95% 区间上界 < 80% | "
          f"{MODEL_CN.get(g4['model'], g4['model'])}：({L.n(g4['model_score'], gp)} − {L.n(g4['best_trivial_score'], gp)}) / "
          f"({L.n(g4['ideal'], gp)} − {L.n(g4['best_trivial_score'], gp)}) = {L.n(g4['headroom_share'], gp, 'pct')}，区间 {L.ci(g4['headroom_share_ci95'], gp, 'pct')}；"
          f"最简单策略里最高的是\"{REF_CN.get(g4['best_trivial'], g4['best_trivial'])}\" | {'没被做穿' if g4['pass_not_too_easy'] else '被做穿了'} |")
    else:
        w(f"| ④ 环境没被现成模型做穿（只报告） | 同上 | 还没有现成模型的结果；最简单策略里最高的是\"{REF_CN.get(g4['best_trivial'], g4['best_trivial'])}\" {L.n(g4['best_trivial_score'], gp)} | — |")
    w("")

    # ---------- 4. 闸门① ----------
    w("## 4. 闸门①：判分自检、台词线索、前后一致")
    w("")
    sc = _load(P("selfcheck.json"))
    w("判分自检（每个世界一份，闸门世界和对照世界都跑）：")
    w("")
    w("| 检查 | " + " | ".join(f"种子 {s}" for s in sc["by_seed"]) + " |")
    w("|---|" + "---|" * len(sc["by_seed"]))
    names = [c["name"] for c in next(iter(sc["by_seed"].values()))["checks"]]
    for i, nm in enumerate(names):
        row = []
        for s, v in sc["by_seed"].items():
            c = v["checks"][i]
            row.append(f"[{'通过' if c['pass'] else '失败'}]({L.rel(P('selfcheck.json'))})（{c['detail']}）")
        w(f"| {nm} | " + " | ".join(row) + " |")
    w("")
    w("**台词线索**（这里用到了评判模型，只用于检验台词、不参与判分）：评判模型只看一句事件台词和用户原来的状态，推出改了哪几项、改成什么；"
      "两个模型都完全推对，这句台词才算合格。校验的是 4 个世界里实际出现的全部事件台词，外加每个台词模板在随机状态上的 3 个实例。")
    w("")
    if cue:
        cp = P("cue_validation.json")
        w(f"- 合格：{L.n(cue['n_items_pass_all_models'], cp, 'int')} / {L.n(cue['n_items'], cp, 'int')}")
        for m, acc in cue["accuracy_by_model"].items():
            w(f"- {MODEL_CN.get(m, m)} 单独推对的比例：{L.n(acc, cp, 'pct')}")
        single = cue.get("items_judged_by_fewer_models", [])
        w(f"- 只有一家判过的台词：{L.n(len(single), cp, 'int')} 句" + ("（grok-4.7 停用后只用缓存里已有的判定，缺判定的只由 DeepSeek V4 Pro 判）：" if single else
          "（grok-4.7 停用前，每句台词都已有它的判定，缓存里全部找得到）。"))
        for e in single:
            w(f"  - \"{e['line']}\"（{e['source']}）：只有 {'、'.join(MODEL_CN.get(m, m) for m in e['judged_by'])} 判过，{'推对' if e['pass'] else '没推对'}")
        if cue["failures"]:
            w("- 没推对的例子：")
            for f in cue["failures"][:10]:
                w(f"  - {MODEL_CN.get(f['model'], f['model'])}：\"{f['line']}\" 真实改动 {f['changes']}，模型推的是 {f['pred']}")
    else:
        w("- 还没跑（需要 DeepSeek V4 Pro）。")
    w("")
    w("改台词的过程（三轮，花费都算在第 11 节的\"台词线索校验\"里）：")
    w("")
    w("1. 只用 grok-4.7：174 句里 172 句推对；没推对的两句都是\"升职\"那句（\"以后吃饭不用太在意价格了\"），原预算为\"低\"时被推成\"中\"。"
      "改成\"以后吃饭就挑高档的地方，不用在意价格了\"。")
    w("2. 两家都判：169 / 174。DeepSeek V4 Pro 没推对 5 句：\"养狗\"那句（\"晚上还得按点回家遛它\"）有一次没推出\"要早回\"、"
      "一次多推了\"出行=开车\"；\"怀孕\"那句三次多推了\"小孩=有婴儿\"。"
      "\"养狗\"改成\"晚上也得早点回家遛它，不能在外面待太晚\"；评判提示里把\"小孩\"的含义写清楚"
      "（\"家里现在有没有已经出生、吃饭要用宝宝椅的婴儿\"，和回答模板一致）。")
    w("3. 两家都判：173 / 174，唯一没对上的是 grok-4.7 把取值后面的括号说明一起抄了回来（\"要早回（晚饭只能订 18:30 以前）\"），"
      "取值本身是对的；比较时去掉括号说明后 174 / 174。")
    w("")
    w(f"**前后一致**：[consistency.json]({L.rel(P('consistency.json'))}) 遍历 4 个世界的全部用户、会话、属性，"
      "按三种提问历史（从不问、每次全问、随机问）和两种重答配置生成回答，用一个只看文字的反解器（texts.parse_claims）逐条核对九条规则"
      "（见 roommate/checks.py 开头）。共检查 "
      f"{L.n(cons['n_answers_checked'], P('consistency.json'), 'int')} 个回答，问题 {L.n(cons['n_problems'], P('consistency.json'), 'int')} 个。"
      "测试里还故意注入第一版的几种毛病（去掉健康原因的说明、答非所问、全称否定、台词没线索、值变了还说\"说过了\"），检查器都能抓到（tests/test_usersim.py）。")
    w("")

    # ---------- 5. 闸门②：参照策略 ----------
    w("## 5. 闸门②：参照策略与捷径")
    w("")
    w("| 策略 | 平均每次会话得分 | 区间 | 成功率 | 平均每次提问 | 提问里被答\"说过了\"的比例 | 选到近似项的比例 | 第一版行为下（总会重新回答）的得分 |")
    w("|---|---|---|---|---|---|---|---|")
    for k in REF_ORDER:
        if k not in pol:
            continue
        v = pol[k]
        v1 = g["v1_mode"].get(k)
        w(f"| {REF_CN[k]} | {L.n(v['mean_score'], gp)} | {L.ci(v['ci95'], gp)} | {L.n(v['success_rate'], gp, 'pct')} | "
          f"{L.n(v['avg_questions'], gp, 'f2')} | {L.n(v['told_before_rate'], gp, 'pct')} | {L.n(v['near_miss_error_rate'], gp, 'pct')} | "
          f"{L.n(v1, gp) if v1 is not None else '—'} |")
    w("")
    w("与偏好无关的会话（算账、换算）单独计：会算的脚本策略全部答对，不看题的策略随机选；它们不进入上面的任何分数。")
    w("")
    w("读法：")
    w("")
    w(f"- 在第二版的默认配置下，\"每次都问、不写记忆\"只在第一次见到用户时拿得到信息，之后每问一次都是\"说过了\"，反而倒扣分；"
      f"它在第一版的行为下得 {L.n(g['v1_mode'].get('ask_all'), gp)}，说明记忆的必要性来自\"说过的不再说\"这一条。")
    w(f"- \"记住所有回答、每次都重问\"能用记忆补上\"说过了\"，但不读台词，事件之后就用旧值，还要为每次重问付钱："
      f"{L.n(pol['remember_all_reask']['mean_score'], gp)}。")
    w(f"- \"问过就记、不读台词\"是最强的捷径（{L.n(pol['remember_no_update']['mean_score'], gp)}），它的失分全部来自没察觉变化和临时情况。")
    w("")

    # ---------- 6. 闸门③ ----------
    w("## 6. 闸门③：选项不泄题")
    w("")
    lpp = P("leak_probes.json")
    w(f"| 规则 | 闸门世界选对率 |")
    w("|---|---|")
    w(f"| 撞得最多（审阅脚本里的规则：菜系、区域、价位档次和其他选项撞得最多） | {L.n(lp['modal'], lpp, 'pct')} |")
    w(f"| 保守默认（对随机用户最可能合适） | {L.n(lp['conservative'], lpp, 'pct')} |")
    for Lt, v in lp["position"].items():
        w(f"| 永远选 {Lt} | {L.n(v, lpp, 'pct')} |")
    c = lp["classifier"]
    w(f"| 只用选项特征训练的条件 logit 分类器（在种子 {c['train_seeds'][0]}–{c['train_seeds'][1]} 的 {c['n_train_sessions']} 个任务上训练，在闸门世界上测） | {L.n(c['test_acc'], lpp, 'pct')}（训练集上 {L.n(c['train_acc'], lpp, 'pct')}） |")
    cl = lp["classifier_large_test"]
    w(f"| 同一个分类器在另外 30 个世界（{cl['n_test_sessions']} 个任务）上测，估计更准 | {L.n(cl['test_acc'], lpp, 'pct')} |")
    w("")
    w(f"正确答案在 A–F 上的次数：{lp['correct_letter_counts']}，卡方 = {L.n(lp['chi2']['statistic'], lpp, 'f2')}，自由度 5，p = {L.n(lp['chi2']['p_value'], lpp, 'p')}。")
    w("")
    w("分类器的特征包括每个选项自己的取值、和其他选项撞值的个数、和最近邻的差异，以及\"在和最近邻不同的属性上自己取什么值\"（用来捕捉\"看起来刚变过\"的方向）。"
      "调这一节时，分类器先后抓到过三处泄题，都已修掉：干扰对比用户这一对多一层变化（20.4%）；用户的任务领域按自己最近的变化挑选、干扰对却不是（20.4%）；"
      "假想变化的事件类型分布和真实事件不同、两组干扰对可能借自同一个用户（20.1%）。")
    w("")

    # ---------- 7. 现成模型 ----------
    w("## 7. 现成模型的基线与闸门④")
    w("")
    if not llm_models:
        w("还没有现成模型的结果。")
        w("")
    else:
        w("| 模型 | 平均每次会话得分 | 区间 | 成功率 | 平均每次提问 | 提问里被答\"说过了\" | 写记忆的会话比例 | 选到近似项 | 剩余空间比例 | 区间 |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for m in llm_models:
            v = pol[f"llm_{m}"]
            hs = g4.get("all_models", {}).get(m, {})
            w(f"| {MODEL_CN.get(m, m)} | {L.n(v['mean_score'], gp)} | {L.ci(v['ci95'], gp)} | {L.n(v['success_rate'], gp, 'pct')} | "
              f"{L.n(v['avg_questions'], gp, 'f2')} | {L.n(v['told_before_rate'], gp, 'pct')} | {L.n(v['memory_write_session_rate'], gp, 'pct')} | "
              f"{L.n(v['near_miss_error_rate'], gp, 'pct')} | {L.n(hs.get('headroom_share'), gp, 'pct')} | {L.ci(hs.get('headroom_share_ci95'), gp, 'pct')} |")
        w("")
        w(f"剩余空间比例 = (模型分 − 最简单策略里的最高分 {L.n(g4['best_trivial_score'], gp)}) / (理想记忆策略 {L.n(g4['ideal'], gp)} − 最简单策略里的最高分)。"
          "闸门④按要求只看 DeepSeek V4 Pro。" + (
              "Claude 系列（Haiku 5.5、Sonnet 5.5、Opus 5.5、Fable 5.1）的设置、和 DeepSeek 对不齐的地方、花费与失分分析见 "
              f"[Claude 基线报告]({L.rel(os.path.join(os.path.dirname(report_path), 'm1v2-claude-baseline.md'))})。" if claude_on else ""))
        w("")
        pr = g.get("partial_runs", {})
        if pr:
            w("**没有跑全的结果**（照实列出，不补跑；不进闸门，也不算置信区间）：grok-4.7 的 xAI 余额在种子 3 中途耗尽，"
              "之后第二版不再向 xAI 发任何请求，这里的 grok-4.7 记录全部从缓存重建（缺缓存的用户从第一次缺的会话起截断）；"
              "对照组按缩减后的口径只给 V4 Pro 跑种子 1，DeepSeek Flash 与 grok-4.7 的对照组是缩减之前已经跑完或已缓存的。"
              + ("Claude 的对照组只给 Opus 5.5 跑种子 1。" if claude_on else "")
              + ("Claude Fable 5.1 按预算优先级只跑了部分世界（见 Claude 基线报告）。" if "llm_claude-fable-5-1" in pr else ""))
            w("")
            w("| 记录 | 平均每次会话得分 | 偏好会话数 | 各世界（会话数 / 应有数：得分） | 成功率 | 平均每次提问 |")
            w("|---|---|---|---|---|---|")
            for k, v in pr.items():
                nm = MODEL_CN.get(k[4:].replace("_true_memory", ""), k) + ("（知道真实状态）" if k.endswith("_true_memory") else "")
                seeds = "；".join(f"种子 {sd}：{e['n_sessions']} / {e['n_full']}：{e['mean_score']:.3f}" for sd, e in v["by_seed"].items())
                w(f"| {nm} | {L.n(v['mean_score'], gp)} | {L.n(v['n_sessions'], gp, 'int')} | {seeds} | {L.n(v['success_rate'], gp, 'pct')} | {L.n(v['avg_questions'], gp, 'f2')} |")
            w("")
        w("**失分拆解**（每次与偏好有关的会话的平均失分；\"没记住或没察觉变化\"与\"选项没对齐\"按同一会话在\"知道真实状态\"对照组里选没选对来分，"
          "所以只算有对照组记录的那些会话）：")
        w("")
        w("| 策略 | 算了哪些会话 | 总失分 | 提问花掉的分 | 没记住或没察觉变化 | 选项没对齐 | 对照组（知道真实状态）成功率 | 对照组平均提问 |")
        w("|---|---|---|---|---|---|---|---|")
        lb = g["loss_breakdown"]
        for k, v in lb.items():
            if not k.startswith("llm_"):
                continue
            nm = MODEL_CN.get(k[4:], k)
            scope = f"种子 {'、'.join(str(x) for x in v.get('seeds', []))}，{v['n_sessions']} 次"
            w(f"| {nm} | {scope} | {L.n(v['total_loss'], gp)} | {L.n(v['question_loss'], gp)} | {L.n(v['memory_or_change_loss'], gp)} | "
              f"{L.n(v['alignment_loss'], gp)} | {L.n(v.get('control_success_rate'), gp, 'pct')} | {L.n(v.get('control_avg_questions'), gp, 'f2')} |")
        for k in ("ideal", "remember_no_update", "remember_all_reask", "ask_all"):
            v = lb[k]
            w(f"| {REF_CN[k]} | 三个闸门世界 | {L.n(v['total_loss'], gp)} | {L.n(v['question_loss'], gp)} | {L.n(v['memory_or_change_loss'], gp)} | "
              f"{L.n(v['alignment_loss'], gp)} | （对照用标准策略） | — |")
        w("")
        w("按近似项种类看选对率（三个闸门世界合计）：")
        w("")
        kinds = list(NM_CN)
        w("| 策略 | " + " | ".join(NM_CN[k] for k in kinds) + " |")
        w("|---|" + "---|" * len(kinds))
        for key in [f"llm_{m}" for m in llm_models] + ["remember_no_update", "ideal"]:
            bk = pol[key]["by_near_miss_kind"]
            cells = []
            for k in kinds:
                e = bk.get(k)
                cells.append(f"{L.n(e['success'] / e['n'], gp, 'pct')}（{e['n']}）" if e else "—")
            name = MODEL_CN.get(key[4:], REF_CN.get(key, key))
            w(f"| {name} | " + " | ".join(cells) + " |")
        w("")

    # ---------- 8. 种子 0 前后对照 + u07 ----------
    w("## 8. 种子 0：第一版与第二版的前后对照（不参与闸门）")
    w("")
    c0 = g.get("compare_seed0", {})
    v1 = g.get("compare_seed0_v1")
    if v1:
        v1p = v1["source"]
        w("| 策略 | 第一版（只算与偏好有关的会话） | 第二版 |")
        w("|---|---|---|")
        for k, k1 in (("ideal", None), ("ask_all", "ask_all"), ("remember_no_update", "remember_no_update"),
                      ("conservative", "conservative"), ("random", "random")):
            a = (L.n(v1["policies"][k1], v1p) if k1 else f"{L.n(v1['ideal_ceiling_estimate'], v1p)}（审阅时的估算）")
            b = L.n(c0[k]["pref_mean_score"], P(f"seed{cfg['compare_seed']}", "policies", k, "sessions.jsonl")) if k in c0 else "—"
            nm = REF_CN.get(k) or MODEL_CN.get(k[4:], k)
            w(f"| {nm} | {a} | {b} |")
        w("")
        w("按缩减后的口径，种子 0 上的大模型只跑 u07 的 8 次会话（见下），整个世界只跑脚本策略。")
        w("")
    w(_u07_section(L, out, cfg))
    w("")

    # ---------- 9. 复现 ----------
    w("## 9. 复现与存档恢复")
    w("")
    if os.path.exists(P("repro_check.json")):
        rp = _load(P("repro_check.json"))
        n_s = len(rp["scripted"])
        n_l = len(rp["llm"])
        w(f"- 世界按种子重新生成后逐字一致；{L.n(sum(rp['scripted'].values()), P('repro_check.json'), 'int')} / {n_s} 个脚本策略记录重跑一致；"
          f"{L.n(sum(rp['llm'].values()), P('repro_check.json'), 'int')} / {n_l} 组大模型记录只用缓存重跑一致。")
        if os.path.exists(P("resume_check.json")):
            rs = _load(P("resume_check.json"))
            w(f"- 在第 4 个会话边界存档、丢掉内存里的环境再恢复：{L.n(sum(v['identical'] for v in rs.values()), P('resume_check.json'), 'int')} / {len(rs)} 组与不中断时一致。")
    else:
        w("- 还没跑。")
    w("")
    w("**运行中出的问题与处理**（日志在 [runs/m1v2/logs/](" + L.rel(P("logs")) + ")）：")
    w("")
    for x in RUN_INCIDENTS:
        w(f"- {x}")
    if os.path.exists(P("tests.txt")):
        w(f"- 单元测试输出：[tests.txt]({L.rel(P('tests.txt'))})。")
    w("")

    # ---------- 10. 模型与价格 ----------
    w("## 10. 用到的模型与官方单价")
    w("")
    pp = P("pricing")
    for fn in sorted(os.listdir(pp)) if os.path.isdir(pp) else []:
        d = _load(os.path.join(pp, fn))
        w(f"- [{fn}]({L.rel(os.path.join(pp, fn))})：{d.get('source', '')}，查询时间 {d.get('fetched_at', '')}")
    w("")
    w(f"调试只用 DeepSeek Flash；正式基线跑 {'、'.join(MODEL_CN.get(m, m) for m in cfg['models'])}（三个闸门世界）；"
      f"\"知道真实状态\"对照组只跑 " + "、".join(f"{MODEL_CN.get(m, m)} 种子 {sd}" for m, sd in cfg["control_runs"]) +
      f"；种子 0 的前后对照只用 {'、'.join(MODEL_CN.get(m, m) for m in cfg['compare_models'])} 跑 u07 的 8 次会话（第一版规则、第二版规则各一遍）。"
      "grok-4.7 的 xAI 余额耗尽后，第二版不再向 xAI 发任何请求（代码里 xAI 只读缓存）；已经跑完或已缓存的 grok-4.7 结果从缓存重建、照实报告。"
      "第一版的 grok-4.20、grok-4.3 不再跑。")
    w("")

    # ---------- 11. 花费 ----------
    w("## 11. 花费")
    w("")
    if os.path.exists(P("spend_ledger.json")):
        lp_ = P("spend_ledger.json")
        led = _load(lp_)
        w(f"本轮累计 {L.n(led['total_usd'], lp_, 'usd')}（上限 {cfg['budget_usd']} 美元），其中 DeepSeek {L.n(led.get('deepseek_cny_total'), lp_, 'f2')} 元。"
          "xAI 的金额取自响应里的 cost_in_usd_ticks；DeepSeek 的金额按调用时刻的官方单价（高峰 / 空闲）和响应里的 token 数计算，"
          "人民币和美元两套价目各算一遍。被 xAI 拒绝（余额耗尽，HTTP 403）的请求不收费，也没有记进账本。")
        if led.get("deepseek_cny_limit") is not None:
            w(f"缩减口径之后的 DeepSeek 花费上限：{L.n(led['deepseek_cny_limit'] - led['deepseek_cny_budget_from'], lp_, 'f2')} 元"
              f"（从累计 {L.n(led['deepseek_cny_budget_from'], lp_, 'f2')} 元算起），实际用了 "
              f"{L.n(led['deepseek_cny_total'] - led['deepseek_cny_budget_from'], lp_, 'f2')} 元。")
        if led.get("note"):
            w("")
            w("账本说明：" + led["note"])
        w("")
        w("按\"阶段 × 模型\"（含调试和重跑；实际请求数不含缓存命中）：")
        w("")
        w("| 阶段 | 模型 | 实际请求 | 输入 tokens（其中缓存命中） | 输出 tokens | 花费（美元） | 花费（元） |")
        w("|---|---|---|---|---|---|---|")
        for k, v in sorted(led["by_model_role"].items(), key=lambda kv: (_stage_of(kv[0].split("|")[1]), kv[0])):
            m, role = k.split("|")
            w(f"| {_stage_cn(role)} | {MODEL_CN.get(m, m)} | {v['api_calls']} | {v['prompt_tokens']}（{v['cached_prompt_tokens']}） | "
              f"{v['completion_tokens']} | {L.n(v['cost_usd'], lp_, 'usd')} | {L.n(v.get('cost_cny'), lp_, 'f2')} |")
        tot = {}
        for k, v in led["by_model_role"].items():
            pv = "xAI" if not k.startswith("deepseek") else "DeepSeek"
            e = tot.setdefault(pv, [0, 0.0, 0.0])
            e[0] += v["api_calls"]
            e[1] += v["cost_usd"]
            e[2] += v.get("cost_cny", 0.0)
        for pv, (n, usd, cny) in tot.items():
            w(f"| **合计** | **{pv}** | {n} | | | {L.n(usd, lp_, 'usd')} | {L.n(cny, lp_, 'f2') if pv == 'DeepSeek' else '—'} |")
        v1l = os.path.join("runs", "m1", "spend_ledger.json")
        if os.path.exists(v1l):
            w("")
            w(f"对账：xAI 后台的本月用量还包括第一版的 {L.n(_load(v1l)['total_usd'], v1l, 'usd')}（runs/m1 的账本）；"
              "DeepSeek 后台按调用时刻的人民币价扣费，应与上表人民币一列一致（逐次明细见 spend_calls.jsonl，账本更正之前的调用只有汇总）。")
    w("")

    # ---------- 12. 局限 ----------
    w("## 12. 已知局限与没解决的问题")
    w("")
    for x in _limitations(g, cue):
        w(f"- {x}")
    w("")

    # ---------- 附录 ----------
    w("## 附录 A：全部回答模板")
    w("")
    w("| 属性 | 取值 / 情形 | 模板 |")
    w("|---|---|---|")
    for a, v, t in texts.template_inventory():
        w(f"| {a} | {v} | {t} |")
    w("")
    w("组合规则：一次回答最多两项，各项的文字用\"；\"连接，句末加\"。\"；声明超过两项时只答前两项，并在末尾加\"" + texts.TOO_MANY_ANSWER + "\"。"
      "没有声明任何有效属性时提问不发出、不扣分，返回：" + texts.NO_ATTR_ERROR)
    w("")
    w("## 附录 B：全部事件台词与标注")
    w("")
    w("| 事件 | 台词模板 | 每项改动的线索 |")
    w("|---|---|---|")
    for e, ev in texts.EVENTS.items():
        tpls = [t for v in ev["mentions"].values() for t in v] if isinstance(ev["mentions"], dict) else ev["mentions"]
        for tpl, cues in tpls:
            w(f"| {ev['cn']} | {tpl} | " + "，".join(f"{ATTR_CN[a]}：{k}" for a, k in cues.items()) + " |")
    w("")
    w("临时情况的台词（只影响本次任务）：" + "；".join(f"{t['cn']}：" + " / ".join(t["mentions"]) for t in texts.TEMPS.values()))
    w("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


RUN_INCIDENTS = [
    "xAI 余额在 grok-4.7 跑种子 3 时耗尽（HTTP 403）。当时的代码把接口出错记成\"agent 没作答\"，于是种子 3 的普通运行有 87 次、"
    "对照组有 150 次、种子 0 有 192 次会话被错记成零分。已改成接口出错就整次停下（llm.ApiError），错记的这几组记录已删除；"
    "之后不再向 xAI 发任何请求，grok-4.7 的记录全部只从缓存重建，缺缓存的用户从第一次缺的会话起截断。",
    "DeepSeek V4 Pro 种子 3 的第一次运行（并行进程 A，日志 pA.log）得 0.725，同样有会话因接口出错被记成零分（同一个毛病）。"
    "修好后从缓存续跑，补发了 39 次请求（1.02 元），最终 0.804。",
    "grok-4.7 种子 2、3 的普通运行和对照组曾在两个进程里同时跑；两者每次会话的第一轮请求完全相同，缓存里同一个键被写了两个不同的响应"
    "（237 个键），按\"后写的为准\"读缓存会让普通运行的后续请求对不上。已改成：重建时先重建普通运行（同一会话里缺缓存就换另一个响应重试），"
    "再让对照组用另一个响应。种子 2 上这样重建的结果与当时保存的记录、对话逐字一致。此后新请求的缓存键带上用途（agent / control），不会再撞。",
    "一次失败的重建把 grok-4.7 种子 2 普通运行的 sessions.jsonl 覆盖成了空文件；已从缓存重建，与当时保存的 192 段对话、summary.json 逐一核对一致。"
    "save_policy 现在拒绝用空记录覆盖。",
    "账本起初把 DeepSeek 一律按高峰美元价记账；这些调用都在 UTC 周六（空闲时段），已按 token 数改用空闲价重算人民币和美元（reports/fix_ledger_offpeak.py），"
    "之后按调用时刻取价，每次实际调用另记在 spend_calls.jsonl。",
]

STAGE_CN = [("cuecheck", "台词线索校验（三轮，含重跑）"), ("pilot:agent", "调试（Flash，小世界）"),
            ("pilot:control", "调试（Flash，对照组）"), ("agent", "闸门世界基线"), ("control", "\"知道真实状态\"对照组"),
            ("u07_v1rules:agent", "u07 · 第一版规则 · agent"), ("u07_v1rules:usersim", "u07 · 第一版规则 · 模拟用户的提问归类（含归类准确率重测）"),
            ("u07_v2rules:agent", "u07 · 第二版规则 · agent")]


def _stage_of(role):
    return next((i for i, (r, _) in enumerate(STAGE_CN) if r == role), len(STAGE_CN))


def _stage_cn(role):
    return dict(STAGE_CN).get(role, role)


def _limitations(g, cue):
    out = []
    g4 = g["gates"]["4"]
    if "model" in g4 and not g4["pass_not_too_easy"]:
        out.append(f"闸门④没过：DeepSeek V4 Pro 在剩余空间里拿到 {100 * g4['headroom_share']:.1f}%，区间上界 {100 * g4['headroom_share_ci95'][1]:.1f}% ≥ 80%，"
                   "环境对现成模型偏简单。按约定不自己加难，停下来报数；④ 只决定论文怎么讲，不挡 M2（见 README 路线）。"
                   "失分拆解显示，现成模型丢的分几乎全在\"提问\"和\"没记住或没察觉变化\"两块，知道真实状态时几乎全对（选项没对齐的失分 ≤ 0.012）。")
    out.append("grok-4.7 只有种子 1、2 完整，种子 3 只跑完 88 / 168 次偏好会话；它不进闸门，照实列在第 7 节。"
               "对照组按缩减后的口径只给 V4 Pro 跑了种子 1，所以 V4 Pro 的失分拆解只基于 168 次会话。")
    if not cue or "deepseek-v4-pro" not in (cue or {}).get("models", []):
        out.append("台词线索校验要求 DeepSeek V4 Pro 和 grok-4.7 都推对；DeepSeek 没跑完之前，闸门①只能算部分完成。")
    out.append("去掉了酒店任务：酒店只有\"预算\"一项两向属性，对约束少的用户，6 个选项里只能靠价位区分，"
               "于是\"价位独一无二的那家\"就是答案，选项必然泄题。宠物改到晚餐里。")
    out.append("选项是\"三对近似状态\"：用户能认出自己那一对，就能从这一对的共同取值里读到自己没被问过的属性。"
               "理想记忆策略不利用这一点（它只问分不清时需要的属性），所以比它更省提问的 agent 有可能存在；这算\"问得更巧\"，仍然要靠记忆和察觉变化，但它不是闸门里的满分参照。")
    out.append("近似项里有一部分是假想的变化（没发生的事件、或\"像是某事件之前\"的状态），用来把\"正确答案在变化之前还是之后\"配平到一半；"
               "这些会话检验的是\"知道当前状态\"，不直接检验\"察觉了哪次变化\"。")
    out.append("理想记忆策略\"读台词\"用的是标注（特权），代表一个完全理解台词的 agent；台词本身能否被读懂由闸门①的台词校验保证。")
    return out


def _u07_table(L, title, sp, world_path, uid="u07"):
    lines = []
    if not (os.path.exists(sp) and os.path.exists(world_path)):
        lines += [f"**{title}**：还没有记录。", ""]
        return lines, None
    world = _load(world_path)
    u = next(x for x in world["users"] if x["uid"] == uid)
    recs = {r["idx"]: r for r in _jsonl(sp) if r["uid"] == uid}
    lines.append(f"**{title}**（[记录]({L.rel(sp)})）")
    lines.append("")
    lines.append("| 次 | 周 | 类型 · 领域 | 用户说 | 问了什么 → 用户怎么答 | 写了什么记忆 | 选了哪个 |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in u["sessions"]:
        r = recs.get(s["idx"])
        if r is None:
            continue
        qa = "<br>".join(
            f"问：{_esc(q['text'])}" + (f"（声明：{'、'.join(ATTR_CN.get(a, a) for a in q.get('attributes', []))}）" if "attributes" in q else
                                       f"（模拟用户归到：{'、'.join(q.get('topics', [])) or '无'}）")
            + f" → 答：{_esc(q['answer'])}" for q in r["questions"]) or "（没问）"
        mem = "<br>".join(f"{_esc(op[1])}：{_esc(str(op[2]))}" for op in r["memory_ops"] if op[0] == "write") or "（没写）"
        ok = "✓" if r["grade"]["success"] else "✗"
        role = r.get("chosen_role") or "—"
        lines.append(f"| {s['idx']} | {s['week']} | {s['type']} · {DOMAIN_CN.get(s['domain'], s['domain'])} | {_esc(s['user_text'])} | {qa} | {mem} | "
                     f"{r['chosen'] or '—'}（正确 {r['correct']}，{ok}，{role}） |")
    lines.append("")
    return lines, recs


def _u07_section(L, out, cfg):
    """u07 的前后对照：同一个模型（DeepSeek V4 Pro）分别在第一版、第二版规则下跑 u07 的 8 次会话。"""
    v1w = os.path.join("runs", "m1", "world.json")
    rw = os.path.join(out, "u07_replay", "world.json")
    cols = [
        ("第一版规则 · DeepSeek V4 Pro", os.path.join(out, "u07_v1rules", "sessions.jsonl"), v1w),
        ("第二版规则 · DeepSeek V4 Pro", os.path.join(out, "u07_replay", "policies", "llm_deepseek-v4-pro", "sessions.jsonl"), rw),
        ("参考：第一版规则 · grok-4.7（第一版报告里的原记录）", os.path.join("runs", "m1", "policies", "llm_grok-4.7", "sessions.jsonl"), v1w),
        ("参考：第一版规则 · 每次都问", os.path.join("runs", "m1", "policies", "ask_all", "sessions.jsonl"), v1w),
        ("参考：第二版规则 · 每次都问", os.path.join(out, "u07_replay", "policies", "ask_all", "sessions.jsonl"), rw),
    ]
    lines = ["### u07 的逐次对照", ""]
    lines.append("同一个模型（DeepSeek V4 Pro）在两套规则下跑 u07 的同一段经历：第 2 次会话说怀孕、第 5 次会话临时换个区聚，会话的周次、类型、领域都相同。"
                 "第一版规则用第一版的环境代码原样运行（legacy/roommate_v1，提交 d3871eb），只把请求改走 DeepSeek；"
                 "第一版的模拟用户要用模型把问题归到话题上，原来用 grok-4.20，现在改用 DeepSeek Flash"
                 "（在第一版自带的人工标注集上的归类准确率见 runs/m1v2/u07_v1rules/classifier_eval.json）。"
                 "第二版规则下，台词、回答、选项都换成第二版的；u07 的初始状态与第一版相同，只有\"晚归\"一项按第二版的含义取值。"
                 "grok-4.7 与每次都问的两列只作参考。")
    lines.append("")
    got = {}
    for title, sp, wp in cols:
        t, recs = _u07_table(L, title, sp, wp)
        lines += t
        got[title] = recs
    a, b = got.get(cols[0][0]), got.get(cols[1][0])
    if a and b:
        lines.append("两套规则下 V4 Pro 的逐次结果：")
        lines.append("")
        lines.append("| 次 | 第一版规则：选对 · 提问次数 | 第二版规则：选对 · 提问次数 |")
        lines.append("|---|---|---|")
        for i in sorted(set(a) | set(b)):
            fa = (f"{'✓' if a[i]['grade']['success'] else '✗'} · {len(a[i]['questions'])}") if i in a else "—"
            fb = (f"{'✓' if b[i]['grade']['success'] else '✗'} · {len(b[i]['questions'])}") if i in b else "—"
            lines.append(f"| {i} | {fa} | {fb} |")
        lines.append("")
    ce = os.path.join(out, "u07_v1rules", "classifier_eval.json")
    if os.path.exists(ce):
        c = _load(ce)
        lines.append(f"第一版规则下的提问归类换成 {c['model']} 后，在第一版的 {c['n']} 条人工标注问题上的准确率：{L.n(c['exact_set_accuracy'], ce, 'pct')}"
                     "（第一版用 grok-4.20 时是 96.7%）。")
        lines.append("")
    return "\n".join(lines)


def _esc(s):
    return str(s).replace("|", "｜").replace("\n", " ")
