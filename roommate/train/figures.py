"""讲解材料和报告里的图：只读仓库里的记录（runs/m2pilot/），画到 reports/m2pilot_figs/。

    python -m roommate.train.figures
"""

import json
import os
import unicodedata

FIG = "reports/m2pilot_figs"
FONT = "WenQuanYi Zen Hei Mono"


def _plt():
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "wqy-zenhei" in f.lower():
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = [FONT, "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _w(ch):
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def mask_png(example, path, cols=110, max_lines=70, skip_prefix_chars=1500):
    """loss 掩码的彩色打印：绿色 = 计入 loss（模型自己生成的），灰色 = 不计入（系统提示、用户的话、工具返回）。
    example = selfcheck 的 material["mask_example"]：{"pieces": [{"loss": bool, "text": str}, ...]}。
    开头的系统提示很长，只留最后一段（前面用"…"代替）。"""
    plt = _plt()
    pieces = [dict(p) for p in example["pieces"]]
    if pieces and not pieces[0]["loss"] and len(pieces[0]["text"]) > skip_prefix_chars:
        pieces[0]["text"] = "…（前面是系统提示和工具说明，略）…\n" + pieces[0]["text"][-400:]
    lines, cur, col = [], [], 0
    for p in pieces:
        for ch in p["text"]:
            if ch == "\n" or col + _w(ch) > cols:
                lines.append(cur)
                cur, col = [], 0
                if ch == "\n":
                    continue
            if cur and cur[-1][1] == p["loss"]:
                cur[-1][0] += ch
            else:
                cur.append([ch, p["loss"]])
            col += _w(ch)
    lines.append(cur)
    lines = lines[:max_lines]
    cw, lh = 0.0605, 0.165          # 每个半角字符宽、每行高（英寸）
    fig = plt.figure(figsize=(cols * cw + 0.4, len(lines) * lh + 0.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, cols * cw + 0.4)
    ax.set_ylim(len(lines) * lh + 0.9, 0)
    ax.axis("off")
    ax.text(0.2, 0.25, "■ 计入 loss（模型生成的）", color="#1a7f37", fontsize=9, va="center")
    ax.text(2.6, 0.25, "■ 不计入（系统提示、用户的话、工具返回）", color="#8c959f", fontsize=9, va="center")
    for i, runs in enumerate(lines):
        x, y = 0.2, 0.6 + i * lh
        for text, loss in runs:
            ax.text(x, y, text, color="#1a7f37" if loss else "#8c959f", fontsize=7.2, va="center",
                    fontweight="bold" if loss else "normal")
            x += sum(_w(c) for c in text) * cw
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def group_ledger_md(g, step=None):
    """GRPO 的一组账本（讲解 4）：同一个用户的 8 次采样，每次两次见面的得分、从这一步往后的累计分（回报）、
    组内标准化后的优势；优势 > 0 的样本被加强，< 0 的被减弱，= 0 的不变。"""
    n = len(g["rewards"])
    k = len(g["rewards"][0])
    means = [sum(g["returns"][i][j] for i in range(n)) / n for j in range(k)]
    head = "| 样本 | " + " | ".join(f"第 {j + 1} 次见面：选了 / 问了几次 / 得分 / 回报 / 优势" for j in range(k)) + " | 这次更新 |"
    sep = "|---" * (k + 2) + "|"
    rows = [head, sep]
    for i in range(n):
        cells = []
        for j in range(k):
            a = g["adv"][i][j]
            cells.append(f"{g['chosen'][i][j] or '—'} / {g['questions'][i][j]} / {g['rewards'][i][j]:+.2f} / "
                         f"{g['returns'][i][j]:+.2f} / **{a:+.2f}**")
        tot = sum(g["adv"][i])
        verdict = "加强" if all(x > 0 for x in g["adv"][i]) else "减弱" if all(x < 0 for x in g["adv"][i]) else \
            ("不变" if all(x == 0 for x in g["adv"][i]) else ("第 1 次加强、第 2 次减弱" if g["adv"][i][0] > 0 > g["adv"][i][1]
                                                           else "第 1 次减弱、第 2 次加强" if g["adv"][i][0] < 0 < g["adv"][i][1]
                                                           else "部分不变"))
        rows.append(f"| {i + 1} | " + " | ".join(cells) + f" | {verdict} |")
    rows.append("| 组内平均回报 | " + " | ".join(f"{m:+.3f}" for m in means) + " | |")
    title = f"世界 {g['world']}，用户 {g['uid']}" + (f"，第 {step} 步" if step else "")
    return f"**{title}**（第 2 次见面要用到第 1 次的信息：{'是' if g.get('needs_first') else '否'}）\n\n" + "\n".join(rows)


def grpo_curves(step_files, path):
    """GRPO 每一步的曲线（讲解 4）：分数、提问数、捷径①②、思考长度、采样端与训练端 logprob 之差。step_files = {标签: steps.jsonl}。"""
    plt = _plt()
    series = {lab: [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()] for lab, f in step_files.items()
              if os.path.exists(f)}
    panels = [("score", "平均分"), ("questions", "平均提问次数"), ("shortcut1", "捷径①：没提过的偏好不问就选"),
              ("shortcut2", "捷径②：提过的偏好又问"), ("think_tokens_per_turn", "每轮思考 token 数"),
              ("mismatch_abs_mean", "采样端与训练端 logprob 之差（均值）")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (key, title) in zip(axes.flat, panels):
        for lab, rows in series.items():
            xs = [r["step"] for r in rows if r.get(key) is not None]
            ys = [r[key] for r in rows if r.get(key) is not None]
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.2, label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("步", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def sft_loss_png(log_file, path):
    """SFT 的 loss 曲线（讲解 3）：每一步的训练 loss，每 10 步一次的验证 loss。"""
    plt = _plt()
    rows = [json.loads(x) for x in open(log_file, encoding="utf-8") if x.strip()]
    tr = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    va = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(*zip(*tr), linewidth=1, label="训练 loss（每步）")
    if va:
        ax.plot(*zip(*va), marker="o", linewidth=1.5, label="验证 loss（留出的用户）")
    ax.set_xlabel("步")
    ax.set_ylabel("每个 token 的平均负 logprob")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    mat = "runs/m2pilot/machine/selfcheck/material.json"
    if os.path.exists(mat):
        m = json.load(open(mat, encoding="utf-8"))
        if m.get("mask_example"):
            print(mask_png(m["mask_example"], os.path.join(FIG, "selfcheck_mask.png")))


if __name__ == "__main__":
    main()
