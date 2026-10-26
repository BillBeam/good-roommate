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


def main():
    mat = "runs/m2pilot/machine/selfcheck/material.json"
    if os.path.exists(mat):
        m = json.load(open(mat, encoding="utf-8"))
        if m.get("mask_example"):
            print(mask_png(m["mask_example"], os.path.join(FIG, "selfcheck_mask.png")))


if __name__ == "__main__":
    main()
