"""Qwen3.5 聊天模板往返测试：逐轮拼接 vs 整段套模板。

一次带思考的多轮工具调用会话（两个用户问题，各带一段工具循环），比较：
  A. 逐轮拼接：模拟 RL 采样时 token 一路累加，模型生成的部分原样保留，
     环境只在后面追加工具返回或新的用户消息。
  B. 整段套模板：会话结束后把全部消息一次交给聊天模板渲染。
另外逐轮检查：第 t+1 轮的提示词（推理服务按模板重新渲染）是否以
"第 t 轮提示词 + 第 t 轮生成"开头。

用法：
  python tools/qwen35_template_roundtrip.py --template runs/m2prep/qwen35_chat_template.jinja \
      [--tiktoken <qwen3_6.tiktoken>] [--out runs/m2prep/template_roundtrip.json]
不给 --tiktoken 时只做文本层比较。
"""
import argparse
import base64
import json
import unicodedata

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

# 与 qwen-tokenizer 0.3.0 中 Qwen3.5/3.6 的切词正则和特殊 token 表一致（非官方移植）
QWEN35_PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
QWEN35_SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>",
    "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>", "<think>", "</think>",
    "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
    "<tts_text_bos_single>", "<|audio_pad|>",
]

TOOLS = [
    {"type": "function", "function": {"name": "ask_user", "description": "向用户提问",
     "parameters": {"type": "object", "properties": {"attributes": {"type": "array", "items": {"type": "string"}},
                    "question": {"type": "string"}}, "required": ["attributes", "question"]}}},
    {"type": "function", "function": {"name": "write_memory", "description": "写一条记忆",
     "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["key", "value"]}}},
    {"type": "function", "function": {"name": "choose_option", "description": "选定一个选项",
     "parameters": {"type": "object", "properties": {"option_id": {"type": "integer"}}, "required": ["option_id"]}}},
]


def tc(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def a(reasoning, content="", calls=None):
    m = {"role": "assistant", "reasoning_content": reasoning, "content": content}
    if calls:
        m["tool_calls"] = calls
    return m


MESSAGES = [
    {"role": "system", "content": "你是用户的长期私人助理。"},
    {"role": "user", "content": "帮我订周五请同事吃饭的餐厅。"},
    a("记忆里没有饮食信息，请客要避开忌口，先问一下。",
      calls=[tc("ask_user", attributes=["饮食限制"], question="有没有忌口或过敏？")]),
    {"role": "tool", "content": "我花生过敏。"},
    a("花生过敏是长期属性，记下来。", calls=[tc("write_memory", key="饮食限制", value="花生过敏")]),
    {"role": "tool", "content": "ok"},
    a("选项 3 是粤菜，不含花生，价位合适。", "订了选项 3。", calls=[tc("choose_option", option_id=3)]),
    {"role": "tool", "content": "已预订。"},
    a("预订成功，告诉用户。", "已为你订好周五晚上的粤菜馆。"),
    {"role": "user", "content": "周末请朋友吃饭，也帮我订一个。"},
    a("记忆里有花生过敏，不用再问；排除含花生的选项。", calls=[tc("choose_option", option_id=5)]),
    {"role": "tool", "content": "已预订。"},
    a("完成。", "订好了，避开了花生。"),
]


def make_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    # 与 transformers 的 tojson 一致：ensure_ascii=False
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    return env


def load_tokenizer(path):
    import tiktoken
    ranks = {}
    with open(path, "rb") as f:
        for line in f.read().splitlines():
            if line:
                tok, rank = line.split()
                ranks[base64.b64decode(tok)] = int(rank)
    specials = {t: 248044 + i for i, t in enumerate(QWEN35_SPECIALS)}
    enc = tiktoken.Encoding("qwen3.5-unofficial", pat_str=QWEN35_PAT, mergeable_ranks=ranks,
                            special_tokens=specials)
    return lambda s: enc.encode(unicodedata.normalize("NFC", s), allowed_special="all")


def generated_text(m):
    """模型在 '<|im_start|>assistant\\n<think>\\n' 之后应生成的文本（按模板格式，含结尾的 <|im_end|>）。"""
    s = m["reasoning_content"] + "\n</think>\n\n" + m["content"]
    for i, call in enumerate(m.get("tool_calls", [])):
        f = call["function"]
        s += ("\n\n" if m["content"] else "") if i == 0 else "\n"
        s += "<tool_call>\n<function=" + f["name"] + ">\n"
        for k, v in f["arguments"].items():
            v = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            s += "<parameter=" + k + ">\n" + v + "\n</parameter>\n"
        s += "</function>\n</tool_call>"
    return s + "<|im_end|>"


def observation_text(msgs):
    """环境在模型说完之后追加的文本：新消息 + 下一轮的生成提示。"""
    s = "\n"
    i = 0
    while i < len(msgs):
        if msgs[i]["role"] == "tool":
            s += "<|im_start|>user"
            while i < len(msgs) and msgs[i]["role"] == "tool":
                s += "\n<tool_response>\n" + msgs[i]["content"] + "\n</tool_response>"
                i += 1
            s += "<|im_end|>\n"
        else:
            s += "<|im_start|>" + msgs[i]["role"] + "\n" + msgs[i]["content"] + "<|im_end|>\n"
            i += 1
    return s + "<|im_start|>assistant\n<think>\n"


def first_diff(x, y):
    n = min(len(x), len(y))
    for i in range(n):
        if x[i] != y[i]:
            return i
    return n if len(x) != len(y) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--tiktoken")
    ap.add_argument("--out")
    args = ap.parse_args()

    tmpl = make_env().from_string(open(args.template, encoding="utf-8").read())
    render = lambda msgs, gen: tmpl.render(messages=msgs, tools=TOOLS, add_generation_prompt=gen)
    enc = load_tokenizer(args.tiktoken) if args.tiktoken else None
    asst_idx = [i for i, m in enumerate(MESSAGES) if m["role"] == "assistant"]
    report = {"assistant_turns": len(asst_idx), "per_turn": []}

    # 逐轮检查：下一轮提示词是否延续上一轮的"提示词 + 生成"
    for t, i in enumerate(asst_idx):
        prompt = render(MESSAGES[:i], True)
        gen = generated_text(MESSAGES[i])
        row = {"turn": t + 1, "msg_index": i}
        if t + 1 < len(asst_idx):
            nxt = render(MESSAGES[:asst_idx[t + 1]], True)
            row["next_prompt_extends"] = nxt.startswith(prompt + gen)
            if not row["next_prompt_extends"]:
                d = first_diff(prompt + gen, nxt)
                row["diverge_at_char"] = d
                row["accumulated_context"] = (prompt + gen)[max(0, d - 40):d + 60]
                row["rerendered_context"] = nxt[max(0, d - 40):d + 60]
        report["per_turn"].append(row)

    # A. 逐轮拼接（token 一路累加）
    chunks = [render(MESSAGES[:asst_idx[0]], True)]
    for t, i in enumerate(asst_idx):
        chunks.append(generated_text(MESSAGES[i]))
        if t + 1 < len(asst_idx):
            chunks.append(observation_text(MESSAGES[i + 1:asst_idx[t + 1]]))
    inc_text = "".join(chunks)
    # B. 整段套模板（去掉结尾那个 '\n'：模型生成到 <|im_end|> 就停）
    full_text = render(MESSAGES, False)
    full_trimmed = full_text[:-1] if full_text.endswith("<|im_end|>\n") else full_text

    report["text_equal"] = inc_text == full_trimmed
    report["text_len"] = {"incremental": len(inc_text), "full_template": len(full_trimmed)}
    d = first_diff(inc_text, full_trimmed)
    report["text_first_diff_char"] = d
    if d is not None:
        report["text_diff_context"] = {"incremental": inc_text[max(0, d - 60):d + 80],
                                       "full_template": full_trimmed[max(0, d - 60):d + 80]}
    # 哪些轮次的思考在整段渲染里还在
    report["reasoning_kept_in_full_template"] = {
        f"turn{t + 1}": (MESSAGES[i]["reasoning_content"] in full_text) for t, i in enumerate(asst_idx)}

    if enc:
        tok_inc = [x for c in chunks for x in enc(c)]  # 分块编码后拼接：生成的 token 原样保留
        tok_inc_joint = enc(inc_text)                  # 同一段文本整体重新编码
        tok_full = enc(full_trimmed)
        report["tokens"] = {
            "incremental_chunks": len(tok_inc), "incremental_rejoined": len(tok_inc_joint),
            "full_template": len(tok_full),
            "chunked_equals_reencoded": tok_inc == tok_inc_joint,
            "incremental_equals_full": tok_inc == tok_full,
            "first_diff_token": first_diff(tok_inc, tok_full),
        }
        # 在最后一个用户问题处按模板重新渲染一次，之后再逐轮累加：是否与整段套模板一致
        last_user = max(i for i, m in enumerate(MESSAGES) if m["role"] == "user")
        k = next(j for j, i in enumerate(asst_idx) if i > last_user)
        tail = [render(MESSAGES[:asst_idx[k]], True)]
        for t in range(k, len(asst_idx)):
            tail.append(generated_text(MESSAGES[asst_idx[t]]))
            if t + 1 < len(asst_idx):
                tail.append(observation_text(MESSAGES[asst_idx[t] + 1:asst_idx[t + 1]]))
        report["tokens"]["rerender_at_last_query_equals_full"] = [x for c in tail for x in enc(c)] == tok_full

    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
