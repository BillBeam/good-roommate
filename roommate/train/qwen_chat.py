"""Qwen3.5 聊天模板、逐 token 拼接、XML 工具调用的解析。

老师轨迹（SFT）、GRPO 采样、评测三处都走这里，所以模板和工具调用格式只有一套：
- 系统提示和工具定义与 M1 第二版的 API 基线逐字相同（roommate.agent_llm.system_prompt / tools）。
- 模板用 Qwen3.5 官方的 chat_template.jinja（经 tokenizer.apply_chat_template 套用），开思考。
- 工具调用是官方的 XML 格式：<tool_call><function=名><parameter=参数>值</parameter></function></tool_call>。

"段"（segment）：模板会删掉"最后一条真实用户消息"之前各轮 assistant 的思考，所以同一次会话里，
一旦插进一条真实用户消息（agent 没调工具时的系统提醒 NUDGE），之后模型看到的前文就和之前不同。
因此每次会话按真实用户消息切成若干段；段内采样端是逐 token 往后拼的（模型生成的 token 原样保留），
训练时一段就是一条序列，保证训练样本和采样时模型看到的逐 token 一致。
"""

import json
import re

IM_START, IM_END, THINK_END = "<|im_start|>", "<|im_end|>", "</think>"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


class QwenChat:
    def __init__(self, tokenizer, tools, enable_thinking=True):
        self.tok = tokenizer
        self.tools = tools
        self.enable_thinking = enable_thinking
        self.schema = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
        self.im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
        self.think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
        eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.stop_ids = [i for i in (self.im_end_id, eot) if isinstance(i, int) and i >= 0]

    # ---------- 模板 ----------
    def render(self, messages, add_generation_prompt=True):
        return self.tok.apply_chat_template(messages, tools=self.tools, tokenize=False,
                                            add_generation_prompt=add_generation_prompt,
                                            enable_thinking=self.enable_thinking)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def gen_prompt(self):
        return "<|im_start|>assistant\n<think>\n" if self.enable_thinking else \
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def tool_suffix(self, contents):
        """模型输出以 <|im_end|> 结束后，模板接着写的文字：换行、若干条工具结果（合在一条 user 消息里）、下一轮的生成提示。"""
        body = "".join(f"\n<tool_response>\n{str(c).strip()}\n</tool_response>" for c in contents)
        return "\n<|im_start|>user" + body + "<|im_end|>\n" + self.gen_prompt()

    # ---------- 解析模型输出 ----------
    def parse(self, text):
        """text：生成提示之后模型写的文字（不含结尾的 <|im_end|>）。
        返回 (assistant 消息, 工具调用 [(名, 参数原文 JSON 字符串)], 问题列表)。
        参数原文按工具定义转类型（数组用 json.loads）；转不了就给出不合法的 JSON 原文，
        交给和 API 基线相同的执行函数（LLMAgent._exec），得到同样的"参数不是合法 JSON"报错。"""
        issues = []
        if self.enable_thinking:
            if THINK_END in text:
                reasoning, rest = text.split(THINK_END, 1)
            else:
                reasoning, rest = text, ""
        else:
            reasoning, rest = "", text
        reasoning = reasoning.strip()
        calls, msg_calls = [], []
        for i, m in enumerate(TOOL_CALL_RE.finditer(rest)):
            name = m.group(1).strip()
            args, bad = {}, False
            props = self.schema.get(name, {})
            for p in PARAM_RE.finditer(m.group(2)):
                k, v = p.group(1).strip(), p.group(2)
                typ = (props.get(k) or {}).get("type", "string")
                if typ in ("array", "object", "integer", "number", "boolean"):
                    try:
                        v = json.loads(v)
                    except ValueError:
                        bad = True
                args[k] = v
            raw = json.dumps(args, ensure_ascii=False) if not bad else "{" + json.dumps(args, ensure_ascii=False)[1:-1]
            if bad:
                issues.append({"kind": "bad_json", "name": name[:80], "raw": m.group(2)[:300]})
            if name not in self.schema:
                issues.append({"kind": "unknown_tool", "name": name[:80]})
            calls.append((name, raw))
            msg_calls.append({"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}})
        content = rest.split("<tool_call>", 1)[0].strip() if msg_calls else rest.strip()
        msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if msg_calls:
            msg["tool_calls"] = msg_calls
        return msg, calls, issues


def assistant_spans(chat, messages):
    """把一段完整对话（system + 用户 + 若干轮 assistant/tool，没有中途的真实用户消息）套模板，
    返回 (ids, mask)：mask = 1 的是 assistant 生成的部分（生成提示之后到 <|im_end|>，含 <|im_end|>）。
    和采样端一样分块编码：前文（含生成提示）一块，模型生成的部分一块，模板接着写的工具结果一块。
    这样训练序列和"采样时模型看到的前文 + 它自己生成的 token"逐 token 对得上；整段一次性编码在思考为空时
    会把生成提示里的换行和生成的换行并成一个 token，对不上。文字层面要求逐轮前缀一致，否则抛错。"""
    ids, mask, text, pending = [], [], "", ""
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        pre = chat.render(messages[:i], add_generation_prompt=True)
        full = chat.render(messages[:i + 1], add_generation_prompt=False)
        if not pre.startswith(text) or not full.startswith(pre):
            raise ValueError(f"prefix mismatch at message {i}")
        add = chat.encode(pending + pre[len(text):])
        gen = full[len(pre):]
        end = gen.rindex(IM_END) + len(IM_END)
        g = chat.encode(gen[:end])
        ids += add + g
        mask += [0] * len(add) + [1] * len(g)
        pending, text = gen[end:], full
    return ids, mask
