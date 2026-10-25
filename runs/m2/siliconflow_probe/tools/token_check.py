"""token_check.py CHAT_TEMPLATE TOKENIZER_JSON PROBE_JSON...
Render each probe's request with the official HF chat template (HF-style jinja env) and compare the
token count with usage.prompt_tokens returned by SiliconFlow. Variants: tools with/without an added
"strict" field (as OpenAI-compatible servers often add), reasoning_content kept/dropped."""
import json, sys, copy
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
tpl, tokf, *probes = sys.argv[1:]
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def _raise(m): raise Exception(m)
env.globals["raise_exception"] = _raise
T = env.from_string(open(tpl).read()); tok = Tokenizer.from_file(tokf)
def count(body, strict, keep_rc):
    b = copy.deepcopy(body)
    for t in b.get("tools") or []:
        if strict: t["function"]["strict"] = False
    for m in b["messages"]:
        if not keep_rc: m.pop("reasoning_content", None)
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"]["arguments"], str): tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    kw = {"enable_thinking": b["enable_thinking"]} if "enable_thinking" in b else {}
    s = T.render(messages=b["messages"], tools=b.get("tools"), add_generation_prompt=True, **kw)
    return len(tok.encode(s, add_special_tokens=False).ids)
out = []
for p in probes:
    d = json.load(open(p)); r = d.get("response") or {}
    if "usage" not in r: continue
    row = {"probe": d["probe"], "api_prompt_tokens": r["usage"]["prompt_tokens"],
           "official_template": count(d["request_body"], False, True)}
    if d["request_body"].get("tools"): row["official_template_tools_plus_strict_field"] = count(d["request_body"], True, True)
    if any("reasoning_content" in m for m in d["request_body"]["messages"]):
        row["official_template_plus_strict_reasoning_dropped"] = count(d["request_body"], True, False)
    out.append(row)
print(json.dumps(out, ensure_ascii=False, indent=1))
