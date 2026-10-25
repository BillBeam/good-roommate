"""render.py TEMPLATE TOKENIZER PROBE_JSON [enable_thinking override] -> rendered prompt + token count (HF-style jinja env)."""
import json, sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
tpl, tokf, probef = sys.argv[1:4]
body = json.load(open(probef))["request_body"]
def raise_exception(m): raise Exception(m)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env.globals["raise_exception"] = raise_exception
t = env.from_string(open(tpl).read())
kw = {}
if "enable_thinking" in body: kw["enable_thinking"] = body["enable_thinking"]
if len(sys.argv) > 4: kw["enable_thinking"] = sys.argv[4] == "true"
s = t.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True, **kw)
tok = Tokenizer.from_file(tokf)
n = len(tok.encode(s, add_special_tokens=False).ids)
print(repr(s[-300:])); print("TOKENS", n, "kwargs", kw)
