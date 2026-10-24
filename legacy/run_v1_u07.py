"""用 DeepSeek V4 Pro 在第一版规则下跑 u07 的 8 次会话（M1 第二版报告里 u07 前后对照的"第一版规则"一列）。

legacy/roommate_v1/ 是第一版环境的原样拷贝（提交 d3871eb 的 roommate/）。这里只换了外面的接口，不改环境本身：
- 请求改走 DeepSeek（第二版的 roommate.llm.LLMClient，同一个缓存与账本，账本里记作 u07_v1rules:*）；
- DeepSeek 的工具调用要求把 reasoning_content 原样带回、也没有 seed 参数，所以 agent 循环里补了这两处；
- 第一版的模拟用户要用模型把问题归到话题上，原来用 grok-4.20；grok 已停用，改用 DeepSeek Flash，
  并在第一版自带的人工标注集上重测归类准确率（classifier_eval.json）。

运行：python -m legacy.run_v1_u07   → 写出 runs/m1v2/u07_v1rules/
"""

import json
import os
import sys

from legacy.roommate_v1 import agent_llm as v1agent
from legacy.roommate_v1.classifier_gold import GOLD
from legacy.roommate_v1.env import Env as V1Env
from legacy.roommate_v1.grading import summarize as v1_summarize
from legacy.roommate_v1.llm import make_classifier
from roommate import run as v2run
from roommate.llm import Ledger, LLMClient

OUT = os.path.join(v2run.OUT, "u07_v1rules")
AGENT_MODEL = "deepseek-v4-pro"
CLASSIFIER_MODEL = "deepseek-flash"
UID = "u07"


class DeepSeekAdapter:
    """把第一版发出的请求体改成 DeepSeek 能接受的样子（去掉 seed），其余原样转给第二版的客户端。"""

    def __init__(self, inner):
        self.inner = inner

    def chat(self, body, role):
        body = dict(body)
        if body["model"].startswith("deepseek"):
            body.pop("seed", None)
        return self.inner.chat(body, role)


class V1AgentEcho(v1agent.LLMAgent):
    """第一版的 agent，只在循环里补上 reasoning_content 的回传。"""

    def run_session(self, ctx):
        messages = [{"role": "system", "content": v1agent.SYSTEM_PROMPT}, {"role": "user", "content": ctx.prompt}]
        turns = 0
        for turns in range(1, v1agent.MAX_TURNS + 1):
            body = {"model": self.model, "messages": messages, "tools": v1agent.TOOLS, "tool_choice": "auto",
                    "temperature": 0, **self.extra}
            resp = self.client.chat(body, role="agent")
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            out = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"):
                out["reasoning_content"] = msg["reasoning_content"]
            if calls:
                out["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["function"]["name"],
                                                   "arguments": c["function"]["arguments"]}} for c in calls]
            messages.append(out)
            if not calls:
                messages.append({"role": "user", "content": v1agent.NUDGE})
                continue
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": self._exec(ctx, c["function"]["name"], c["function"]["arguments"])})
            if ctx.done:
                break
        ctx.notes["turns"] = turns
        ctx.notes["transcript"] = messages[1:]


def main(offline=False):
    v2run.load_deepseek_prices()
    ledger = Ledger(os.path.join(v2run.OUT, "spend_ledger.json"), 40.0) if not offline else Ledger(None, 0)
    client = DeepSeekAdapter(LLMClient(os.path.join(v2run.OUT, "llm_cache"), ledger, offline=offline,
                                       role_prefix="u07_v1rules:", blocked_providers={"xai"}))
    with open(os.path.join("runs", "m1", "world.json"), encoding="utf-8") as f:
        world = json.load(f)
    world["users"] = [u for u in world["users"] if u["uid"] == UID]
    world["n_users"] = 1
    classify = make_classifier(client, model=CLASSIFIER_MODEL)
    env = V1Env(world, classify)
    env.run(V1AgentEcho(client, AGENT_MODEL, seed=0))
    recs = env.records()
    plain, trans = v2run.split_transcripts(recs)
    os.makedirs(OUT, exist_ok=True)
    v2run.dump_jsonl(os.path.join(OUT, "sessions.jsonl"), plain)
    v2run.dump_jsonl(os.path.join(OUT, "transcripts.jsonl"), trans)
    v2run.dump(os.path.join(OUT, "summary.json"), v1_summarize(plain))
    rows, exact = [], 0
    for q, gold in GOLD:
        pred = classify(q)
        ok = set(pred) == set(gold)
        exact += ok
        rows.append({"question": q, "gold": gold, "pred": pred, "match": ok})
    v2run.dump(os.path.join(OUT, "classifier_eval.json"),
               {"model": CLASSIFIER_MODEL, "n": len(GOLD), "exact_set_accuracy": exact / len(GOLD), "items": rows})
    v2run.dump(os.path.join(OUT, "config.json"), {
        "rules": "第一版（legacy/roommate_v1，提交 d3871eb）", "world": "runs/m1/world.json 里的 u07",
        "agent_model": AGENT_MODEL, "classifier_model": CLASSIFIER_MODEL,
        "classifier_model_v1_original": "grok-4.20-0309-non-reasoning（已停用）",
        "transport_changes": ["请求改走 DeepSeek", "agent 循环回传 reasoning_content", "去掉 seed 参数"]})
    print({k: v for k, v in v1_summarize(plain).items() if k in ("mean_score", "success_rate", "avg_questions")},
          "归类准确率", exact / len(GOLD))


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
