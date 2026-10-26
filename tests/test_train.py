"""M2 试跑的训练代码：模板与逐 token 拼接、本地 agent 的分段与 loss 掩码、奖励与优势、捷径指标、训练端。

需要 Qwen3.5 的分词器文件（环境变量 M2P_TOKENIZER 指向含 tokenizer.json、chat_template.jinja 的目录）；
训练端的测试另外需要 torch 与 peft，用一个随机初始化的小 Qwen3.5 模型在 CPU 上跑。缺哪样就跳过哪些测试。
"""

import json
import math
import os
import random
import shutil
import tempfile
import unittest

from roommate.agent_llm import NUDGE, tools
from roommate.env import Env
from roommate.train.grpo import advantages, build_items, returns, shortcut_stop
from roommate.train.metrics import session_row, summarize_rows
from roommate.train.shortrange import needs_first, short_world
from roommate.world import generate_world

TOK = os.environ.get("M2P_TOKENIZER")


def have(*mods):
    try:
        for m in mods:
            __import__(m)
        return True
    except ImportError:
        return False


def chat():
    from transformers import AutoTokenizer
    from roommate.train.qwen_chat import QwenChat
    return QwenChat(AutoTokenizer.from_pretrained(TOK), tools(0.15))


class ScriptedSampler:
    """按"这是第几轮 assistant"给出写好的输出：读记忆 →（第 1 次会话）只说话不调工具 → 提问 → 写记忆 → 选 A。"""

    def __init__(self, c):
        self.c = c
        self.calls = []

    def complete(self, prompt_ids, max_tokens, temperature, top_p=1.0, seed=None, stop_ids=None, top_logprobs=1,
                 ignore_eos=False):
        text = self.c.decode(prompt_ids)
        n = text.count("<|im_start|>assistant") - 1
        first = "（记忆为空）" in text or n == 0       # 第 1 次会话读到的记忆为空
        script = ["读一下记忆。\n</think>\n\n<tool_call>\n<function=memory_read>\n</function>\n</tool_call>",
                  "我想想。\n</think>\n\n我先问一下用户。",
                  "问菜系。\n</think>\n\n<tool_call>\n<function=ask_user>\n<parameter=question>\n你爱吃什么？\n"
                  "</parameter>\n<parameter=attributes>\n[\"菜系\"]\n</parameter>\n</function>\n</tool_call>",
                  "记下来。\n</think>\n\n<tool_call>\n<function=memory_write>\n<parameter=key>\n菜系\n</parameter>\n"
                  "<parameter=value>\n见回答\n</parameter>\n</function>\n</tool_call>",
                  "选 A。\n</think>\n\n<tool_call>\n<function=choose_option>\n<parameter=option>\nA\n</parameter>\n"
                  "</function>\n</tool_call>"]
        if not first:
            script = script[:1] + script[2:]
        out = script[min(n, len(script) - 1)]
        ids = self.c.encode(out) + [self.c.im_end_id]
        self.calls.append(list(prompt_ids))
        rng = random.Random(len(prompt_ids))
        return {"ids": ids, "lp": [-rng.random() for _ in ids], "top": [], "finish_reason": "stop",
                "stop_reason": None, "appended_stop": False}


@unittest.skipUnless(TOK and have("transformers"), "需要 M2P_TOKENIZER 与 transformers")
class TestTokens(unittest.TestCase):
    def setUp(self):
        self.c = chat()
        w = generate_world(301)
        self.sw = short_world(w, starts=[0])
        self.sw["users"] = self.sw["users"][:2]
        self.sw["n_users"] = 2

    def run_agent(self):
        from roommate.train.agent_local import LocalAgent
        sp = ScriptedSampler(self.c)
        env = Env(self.sw)
        env.run(LocalAgent(sp, self.c, temperature=1.0), workers=1)
        return env.records(), sp

    def test_segments_and_mask(self):
        recs, sp = self.run_agent()
        r0 = recs[0]
        self.assertIsNotNone(r0["chosen"])
        self.assertEqual(len(r0["segments"]), 2)          # 第 1 次会话中途插了一条提醒：切成两段
        self.assertEqual([i["kind"] for i in r0["format_issues"]], ["no_tool_call"])
        for r in recs:
            for s in r["segments"]:
                self.assertEqual(len(s["ids"]), len(s["mask"]))
                self.assertEqual(len(s["ids"]), len(s["lp"]))
                # 每次发给采样端的前文，都是这段序列在那一刻的前缀
                for p in s["prompts"]:
                    self.assertIn(s["ids"][:p], sp.calls)
                txt = self.c.decode([t for t, m in zip(s["ids"], s["mask"]) if m])
                self.assertNotIn("<tool_response>", txt)
                self.assertNotIn("<|im_start|>", txt)
                self.assertTrue(all(v is not None for v, m in zip(s["lp"], s["mask"]) if m))
        # 第二段按模板重新渲染：第一轮的思考被删掉
        seg2 = self.c.decode(r0["segments"][1]["ids"])
        self.assertNotIn("读一下记忆", seg2)
        self.assertIn(NUDGE, seg2)

    def test_incremental_equals_template(self):
        recs, _ = self.run_agent()
        from roommate.agent_llm import system_prompt
        r = next(x for x in recs if len(x["segments"]) == 1)   # 没有插提醒的会话：整段一致
        msgs = [{"role": "system", "content": system_prompt(0.15, "refuse_told")}] + r["transcript"]
        full = self.c.encode(self.c.render(msgs, add_generation_prompt=False))
        seg = r["segments"][0]["ids"]
        self.assertEqual(full[:len(seg)], seg)

    def test_assistant_spans(self):
        from roommate.train.qwen_chat import assistant_spans
        from roommate.agent_llm import system_prompt
        recs, _ = self.run_agent()
        r = next(x for x in recs if len(x["segments"]) == 1)
        msgs = [{"role": "system", "content": system_prompt(0.15, "refuse_told")}] + r["transcript"]
        ids, mask = assistant_spans(self.c, msgs)
        seg = r["segments"][0]
        self.assertEqual(ids[:len(seg["ids"])], seg["ids"])
        self.assertEqual(mask[:len(seg["ids"])], seg["mask"])


class TestGrpoMath(unittest.TestCase):
    def test_returns_advantages(self):
        R = [[1.0, 0.85], [0.0, 1.0], [1.0, 0.0], [0.85, 0.85]]
        G = [returns(x) for x in R]
        self.assertEqual(G[0], [1.85, 0.85])
        A = advantages(G)
        for k in range(2):
            col = [g[k] for g in G]
            mu = sum(col) / 4
            sd = math.sqrt(sum((x - mu) ** 2 for x in col) / 4)
            for i in range(4):
                self.assertAlmostEqual(A[i][k], (G[i][k] - mu) / (sd + 1e-6))
        self.assertEqual(advantages([[1.0, 1.0]] * 3), [[0.0, 0.0]] * 3)

    def test_build_items(self):
        trajs = [[{"segments": [{"ids": [1, 2, 3], "mask": [0, 1, 1], "lp": [None, -1, -1]}]},
                  {"segments": [{"ids": [1, 2], "mask": [0, 1], "lp": [None, -1]},
                                {"ids": [1, 2], "mask": [0, 0], "lp": [None, None]}]}]] * 2
        A = [[0.5, -1.0], [-0.5, 1.0]]
        items, dropped = build_items(trajs, A)
        self.assertEqual(len(items), 4)
        self.assertEqual([it["adv"] for it in items], [0.5, -1.0, -0.5, 1.0])

    def test_stop_rule(self):
        base = [{"score": 0.2, "shortcut1": 0.3, "questions": 1.5}] * 5
        later = [{"score": 0.4, "shortcut1": 0.6, "questions": 0.2}] * 5
        self.assertTrue(shortcut_stop(base + later)[0])
        honest = [{"score": 0.4, "shortcut1": 0.3, "questions": 1.4}] * 5
        self.assertFalse(shortcut_stop(base + honest)[0])


class TestShortRange(unittest.TestCase):
    def test_window_state(self):
        w = generate_world(301)
        sw = short_world(w)
        self.assertEqual(sw["n_users"], 24 * 7)
        u = sw["users"][3]
        self.assertEqual(len(u["sessions"]), 2)
        flags = [needs_first(x)[0] for x in sw["users"]]
        self.assertTrue(0.3 < sum(flags) / len(flags) < 0.95)

    def test_shortcut_metrics_on_scripted(self):
        from roommate import policies as P
        w = generate_world(2)
        for pol, s1, s2 in ((P.AskAll(), 0.0, 1.0), (P.Oracle(), None, None)):
            env = Env(w)
            env.run(pol)
            from roommate.train.metrics import replay_told
            plans = {u["uid"]: u for u in w["users"]}
            rows = []
            by = {}
            for r in env.records():
                by.setdefault(r["uid"], []).append(r)
            for uid, rs in by.items():
                info = replay_told(plans[uid], rs)
                sess = {s["idx"]: s for s in plans[uid]["sessions"]}
                rows += [(2, uid, session_row(r, sess[r["idx"]], info[r["idx"]])) for r in rs]
            p = summarize_rows(rows, n_boot=0)["point"]
            if s1 is not None:
                self.assertEqual(p["shortcut1_guess_unmentioned"], s1)
                self.assertEqual(p["shortcut2_reask_told"], s2)


@unittest.skipUnless(TOK and have("torch", "peft", "transformers"), "需要 torch、peft 与分词器")
class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
        cls.d = tempfile.mkdtemp()
        cfg = json.load(open(os.path.join(TOK, "config.json")))
        tc = cfg["text_config"]
        tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
                  linear_num_key_heads=2, linear_num_value_heads=4, mtp_num_hidden_layers=0)
        cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
        json.dump(cfg, open(os.path.join(cls.d, "config.json"), "w"))
        torch.manual_seed(0)
        Qwen3_5ForConditionalGeneration(AutoConfig.from_pretrained(cls.d)).to(torch.bfloat16).save_pretrained(cls.d)
        for f in os.listdir(TOK):
            if f != "config.json":
                shutil.copy(os.path.join(TOK, f), cls.d)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def test_sign_and_resume(self):
        import torch
        from roommate.train.trainer import Trainer
        tr = Trainer(self.d, lr=1e-2, device="cpu")
        ids, mask = list(range(500, 700)), [0] * 50 + [1] * 150
        lp = tr.logprobs(ids, mask)
        saved = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
        for adv, sign in ((1.0, 1), (-1.0, -1)):
            for n, p in tr.model.named_parameters():
                if n in saved:
                    p.data.copy_(saved[n])
            before = float(tr.logprobs(ids, mask).mean())
            tr.opt = torch.optim.SGD(tr.params, lr=1.0)
            tr.train_step([{"ids": ids, "mask": mask, "lp": [None] * 50 + lp.tolist(), "adv": adv}], "grpo",
                          max_grad_norm=1e9)
            after = float(tr.logprobs(ids, mask).mean())
            self.assertGreater(sign * (after - before), 0)
        d = tempfile.mkdtemp()
        tr.save(d, {"step": 3})
        tr2 = Trainer(self.d, lr=1e-2, device="cpu", adapter=d)
        self.assertEqual(tr2.load_state(d)["step"], 3)
        self.assertEqual(float((tr2.logprobs(ids, mask) - tr.logprobs(ids, mask)).abs().max()), 0.0)
        shutil.rmtree(d)

    def test_padding_and_export(self):
        from roommate.train.trainer import Trainer
        tr = Trainer(self.d, lr=1e-3, device="cpu")
        a = {"ids": list(range(300, 420)), "mask": [0] * 20 + [1] * 100}
        b = {"ids": list(range(300, 360)), "mask": [0] * 20 + [1] * 40}
        pa, pb = tr.logprobs_padded([a, b])
        self.assertLess(float((pa - tr.logprobs(a["ids"], a["mask"])).abs().max()), 1e-3)
        self.assertLess(float((pb - tr.logprobs(b["ids"], b["mask"])).abs().max()), 1e-3)
        w = tr.merged_weights()
        self.assertTrue(all(k.startswith("model.language_model.layers.") and k.endswith(".weight") for k in w))
        self.assertEqual(len(w), sum(1 for _ in tr.lora_modules()))


if __name__ == "__main__":
    unittest.main()
