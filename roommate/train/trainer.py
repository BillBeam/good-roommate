"""训练端：HF transformers + PEFT LoRA（只加在语言模型的线性层上），BF16。

- 一条序列一次前向（微批 = 1 条序列，不做填充），梯度在一步里的全部序列上累加；
  loss 按 token 平均：一步里所有计入 loss 的 token 一起平均（长短序列的 token 权重相同）。
- logprob：先算最后一层的隐状态，再对需要的位置分块过输出层、取 log_softmax（float32），
  分块并做重算，避免 248K 词表的 logits 占满显存。
- SFT：loss = −平均 logprob（只算 mask = 1 的 token，即模型生成的部分）。
- GRPO：loss = −平均 [w · A · logprob]，A 是这条样本所在段的优势，
  w = min(exp(logp_训练端 − logp_采样端), TIS_CAP)（截断的重要性采样，修正采样端和训练端的数值差；不回传梯度）。
  每批数据只更新一次，所以不需要 PPO 的比值截断。
- 和采样端同步：把 LoRA 合并进对应的线性层权重（W + scaling·B·A，float32 计算后转 BF16），
  按检查点里的权重名写进 safetensors，交给 vLLM 载入。
"""

import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

TIS_CAP = 2.0
CHUNK = 1024


def load_base(model_path, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype)
    inner = getattr(model, "model", model)
    if hasattr(inner, "visual"):
        inner.visual = None          # 只训练、只用文字部分
    return model.to(device), cfg


def hf_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def backbone_of(model):
    """(语言模型主干, 输出层)。纯文字版的模型是 model.model；多模态版是 model.model.language_model。"""
    m = hf_model(model)
    inner = m.model
    return getattr(inner, "language_model", inner), m.lm_head


def linear_targets(model):
    """语言模型各层里的全部线性层（不含输出层、嵌入、视觉部分）。"""
    bb, _ = backbone_of(model)
    names = set()
    for n, mod in bb.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in f".{n}":
            names.add(n.split(".")[-1])
    return sorted(names)


class Trainer:
    def __init__(self, model_path, lr=1e-4, lora_r=16, lora_alpha=32, device="cuda", grad_ckpt=True,
                 adapter=None, weight_decay=0.0, seed=0):
        from peft import LoraConfig, PeftModel, get_peft_model
        torch.manual_seed(seed)
        self.device = device
        if str(device).startswith("cuda") and os.environ.get("M2P_TRAIN_MEM_FRAC", "0.5") != "1":
            # 和采样端（vLLM）共用一张卡：训练端最多用这么多显存，给 vLLM 的临时内存（如整段 logprob）留出余地
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("M2P_TRAIN_MEM_FRAC", "0.5")))
        self.model_path = model_path
        base, self.cfg = load_base(model_path, device=device)
        if grad_ckpt:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        if adapter:
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
        else:
            targets = linear_targets(base)
            lc = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules=targets,
                            exclude_modules=r".*visual.*", bias="none")
            self.model = get_peft_model(base, lc)
        self.model.config.use_cache = False
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-8)
        self.lr = lr
        self.step_no = 0
        self._names = None

    # ---------- logprob ----------
    def _hidden(self, ids):
        bb, _ = backbone_of(self.model)
        out = bb(input_ids=ids, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def _chunk_lp(self, h, tgt):
        _, head = backbone_of(self.model)
        logits = head(h).float()
        return torch.gather(F.log_softmax(logits, dim=-1), 1, tgt[:, None])[:, 0]

    def logprobs(self, ids, mask, grad=False):
        """ids、mask：同长的列表。返回 mask = 1 的各位置 t 上 logp(ids[t] | ids[:t])（float32 张量）。"""
        pos = [t for t, m in enumerate(mask) if m and t > 0]
        x = torch.tensor([ids], device=self.device)
        self.model.train(grad)     # 梯度检查点只在训练模式下生效（没有 dropout，两种模式算出的数相同）
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            h = self._hidden(x)[0]
            p = torch.tensor(pos, device=self.device)
            hs, tg = h[p - 1], x[0, p]
            outs = []
            for i in range(0, len(pos), CHUNK):
                if grad:
                    outs.append(checkpoint(self._chunk_lp, hs[i:i + CHUNK], tg[i:i + CHUNK], use_reentrant=False))
                else:
                    outs.append(self._chunk_lp(hs[i:i + CHUNK], tg[i:i + CHUNK]))
            return torch.cat(outs) if outs else torch.zeros(0, device=self.device)

    def logprobs_padded(self, seqs):
        """自检用：多条序列右侧填充成一个批次一起前向，返回各序列 mask=1 位置的 logprob。"""
        L = max(len(s["ids"]) for s in seqs)
        pad = 0
        x = torch.tensor([s["ids"] + [pad] * (L - len(s["ids"])) for s in seqs], device=self.device)
        am = torch.tensor([[1] * len(s["ids"]) + [0] * (L - len(s["ids"])) for s in seqs], device=self.device)
        bb, _ = backbone_of(self.model)
        out = []
        with torch.no_grad():
            h = bb(input_ids=x, attention_mask=am, use_cache=False).last_hidden_state
            for b, s in enumerate(seqs):
                pos = torch.tensor([t for t, m in enumerate(s["mask"]) if m and t > 0], device=self.device)
                out.append(self._chunk_lp(h[b, pos - 1], x[b, pos]))
        return out

    # ---------- 一步更新 ----------
    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def train_step(self, items, kind, max_grad_norm=1.0):
        """items：[{"ids", "mask", "lp"(采样端, GRPO), "adv"(GRPO)}]。返回统计。"""
        self.model.train()
        ntok = sum(sum(it["mask"]) for it in items)
        self.opt.zero_grad(set_to_none=True)
        st = {"tokens": ntok, "loss": 0.0, "diffs": [], "tis_clipped": 0, "tis_n": 0, "lp_mean": 0.0}
        t0 = time.time()
        for it in items:
            lp = self.logprobs(it["ids"], it["mask"], grad=True)
            if kind == "sft":
                loss = -lp.sum() / ntok
            else:
                samp = [v for v, m in zip(it["lp"], it["mask"]) if m]
                samp = samp[len(samp) - len(lp):]   # 第 0 个位置没有前文，logprobs 里没有它
                have = torch.tensor([v is not None for v in samp], device=self.device)
                sv = torch.tensor([v if v is not None else 0.0 for v in samp], device=self.device)
                with torch.no_grad():
                    diff = torch.where(have, lp.detach() - sv, torch.zeros_like(sv))
                    w = torch.clamp(torch.exp(diff), max=TIS_CAP)
                    st["diffs"] += diff[have].abs().tolist()
                    st["tis_clipped"] += int((torch.exp(diff[have]) > TIS_CAP).sum())
                    st["tis_n"] += int(have.sum())
                loss = -(w * it["adv"] * lp).sum() / ntok
            loss.backward()
            st["loss"] += loss.item()
            st["lp_mean"] += float(lp.detach().sum()) / ntok
        gn = torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.step_no += 1
        st["grad_norm"] = float(gn)
        st["seconds"] = time.time() - t0
        d = sorted(st.pop("diffs"))
        if d:
            st["mismatch_abs_mean"] = sum(d) / len(d)
            st["mismatch_abs_p99"] = d[min(len(d) - 1, int(0.99 * len(d)))]
            st["mismatch_abs_max"] = d[-1]
        return st

    # ---------- 存档 ----------
    def save(self, d, extra=None):
        os.makedirs(d, exist_ok=True)
        self.model.save_pretrained(d)
        torch.save({"opt": self.opt.state_dict(), "step": self.step_no, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "py_rng": random.getstate()}, os.path.join(d, "trainer_state.pt"))
        with open(os.path.join(d, "extra.json"), "w") as f:
            json.dump(extra or {}, f, ensure_ascii=False, indent=1)

    def load_state(self, d):
        s = torch.load(os.path.join(d, "trainer_state.pt"), weights_only=False)
        self.opt.load_state_dict(s["opt"])
        self.step_no = s["step"]
        torch.set_rng_state(s["torch_rng"])
        if s.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s["cuda_rng"])
        random.setstate(s["py_rng"])
        with open(os.path.join(d, "extra.json")) as f:
            return json.load(f)

    # ---------- 给采样端的权重 ----------
    def _checkpoint_names(self):
        if self._names is None:
            idx = os.path.join(self.model_path, "model.safetensors.index.json")
            if os.path.exists(idx):
                keys = list(json.load(open(idx))["weight_map"])
            else:
                from safetensors import safe_open
                with safe_open(os.path.join(self.model_path, "model.safetensors"), "pt") as f:
                    keys = list(f.keys())
            self._names = keys
        return self._names

    def lora_modules(self):
        bb, _ = backbone_of(self.model)
        for n, mod in bb.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and "default" in getattr(mod, "lora_A", {}):
                yield n, mod

    def _ckpt_name(self, rel):
        rel = rel.replace(".base_layer", "")
        hits = [k for k in self._checkpoint_names() if k.endswith("." + rel + ".weight") and "visual" not in k
                and not k.startswith("mtp.")]          # 检查点里还有多 token 预测层 mtp.layers.*，名字的后缀相同
        assert len(hits) == 1, (rel, hits[:3])
        return hits[0]

    @torch.no_grad()
    def merged_weights(self, with_lora=True):
        out = {}
        for n, mod in self.lora_modules():
            W = mod.base_layer.weight
            if with_lora:
                A, B = mod.lora_A["default"].weight, mod.lora_B["default"].weight
                W = (W.float() + mod.scaling["default"] * (B.float() @ A.float())).to(W.dtype)
            out[self._ckpt_name(n)] = W.contiguous()
        return out

    def export(self, path, with_lora=True):
        from safetensors.torch import save_file
        w = self.merged_weights(with_lora)
        tmp = path + ".tmp"
        save_file({k: v.cpu() for k, v in w.items()}, tmp)
        os.replace(tmp, path)
        return len(w)


def cosine_lr(step, total, base, warmup=10, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
