"""envcheck.sh 的 Python 部分：版本、vLLM 载入并生成、GPU 上 BF16 计算。写成文件（不是从标准输入读）：
vLLM 起引擎进程用 spawn 方式时要按文件路径重新导入主模块；vLLM 也要在本进程碰 CUDA 之前起。"""
import importlib
import json
import time

WANT = {"vllm": "0.31.0", "transformers": "5.17.0", "peft": "0.21.2", "fla": "0.6.0", "safetensors": "0.8.0"}


def main():
    got = {m: importlib.import_module(m).__version__ for m in WANT}
    assert got == WANT, (got, WANT)
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM("/root/work/models/Qwen3.5-4B", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.5,
              generation_config="vllm")
    out = llm.generate(["<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"],
                       SamplingParams(temperature=0, max_tokens=16))
    load_s = round(time.time() - t0, 1)
    del llm
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ x.float()
    err = float(((x @ x).float() - ref).abs().max() / ref.abs().max())
    res = {"versions": got, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "bf16_matmul_rel_err": err,
           "vllm_load_and_generate_s": load_s, "sample": out[0].outputs[0].text[:80]}
    with open("/root/work/out/envcheck/envcheck.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("ENV_OK", json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
