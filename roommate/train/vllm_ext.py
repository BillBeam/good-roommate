"""vLLM 的 worker 扩展：从 safetensors 文件载入（部分）权重，用于把训练端合并了 LoRA 的权重同步给采样端。

启动服务时加 --worker-extension-cls roommate.train.vllm_ext.WeightSyncExtension，
再经 /collective_rpc（VLLM_SERVER_DEV_MODE=1）调 update_weights_from_path。
权重名用 HF 检查点里的名字，由模型自己的 load_weights 处理拼接（如 q/k/v 合成一个矩阵）。
"""

import torch


class WeightSyncExtension:
    def update_weights_from_path(self, path):
        from safetensors import safe_open
        dev = getattr(self, "device", None) or torch.device("cuda", torch.cuda.current_device())
        with safe_open(path, framework="pt", device=str(dev)) as f:
            weights = [(k, f.get_tensor(k)) for k in f.keys()]
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()
        return len(weights)
