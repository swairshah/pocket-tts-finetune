import math


class LoRALinear:
    """Drop-in replacement for nn.Linear with low-rank adaptation."""

    @staticmethod
    def create(original, r=16, alpha=16):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _LoRALinear(nn.Module):
            def __init__(self, orig, r, alpha):
                super().__init__()
                self.original = orig
                self.r = r
                self.scale = alpha / r

                self.original.weight.requires_grad = False
                if self.original.bias is not None:
                    self.original.bias.requires_grad = False

                self.lora_A = nn.Parameter(
                    torch.zeros(
                        r,
                        orig.in_features,
                        device=orig.weight.device,
                        dtype=orig.weight.dtype,
                    )
                )
                self.lora_B = nn.Parameter(
                    torch.zeros(
                        orig.out_features,
                        r,
                        device=orig.weight.device,
                        dtype=orig.weight.dtype,
                    )
                )
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            @property
            def weight(self):
                return self.original.weight

            @property
            def bias(self):
                return self.original.bias

            @property
            def in_features(self):
                return self.original.in_features

            @property
            def out_features(self):
                return self.original.out_features

            def forward(self, x):
                result = self.original(x)
                lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
                return result + lora_out

            def merge_weights(self):
                with torch.no_grad():
                    self.original.weight.add_((self.lora_B @ self.lora_A) * self.scale)
                return self.original

        return _LoRALinear(original, r, alpha)


def apply_lora(model, target_names, r=16, alpha=16):
    import torch.nn as nn

    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        leaf_name = parts[-1]
        if leaf_name not in target_names:
            continue

        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        lora_layer = LoRALinear.create(module, r=r, alpha=alpha)
        setattr(parent, leaf_name, lora_layer)
        replaced.append(name)

    return replaced


def merge_lora(model):
    for name, module in list(model.named_modules()):
        if not hasattr(module, "merge_weights"):
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        merged = module.merge_weights()
        setattr(parent, parts[-1], merged)


def save_finetuned_weights(flow_lm, path):
    import json
    import os

    import safetensors.torch

    os.makedirs(path, exist_ok=True)

    state = {}
    for name, param in flow_lm.named_parameters():
        if param.requires_grad:
            state[name] = param.data

    safetensors.torch.save_file(state, f"{path}/finetuned_weights.safetensors")

    config = {
        "target_names": ["in_proj", "out_proj", "linear1", "linear2"],
        "r": 32,
        "alpha": 32,
        "finetuned_modules": ["flow_net", "out_eos", "speaker_proj_weight"],
    }
    with open(f"{path}/lora_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved {len(state)} tensors ({sum(v.numel() for v in state.values()):,} params)")


def load_finetuned_weights(flow_lm, path):
    import safetensors.torch

    state = safetensors.torch.load_file(f"{path}/finetuned_weights.safetensors")
    loaded = 0
    for name, param in flow_lm.named_parameters():
        if name in state:
            param.data = state[name].to(param.device, dtype=param.dtype)
            loaded += 1

    print(f"  Loaded {loaded} tensors from {path}")
