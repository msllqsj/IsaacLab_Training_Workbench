"""Read checkpoint metadata with the training interpreter, on CPU."""
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("path")
a = p.parse_args()
import torch

data = torch.load(a.path, map_location="cpu", weights_only=True)
if not isinstance(data, dict) or "model_state_dict" not in data or "iter" not in data:
    raise ValueError("不是当前 RSL-RL 支持的训练 checkpoint")
print(json.dumps({"iteration": int(data["iter"]), "has_optimizer": "optimizer_state_dict" in data,
                  "parameters": sum(v.numel() for v in data["model_state_dict"].values() if torch.is_tensor(v))}))
