"""A deterministic subprocess fixture, never presented as real training."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
for name in ("run_name", "experiment_name", "mode", "checkpoint"):
    p.add_argument("--" + name, default="")
p.add_argument("--max_iterations", type=int, default=3)
a = p.parse_args()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
out = Path(a.experiment_name) / (timestamp + "_" + a.run_name)
out.mkdir(parents=True, exist_ok=True)
(out / "params").mkdir(exist_ok=True)
(out / "params/env.yaml").write_text("mock: true\n", encoding="utf-8")
start = json.loads(Path(a.checkpoint).read_text())["iteration"] if a.mode == "resume" else 0
print(f"[INFO] Logging experiment in directory: {a.experiment_name}", flush=True)
print("Exact experiment name requested from command line: " + timestamp, flush=True)
for i in range(start, start + a.max_iterations):
    print(f"\x1b[1m Learning iteration {i}/{start+a.max_iterations}\x1b[0m", flush=True)
    print("Mean reward: 1.0", flush=True)
    time.sleep(0.05)
    if a.run_name.startswith("fail"):
        raise RuntimeError("Intentional test failure")
    (out / f"model_{i}.pt").write_text(json.dumps({"iteration": i, "has_optimizer": True}), encoding="utf-8")
