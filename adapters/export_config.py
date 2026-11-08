"""Run with the TRAINING interpreter. Instantiates configs, never gym.make()."""
import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--imports", default="ame_locomotion.tasks")
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, args.project)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    import isaaclab_tasks
    for module in filter(None, args.imports.split(",")):
        importlib.import_module(module)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    ed = env.to_dict()
    ad = agent if isinstance(agent, dict) else agent.to_dict()
    fields = []
    def add(path, value, group):
        kind = "none" if value is None else "bool" if type(value) is bool else "int" if type(value) is int else "float"
        fields.append({"path": path, "value": value, "kind": kind, "group": group,
                       "nullable": path.endswith((".max_level", ".end_iteration")),
                       "label": path.removeprefix("env.rewards.").removesuffix(".weight") if group == "rewards" else path.removeprefix("env.reward_gates.")})
    for name, term in ed.get("rewards", {}).items():
        if isinstance(term, dict) and "weight" in term:
            add(f"env.rewards.{name}.weight", term["weight"], "rewards")
    gates = ed.get("reward_gates", {})
    def walk(d, prefix):
        for k, v in d.items():
            path = prefix + "." + k
            if isinstance(v, dict):
                walk(v, path)
            elif v is None or type(v) in (bool, int, float):
                add(path, v, "gates")
    walk({k: v for k, v in gates.items() if k in ("enabled", "groups", "overrides")}, "env.reward_gates")
    versions = {}
    for name in ("isaaclab", "isaaclab-tasks", "rsl-rl-lib", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    data = {"task": args.task, "experiment_name": ad.get("experiment_name", args.task), "stage": (2 if ed["finetune"] else 1) if "finetune" in ed else None,
            "fields": fields, "training": {"run_name": "experiment", "max_iterations": ad["max_iterations"],
            "num_envs": ed["scene"]["num_envs"], "seed": ad.get("seed", 42), "device": ad.get("device", "cuda:0"),
            "save_interval": ad.get("save_interval", 100), "headless": True, "video": False,
            "video_length": 200, "video_interval": 24000}, "versions": versions}
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WORKBENCH_CONFIG_EXPORTED " + args.output, flush=True)
finally:
    app.close()
