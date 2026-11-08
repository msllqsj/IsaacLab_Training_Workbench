import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from .models import Experiment, Project
from .paths import DATA, ROOT, read_json

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def training_environment(p: Project):
    """Reproduce the explicit runtime environment without executing shell profiles."""
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
    prefix = Path(p.python).resolve().parent
    if (prefix / "conda-meta").is_dir():
        env["CONDA_PREFIX"] = str(prefix)
        bins = [prefix, prefix / "Scripts", prefix / "Library/bin"] if os.name == "nt" else [prefix / "bin"]
        env["PATH"] = os.pathsep.join(map(str, bins)) + os.pathsep + env.get("PATH", "")
    if p.isaac_sim:
        sim = Path(p.isaac_sim).expanduser().resolve()
        if not (sim / "site").is_dir():
            raise ValueError("Isaac Sim 独立安装目录缺少 site 文件夹")
        env.update(ISAAC_PATH=str(sim), CARB_APP_PATH=str(sim / "kit"), EXP_PATH=str(sim / "apps"))
        env["PYTHONPATH"] = str(sim / "site") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(p.environment)
    return env


def fingerprint(p: Project):
    root = Path(p.root).expanduser().resolve()
    files = {}
    directories = (root / "source", root / "scripts", root / "rsl_rl")
    for folder in directories:
        if not folder.exists():
            continue
        for file in sorted(folder.rglob("*")):
            if file.suffix not in (".py", ".toml", ".yaml", ".yml", ".json") or any(x in file.parts for x in ("__pycache__", "node_modules", ".venv", "logs", "outputs", ".git")):
                continue
            files[str(file.relative_to(root)).replace("\\", "/")] = hashlib.sha256(file.read_bytes()).hexdigest()
    entry = root / p.train_entry
    if entry.is_file():
        files[p.train_entry] = hashlib.sha256(entry.read_bytes()).hexdigest()
    payload = {"files": files, "python": p.python, "task": p.task, "imports": p.imports, "entry": p.train_entry, "adapter": p.adapter, "isaac_sim": p.isaac_sim, "environment": p.environment}
    return {"hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), **payload}


def changed_files(a, b):
    return [k for k in sorted(set(a.get("files", {})) | set(b.get("files", {}))) if a.get("files", {}).get(k) != b.get("files", {}).get(k)]


def schema_path(pid):
    return DATA / "schemas" / (pid + ".json")


def schema(p):
    return read_json(schema_path(p.id))


def validate_project(p):
    root = Path(p.root).expanduser().resolve()
    py = Path(p.python).expanduser()
    entry = (root / p.train_entry).resolve()
    if not root.is_dir():
        raise ValueError("训练项目目录不存在")
    if not py.is_file():
        raise ValueError("请选择训练环境中的 Python 可执行文件完整路径")
    if not entry.is_relative_to(root) or not entry.is_file() or entry.suffix != ".py":
        raise ValueError("训练入口必须是项目目录内的 Python 文件")


def validate_config(p, spec: Experiment, known_schema=None):
    validate_project(p)
    s = known_schema or schema(p)
    if not s:
        raise ValueError("请先刷新项目配置")
    current = fingerprint(p)
    if s.get("fingerprint", {}).get("hash") != current["hash"]:
        raise ValueError("项目源码已变化，请刷新配置后保存实验")
    fields = {x["path"]: x for x in s.get("fields", [])}
    for key, val in spec.overrides.items():
        f = fields.get(key)
        if not f:
            raise ValueError(f"配置字段未在当前项目中导出: {key}")
        if val is None:
            if not f.get("nullable"):
                raise ValueError(f"该字段不能设为无限制: {key}")
        elif f["kind"] == "bool":
            if type(val) is not bool:
                raise ValueError(f"需要布尔值: {key}")
        elif f["kind"] == "int":
            if type(val) is not int or val < 0:
                raise ValueError(f"需要非负整数: {key}")
        elif f["kind"] == "none":
            raise ValueError(f"当前 Hydra 不支持将默认 None 改为数值: {key}")
        elif type(val) not in (int, float):
            raise ValueError(f"需要数值: {key}")
        elif f["kind"] == "float":
            spec.overrides[key] = float(val)
        if "reward_gates" in key and type(val) in (int, float) and val < 0:
            raise ValueError(f"门控系数不能为负: {key}")
    if s.get("stage") == 2 and spec.mode == "new":
        raise ValueError("当前为阶段二，必须选择恢复训练或仅加载权重")
    if spec.checkpoint.kind == "file" and not Path(spec.checkpoint.path).expanduser().is_file():
        raise ValueError("checkpoint 文件不存在")
    return current


def training_output(p: Project):
    root = Path(p.root).expanduser().resolve() / "logs" / "rsl_rl"
    name = (schema(p) or {}).get("experiment_name") or p.task
    output = (root / name).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("实验目录名必须位于项目 logs/rsl_rl 内")
    return output


def logged_run_directory(output: Path, run_name: str, progress: dict):
    timestamp = progress.get("run_timestamp")
    if not timestamp:
        return None
    output = output.resolve()
    directory = (output / (timestamp + ("_" + run_name if run_name else ""))).resolve()
    if directory.parent != output:
        raise ValueError("训练报告的日志目录无效")
    return directory


def build_command(p: Project, spec: Experiment, output_root: Path, checkpoint=None):
    t = spec.training
    if p.adapter == "mock":
        cmd = [p.python, str(Path(p.root) / p.train_entry), "--run_name", t.run_name, "--max_iterations", str(t.max_iterations), "--experiment_name", str(output_root), "--mode", spec.mode]
        if checkpoint:
            cmd += ["--checkpoint", str(checkpoint)]
        return cmd
    cmd = [p.python, "-s", "-u", str(Path(p.root) / p.train_entry), "--task", p.task,
           "--num_envs", str(t.num_envs), "--max_iterations", str(t.max_iterations), "--seed", str(t.seed),
           "--device", t.device, "--run_name", t.run_name, "--experiment_name", str(output_root), "--logger", "tensorboard"]
    if t.headless:
        cmd.append("--headless")
    if t.video:
        cmd += ["--video", "--video_length", str(t.video_length), "--video_interval", str(t.video_interval)]
    if spec.mode == "resume":
        cmd += ["--resume", "--checkpoint", str(Path(checkpoint).resolve())]
    elif spec.mode == "warm_start":
        cmd += ["--finetune_from", str(Path(checkpoint).resolve())]
    cmd.append(f"agent.save_interval={t.save_interval}")
    for key, value in sorted(spec.overrides.items()):
        if type(value) in (int, float) and (key.endswith(".weight") or not key.endswith((".max_level", ".end_iteration"))):
            value = float(value)
        cmd.append(key + "=" + json.dumps(value, allow_nan=False, separators=(",", ":")))
    return cmd


def display_command(cmd):
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def parse_line(line, progress):
    clean = ANSI.sub("", line).strip()
    m = re.search(r"Learning iteration\s+(\d+)/(\d+)", clean)
    if m:
        now, end = map(int, m.groups())
        first = progress.setdefault("start_iteration", now)
        progress.update(iteration=now, total_iteration=end, completed_updates=now - first + 1,
                        progress=min(1, (now - first + 1) / max(1, end - first)), phase="training")
    if "[INFO] Logging experiment in directory:" in clean:
        progress["log_root"] = clean.split("directory:", 1)[1].strip()
    if clean.startswith("Exact experiment name requested from command line:"):
        progress["run_timestamp"] = clean.split(":", 1)[1].strip()
    if "Time elapsed:" in clean:
        progress["elapsed_display"] = clean.split("Time elapsed:", 1)[1].strip()
    if "ETA:" in clean:
        progress["eta"] = clean.split("ETA:", 1)[1].strip()
    if "Mean reward:" in clean:
        try:
            progress["mean_reward"] = float(clean.split("Mean reward:", 1)[1])
        except ValueError:
            pass
    return clean


def expand_grid(base: Experiment, axes: dict):
    if not axes or any(not isinstance(v, list) or not v for v in axes.values()):
        raise ValueError("每个参数轴至少需要一个候选值")
    if len(list(itertools.islice(itertools.product(*axes.values()), 1001))) > 1000:
        raise ValueError("单次最多生成 1000 项，请缩小参数网格")
    result = []
    for i, values in enumerate(itertools.product(*axes.values())):
        data = base.model_dump()
        changes = dict(zip(axes, values))
        for key, val in changes.items():
            if key.startswith("training.") and key != "training.run_name":
                field = key.split(".", 1)[1]
                if field not in data["training"]:
                    raise ValueError(f"未知训练字段: {key}")
                data["training"][field] = val
            elif key.startswith("env."):
                data["overrides"][key] = val
            else:
                raise ValueError(f"不支持的参数轴: {key}")
        data["training"]["run_name"] = f"{base.training.run_name[:70]}_{i + 1:03d}_s{data['training']['seed']}"
        result.append({"config": Experiment.model_validate(data).model_dump(), "changes": changes})
    return result
