import json
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout

from . import adapters, db
from .paths import DATA, ROOT, atomic_json, prepare, read_json
from .processes import training_processes


def run(pid):
    prepare(); db.initialize()
    statefile = DATA / "schemas" / f"{pid}.status.json"
    p = db.project(pid)
    atomic_json(statefile, {"status": "running", "message": "正在读取真实 Isaac Lab 默认配置…", "started": time.time()})
    try:
        with FileLock(str(DATA / "gpu.lock"), timeout=0):
            adapters.validate_project(p)
            if training_processes(p):
                raise ValueError("存在正在运行的训练，请结束后刷新配置")
            baseline = adapters.fingerprint(p)
            temp = DATA / "schemas" / f"{pid}.export.json"
            logpath = DATA / "schemas" / f"{pid}.log"
            env = adapters.training_environment(p)
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            if p.adapter == "mock":
                data = {"task": p.task, "experiment_name": "mock_task", "stage": 1, "fields": [{"path": "env.rewards.feet_gait.weight", "value": 0.2, "kind": "float", "nullable": False, "group": "rewards", "label": "feet_gait"}], "training": {"run_name": "experiment", "max_iterations": 3, "num_envs": 4, "seed": 42, "device": "cpu", "save_interval": 1, "headless": True, "video": False, "video_length": 200, "video_interval": 24000}, "versions": {"adapter": "test"}}
            else:
                cmd = [p.python, "-s", "-u", str(ROOT / "adapters/export_config.py"), "--project", p.root, "--task", p.task, "--imports", ",".join(p.imports), "--output", str(temp)]
                with logpath.open("wb") as log:
                    process = subprocess.Popen(cmd, cwd=p.root, env=env, stdout=log, stderr=subprocess.STDOUT, **flags)
                    atomic_json(statefile, {"status": "running", "message": "Isaac Lab 配置初始化中（通常需要数十秒）", "pid": process.pid, "started": time.time()})
                    code = process.wait(timeout=300)
                if code:
                    raise ValueError(f"配置导出失败，查看配置日志（退出码 {code}）")
                data = read_json(temp)
                if not data:
                    raise ValueError("配置导出未生成结果")
            if adapters.fingerprint(p)["hash"] != baseline["hash"]:
                raise ValueError("导出期间源码发生变化，请重新刷新")
            data.update(fingerprint=baseline, exported_at=time.time())
            atomic_json(adapters.schema_path(pid), data)
            atomic_json(statefile, {"status": "ready", "message": "配置已就绪", "finished": time.time()})
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            from .processes import identity, terminate_tree
            terminate_tree(identity(process.pid))
        atomic_json(statefile, {"status": "failed", "message": "GPU 正被工作台占用" if isinstance(exc, Timeout) else str(exc), "finished": time.time()})


if __name__ == "__main__":
    run(sys.argv[1])
