import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from filelock import FileLock

from . import adapters, console
from .models import Experiment, Project
from .paths import DATA, ROOT, atomic_json, read_json
from .processes import identity, terminate_tree


def checkpoint_info(project, path):
    if project.adapter == "mock":
        data = read_json(path)
        if not data or "iteration" not in data:
            raise ValueError("Invalid mock checkpoint")
        return data
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    p = subprocess.run([project.python, "-s", str(ROOT / "adapters/checkpoint_info.py"), str(path)], env=adapters.training_environment(project), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90, **flags)
    if p.returncode:
        raise ValueError("checkpoint 无法读取: " + p.stderr[-1200:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def run(recipe_path):
    recipe = read_json(recipe_path)
    directory = Path(recipe_path).parent
    p = Project.model_validate(recipe["project"])
    spec = Experiment.model_validate(recipe["config"])
    progress = {"phase": "initializing", "progress": 0, "completed_updates": 0, "started": time.time(), "supervisor": identity(os.getpid())}
    child = None
    result = {"status": "failed"}
    lock = threading.Lock()
    def save_progress():
        try:
            atomic_json(directory / "progress.json", progress)
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            print(f"[workbench] Progress snapshot locked; training continues: {exc}", flush=True)
    try:
        with FileLock(str(DATA / "gpu.lock"), timeout=0):
            if recipe.get("checkpoint"):
                meta = checkpoint_info(p, recipe["checkpoint"])
                if spec.mode == "resume" and not meta.get("has_optimizer", True):
                    raise ValueError("该 checkpoint 不包含优化器，请使用仅加载权重")
                progress["loaded_iteration"] = meta["iteration"]
            env = adapters.training_environment(p)
            flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            child = subprocess.Popen(recipe["command"], cwd=p.root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **flags)
            progress["process"] = identity(child.pid)
            save_progress()
            def consume():
                pending = b""
                for chunk in iter(lambda: child.stdout.read1(65536), b""):
                    try:
                        console.append(directory, chunk)
                    except OSError as exc:
                        print(f"[workbench] Console write failed; training continues: {exc}", flush=True)
                    lines = (pending + chunk).split(b"\n")
                    pending = lines.pop()
                    for raw in lines:
                        with lock:
                            adapters.parse_line(raw.decode("utf-8", errors="replace"), progress)
                if pending:
                    with lock:
                        adapters.parse_line(pending.decode("utf-8", errors="replace"), progress)
            reader = threading.Thread(target=consume, daemon=True); reader.start()
            stopping = False
            while child.poll() is None:
                if (directory / "stop.request").exists():
                    stopping = True
                    with lock:
                        progress["phase"] = "stopping"
                    terminate_tree(progress.get("process"))
                with lock:
                    progress["heartbeat"] = time.time()
                    save_progress()
                time.sleep(0.5)
            reader.join(timeout=5)
            result.update(exit_code=child.returncode, finished=time.time(), stopped=stopping)
            if stopping:
                result["status"] = "cancelled"
            elif child.returncode:
                result.update(status="failed", message=f"训练退出码 {child.returncode}")
            else:
                output = adapters.logged_run_directory(Path(recipe["output_root"]), spec.training.run_name, progress)
                if output is None:
                    raise ValueError("训练未报告本次时间戳目录，无法确认最终 checkpoint")
                candidates = [f for f in output.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", f.name)]
                if not candidates:
                    raise ValueError("训练已退出，但没有找到最终 checkpoint")
                final = max(candidates, key=lambda f: (int(f.stem.split("_")[-1]), f.stat().st_mtime_ns))
                meta = checkpoint_info(p, final)
                expected_start = progress.get("loaded_iteration", 0) if spec.mode == "resume" else 0
                expected = expected_start + spec.training.max_iterations - 1
                if meta["iteration"] != expected:
                    raise ValueError(f"最终 checkpoint 迭代不符：预期 {expected}，实际 {meta['iteration']}")
                result.update(status="succeeded", final_checkpoint=str(final.resolve()), run_dir=str(final.parent.resolve()), iteration=meta["iteration"], checkpoint_info=meta)
    except Exception as exc:
        if child and child.poll() is None:
            terminate_tree(identity(child.pid))
        result.update(status="failed", message=str(exc), finished=time.time())
    finally:
        progress.update(finished=result.get("finished", time.time()), phase=result["status"])
        if result["status"] == "succeeded":
            progress.update(progress=1, completed_updates=spec.training.max_iterations)
        try:
            save_progress()
        finally:
            atomic_json(directory / "result.json", {**result, "progress": progress})


if __name__ == "__main__":
    run(sys.argv[1])
