import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


def identity(pid):
    try:
        p = psutil.Process(pid)
        return {"pid": p.pid, "created": p.create_time()}
    except psutil.Error:
        return None


def alive(info):
    if not info:
        return False
    try:
        p = psutil.Process(info["pid"])
        return abs(p.create_time() - info["created"]) < 0.01 and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError):
        return False


def training_processes(project):
    root = Path(project.root).resolve()
    entry = Path(project.train_entry).name
    result = []
    for p in psutil.process_iter(["pid", "create_time", "cmdline", "cwd", "name"]):
        try:
            cmd = p.info["cmdline"] or []
            if not any(Path(x).name == entry for x in cmd[1:]):
                continue
            matches = Path(p.info["cwd"] or "").resolve() == root or any(str(root).lower() in x.lower() for x in cmd)
            if matches:
                result.append({"pid": p.pid, "created": p.info["create_time"], "command": cmd, "name": p.info["name"]})
        except (psutil.Error, OSError):
            pass
    return result


def terminate_tree(info, grace=4):
    if not alive(info):
        return
    p = psutil.Process(info["pid"])
    children = p.children(recursive=True)
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            p.wait(timeout=grace)
        except (ProcessLookupError, psutil.Error):
            pass
    for q in [*reversed(children), p]:
        try:
            q.terminate()
        except psutil.Error:
            pass
    _, remaining = psutil.wait_procs([*children, p], timeout=grace)
    for q in remaining:
        try:
            q.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(remaining, timeout=grace)


def gpu_info():
    try:
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], timeout=5, text=True, **flags)
        return [{"index": int(a[0]), "name": a[1].strip(), "total_mb": int(a[2]), "used_mb": int(a[3]), "utilization": int(a[4])} for line in out.splitlines() if len(a := line.split(",")) == 5]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
