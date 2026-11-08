import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ.get("WORKBENCH_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA = Path(os.environ.get("WORKBENCH_DATA_DIR", ROOT / "data")).resolve()
OUTPUTS = Path(os.environ.get("WORKBENCH_OUTPUT_DIR", ROOT / "outputs")).resolve()


def prepare():
    for p in (DATA, OUTPUTS, DATA / "schemas", OUTPUTS / "attempts"):
        p.mkdir(parents=True, exist_ok=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        for attempt in range(10):
            try:
                os.replace(name, path)
                break
            except PermissionError as exc:
                # Windows readers and antivirus can briefly prevent replacement.
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 9:
                    raise
                time.sleep(min(0.01 * 2**attempt, 0.25))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def detached(module, *args, log_name="services.log"):
    prepare()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "backend") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (DATA / log_name).open("ab", buffering=0) as log:
        return subprocess.Popen([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **flags)
