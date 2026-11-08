"""Cross-platform bootstrap; creates only this application's environment."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description="安装独立训练工作台")
p.add_argument("--skip-frontend", action="store_true")
args = p.parse_args()
if sys.version_info < (3, 11):
    raise SystemExit("需要 Python 3.11 或更新版本")
venv = root / ".venv"
python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.exists():
    subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
if (root / "requirements.lock").exists():
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", "requirements.lock"], cwd=root)
    subprocess.check_call([str(python), "-m", "pip", "install", "--no-deps", "-e", ".[test]"], cwd=root)
else:
    subprocess.check_call([str(python), "-m", "pip", "install", "-e", ".[test]"], cwd=root)
if not args.skip_frontend:
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        raise SystemExit("请先安装 Node.js 22.12+ 或 24 LTS，再运行安装脚本")
    subprocess.check_call([npm, "ci" if (root / "frontend/package-lock.json").exists() else "install", "--no-audit", "--no-fund"], cwd=root / "frontend")
    subprocess.check_call([npm, "run", "build"], cwd=root / "frontend")
print("安装完成。运行 start.cmd 或 bash start.sh 启动。")
