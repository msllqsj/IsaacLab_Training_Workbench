import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from . import db
from .paths import DATA, atomic_json, detached, prepare, read_json
from .processes import alive, identity, terminate_tree


def main():
    parser = argparse.ArgumentParser(description="Isaac Lab 本地训练工作台")
    parser.add_argument("action", choices=("start", "serve", "status", "stop"), nargs="?", default="start")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WORKBENCH_PORT", "8765")))
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    prepare(); db.initialize()
    server = read_json(DATA / "server.json")
    if args.action == "status":
        print({"server": {"alive": alive(server), **(server or {})}, "worker": {"alive": alive(read_json(DATA / 'worker.json'))}, "queue": db.state("queue")})
        return
    if args.action == "stop":
        q = db.state("queue"); q.update(enabled=False, message="服务已停止，队列暂停"); db.set_state("queue", q)
        worker = read_json(DATA / "worker.json")
        if alive(worker):
            import psutil
            try:
                process = psutil.Process(worker["pid"])
                process.terminate()
                process.wait(timeout=10)
            except psutil.Error:
                pass
        # Do not terminate supervised training when stopping only the web service.
        if server and alive(server):
            import psutil
            try:
                process = psutil.Process(server["pid"])
                process.terminate()
                process.wait(timeout=10)
            except psutil.Error:
                pass
        print("服务已停止；独立监督的训练进程保持运行。")
        return
    if args.action == "serve":
        atomic_json(DATA / "server.json", {**identity(os.getpid()), "port": args.port, "url": f"http://127.0.0.1:{args.port}"})
        import uvicorn
        uvicorn.run("workbench.app:app", host=args.host, port=args.port, access_log=False)
        return
    url = server.get("url") if alive(server) else f"http://127.0.0.1:{args.port}"
    if not alive(server):
        detached("workbench.cli", "serve", "--port", args.port, log_name="server.log")
    for _ in range(60):
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as res:
                if res.status == 200:
                    health = json.load(res)
                    if not health.get("worker", {}).get("alive") and os.environ.get("WORKBENCH_NO_WORKER") != "1":
                        time.sleep(0.25)
                        continue
                    print("Workbench: " + url)
                    if args.open:
                        webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise SystemExit("服务启动超时，请检查 data/server.log")


if __name__ == "__main__":
    main()
