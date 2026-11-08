import asyncio
import itertools
import json
import os
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from filelock import FileLock
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import adapters, console, db, records
from .models import Experiment, ExperimentSave, GridRequest, Project, WorkflowCreate, uid
from .paths import DATA, OUTPUTS, ROOT, atomic_json, detached, prepare, read_json
from .processes import alive, gpu_info, identity, terminate_tree, training_processes
from .worker import ACTIVE


def ensure_worker():
    info = read_json(DATA / "worker.json")
    if not alive(info):
        detached("workbench.worker", log_name="worker.log")


@asynccontextmanager
async def lifespan(app):
    db.initialize()
    if os.environ.get("WORKBENCH_NO_WORKER") != "1":
        ensure_worker()
    yield


app = FastAPI(title="Isaac Lab Training Workbench", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?", allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
        return JSONResponse({"detail": "Only local origins are allowed"}, status_code=403)
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


def schema_response(p):
    s = adapters.schema(p)
    status = read_json(DATA / "schemas" / f"{p.id}.status.json", {"status": "empty", "message": "尚未读取项目配置"})
    if status.get("status") == "running" and status.get("pid"):
        try:
            import psutil
            if not psutil.pid_exists(status["pid"]):
                status = {"status": "failed", "message": "配置导出进程已结束，请查看日志或重试"}
        except OSError:
            pass
    if s:
        s = {**s, "stale": s.get("fingerprint", {}).get("hash") != adapters.fingerprint(p)["hash"]}
    return {"schema": s, "probe": status}


def enriched_run(r):
    saved = r["data"].get("progress", {})
    progress = read_json(Path(r["data"]["directory"]) / "progress.json", saved) if r["status"] in ACTIVE or not saved else saved
    return {**r, "progress": progress, "command_display": adapters.display_command(r["data"]["command"])}


@app.get("/api/health")
def health():
    return {"ok": True, "version": "0.1.0", "root": str(ROOT), "worker": {"alive": alive(read_json(DATA / "worker.json")), **(read_json(DATA / "worker.json", {}))}}


@app.get("/api/state")
def snapshot():
    workflows = db.workflows()
    all_runs = db.runs(include_deleted=True)
    by_job = {}
    for run in all_runs:
        by_job.setdefault(run["job_id"], []).append(run)
    return {"projects": db.projects(), "workflows": workflows, "queue": db.state("queue"),
            "jobs": {w["id"]: [records.job_summary(job, by_job.get(job["id"], [])) for job in db.jobs(w["id"])] for w in workflows},
            "runs": [enriched_run(r) for r in all_runs if r["deleted_at"] is None], "worker_alive": alive(read_json(DATA / "worker.json"))}


@app.get("/api/gpus")
def gpus():
    return gpu_info()


@app.put("/api/projects/{pid}")
def save_project(pid: str, p: Project):
    if pid != p.id:
        raise ValueError("Project id mismatch")
    if any(r["status"] in ACTIVE and r["data"]["project"]["id"] == pid for r in db.runs()):
        raise ValueError("训练运行期间不能修改项目设置")
    adapters.validate_project(p)
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        existing = c.execute("SELECT deleted_at FROM projects WHERE id=?", (pid,)).fetchone()
        if existing and existing["deleted_at"] is not None:
            raise ValueError("此项目连接已移除，请添加新的项目连接")
        c.execute("INSERT INTO projects(id,data,updated) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data,updated=excluded.updated", (pid, p.model_dump_json(), time.time()))
    return p


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    return records.delete_project(pid)


@app.get("/api/projects/{pid}/schema")
def get_schema(pid: str):
    return schema_response(db.project(pid))


@app.post("/api/projects/{pid}/schema/refresh")
def refresh_schema(pid: str):
    p = db.project(pid); adapters.validate_project(p)
    status = schema_response(p)["probe"]
    if status["status"] == "running":
        return status
    if any(r["status"] in ACTIVE for r in db.runs()) or training_processes(p):
        raise ValueError("请在现有训练结束后刷新配置")
    atomic_json(DATA / "schemas" / f"{pid}.status.json", {"status": "running", "message": "正在启动配置导出", "started": time.time()})
    child = detached("workbench.probe", pid, log_name="probes.log")
    return {"status": "running", "pid": child.pid}


@app.get("/api/projects/{pid}/schema/log")
def schema_log(pid: str):
    db.project(pid)
    path = DATA / "schemas" / f"{pid}.log"
    if not path.exists():
        return {"text": ""}
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 24000))
        return {"text": f.read().decode("utf-8", errors="replace")}


@app.get("/api/projects/{pid}/processes")
def external_processes(pid: str):
    return training_processes(db.project(pid))


@app.post("/api/workflows")
def create_workflow(data: WorkflowCreate):
    db.project(data.project_id)
    wid = uid()
    with db.connect() as c:
        c.execute("INSERT INTO workflows VALUES(?,?,?,?)", (wid, data.name, data.project_id, time.time()))
    return db.workflow(wid)


def validate_dependencies(wid, spec, jid=None, position=None):
    jobs = db.jobs(wid)
    if position is None:
        position = next((j["position"] for j in jobs if j["id"] == jid), len(jobs))
    if any(j["config"]["training"]["run_name"] == spec.training.run_name and j["id"] != jid for j in jobs):
        raise ValueError("同一工作流中的实验名称必须唯一")
    if spec.checkpoint.kind == "job":
        parent = next((j for j in jobs if j["id"] == spec.checkpoint.job_id), None)
        if not parent or parent["position"] >= position:
            raise ValueError("checkpoint 依赖必须指向本工作流中排在前面的实验")


@app.post("/api/workflows/{wid}/jobs")
def create_job(wid: str, spec: ExperimentSave):
    w = db.workflow(wid); p = db.project(w["project_id"])
    baseline = adapters.validate_config(p, spec)
    validate_dependencies(wid, spec)
    return db.job(db.add_job(wid, spec.model_dump(exclude={"note"}), baseline, spec.note if "note" in spec.model_fields_set else None))


@app.put("/api/jobs/{jid}")
def edit_job(jid: str, spec: ExperimentSave):
    j = db.job(jid)
    if j["status"] in (*ACTIVE, "succeeded", "skipped") or db.runs(jid, include_deleted=True):
        raise ValueError("已执行的实验配置不可改写，请复制为新实验")
    p = db.project(db.workflow(j["workflow_id"])["project_id"])
    baseline = adapters.validate_config(p, spec)
    validate_dependencies(j["workflow_id"], spec, jid)
    note = {"note": spec.note, "note_updated": time.time()} if "note" in spec.model_fields_set else {}
    db.update_job(jid, config=spec.model_dump(exclude={"note"}), baseline=baseline, status="draft", message="", **note)
    return db.job(jid)


@app.get("/api/jobs/{jid}/note")
def job_note(jid: str):
    return db.job_note(jid)


@app.put("/api/jobs/{jid}/note")
def save_job_note(jid: str, text: str = Body(embed=True, max_length=20000)):
    return db.save_job_note(jid, text)


@app.post("/api/workflows/{wid}/reorder")
def reorder(wid: str, ids: list[str] = Body()):
    jobs = db.jobs(wid)
    if set(ids) != {j["id"] for j in jobs} or len(ids) != len(jobs):
        raise ValueError("排序列表不完整")
    if any(j["status"] in ACTIVE for j in jobs):
        raise ValueError("当前工作流正在训练，请完成后排序")
    positions = {jid: i for i, jid in enumerate(ids)}
    for j in jobs:
        cp = j["config"]["checkpoint"]
        if cp["kind"] == "job" and positions.get(cp["job_id"], 10**9) >= positions[j["id"]]:
            raise ValueError("此顺序会把依赖放在后面")
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        for jid, position in positions.items():
            c.execute("UPDATE jobs SET position=?,updated=? WHERE id=?", (position, time.time(), jid))
    return {"ok": True}


@app.post("/api/workflows/{wid}/grid/preview")
def grid_preview(wid: str, data: GridRequest):
    w = db.workflow(wid); p = db.project(w["project_id"])
    rows = adapters.expand_grid(data.base, data.axes)
    existing = {j["config"]["training"]["run_name"] for j in db.jobs(wid)}
    for row in rows:
        spec = Experiment.model_validate(row["config"])
        if spec.training.run_name in existing:
            raise ValueError("生成名称与已有实验重复，请修改基础名称")
        adapters.validate_config(p, spec)
        validate_dependencies(wid, spec)
        row["config"] = spec.model_dump()
    return rows


@app.post("/api/workflows/{wid}/grid/commit")
def grid_commit(wid: str, data: GridRequest):
    rows = grid_preview(wid, data)
    p = db.project(db.workflow(wid)["project_id"])
    baseline = adapters.fingerprint(p)
    return [db.job(db.add_job(wid, row["config"], baseline)) for row in rows]


@app.post("/api/workflows/{wid}/start")
def start_workflow(wid: str, data: dict = Body(default={})):
    db.workflow(wid)
    q = db.state("queue")
    if any(r["status"] in ACTIVE for r in db.runs()) and q.get("workflow_id") != wid:
        raise ValueError("另一个工作流仍在运行")
    jobs = db.jobs(wid)
    if not jobs:
        raise ValueError("请先添加实验")
    p = db.project(db.workflow(wid)["project_id"])
    for j in jobs:
        if j["status"] in ("failed", "interrupted", "blocked"):
            raise ValueError("请先重试、编辑或跳过失败/阻塞实验")
        if j["status"] in ("draft", "queued"):
            adapters.validate_config(p, Experiment.model_validate(j["config"]))
            if j["baseline"]["hash"] != adapters.fingerprint(p)["hash"]:
                raise ValueError("源码已变化，请更新待运行实验基线")
    with db.connect() as c:
        c.execute("UPDATE jobs SET status='queued',updated=? WHERE workflow_id=? AND status='draft' AND deleted_at IS NULL", (time.time(), wid))
    q.update(enabled=True, workflow_id=wid, message="等待启动下一项实验", wait_for=data.get("wait_for", q.get("wait_for")))
    db.set_state("queue", q)
    if os.environ.get("WORKBENCH_NO_WORKER") != "1":
        ensure_worker()
    return q


@app.post("/api/queue/pause")
def pause_queue():
    q = db.state("queue"); q.update(enabled=False, message="队列已暂停，当前训练会继续完成"); db.set_state("queue", q); return q


@app.post("/api/queue/wait")
def wait_for_process(data: dict = Body()):
    info = {"pid": int(data["pid"]), "created": float(data["created"])} if data else None
    if info and not alive(info):
        raise ValueError("该进程已经退出，请重新检查")
    q = db.state("queue")
    q.update(wait_for=info, message=f"已设置等待 PID {info['pid']}" if info else "已清除进程等待")
    db.set_state("queue", q)
    return q


@app.get("/api/workflows/{wid}/export")
def export_workflow(wid: str):
    w = db.workflow(wid)
    jobs = db.jobs(wid)
    positions = {j["id"]: i for i, j in enumerate(jobs)}
    result = []
    for j in jobs:
        spec = Experiment.model_validate(j["config"]).model_dump()
        cp = spec["checkpoint"]
        dependency = positions[cp["job_id"]] if cp["kind"] == "job" else None
        if dependency is not None:
            cp["job_id"] = ""
        result.append({"config": spec, "dependency_index": dependency, "note": db.job_note(j["id"])["text"]})
    return {"format": "isaac-workbench-workflow-v1", "name": w["name"], "jobs": result}


@app.post("/api/projects/{pid}/workflows/import")
def import_workflow(pid: str, data: dict = Body()):
    p = db.project(pid)
    if data.get("format") != "isaac-workbench-workflow-v1" or not 1 <= len(data.get("jobs", [])) <= 1000:
        raise ValueError("工作流文件格式错误或实验数量不在 1–1000 范围")
    items = []
    ids = [uid() for _ in data["jobs"]]
    names = set()
    baseline = adapters.fingerprint(p)
    for index, item in enumerate(data["jobs"]):
        raw = json.loads(json.dumps(item["config"]))
        dep = item.get("dependency_index")
        if raw["checkpoint"]["kind"] == "job":
            if type(dep) is not int or not 0 <= dep < index:
                raise ValueError("导入文件含无效依赖，必须引用前面的实验")
            raw["checkpoint"]["job_id"] = ids[dep]
        spec = ExperimentSave.model_validate({**raw, "note": item.get("note", "")})
        adapters.validate_config(p, spec)
        if spec.training.run_name in names:
            raise ValueError("导入文件含重复实验名称")
        names.add(spec.training.run_name)
        items.append((spec.model_dump_json(exclude={"note"}), spec.note))
    name = str(data.get("name", "导入的工作流"))[:110] + " · 导入"
    wid = uid()
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        c.execute("INSERT INTO workflows VALUES(?,?,?,?)", (wid, name, pid, time.time()))
        for index, (spec, note) in enumerate(items):
            now = time.time()
            c.execute("INSERT INTO jobs(id,workflow_id,position,config,baseline,status,message,updated,note,note_updated) VALUES(?,?,?,?,?,?,?,?,?,?)", (ids[index], wid, index, spec, json.dumps(baseline), "draft", "", now, note, now))
    return db.workflow(wid)


@app.post("/api/jobs/{jid}/{action}")
def job_action(jid: str, action: str):
    j = db.job(jid)
    if j["status"] in ACTIVE:
        raise ValueError("请使用停止当前训练")
    if action == "retry":
        if j["status"] not in ("failed", "interrupted", "cancelled", "blocked"):
            raise ValueError("该实验无需重试")
        p = db.project(db.workflow(j["workflow_id"])["project_id"])
        adapters.validate_config(p, Experiment.model_validate(j["config"]))
        if j["baseline"]["hash"] != adapters.fingerprint(p)["hash"]:
            raise ValueError("源码已变化，请先更新基线")
        db.update_job(jid, status="draft", message="")
    elif action in ("skip", "cancel"):
        if j["status"] == "succeeded":
            raise ValueError("不能跳过已成功的实验")
        db.update_job(jid, status="skipped" if action == "skip" else "cancelled", message="用户跳过" if action == "skip" else "用户取消")
    else:
        raise ValueError("未知操作")
    return db.job(jid)


@app.post("/api/workflows/{wid}/baseline")
def accept_baseline(wid: str):
    p = db.project(db.workflow(wid)["project_id"])
    s = adapters.schema(p)
    if not s or s["fingerprint"]["hash"] != adapters.fingerprint(p)["hash"]:
        raise ValueError("请先刷新项目配置")
    current = adapters.fingerprint(p)
    changes = []
    for j in db.jobs(wid):
        if j["status"] in ("draft", "queued", "blocked"):
            adapters.validate_config(p, Experiment.model_validate(j["config"]))
            changes.extend(adapters.changed_files(j["baseline"], current))
            db.update_job(j["id"], baseline=current, status="draft", message="")
    return {"updated": True, "changed_files": sorted(set(changes))}


@app.delete("/api/runs/{rid}")
def delete_run(rid: str, delete_files: bool = False):
    return records.delete_run(rid, delete_files=delete_files)


@app.get("/api/runs/{rid}/note")
def run_note(rid: str):
    return db.job_note(db.run(rid)["job_id"])


@app.put("/api/runs/{rid}/note")
def save_run_note(rid: str, text: str = Body(embed=True, max_length=20000)):
    return db.save_job_note(db.run(rid)["job_id"], text, run_id=rid)


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str, delete_files: bool = False):
    return records.delete_job(jid, delete_files=delete_files)


@app.post("/api/runs/{rid}/stop")
def stop_run(rid: str):
    r = db.run(rid)
    if r["status"] not in ACTIVE:
        raise ValueError("该训练已结束")
    pause_queue()
    directory = Path(r["data"]["directory"])
    (directory / "stop.request").write_text("stop", encoding="ascii")
    if r["data"].get("orphaned"):
        progress = read_json(directory / "progress.json", {})
        terminate_tree(progress.get("process"))
        atomic_json(directory / "result.json", {"status": "cancelled", "stopped": True, "finished": time.time()})
    db.update_run(rid, status="stopping")
    return {"stopping": True, "checkpoint_policy": "仅保留已经成功保存的 checkpoint"}


@app.get("/api/runs/{rid}/logs")
def logs(rid: str, offset: int = Query(default=0, ge=0), tail: bool = False, generation: str | None = None):
    r = db.run(rid)
    result = console.read(r["data"]["directory"], offset, tail, generation)
    result["text"] = adapters.ANSI.sub("", result["text"])
    return result


@app.delete("/api/runs/{rid}/logs")
def clear_logs(rid: str):
    return console.clear(db.run(rid)["data"]["directory"])


def run_directory(r):
    path = r["data"].get("run_dir")
    if path:
        return Path(path)
    output = Path(r["data"]["output_root"])
    return adapters.logged_run_directory(output, r["data"]["config"]["training"]["run_name"], enriched_run(r)["progress"])


@app.get("/api/runs/{rid}/artifacts")
def artifacts(rid: str):
    r = db.run(rid); root = run_directory(r)
    return {"run_dir": str(root) if root else "", "files": [{"path": str(p.relative_to(root)).replace("\\", "/"), "size": p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".pt", ".yaml", ".mp4", ".json")] if root and root.exists() else []}


@app.get("/api/runs/{rid}/artifact")
def artifact(rid: str, path: str):
    root = run_directory(db.run(rid))
    if root is None:
        raise HTTPException(404)
    root = root.resolve(); p = (root / path).resolve()
    if not p.is_relative_to(root) or not p.is_file() or p.suffix not in (".pt", ".yaml", ".mp4", ".json"):
        raise HTTPException(404)
    return FileResponse(p, filename=p.name if p.suffix != ".mp4" else None)


@app.get("/api/files")
def files(project_id: str, path: str = ""):
    p = db.project(project_id)
    roots = [Path(p.root).resolve(), OUTPUTS.resolve()]
    folder = Path(path).expanduser().resolve() if path else roots[0] / "logs"
    if not folder.exists() and not path:
        folder = roots[0]
    if not folder.is_dir() or not any(folder.is_relative_to(r) for r in roots):
        raise ValueError("请选择项目目录或工作台 outputs 目录中的文件夹")
    entries = []
    for f in sorted(folder.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        if f.name.startswith(".") or not any(f.resolve().is_relative_to(r) for r in roots):
            continue
        if f.is_dir() or f.suffix == ".pt":
            entries.append({"name": f.name, "path": str(f), "directory": f.is_dir(), "size": f.stat().st_size if f.is_file() else None})
    parent = str(folder.parent) if any(folder.parent.is_relative_to(r) for r in roots) else None
    return {"path": str(folder), "parent": parent, "roots": [str(r) for r in roots], "entries": entries[:1000]}


@app.post("/api/command/preview")
def preview(data: dict = Body()):
    p = db.project(data["project_id"]); spec = Experiment.model_validate(data["config"])
    checkpoint = spec.checkpoint.path if spec.checkpoint.kind == "file" else str(OUTPUTS / "<上游实验最终checkpoint>.pt")
    cmd = adapters.build_command(p, spec, adapters.training_output(p), checkpoint)
    return {"args": cmd, "display": adapters.display_command(cmd)}


def wait_tensorboard(info, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive(info):
            raise ValueError(f"TensorBoard 启动失败，进程已退出。启动日志：{info['log']}")
        try:
            with urlopen(info["url"] + "/data/environment", timeout=1) as response:
                if response.status == 200:
                    return info
        except OSError:
            pass
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
    raise ValueError(f"TensorBoard 尚未就绪，请稍后重试。启动日志：{info['log']}")


@app.post("/api/runs/{rid}/tensorboard")
def start_tensorboard(rid: str):
    r = db.run(rid); root = run_directory(r)
    if root is None or not root.exists():
        raise ValueError("训练尚未创建日志目录")
    key = "tensorboard:" + rid
    with FileLock(str(DATA / f"tensorboard-{rid}.lock"), timeout=5):
        result = db.state(key)
        if not alive(result):
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            logpath = Path(r["data"]["directory"]) / "tensorboard.log"
            with logpath.open("ab") as log:
                child = subprocess.Popen([r["data"]["project"]["python"], "-s", "-m", "tensorboard.main", "--logdir", str(root), "--host", "127.0.0.1", "--port", str(port), "--load_fast", "false"], env=adapters.training_environment(Project.model_validate(r["data"]["project"])), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **flags)
            process = identity(child.pid)
            if process is None:
                raise ValueError(f"TensorBoard 启动失败，进程已退出。启动日志：{logpath}")
            result = {**process, "url": f"http://127.0.0.1:{port}", "log": str(logpath)}
            db.set_state(key, result)
    return wait_tensorboard(result)


@app.post("/api/runs/{rid}/tensorboard/stop")
def stop_tensorboard(rid: str):
    key = "tensorboard:" + rid
    with FileLock(str(DATA / f"tensorboard-{rid}.lock"), timeout=5):
        terminate_tree(db.state(key)); db.set_state(key, None)
    return {"stopped": True}


@app.get("/api/events")
async def events(request: Request):
    async def stream():
        previous = None
        while not await request.is_disconnected():
            with db.connect() as c:
                revision = c.execute("SELECT COALESCE(MAX(updated),0) FROM runs").fetchone()[0]
                revision2 = c.execute("SELECT COALESCE(MAX(updated),0) FROM jobs").fetchone()[0]
            signature = json.dumps([revision, revision2, db.state("queue")], sort_keys=True)
            if signature != previous:
                yield "event: state\ndata: " + json.dumps({"revision": time.time()}) + "\n\n"
                previous = signature
            else:
                yield ": heartbeat\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


dist = ROOT / "frontend/dist"
if (dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")


@app.get("/{path:path}")
def frontend(path: str):
    if path.startswith("api/"):
        raise HTTPException(404)
    if (dist / "index.html").exists():
        return FileResponse(dist / "index.html")
    return JSONResponse({"message": "前端尚未构建，请执行 npm.cmd run build", "api": "/docs"})
