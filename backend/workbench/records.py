"""Remove visible records while retaining files and immutable execution evidence."""
import json
import shutil
import time
from pathlib import Path

from . import adapters, console, db
from .paths import OUTPUTS, read_json
from .processes import alive, terminate_tree

ACTIVE = ("starting", "running", "stopping")


def job_summary(job, runs=None):
    runs = db.runs(job["id"], include_deleted=True) if runs is None else runs
    latest = runs[0] if runs else None
    result = {**job, "attempt_count": len(runs)}
    if latest:
        data = latest["data"]
        # A removed history entry still supplies queue progress and dependencies.
        result["last_result"] = {"status": latest["status"], "progress": data.get("progress", {}),
                                 "final_checkpoint": None if data.get("files_deleted") else data.get("final_checkpoint"), "checkpoint": data.get("checkpoint")}
    return result


def ensure_finished(run):
    data = run["data"]
    progress = read_json(Path(data["directory"]) / "progress.json", data.get("progress", {}))
    if run["status"] in ACTIVE or any(alive(info) for info in (data.get("supervisor"), progress.get("supervisor"), progress.get("process"))):
        raise ValueError("运行中的训练不能删除，请先停止训练并等待退出")


def stop_owned_tensorboards(ids, connection=None):
    for rid in ids:
        key = "tensorboard:" + rid
        if connection is None:
            terminate_tree(db.state(key))
            db.set_state(key, None)
        else:
            row = connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            terminate_tree(json.loads(row[0]) if row else None)
            connection.execute("DELETE FROM state WHERE key=?", (key,))


def purge_files(connection, runs, deleting_jobs=()):
    ids = {run["id"] for run in runs}
    targets = set()
    for run in runs:
        data = run["data"]
        progress = data.get("progress") or read_json(Path(data["directory"]) / "progress.json", {})
        root = Path(data["output_root"]).resolve()
        expected = adapters.logged_run_directory(root, data["config"]["training"]["run_name"], progress)
        target = Path(data["run_dir"]).resolve() if data.get("run_dir") else expected
        if target is None:
            continue
        if expected is None:
            raise ValueError("缺少本次训练时间戳，无法安全确认删除范围；可仅删除记录")
        allowed = ((Path(data["project"]["root"]).resolve() / "logs").resolve(), (OUTPUTS / "training").resolve())
        if target.parent != root or (expected and target != expected) or not any(target != folder and target.is_relative_to(folder) for folder in allowed):
            raise ValueError("无法安全确认本次训练目录，未删除文件")
        targets.add(target)
    def inside(path):
        return bool(path) and any(Path(path).expanduser().resolve().is_relative_to(target) for target in targets)
    for row in connection.execute("SELECT id,config FROM jobs WHERE deleted_at IS NULL"):
        if row["id"] in deleting_jobs:
            continue
        config = json.loads(row["config"])
        cp = config["checkpoint"]
        referenced = cp["path"] if cp["kind"] == "file" else None
        if cp["kind"] == "job":
            sources = connection.execute("SELECT data FROM runs WHERE job_id=? AND status='succeeded' ORDER BY created DESC", (cp["job_id"],)).fetchall()
            for source in sources:
                data = json.loads(source[0])
                if not data.get("files_deleted"):
                    referenced = data.get("final_checkpoint")
                    break
        if inside(referenced):
            raise ValueError(f"{config['training']['run_name']} 仍依赖此目录中的 checkpoint；请先修改来源，或仅删除记录")
    for row in connection.execute("SELECT id,data FROM runs"):
        if row["id"] not in ids and json.loads(row["data"]).get("run_dir") and Path(json.loads(row["data"])["run_dir"]).resolve() in targets:
            raise ValueError("此训练目录仍被其他运行记录使用，未删除文件")
    for run in runs:
        directory = Path(run["data"]["directory"]).resolve()
        if directory != (OUTPUTS / "attempts" / run["id"]).resolve():
            raise ValueError("控制日志目录无效，未删除文件")
    stop_owned_tensorboards(ids, connection)
    try:
        for target in targets:
            if target.exists():
                shutil.rmtree(target)
        for run in runs:
            directory = Path(run["data"]["directory"]).resolve()
            console.clear(directory)
            data = {**run["data"], "files_deleted": True}
            connection.execute("UPDATE runs SET data=? WHERE id=?", (json.dumps(data, ensure_ascii=False), run["id"]))
    except OSError as exc:
        raise ValueError(f"清理训练文件失败，请检查路径；部分文件可能已被移除：{exc}") from exc


def delete_run(rid, delete_files=False):
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute("SELECT * FROM runs WHERE id=? AND deleted_at IS NULL", (rid,)).fetchone()
        if not row:
            raise ValueError("运行记录不存在或已删除")
        run = {**dict(row), "data": json.loads(row["data"])}
        ensure_finished(run)
        if delete_files:
            purge_files(c, [run])
        now = time.time()
        c.execute("UPDATE runs SET deleted_at=?,updated=? WHERE id=?", (now, now, rid))
    stop_owned_tensorboards([rid])
    return {"deleted": True, "files_deleted": delete_files, "id": rid}


def delete_job(jid, delete_files=False):
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        job = c.execute("SELECT * FROM jobs WHERE id=? AND deleted_at IS NULL", (jid,)).fetchone()
        if not job:
            raise ValueError("实验不存在或已删除")
        if job["status"] in ACTIVE:
            raise ValueError("运行中的实验不能删除，请先停止训练并等待退出")
        dependents = []
        for other in c.execute("SELECT config FROM jobs WHERE workflow_id=? AND id<>? AND deleted_at IS NULL", (job["workflow_id"], jid)):
            config = json.loads(other["config"])
            if config["checkpoint"]["kind"] == "job" and config["checkpoint"]["job_id"] == jid:
                dependents.append(config["training"]["run_name"])
        if dependents:
            raise ValueError("以下实验依赖此任务，请先修改来源或删除依赖项：" + "、".join(dependents[:8]))
        runs = c.execute("SELECT * FROM runs WHERE job_id=?", (jid,)).fetchall()
        for run in runs:
            ensure_finished({**dict(run), "data": json.loads(run["data"])})
        if delete_files:
            purge_files(c, [{**dict(run), "data": json.loads(run["data"])} for run in runs], deleting_jobs={jid})
        now = time.time()
        c.execute("UPDATE jobs SET deleted_at=?,updated=? WHERE id=?", (now, now, jid))
        c.execute("UPDATE runs SET deleted_at=COALESCE(deleted_at,?),updated=? WHERE job_id=?", (now, now, jid))
        remaining = c.execute("SELECT id FROM jobs WHERE workflow_id=? AND deleted_at IS NULL ORDER BY position", (job["workflow_id"],)).fetchall()
        for position, row in enumerate(remaining):
            c.execute("UPDATE jobs SET position=?,updated=? WHERE id=?", (position, now, row["id"]))
    stop_owned_tensorboards([run["id"] for run in runs])
    return {"deleted": True, "files_deleted": delete_files, "id": jid, "records_removed": len(runs)}


def delete_project(pid):
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        project = c.execute("SELECT id FROM projects WHERE id=? AND deleted_at IS NULL", (pid,)).fetchone()
        if not project:
            raise ValueError("项目连接不存在或已移除")
        workflows = {row["id"] for row in c.execute("SELECT id FROM workflows WHERE project_id=?", (pid,))}
        jobs = {row["id"] for row in c.execute("SELECT j.id FROM jobs j JOIN workflows w ON w.id=j.workflow_id WHERE w.project_id=?", (pid,))}
        if c.execute("SELECT 1 FROM jobs j JOIN workflows w ON w.id=j.workflow_id WHERE w.project_id=? AND j.status IN ('starting','running','stopping')", (pid,)).fetchone():
            raise ValueError("运行中的项目不能移除，请先停止训练并等待退出")
        runs = []
        for row in c.execute("SELECT * FROM runs"):
            run = {**dict(row), "data": json.loads(row["data"])}
            if row["job_id"] in jobs or run["data"]["project"]["id"] == pid:
                ensure_finished(run)
                runs.append(run)
        now = time.time()
        c.execute("UPDATE projects SET deleted_at=?,updated=? WHERE id=?", (now, now, pid))
        c.execute("UPDATE jobs SET deleted_at=COALESCE(deleted_at,?),updated=? WHERE workflow_id IN (SELECT id FROM workflows WHERE project_id=?)", (now, now, pid))
        for run in runs:
            c.execute("UPDATE runs SET deleted_at=COALESCE(deleted_at,?),updated=? WHERE id=?", (now, now, run["id"]))
        queue_row = c.execute("SELECT value FROM state WHERE key='queue'").fetchone()
        queue = json.loads(queue_row[0]) if queue_row else {}
        if queue.get("workflow_id") in workflows:
            queue.update(enabled=False, workflow_id=None, wait_for=None, message="项目连接已移除，队列已暂停")
            c.execute("UPDATE state SET value=? WHERE key='queue'", (json.dumps(queue, ensure_ascii=False),))
    stop_owned_tensorboards([run["id"] for run in runs])
    return {"deleted": True, "files_deleted": False, "id": pid}
