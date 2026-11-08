import json
import os
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout

from . import adapters, db
from .models import Experiment, uid
from .paths import DATA, OUTPUTS, atomic_json, detached, read_json
from .processes import alive, identity, training_processes

ACTIVE = ("starting", "running", "stopping")


def pause(message):
    q = db.state("queue"); q.update(enabled=False, message=message); db.set_state("queue", q)


def tick():
    active = [r for r in db.runs() if r["status"] in ACTIVE]
    if active:
        for run in active:
            directory = Path(run["data"]["directory"])
            progress = read_json(directory / "progress.json", {})
            result = read_json(directory / "result.json")
            supervisor = progress.get("supervisor") or run["data"].get("supervisor")
            if result and not alive(supervisor):
                db.update_run(run["id"], **{**result, "progress": result.get("progress", progress)})
                db.update_job(run["job_id"], status=result["status"], message=result.get("message", ""))
                if result["status"] != "succeeded":
                    pause(result.get("message", "训练已停止，队列暂停"))
            elif alive(supervisor):
                status = "stopping" if progress.get("phase") == "stopping" else "running"
                db.update_run(run["id"], status=status, progress=progress, supervisor=supervisor)
                if db.job(run["job_id"])["status"] != status:
                    db.update_job(run["job_id"], status=status)
            elif time.time() - run["created"] > 15:
                child = progress.get("process")
                if alive(child):
                    db.update_run(run["id"], status="running", progress=progress, orphaned=True)
                    pause("训练监督进程中断，仍有训练进程存活；请检查后停止或等待，避免重复启动")
                else:
                    db.update_run(run["id"], status="interrupted", progress=progress, message="运行进程已消失且没有可靠完成记录")
                    db.update_job(run["job_id"], status="interrupted", message="进程中断")
                    pause("检测到中断任务，请选择重试或从 checkpoint 恢复")
        return
    q = db.state("queue")
    if not q["enabled"] or not q.get("workflow_id"):
        return
    wid = q["workflow_id"]
    workflow = db.workflow(wid)
    p = db.project(workflow["project_id"])
    if q.get("wait_for") and alive(q["wait_for"]):
        q["message"] = f"等待已有训练 PID {q['wait_for']['pid']} 退出"; db.set_state("queue", q); return
    others = training_processes(p)
    if others:
        q["message"] = f"等待本项目已有训练 PID {others[0]['pid']} 退出"; db.set_state("queue", q); return
    # A configuration export owns the same GPU reservation as a training supervisor.
    try:
        with FileLock(str(DATA / "gpu.lock"), timeout=0):
            pass
    except Timeout:
        q["message"] = "等待配置初始化释放 GPU"; db.set_state("queue", q); return
    jobs = db.jobs(wid)
    pending = [j for j in jobs if j["status"] not in ("succeeded", "skipped", "cancelled")]
    if not pending:
        pause("工作流已完成"); return
    j = pending[0]
    if j["status"] == "draft":
        pause("存在未入队的草稿，请检查后启动工作流"); return
    if j["status"] in ("failed", "interrupted"):
        pause("实验失败或中断，请重试或跳过"); return
    current = adapters.fingerprint(p)
    if current["hash"] != j["baseline"]["hash"]:
        changes = adapters.changed_files(j["baseline"], current)
        db.update_job(j["id"], message="源码已变化: " + ", ".join(changes[:8]))
        pause("检测到源码或项目配置变化，请刷新配置并更新待运行任务基线"); return
    spec = Experiment.model_validate(j["config"])
    checkpoint = None
    if spec.checkpoint.kind == "job":
        parent = next((x for x in jobs if x["id"] == spec.checkpoint.job_id), None)
        if not parent or parent["position"] >= j["position"] or parent["status"] != "succeeded":
            db.update_job(j["id"], status="blocked", message="上游实验尚未成功，请修改 checkpoint 来源")
            pause("checkpoint 依赖不可用"); return
        checkpoint = next((r["data"].get("final_checkpoint") for r in db.runs(parent["id"], include_deleted=True) if r["status"] == "succeeded" and not r["data"].get("files_deleted")), None)
    elif spec.checkpoint.kind == "file":
        checkpoint = str(Path(spec.checkpoint.path).expanduser().resolve())
    if spec.mode != "new" and (not checkpoint or not Path(checkpoint).is_file()):
        db.update_job(j["id"], status="blocked", message="checkpoint 不存在")
        pause("checkpoint 文件不可用"); return
    adapters.validate_config(p, spec)
    rid = uid()
    directory = OUTPUTS / "attempts" / rid
    output = adapters.training_output(p)
    command = adapters.build_command(p, spec, output, checkpoint)
    data = {"directory": str(directory), "output_root": str(output), "command": command, "project": p.model_dump(), "config": spec.model_dump(), "checkpoint": checkpoint, "source_fingerprint": current}
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "recipe.json", data)
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        if c.execute("SELECT 1 FROM runs WHERE status IN ('starting','running','stopping')").fetchone():
            return
        current_job = c.execute("SELECT status,deleted_at FROM jobs WHERE id=?", (j["id"],)).fetchone()
        if not current_job or current_job["deleted_at"] is not None or current_job["status"] != "queued":
            return
        number = c.execute("SELECT COUNT(*)+1 FROM runs WHERE job_id=?", (j["id"],)).fetchone()[0]
        c.execute("INSERT INTO runs(id,job_id,number,data,status,created,updated) VALUES(?,?,?,?,?,?,?)", (rid, j["id"], number, json.dumps(data, ensure_ascii=False), "starting", time.time(), time.time()))
        c.execute("UPDATE jobs SET status='starting',message='',updated=? WHERE id=?", (time.time(), j["id"]))
    supervisor = detached("workbench.supervisor", directory / "recipe.json", log_name="supervisors.log")
    db.update_run(rid, supervisor=identity(supervisor.pid))
    q.update(message="正在启动 " + spec.training.run_name, wait_for=None); db.set_state("queue", q)


def main():
    db.initialize()
    try:
        with FileLock(str(DATA / "worker.lock"), timeout=0):
            while True:
                atomic_json(DATA / "worker.json", {**identity(os.getpid()), "heartbeat": time.time()})
                try:
                    tick()
                except Exception as exc:
                    pause("调度检查失败: " + str(exc))
                time.sleep(1)
    except Timeout:
        pass


if __name__ == "__main__":
    main()
