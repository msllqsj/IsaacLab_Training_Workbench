import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .models import Project, uid
from .paths import DATA, ROOT, prepare


@contextmanager
def connect():
    prepare()
    con = sqlite3.connect(DATA / "workbench.sqlite", timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=20000")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize():
    with connect() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS workflows (id TEXT PRIMARY KEY, name TEXT NOT NULL, project_id TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, position INTEGER NOT NULL,
            config TEXT NOT NULL, baseline TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL DEFAULT '', updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, number INTEGER NOT NULL,
            data TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS jobs_workflow ON jobs(workflow_id,position);
        CREATE INDEX IF NOT EXISTS runs_job ON runs(job_id,number);
        """)
        for table in ("projects", "jobs", "runs"):
            if "deleted_at" not in {row[1] for row in c.execute(f"PRAGMA table_info({table})")}:
                c.execute(f"ALTER TABLE {table} ADD COLUMN deleted_at REAL")
        columns = {row[1] for row in c.execute("PRAGMA table_info(jobs)")}
        for column, kind in (("note", "TEXT"), ("note_updated", "REAL")):
            if column not in columns:
                c.execute(f"ALTER TABLE jobs ADD COLUMN {column} {kind}")
        if not c.execute("SELECT 1 FROM projects LIMIT 1").fetchone():
            default_root = os.environ.get("WORKBENCH_PROJECT", str(ROOT.parent / "AME_Locomotion"))
            train_python = os.environ.get("WORKBENCH_TRAIN_PYTHON", str(Path.home() / ".conda/envs/env_ame/python.exe") if os.name == "nt" else "")
            sim = Path(default_root).parent / "MyLocoInfra/IsaacLab/_isaac_sim"
            sim_path = os.environ.get("WORKBENCH_ISAAC_SIM", str(sim.resolve()) if sim.is_dir() else "")
            p = Project(id="go2-local", root=default_root, python=train_python, isaac_sim=sim_path)
            c.execute("INSERT INTO projects(id,data,updated) VALUES (?,?,?)", (p.id, p.model_dump_json(), time.time()))
        for key, value in {"queue": {"enabled": False, "workflow_id": None, "message": "队列尚未启动", "wait_for": None}}.items():
            c.execute("INSERT OR IGNORE INTO state VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))


def state(key, default=None):
    with connect() as c:
        row = c.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default


def set_state(key, value):
    with connect() as c:
        c.execute("INSERT INTO state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value, ensure_ascii=False)))


def projects():
    with connect() as c:
        return [json.loads(r[0]) for r in c.execute("SELECT data FROM projects WHERE deleted_at IS NULL ORDER BY updated")]


def project(pid):
    with connect() as c:
        r = c.execute("SELECT data FROM projects WHERE id=? AND deleted_at IS NULL", (pid,)).fetchone()
    if not r:
        raise ValueError("项目不存在")
    return Project.model_validate_json(r[0])


def workflows():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT w.* FROM workflows w JOIN projects p ON p.id=w.project_id WHERE p.deleted_at IS NULL ORDER BY w.created")]


def workflow(wid):
    with connect() as c:
        r = c.execute("SELECT w.* FROM workflows w JOIN projects p ON p.id=w.project_id WHERE w.id=? AND p.deleted_at IS NULL", (wid,)).fetchone()
    if not r:
        raise ValueError("工作流不存在")
    return dict(r)


def decode_job(r):
    out = dict(r)
    out["config"] = json.loads(out["config"])
    out["baseline"] = json.loads(out["baseline"])
    return out


def jobs(wid, include_deleted=False):
    with connect() as c:
        condition = "" if include_deleted else " AND deleted_at IS NULL"
        return [decode_job(r) for r in c.execute("SELECT * FROM jobs WHERE workflow_id=?" + condition + " ORDER BY position", (wid,))]


def job(jid, include_deleted=False):
    with connect() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?" + ("" if include_deleted else " AND deleted_at IS NULL"), (jid,)).fetchone()
    if not r:
        raise ValueError("实验不存在")
    return decode_job(r)


def add_job(wid, config, baseline, note=None):
    jid = uid()
    with connect() as c:
        c.execute("BEGIN IMMEDIATE")
        pos = c.execute("SELECT COALESCE(MAX(position),-1)+1 FROM jobs WHERE workflow_id=? AND deleted_at IS NULL", (wid,)).fetchone()[0]
        now = time.time()
        c.execute("INSERT INTO jobs(id,workflow_id,position,config,baseline,status,updated,note,note_updated) VALUES(?,?,?,?,?,'draft',?,?,?)",
                  (jid, wid, pos, json.dumps(config, ensure_ascii=False), json.dumps(baseline), now, note, now if note is not None else None))
    return jid


def update_job(jid, **values):
    allowed = {"status", "message", "config", "baseline", "position", "note", "note_updated"}
    if set(values) - allowed:
        raise ValueError("Invalid job update")
    values = {k: json.dumps(v, ensure_ascii=False) if k in ("config", "baseline") else v for k, v in values.items()}
    values["updated"] = time.time()
    with connect() as c:
        c.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?", (*values.values(), jid))


def runs(jid=None, include_deleted=False):
    with connect() as c:
        conditions = (["job_id=?"] if jid else []) + ([] if include_deleted else ["deleted_at IS NULL"])
        rows = c.execute("SELECT * FROM runs" + (" WHERE " + " AND ".join(conditions) if conditions else "") + " ORDER BY created DESC", (jid,) if jid else ()).fetchall()
    return [{**dict(r), "data": json.loads(r["data"])} for r in rows]


def run(rid, include_deleted=False):
    with connect() as c:
        r = c.execute("SELECT * FROM runs WHERE id=?" + ("" if include_deleted else " AND deleted_at IS NULL"), (rid,)).fetchone()
    if not r:
        raise ValueError("运行记录不存在")
    return {**dict(r), "data": json.loads(r["data"])}


def update_run(rid, status=None, **patch):
    with connect() as c:
        c.execute("BEGIN IMMEDIATE")
        r = c.execute("SELECT data,status FROM runs WHERE id=?", (rid,)).fetchone()
        if not r:
            return
        data = json.loads(r[0]); data.update(patch)
        c.execute("UPDATE runs SET data=?,status=?,updated=? WHERE id=?", (json.dumps(data, ensure_ascii=False), status or r[1], time.time(), rid))


def job_note(jid):
    j = job(jid)
    if j["note"] is not None:
        return {"text": j["note"], "updated": j["note_updated"]}
    # Older versions stored notes on attempts; retain that data and read the latest edit.
    legacy = [r for r in runs(jid, include_deleted=True) if "note" in r["data"]]
    if legacy:
        r = max(legacy, key=lambda r: r["data"].get("note_updated") or r["created"])
        return {"text": r["data"]["note"], "updated": r["data"].get("note_updated")}
    return {"text": "", "updated": None}


def save_job_note(jid, text, run_id=None):
    with connect() as c:
        c.execute("BEGIN IMMEDIATE")
        if run_id and not c.execute("SELECT 1 FROM runs WHERE id=? AND job_id=? AND deleted_at IS NULL", (run_id, jid)).fetchone():
            raise ValueError("运行记录不存在或已删除")
        now = time.time()
        changed = c.execute("UPDATE jobs SET note=?,note_updated=?,updated=? WHERE id=? AND deleted_at IS NULL", (text, now, now, jid))
        if not changed.rowcount:
            raise ValueError("实验不存在或已删除")
    return {"text": text, "updated": now}
