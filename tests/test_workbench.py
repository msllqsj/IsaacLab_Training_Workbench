import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from workbench import adapters, db, probe, worker
from workbench.app import app
from workbench.models import Experiment, Project
from workbench.paths import atomic_json
from workbench.processes import alive, identity, terminate_tree


@pytest.fixture
def client(tmp_path):
    db.initialize()
    with db.connect() as c:
        for table in ("runs", "jobs", "workflows", "projects", "state"):
            c.execute(f"DELETE FROM {table}")
    db.initialize()
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(Path(__file__).parent / "fixtures/mock_train.py", root / "scripts/train.py")
    p = Project(id="mock-local", name="Test fixture", root=str(root), python=sys.executable, train_entry="scripts/train.py", task="Mock", imports=[], adapter="mock")
    with TestClient(app) as client:
        assert client.put(f"/api/projects/{p.id}", json=p.model_dump()).status_code == 200
        probe.run(p.id)
        assert client.get(f"/api/projects/{p.id}/schema").json()["schema"]
        yield client, p
    for r in db.runs():
        terminate_tree(r["data"].get("supervisor"))


def workflow(client, p):
    return client.post("/api/workflows", json={"name": "Test", "project_id": p.id}).json()["id"]


def spec(name="test", iterations=3, mode="new", parent=None):
    return {"training": {"run_name": name, "max_iterations": iterations, "num_envs": 4, "seed": 42, "device": "cpu", "save_interval": 1},
            "mode": mode, "checkpoint": {"kind": "job", "job_id": parent} if parent else {"kind": "none"}, "overrides": {}}


def add(client, wid, data):
    response = client.post(f"/api/workflows/{wid}/jobs", json=data)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def finish(wid, timeout=25):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        worker.tick()
        if not db.state("queue")["enabled"] and not any(r["status"] in worker.ACTIVE for r in db.runs()):
            return
        time.sleep(0.1)
    raise AssertionError(f"Queue timeout: {db.state('queue')}; {db.jobs(wid)}")


def test_remove_empty_project_preserves_other_queue_and_files(client):
    c, p = client
    wid = workflow(c, p)
    add(c, wid, spec())
    c.post(f"/api/workflows/{wid}/start", json={})
    queue = db.state("queue")
    assert c.delete("/api/projects/go2-local").status_code == 200
    assert db.state("queue") == queue
    db.initialize()
    assert [project["id"] for project in db.projects()] == [p.id]
    assert (Path(p.root) / p.train_entry).is_file()


def test_remove_project_hides_owned_history_and_preserves_training_files(client):
    c, p = client
    wid = workflow(c, p)
    jid = add(c, wid, spec())
    c.post(f"/api/workflows/{wid}/start", json={})
    finish(wid)
    run = db.runs(jid)[0]
    checkpoint = Path(run["data"]["final_checkpoint"])
    assert checkpoint.is_file()
    response = c.delete(f"/api/projects/{p.id}")
    assert response.status_code == 200, response.text
    assert response.json()["files_deleted"] is False
    assert checkpoint.is_file() and (Path(p.root) / p.train_entry).is_file()
    state = c.get("/api/state").json()
    assert not state["workflows"] and not state["jobs"] and not state["runs"]
    assert not state["queue"]["enabled"] and state["queue"]["workflow_id"] is None
    assert c.get(f"/api/projects/{p.id}/schema").status_code == 400
    assert c.put(f"/api/projects/{p.id}", json=p.model_dump()).status_code == 400
    assert c.delete(f"/api/projects/{p.id}").status_code == 400


def test_remove_project_rejects_active_training(client):
    c, p = client
    wid = workflow(c, p)
    add(c, wid, spec(iterations=400))
    c.post(f"/api/workflows/{wid}/start", json={})
    worker.tick()
    assert c.delete(f"/api/projects/{p.id}").status_code == 400
    assert db.project(p.id) and db.workflows() and db.runs()


def test_remove_last_project_stays_empty_and_allows_new_connection(client):
    c, p = client
    for project in db.projects():
        assert c.delete(f"/api/projects/{project['id']}").status_code == 200
    db.initialize()
    assert c.get("/api/state").json()["projects"] == []
    new = p.model_copy(update={"id": "new-connection"})
    assert c.put(f"/api/projects/{new.id}", json=new.model_dump()).status_code == 200
    assert db.project(new.id).root == p.root


def test_remove_project_prevents_stale_scheduler_launch(client, monkeypatch):
    c, p = client
    wid = workflow(c, p)
    add(c, wid, spec())
    c.post(f"/api/workflows/{wid}/start", json={})
    original = adapters.build_command
    def remove_before_claim(*args, **kwargs):
        command = original(*args, **kwargs)
        assert c.delete(f"/api/projects/{p.id}").status_code == 200
        return command
    monkeypatch.setattr(adapters, "build_command", remove_before_claim)
    worker.tick()
    assert not db.runs() and not db.state("queue")["enabled"]


def test_command_preserves_paths_and_float_zero(client, tmp_path):
    c, p = client
    p.adapter = "isaaclab_rsl_rl"
    e = Experiment.model_validate(spec())
    e.overrides["env.rewards.feet_gait.weight"] = 0
    cmd = adapters.build_command(p, e, tmp_path / "output with spaces")
    assert cmd[cmd.index("--experiment_name") + 1] == str(tmp_path / "output with spaces")
    assert "env.rewards.feet_gait.weight=0.0" in cmd
    assert "--load_run" not in cmd
    assert cmd[0] == sys.executable


def test_run_notes_persist_through_training_and_remain_isolated(client):
    c, p = client
    wid = workflow(c, p)
    first = add(c, wid, spec("with_notes"))
    second = add(c, wid, spec("other_run"))
    c.post(f"/api/workflows/{wid}/start", json={})
    worker.tick()
    rid = db.runs(first)[0]["id"]
    url = f"/api/runs/{rid}/note"
    assert c.get(url).json() == {"text": "", "updated": None}
    text = "实验目的：验证门控\n观察：抬腿更稳定。\n<script>alert('plain text')</script>"
    response = c.put(url, json={"text": text})
    assert response.status_code == 200
    assert response.json()["updated"] > 0
    finish(wid)
    db.initialize()
    assert c.get(url).json() == response.json()
    assert db.run(rid)["status"] == "succeeded"
    assert c.get(f"/api/runs/{db.runs(second)[0]['id']}/note").json()["text"] == ""
    assert c.put(url, json={"text": "修改后的备注"}).json()["text"] == "修改后的备注"
    assert c.put(url, json={"text": ""}).status_code == 200
    assert c.get(url).json()["text"] == ""


def test_run_notes_validate_input_and_reject_deleted_records(client):
    c, p = client
    wid = workflow(c, p)
    jid = add(c, wid, spec())
    c.post(f"/api/workflows/{wid}/start", json={})
    finish(wid)
    rid = db.runs(jid)[0]["id"]
    url = f"/api/runs/{rid}/note"
    assert c.put(url, json={"text": "字"*20000}).status_code == 200
    for invalid in ({"text": "字"*20001}, {"text": None}, {"text": 42}, {}):
        assert c.put(url, json=invalid).status_code == 422
    assert c.get(url).json()["text"] == "字"*20000
    assert c.delete(f"/api/runs/{rid}").status_code == 200
    for target in (url, "/api/runs/missing/note"):
        assert c.get(target).status_code == 400
        assert c.put(target, json={"text": "不应保存"}).status_code == 400


def test_editor_notes_are_shared_with_run_details_and_export(client):
    c, p = client
    wid = workflow(c, p)
    jid = add(c, wid, {**spec("shared_notes"), "note": "训练前的计划\n对比步态奖励"})
    url = f"/api/jobs/{jid}/note"
    assert c.get(url).json()["text"] == "训练前的计划\n对比步态奖励"
    assert "note" not in db.job(jid)["config"]
    assert c.put(f"/api/jobs/{jid}", json={**spec("shared_notes"), "note": "编辑配置后的备注"}).status_code == 200
    assert c.get(url).json()["text"] == "编辑配置后的备注"
    # Older callers saving only configuration must retain the shared note.
    assert c.put(f"/api/jobs/{jid}", json=spec("shared_notes")).status_code == 200
    assert c.get(url).json()["text"] == "编辑配置后的备注"
    c.post(f"/api/workflows/{wid}/start", json={})
    finish(wid)
    rid = db.runs(jid)[0]["id"]
    run_url = f"/api/runs/{rid}/note"
    assert c.get(run_url).json() == c.get(url).json()
    assert c.put(run_url, json={"text": "详情里追加的观察"}).status_code == 200
    assert c.get(url).json()["text"] == "详情里追加的观察"
    assert c.put(url, json={"text": "共享修改"}).status_code == 200
    assert c.get(run_url).json()["text"] == "共享修改"
    assert "note" not in db.run(rid)["data"]["config"]
    exported = c.get(f"/api/workflows/{wid}/export").json()
    imported = c.post(f"/api/projects/{p.id}/workflows/import", json=exported)
    assert imported.status_code == 200, imported.text
    copied = db.jobs(imported.json()["id"])[0]["id"]
    assert c.get(f"/api/jobs/{copied}/note").json()["text"] == "共享修改"
    assert c.put(f"/api/jobs/{copied}/note", json={"text": "独立实验"}).status_code == 200
    assert c.get(url).json()["text"] == "共享修改"


def test_legacy_attempt_notes_use_latest_edit_and_share_across_retries(client):
    c, p = client
    wid = workflow(c, p)
    jid = add(c, wid, spec("fail_legacy_notes"))
    c.post(f"/api/workflows/{wid}/start", json={})
    finish(wid)
    first = db.runs(jid)[0]["id"]
    db.update_run(first, note="旧备注", note_updated=10)
    assert c.post(f"/api/jobs/{jid}/retry", json={}).status_code == 200
    c.post(f"/api/workflows/{wid}/start", json={})
    finish(wid)
    second = db.runs(jid)[0]["id"]
    assert second != first
    db.update_run(second, note="最新旧版备注", note_updated=20)
    db.initialize()
    url = f"/api/jobs/{jid}/note"
    assert c.get(url).json() == {"text": "最新旧版备注", "updated": 20}
    assert c.get(f"/api/runs/{first}/note").json() == c.get(url).json()
    assert db.run(first)["data"]["note"] == "旧备注"
    assert db.run(second)["data"]["note"] == "最新旧版备注"
    assert c.put(url, json={"text": ""}).status_code == 200
    db.update_run(first, note="滞后的旧进程写入", note_updated=30)
    db.initialize()
    for rid in (first, second):
        assert c.get(f"/api/runs/{rid}/note").json()["text"] == ""


def test_editor_notes_validate_before_saving_and_reject_deleted_jobs(client):
    c, p = client
    wid = workflow(c, p)
    jid = add(c, wid, {**spec(), "note": "保留这条"})
    url = f"/api/jobs/{jid}/note"
    for text in (None, 42, "字"*20001):
        assert c.put(url, json={"text": text}).status_code == 422
        assert c.put(f"/api/jobs/{jid}", json={**spec(), "note": text}).status_code == 422
        assert c.post(f"/api/workflows/{wid}/jobs", json={**spec("invalid_note"), "note": text}).status_code == 422
    invalid = spec()
    invalid["training"]["num_envs"] = 0
    assert c.put(f"/api/jobs/{jid}", json={**invalid, "note": "不应保存"}).status_code == 422
    assert c.get(url).json()["text"] == "保留这条"
    assert len(db.jobs(wid)) == 1
    assert c.delete(f"/api/jobs/{jid}").status_code == 200
    assert c.get(url).status_code == 400
    assert c.put(url, json={"text": "不应保存"}).status_code == 400


def test_runtime_environment_supports_standalone_install(client, tmp_path):
    _, p = client
    sim = tmp_path / "Isaac Sim"
    (sim / "site").mkdir(parents=True)
    p.isaac_sim = str(sim)
    p.environment = {"QA_PROFILE_VALUE": "example"}
    env = adapters.training_environment(p)
    assert env["ISAAC_PATH"] == str(sim)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(sim / "site")
    assert env["QA_PROFILE_VALUE"] == "example"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_output_path_and_command_preview_use_connected_project_logs(client):
    c, p = client
    expected = Path(p.root).resolve() / "logs/rsl_rl/mock_task"
    assert adapters.training_output(p) == expected
    preview = c.post("/api/command/preview", json={"project_id": p.id, "config": spec()})
    assert preview.status_code == 200
    args = preview.json()["args"]
    assert args[args.index("--experiment_name") + 1] == str(expected)


def test_workflow_export_import_remaps_dependencies(client):
    c, p = client; wid = workflow(c, p)
    first = add(c, wid, spec("one"))
    add(c, wid, spec("two", mode="resume", parent=first))
    data = c.get(f"/api/workflows/{wid}/export").json()
    assert data["jobs"][1]["dependency_index"] == 0
    response = c.post(f"/api/projects/{p.id}/workflows/import", json=data)
    assert response.status_code == 200, response.text
    imported = db.jobs(response.json()["id"])
    assert imported[1]["config"]["checkpoint"]["job_id"] == imported[0]["id"]
    assert imported[0]["id"] != first
    data["jobs"][1]["dependency_index"] = 1
    assert c.post(f"/api/projects/{p.id}/workflows/import", json=data).status_code == 400


def test_explicit_process_wait_survives_queue_start(client):
    c, p = client; wid = workflow(c, p)
    add(c, wid, spec("wait"))
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        info = identity(process.pid)
        assert c.post("/api/queue/wait", json=info).status_code == 200
        c.post(f"/api/workflows/{wid}/start", json={})
        worker.tick()
        assert not db.runs()
        assert str(process.pid) in db.state("queue")["message"]
        c.post("/api/queue/wait", json={})
        finish(wid)
        assert db.jobs(wid)[0]["status"] == "succeeded"
    finally:
        process.terminate(); process.wait(timeout=10)


def test_service_stop_preserves_training_and_recovers_once(client):
    c, p = client; wid = workflow(c, p)
    first = add(c, wid, spec("persistent", iterations=160))
    add(c, wid, spec("next"))
    c.post(f"/api/workflows/{wid}/start", json={})
    background = subprocess.Popen([sys.executable, "-m", "workbench.worker"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            runs = db.runs(first)
            if runs:
                from workbench.paths import read_json
                progress = read_json(Path(runs[0]["data"]["directory"]) / "progress.json", {})
                if alive(progress.get("process")):
                    break
            time.sleep(.05)
        else:
            raise AssertionError("Supervised training did not start")
        subprocess.run([sys.executable, "-m", "workbench.cli", "stop"], check=True, capture_output=True, timeout=15)
        background.wait(timeout=10)
        assert alive(progress["supervisor"])
        assert alive(progress["process"])
        assert not db.state("queue")["enabled"]
        # Reconcile exactly as a newly started worker does, then resume the queue.
        worker.tick()
        assert c.post(f"/api/workflows/{wid}/start", json={}).status_code == 200
        finish(wid)
        assert len(db.runs(first)) == 1
        assert all(j["status"] == "succeeded" for j in db.jobs(wid))
    finally:
        if background.poll() is None:
            background.terminate(); background.wait(timeout=10)


def test_invalid_modes_and_nonfinite():
    with pytest.raises(ValueError):
        Experiment.model_validate({**spec(), "mode": "resume"})
    with pytest.raises(ValueError):
        Experiment.model_validate({**spec(), "overrides": {"env.rewards.feet_gait.weight": float("nan")}})


def test_grid_types_and_count(client):
    c, p = client; wid = workflow(c, p)
    data = {"base": spec("grid"), "axes": {"training.seed": [42, 43], "env.rewards.feet_gait.weight": [-0.1, 0, 0.2]}}
    result = c.post(f"/api/workflows/{wid}/grid/preview", json=data)
    assert result.status_code == 200, result.text
    rows = result.json()
    assert len(rows) == 6
    assert len({r["config"]["training"]["run_name"] for r in rows}) == 6
    assert rows[1]["config"]["overrides"]["env.rewards.feet_gait.weight"] == 0.0
    assert c.post(f"/api/workflows/{wid}/grid/commit", json=data).status_code == 200
    assert len(db.jobs(wid)) == 6


def test_serial_new_resume_and_warm_start(client):
    c, p = client; wid = workflow(c, p)
    a = add(c, wid, spec("a", 3))
    b = add(c, wid, spec("b", 2, "resume", a))
    d = add(c, wid, spec("c", 2, "warm_start", b))
    before = adapters.fingerprint(p)
    assert c.post(f"/api/workflows/{wid}/start", json={}).status_code == 200
    finish(wid)
    assert [j["status"] for j in db.jobs(wid)] == ["succeeded"] * 3
    runs = sorted(db.runs(), key=lambda r: r["created"])
    assert [r["data"]["iteration"] for r in runs] == [2, 3, 1]
    assert runs[1]["data"]["checkpoint"] == runs[0]["data"]["final_checkpoint"]
    assert runs[2]["data"]["checkpoint"] == runs[1]["data"]["final_checkpoint"]
    for first, second in zip(runs, runs[1:]):
        assert second["created"] >= first["data"]["finished"]
    for run in runs:
        assert Path(run["data"]["output_root"]) == adapters.training_output(p)
        assert Path(run["data"]["final_checkpoint"]).is_relative_to(Path(p.root) / "logs")
    assert adapters.fingerprint(p) == before
    assert c.put(f"/api/jobs/{a}", json=spec("overwrite")).status_code == 400
    assert c.get(f"/api/runs/{runs[0]['id']}/logs").json()["text"]
    assert "files" in c.get(f"/api/runs/{runs[0]['id']}/artifacts").json()
    assert len({run["data"]["run_dir"] for run in runs}) == 3


def test_legacy_output_paths_remain_browsable_and_can_feed_new_project_log_runs(client):
    from workbench.paths import OUTPUTS
    c, p = client; wid = workflow(c, p)
    parent = add(c, wid, spec("legacy"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(parent)[0]
    legacy_root = OUTPUTS / "training" / run["id"]
    legacy_dir = legacy_root / "old_timestamp_legacy"
    (legacy_dir / "params").mkdir(parents=True)
    (legacy_dir / "params/env.yaml").write_text("legacy: true\n", encoding="utf-8")
    checkpoint = legacy_dir / "model_2.pt"
    checkpoint.write_text(json.dumps({"iteration": 2, "has_optimizer": True}), encoding="utf-8")
    db.update_run(run["id"], output_root=str(legacy_root), run_dir=str(legacy_dir), final_checkpoint=str(checkpoint))
    artifacts = c.get(f"/api/runs/{run['id']}/artifacts").json()
    assert artifacts["run_dir"] == str(legacy_dir)
    assert any(file["path"] == "model_2.pt" for file in artifacts["files"])
    assert c.get(f"/api/runs/{run['id']}/artifact", params={"path": "model_2.pt"}).content == checkpoint.read_bytes()
    child = add(c, wid, spec("new_location", 2, "resume", parent))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    resumed = db.runs(child)[0]
    assert resumed["status"] == "succeeded"
    assert resumed["data"]["checkpoint"] == str(checkpoint)
    assert Path(resumed["data"]["output_root"]).is_relative_to(Path(p.root) / "logs")
    assert db.run(run["id"])["data"]["output_root"] == str(legacy_root)


def test_shared_task_folder_ignores_other_runs_higher_checkpoint_numbers(client):
    c, p = client; wid = workflow(c, p)
    old = adapters.training_output(p) / "old_timestamp_other_experiment"
    old.mkdir(parents=True)
    (old / "model_99999.pt").write_text(json.dumps({"iteration": 99999, "has_optimizer": True}), encoding="utf-8")
    jid = add(c, wid, spec("current", 3))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(jid)[0]
    assert run["status"] == "succeeded"
    assert Path(run["data"]["final_checkpoint"]).name == "model_2.pt"
    assert Path(run["data"]["run_dir"]) != old


def test_unknown_run_timestamp_cannot_show_other_experiments_artifacts(client, monkeypatch):
    c, p = client; wid = workflow(c, p)
    parent = add(c, wid, spec("previous"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    old = db.runs(parent)[0]
    child = add(c, wid, spec("not_initialized"))
    monkeypatch.setattr(worker, "detached", lambda *args, **kwargs: type("PendingSupervisor", (), {"pid": 999999999})())
    c.post(f"/api/workflows/{wid}/start", json={}); worker.tick()
    rid = db.runs(child)[0]["id"]
    response = c.get(f"/api/runs/{rid}/artifacts")
    assert response.json() == {"run_dir": "", "files": []}
    assert c.post(f"/api/runs/{rid}/tensorboard").status_code == 400
    assert c.get(f"/api/runs/{old['id']}/artifacts").json()["files"]


def test_native_output_names_and_reported_timestamps_cannot_escape_log_root(client):
    c, p = client
    cache = adapters.schema(p)
    cache["experiment_name"] = "../../outside"
    atomic_json(adapters.schema_path(p.id), cache)
    with pytest.raises(ValueError):
        adapters.training_output(p)
    with pytest.raises(ValueError):
        adapters.logged_run_directory(Path(p.root) / "logs", "test", {"run_timestamp": "../../outside"})
    assert adapters.logged_run_directory(Path(p.root) / "logs", "test", {}) is None


def test_deleted_record_stays_hidden_after_stale_worker_update_and_new_client(client):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("persist_delete"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(jid)[0]
    assert c.delete(f"/api/runs/{run['id']}").status_code == 200
    db.update_run(run["id"], status="succeeded", progress=run["data"]["progress"])
    with TestClient(app) as other:
        response = other.get("/api/state")
        assert response.headers["Cache-Control"] == "no-store"
        assert all(item["id"] != run["id"] for item in response.json()["runs"])


def test_clear_log_endpoint_does_not_restore_text_after_polling_or_refresh(client):
    from workbench import console
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("clear_log"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(jid)[0]; rid = run["id"]
    old = c.get(f"/api/runs/{rid}/logs").json()
    assert old["text"]
    assert c.delete(f"/api/runs/{rid}/logs").status_code == 200
    assert c.get(f"/api/runs/{rid}/logs", params={"tail": True}).json()["text"] == ""
    console.append(run["data"]["directory"], b"fresh output\n")
    page = c.get(f"/api/runs/{rid}/logs", params={"offset": old["offset"], "generation": old["generation"]}).json()
    assert page["reset"] and page["text"] == "fresh output\n"
    assert Path(run["data"]["final_checkpoint"]).exists()


def test_delete_with_files_removes_only_selected_training_directory(client):
    c, p = client; wid = workflow(c, p)
    first = add(c, wid, spec("purge_this")); second = add(c, wid, spec("keep_this"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    removed = db.runs(first)[0]; kept = db.runs(second)[0]
    directory = Path(removed["data"]["run_dir"])
    response = c.delete(f"/api/runs/{removed['id']}", params={"delete_files": True})
    assert response.status_code == 200, response.text
    assert response.json()["files_deleted"]
    assert not directory.exists()
    assert Path(kept["data"]["final_checkpoint"]).exists()
    assert (Path(removed["data"]["directory"]) / "console.log").stat().st_size == 0
    assert (Path(removed["data"]["directory"]) / "recipe.json").exists()


def test_file_purge_protects_checkpoint_dependencies(client):
    c, p = client; wid = workflow(c, p)
    parent = add(c, wid, spec("source"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(parent)[0]
    add(c, wid, spec("dependent", 2, "resume", parent))
    response = c.delete(f"/api/runs/{run['id']}", params={"delete_files": True})
    assert response.status_code == 400
    assert Path(run["data"]["final_checkpoint"]).exists()
    assert db.run(run["id"])
    assert c.delete(f"/api/runs/{run['id']}").status_code == 200


def test_file_purge_rejects_log_root_and_external_paths_before_removing_anything(client, tmp_path):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("guard"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(jid)[0]
    source = Path(run["data"]["final_checkpoint"])
    db.update_run(run["id"], run_dir=str(adapters.training_output(p)))
    assert c.delete(f"/api/runs/{run['id']}", params={"delete_files": True}).status_code == 400
    assert source.exists()
    outside = tmp_path / "unrelated"; outside.mkdir()
    marker = outside / "keep.txt"; marker.write_text("keep", encoding="utf-8")
    db.update_run(run["id"], run_dir=str(outside))
    assert c.delete(f"/api/runs/{run['id']}", params={"delete_files": True}).status_code == 400
    assert marker.exists() and source.exists()


def test_delete_experiment_with_files_cleans_all_attempts_without_affecting_other_jobs(client):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("purge_job"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(jid)[0]
    response = c.delete(f"/api/jobs/{jid}", params={"delete_files": True})
    assert response.status_code == 200, response.text
    assert not Path(run["data"]["run_dir"]).exists()
    assert not db.jobs(wid)


def test_tensorboard_parallel_clicks_and_reuse_wait_for_the_same_process(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import workbench.app as application
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("tb_start"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    rid = db.runs(jid)[0]["id"]
    spawned = []; waited = []
    def spawn(*args, **kwargs):
        spawned.append(args)
        time.sleep(.05)
        return SimpleNamespace(pid=999999999)
    monkeypatch.setattr(application.subprocess, "Popen", spawn)
    monkeypatch.setattr(application, "identity", lambda pid: {"pid": pid, "created": 1})
    monkeypatch.setattr(application, "alive", lambda info: bool(info and info["pid"] == 999999999))
    monkeypatch.setattr(application, "wait_tensorboard", lambda info: waited.append(info["pid"]) or info)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: c.post(f"/api/runs/{rid}/tensorboard"), range(2)))
    responses.append(c.post(f"/api/runs/{rid}/tensorboard"))
    assert all(response.status_code == 200 for response in responses)
    assert len(spawned) == 1
    assert len({response.json()["url"] for response in responses}) == 1
    assert waited == [999999999] * 3


def test_failure_pauses_and_blocks_dependencies(client):
    c, p = client; wid = workflow(c, p)
    a = add(c, wid, spec("fail_expected"))
    b = add(c, wid, spec("dependent", 2, "resume", a))
    c.post(f"/api/workflows/{wid}/start", json={})
    finish(wid)
    assert db.job(a)["status"] == "failed"
    assert not db.runs(b)
    assert c.post(f"/api/jobs/{a}/skip").status_code == 200
    assert c.post(f"/api/workflows/{wid}/start", json={}).status_code == 200
    finish(wid)
    assert db.job(b)["status"] == "blocked"
    assert not db.runs(b)


def test_source_change_prevents_launch(client):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec())
    c.post(f"/api/workflows/{wid}/start", json={})
    with (Path(p.root) / p.train_entry).open("a") as f:
        f.write("\n# changed after enqueue\n")
    worker.tick()
    assert not db.runs(jid)
    assert not db.state("queue")["enabled"]
    assert "源码" in db.state("queue")["message"]


def test_reorder_rejects_forward_dependency(client):
    c, p = client; wid = workflow(c, p)
    a = add(c, wid, spec("a")); b = add(c, wid, spec("b", 2, "resume", a))
    assert c.post(f"/api/workflows/{wid}/reorder", json=[b, a]).status_code == 400


def test_pause_and_stop_preserve_completed_checkpoints(client):
    c, p = client; wid = workflow(c, p)
    a = add(c, wid, spec("long", 400)); b = add(c, wid, spec("later"))
    c.post(f"/api/workflows/{wid}/start", json={}); worker.tick()
    start = time.monotonic()
    while time.monotonic() - start < 10:
        worker.tick()
        r = db.runs(a)[0]
        if r["data"].get("progress", {}).get("completed_updates", 0) > 3:
            break
        time.sleep(.1)
    assert c.post(f"/api/runs/{r['id']}/stop").status_code == 200
    finish(wid)
    assert db.job(a)["status"] == "cancelled"
    assert not db.runs(b)
    assert list(Path(r["data"]["output_root"]).rglob("model_*.pt"))


def test_pid_identity_and_log_parser():
    me = identity(os.getpid()); assert alive(me)
    assert not alive({**me, "created": me["created"] + 1})
    progress = {}
    adapters.parse_line("\x1b[1m Learning iteration 3000/5000\x1b[0m", progress)
    assert progress["completed_updates"] == 1
    assert progress["progress"] == pytest.approx(1/2000)
    adapters.parse_line("unrelated output", progress)
    assert progress["phase"] == "training"


def test_schema_and_local_origin(client):
    c, p = client
    assert c.get("/api/state").status_code == 200
    assert c.post("/api/queue/pause", headers={"Origin": "https://unrelated.example"}).status_code == 403
    assert c.get(f"/api/files?project_id={p.id}&path={p.root}").status_code == 200
    assert c.get(f"/api/files?project_id={p.id}&path={Path(p.root).parent}").status_code == 400


def test_delete_run_keeps_files_queue_progress_and_resume_dependency(client):
    c, p = client; wid = workflow(c, p)
    parent = add(c, wid, spec("original", 3))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(parent)[0]
    checkpoint = Path(run["data"]["final_checkpoint"])
    log = Path(run["data"]["directory"]) / "console.log"
    original = (checkpoint.read_bytes(), log.read_bytes())
    response = c.delete(f"/api/runs/{run['id']}")
    assert response.status_code == 200, response.text
    assert response.json()["files_deleted"] is False
    assert not db.runs(parent)
    assert len(db.runs(parent, include_deleted=True)) == 1
    assert (checkpoint.read_bytes(), log.read_bytes()) == original
    state = c.get("/api/state").json()
    assert state["runs"] == []
    summary = state["jobs"][wid][0]
    assert summary["attempt_count"] == 1
    assert summary["last_result"]["progress"]["completed_updates"] == 3
    assert summary["last_result"]["final_checkpoint"] == str(checkpoint)
    child = add(c, wid, spec("continue", 2, "resume", parent))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    assert db.job(child)["status"] == "succeeded"
    assert db.runs(child)[0]["data"]["checkpoint"] == str(checkpoint)


def test_cannot_delete_active_run_or_experiment(client):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("active", 400))
    c.post(f"/api/workflows/{wid}/start", json={}); worker.tick()
    run = db.runs(jid)[0]
    assert c.delete(f"/api/runs/{run['id']}").status_code == 400
    assert c.delete(f"/api/jobs/{jid}").status_code == 400
    assert db.job(jid)
    assert db.run(run["id"])


def test_delete_experiment_protects_dependencies_and_renumbers_queue(client):
    c, p = client; wid = workflow(c, p)
    first = add(c, wid, spec("root"))
    second = add(c, wid, spec("child", 2, "resume", first))
    third = add(c, wid, spec("independent"))
    response = c.delete(f"/api/jobs/{first}")
    assert response.status_code == 400
    assert "child" in response.json()["detail"]
    assert c.delete(f"/api/jobs/{second}").status_code == 200
    assert [(j["id"], j["position"]) for j in db.jobs(wid)] == [(first, 0), (third, 1)]
    assert c.delete(f"/api/jobs/{first}").status_code == 200
    assert c.delete(f"/api/jobs/{third}").status_code == 200
    new = add(c, wid, spec("fresh"))
    assert db.job(new)["position"] == 0


def test_delete_finished_experiment_removes_history_but_keeps_artifacts(client):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("finished"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(jid)[0]
    checkpoint = Path(run["data"]["final_checkpoint"])
    response = c.delete(f"/api/jobs/{jid}")
    assert response.status_code == 200
    assert response.json()["records_removed"] == 1
    assert not c.get("/api/state").json()["jobs"][wid]
    assert not db.runs(jid)
    assert db.job(jid, include_deleted=True)["deleted_at"] is not None
    assert checkpoint.is_file()
    assert (Path(run["data"]["directory"]) / "recipe.json").is_file()


def test_removed_history_cannot_reset_attempt_numbers_or_edit_executed_config(client):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("fail_attempt"))
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    run = db.runs(jid)[0]
    assert c.delete(f"/api/runs/{run['id']}").status_code == 200
    assert c.put(f"/api/jobs/{jid}", json=spec("overwrite")).status_code == 400
    assert c.post(f"/api/jobs/{jid}/retry").status_code == 200
    c.post(f"/api/workflows/{wid}/start", json={}); finish(wid)
    assert db.runs(jid)[0]["number"] == 2
    assert [r["number"] for r in db.runs(jid, include_deleted=True)] == [2, 1]


def test_deleted_job_cannot_be_claimed_from_stale_scheduler_snapshot(client, monkeypatch):
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("remove_before_claim"))
    c.post(f"/api/workflows/{wid}/start", json={})
    build = adapters.build_command
    def remove_before_claim(*args, **kwargs):
        assert c.delete(f"/api/jobs/{jid}").status_code == 200
        return build(*args, **kwargs)
    monkeypatch.setattr(adapters, "build_command", remove_before_claim)
    worker.tick()
    assert not db.runs(include_deleted=True)
    assert not db.jobs(wid)


def test_locked_progress_snapshot_does_not_abort_training_or_hide_final_progress(client, monkeypatch):
    from workbench import supervisor
    from workbench.app import enriched_run, run_directory
    c, p = client; wid = workflow(c, p)
    jid = add(c, wid, spec("locked_progress", 3))
    c.post(f"/api/workflows/{wid}/start", json={})
    # Run the real supervisor in this process so the progress-only write fault is injected.
    def supervise_inline(module, recipe, **kwargs):
        supervisor.run(recipe)
        return type("CompletedSupervisor", (), {"pid": 999999999})()
    atomic = supervisor.atomic_json
    def locked_snapshot(path, value):
        if Path(path).name == "progress.json":
            error = PermissionError("Windows progress handle remains locked")
            error.winerror = 5
            raise error
        return atomic(path, value)
    monkeypatch.setattr(supervisor, "atomic_json", locked_snapshot)
    monkeypatch.setattr(worker, "detached", supervise_inline)
    worker.tick()
    # The inline supervisor has exited; the worker must recover its final snapshot.
    worker.tick()
    run = db.runs(jid)[0]
    assert run["status"] == "succeeded"
    assert Path(run["data"]["final_checkpoint"]).is_file()
    assert run["data"]["progress"]["completed_updates"] == 3
    assert not (Path(run["data"]["directory"]) / "progress.json").exists()
    atomic(Path(run["data"]["directory"]) / "progress.json", {"phase": "initializing", "completed_updates": 0})
    assert enriched_run(run)["progress"]["completed_updates"] == 3
    without_cached_directory = {**run, "data": {**run["data"], "run_dir": None}}
    assert run_directory(without_cached_directory) == Path(run["data"]["final_checkpoint"]).parent
