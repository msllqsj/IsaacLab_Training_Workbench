"""Opt-in real GPU acceptance through the running workbench API (not a pytest test)."""
import copy
import json
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8765/api"


def api(path, method="GET", data=None):
    req = urllib.request.Request(URL + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


if __name__ == "__main__":
    schema = api("/projects/go2-local/schema")["schema"]
    assert schema and not schema["stale"] and schema["stage"] == 1
    workflow = api("/workflows", "POST", {"name": "本机验收 · Go2 真实短训练", "project_id": "go2-local"})
    wid = workflow["id"]
    spec = {"training": {**schema["training"], "run_name": "qa_new", "max_iterations": 2,
                          "num_envs": 4, "save_interval": 1, "video": False, "headless": True},
            "mode": "new", "checkpoint": {"kind": "none", "path": "", "job_id": ""},
            "overrides": {"env.rewards.feet_gait.weight": 0.15,
                          "env.reward_gates.groups.gait_guidance.end_iteration": None,
                          "env.reward_gates.overrides.lin_vel_z_l2.terrain_scales.discrete": 0.5}}
    first = api(f"/workflows/{wid}/jobs", "POST", spec)
    spec = copy.deepcopy(spec)
    spec.update(mode="resume", checkpoint={"kind": "job", "job_id": first["id"], "path": ""})
    spec["training"]["run_name"] = "qa_resume"
    second = api(f"/workflows/{wid}/jobs", "POST", spec)
    spec = copy.deepcopy(spec)
    spec.update(mode="warm_start", checkpoint={"kind": "job", "job_id": second["id"], "path": ""})
    spec["training"]["run_name"] = "qa_warm_start"
    third = api(f"/workflows/{wid}/jobs", "POST", spec)
    api(f"/workflows/{wid}/start", "POST", {})
    print("WORKFLOW " + wid, flush=True)
    last = None
    for _ in range(900):
        state = api("/state")
        jobs = state["jobs"][wid]
        summary = [(j["config"]["training"]["run_name"], j["status"]) for j in jobs]
        if summary != last:
            print(json.dumps(summary), flush=True); last = summary
        if all(j["status"] == "succeeded" for j in jobs) or any(j["status"] in ("failed", "interrupted", "blocked") for j in jobs):
            runs = [r for r in state["runs"] if r["job_id"] in {first["id"], second["id"], third["id"]}]
            report = {"workflow": workflow, "jobs": jobs, "runs": runs, "queue": state["queue"]}
            path = Path(__file__).resolve().parents[1] / "data/real-acceptance.json"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("REPORT " + str(path), flush=True)
            if not all(j["status"] == "succeeded" for j in jobs):
                for r in runs:
                    if r["status"] == "failed":
                        print(r["data"].get("message"), flush=True)
                        text = api(f"/runs/{r['id']}/logs?tail=true")["text"]
                        lines = text.splitlines()
                        for i, line in enumerate(lines):
                            if "Traceback" in line or "Error:" in line or "Error executing" in line:
                                print("\n".join(lines[max(0, i-2):i+25]), flush=True)
                raise SystemExit(1)
            print("REAL ACCEPTANCE PASSED", flush=True)
            break
        time.sleep(2)
    else:
        raise SystemExit("Acceptance timeout; training remains under workbench supervision")
