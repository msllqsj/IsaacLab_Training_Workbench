"""Read real artifacts and exercise HTTP/TensorBoard; never starts training."""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from local_acceptance import api


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for _ in range(30):
        health = api("/health")
        if health["worker"]["alive"]:
            break
        time.sleep(.5)
    report = {"health": health, "verified_at": time.time()}
    assert report["health"]["worker"]["alive"]
    state = api("/state")
    project = next(p for p in state["projects"] if p["id"] == "go2-local")
    project["name"] = "AME Locomotion · Go2"
    api("/projects/go2-local", "PUT", project)
    schema = api("/projects/go2-local/schema")["schema"]
    assert schema and not schema["stale"]
    acceptance = json.loads((root / "data/real-acceptance.json").read_text(encoding="utf-8"))
    checks = []
    for run in sorted(acceptance["runs"], key=lambda r: r["created"]):
        data = run["data"]
        env = yaml.load((Path(data["run_dir"]) / "params/env.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        agent = yaml.load((Path(data["run_dir"]) / "params/agent.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        check = {"name": data["config"]["training"]["run_name"], "mode": data["config"]["mode"],
                 "iteration": data["iteration"], "loaded_iteration": run["progress"].get("loaded_iteration"),
                 "num_envs": env["scene"]["num_envs"], "gait_weight": env["rewards"]["feet_gait"]["weight"],
                 "end_iteration": env["reward_gates"]["groups"]["gait_guidance"]["end_iteration"],
                 "lin_vel_z_discrete_scale": env["reward_gates"]["overrides"]["lin_vel_z_l2"]["terrain_scales"]["discrete"],
                 "save_interval": agent["save_interval"], "run_dir": data["run_dir"],
                 "source_unchanged": data["source_fingerprint"]["files"] == schema["fingerprint"]["files"]}
        assert check["num_envs"] == "4" and check["gait_weight"] == "0.15"
        assert check["end_iteration"] == "null" and check["lin_vel_z_discrete_scale"] == "0.5"
        assert check["save_interval"] == "1" and check["source_unchanged"]
        assert any(Path(data["run_dir"]).is_relative_to(folder) for folder in (root / "outputs", Path(data["project"]["root"]) / "logs"))
        assert Path(data["final_checkpoint"]).is_file()
        artifacts = api(f"/runs/{run['id']}/artifacts")
        assert any(f["path"].endswith(".pt") for f in artifacts["files"])
        checks.append(check)
    assert [c["iteration"] for c in checks] == [1, 2, 1]
    report["training"] = checks
    with urllib.request.urlopen("http://127.0.0.1:8765", timeout=10) as response:
        html = response.read().decode()
        assert "<div id=\"root\"></div>" in html
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
    assert len(assets) >= 2
    report["frontend_assets"] = []
    for asset in assets:
        with urllib.request.urlopen("http://127.0.0.1:8765" + asset, timeout=10) as response:
            report["frontend_assets"].append({"path": asset, "bytes": len(response.read()), "status": response.status})
    export = api(f"/workflows/{acceptance['workflow']['id']}/export")
    assert export["jobs"][1]["dependency_index"] == 0 and export["jobs"][2]["dependency_index"] == 1
    report["workflow_export"] = "passed"
    run_id = acceptance["runs"][0]["id"]
    tb = api(f"/runs/{run_id}/tensorboard", "POST", {})
    try:
        for _ in range(45):
            try:
                with urllib.request.urlopen(tb["url"] + "/data/plugin/scalars/tags", timeout=2) as response:
                    tags = json.load(response)
                    if not tags:
                        time.sleep(1)
                        continue
                    report["tensorboard"] = {"url": tb["url"], "http_status": response.status, "run_count": len(tags),
                                             "sample_tags": list(next(iter(tags.values())).keys())[:8]}
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        else:
            log = Path(tb["log"]).read_text(encoding="utf-8", errors="replace")
            raise AssertionError("TensorBoard did not start: " + log[-3000:])
    finally:
        api(f"/runs/{run_id}/tensorboard/stop", "POST", {})
    state = api("/state")
    assert not any(r["status"] in ("starting", "running", "stopping") for r in state["runs"])
    report["idle_after_acceptance"] = True
    (root / "data/final-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verified": True, "real_runs": len(checks), "iterations": [c["iteration"] for c in checks],
                      "source_unchanged": True, "frontend_assets": len(assets), "tensorboard": report["tensorboard"]}, ensure_ascii=False), flush=True)
