import os
import tempfile
from pathlib import Path


def pytest_configure(config):
    runtime = Path(tempfile.mkdtemp(prefix="isaac-workbench-test-"))
    os.environ["WORKBENCH_DATA_DIR"] = str(runtime / "data")
    os.environ["WORKBENCH_OUTPUT_DIR"] = str(runtime / "outputs")
    os.environ["WORKBENCH_NO_WORKER"] = "1"
    # Avoid inheriting ACLs from old shared pytest-of-USER directories on Windows.
    if not config.option.basetemp:
        config.option.basetemp = str(Path(tempfile.mkdtemp(prefix="workbench-pytest-")) / "cases")
    Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)
