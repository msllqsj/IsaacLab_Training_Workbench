from concurrent.futures import ThreadPoolExecutor

from workbench import console


def test_already_removed_log_directory_can_be_read_and_cleared(tmp_path):
    directory = tmp_path / "removed"
    assert console.read(directory)["text"] == ""
    assert console.clear(directory)["reset"]
    assert console.read(directory, tail=True)["text"] == ""


def test_clear_is_persistent_and_only_new_output_returns(tmp_path):
    console.append(tmp_path, b"old logs\n")
    old = console.read(tmp_path)
    cleared = console.clear(tmp_path)
    assert cleared["generation"] != old["generation"]
    assert (tmp_path / "console.log").stat().st_size == 0
    console.append(tmp_path, b"new logs\n")
    page = console.read(tmp_path, old["offset"], generation=old["generation"])
    assert page["reset"] and page["text"] == "new logs\n"
    assert console.read(tmp_path, tail=True)["text"] == "new logs\n"


def test_old_append_handle_does_not_restore_deleted_logs(tmp_path):
    with (tmp_path / "console.log").open("ab", buffering=0) as writer:
        writer.write(b"old logs\n")
        console.clear(tmp_path)
        writer.write(b"new logs\n")
    assert console.read(tmp_path)["text"] == "new logs\n"


def test_console_keeps_latest_content_and_notifies_other_readers(tmp_path, monkeypatch):
    monkeypatch.setattr(console, "MAX_BYTES", 100)
    monkeypatch.setattr(console, "KEEP_BYTES", 50)
    console.append(tmp_path, b"old\n" * 20)
    old = console.read(tmp_path)
    console.append(tmp_path, b"new\n" * 20)
    assert (tmp_path / "console.log").stat().st_size <= 100
    page = console.read(tmp_path, old["offset"], generation=old["generation"])
    assert page["reset"] and page["text"].endswith("new\n")
    assert "old" not in page["text"]


def test_concurrent_append_read_and_clear_leave_a_valid_bounded_log(tmp_path, monkeypatch):
    monkeypatch.setattr(console, "MAX_BYTES", 1000)
    monkeypatch.setattr(console, "KEEP_BYTES", 500)
    def write():
        for i in range(80):
            console.append(tmp_path, f"line {i}\n".encode())
    def read():
        for _ in range(30):
            console.read(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(write), pool.submit(read), pool.submit(console.clear, tmp_path)]
        for future in futures:
            future.result(timeout=20)
    assert (tmp_path / "console.log").stat().st_size <= 1000
