import errno
import json
import os

import pytest

from workbench import paths


def windows_lock():
    error = PermissionError(errno.EACCES, "File is temporarily in use")
    error.winerror = 5
    return error


def test_atomic_write_retries_windows_lock_without_exposing_partial_json(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    paths.atomic_json(target, {"iteration": 1})
    replace = paths.os.replace
    attempts = []
    delays = []
    def locked_then_released(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text()) == {"iteration": 1}
        if len(attempts) <= 3:
            raise windows_lock()
        replace(source, destination)
    monkeypatch.setattr(paths.os, "replace", locked_then_released)
    monkeypatch.setattr(paths.time, "sleep", delays.append)
    paths.atomic_json(target, {"iteration": 2})
    assert json.loads(target.read_text()) == {"iteration": 2}
    assert delays == [.01, .02, .04]
    assert len(set(attempts)) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_windows_lock_is_bounded_and_keeps_previous_snapshot(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    paths.atomic_json(target, {"iteration": 1})
    attempts = []
    def locked(*args):
        attempts.append(args)
        raise windows_lock()
    monkeypatch.setattr(paths.os, "replace", locked)
    monkeypatch.setattr(paths.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        paths.atomic_json(target, {"iteration": 2})
    assert len(attempts) == 10
    assert json.loads(target.read_text()) == {"iteration": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_other_permission_errors_are_reported_without_retry(tmp_path, monkeypatch):
    attempts = []
    def denied(*args):
        attempts.append(args)
        raise PermissionError(errno.EACCES, "Permission denied")
    monkeypatch.setattr(paths.os, "replace", denied)
    with pytest.raises(PermissionError):
        paths.atomic_json(tmp_path / "progress.json", {})
    assert len(attempts) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_real_windows_reader_lock_is_retried(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes
    from threading import Timer

    target = tmp_path / "progress.json"
    paths.atomic_json(target, {"iteration": 1})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(target), 0x80000000, 0x3, None, 3, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    replace = paths.os.replace
    conflicts = []
    def observed_replace(source, destination):
        try:
            replace(source, destination)
        except PermissionError as error:
            conflicts.append(error.winerror)
            raise
    monkeypatch.setattr(paths.os, "replace", observed_replace)
    release = Timer(.1, lambda: kernel.CloseHandle(handle))
    release.start()
    try:
        paths.atomic_json(target, {"iteration": 2})
    finally:
        release.join(timeout=2)
    assert conflicts
    assert json.loads(target.read_text()) == {"iteration": 2}
