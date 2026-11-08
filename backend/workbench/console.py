"""Bounded text logs, coordinated between the supervisor and web clients."""
import os
import uuid
from pathlib import Path

from filelock import FileLock

from .paths import atomic_json, read_json

MAX_BYTES = 10 * 1024 * 1024
KEEP_BYTES = MAX_BYTES // 2


def _generation(directory):
    return read_json(directory / "console.meta.json", {}).get("generation", "0")


def _changed(directory):
    value = uuid.uuid4().hex
    atomic_json(directory / "console.meta.json", {"generation": value})
    return value


def append(directory, data):
    directory = Path(directory)
    with FileLock(str(directory / "console.lock"), timeout=5):
        path = directory / "console.log"
        with path.open("ab") as stream:
            stream.write(data)
        if path.stat().st_size > MAX_BYTES:
            with path.open("r+b") as stream:
                stream.seek(-KEEP_BYTES, os.SEEK_END)
                tail = stream.read()
                newline = tail.find(b"\n")
                tail = tail[newline + 1:] if newline >= 0 else tail.decode("utf-8", errors="ignore").encode("utf-8")
                stream.seek(0); stream.write(tail); stream.truncate()
            _changed(directory)


def read(directory, offset=0, tail=False, generation=None):
    directory = Path(directory)
    if not directory.exists():
        return {"text": "", "offset": 0, "size": 0, "generation": "0", "reset": True}
    with FileLock(str(directory / "console.lock"), timeout=5):
        current = _generation(directory)
        path = directory / "console.log"
        if not path.exists():
            return {"text": "", "offset": 0, "size": 0, "generation": current, "reset": True}
        with path.open("rb") as stream:
            size = path.stat().st_size
            reset = tail or (generation is not None and generation != current) or offset > size
            stream.seek(max(0, size - 36000) if reset else offset)
            data = stream.read(64000)
            return {"text": data.decode("utf-8", errors="replace"), "offset": stream.tell(), "size": size,
                    "generation": current, "reset": reset}


def clear(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "console.lock"), timeout=5):
        path = directory / "console.log"
        # Truncate in place so existing Windows append handles cannot recreate old data.
        if path.exists():
            with path.open("r+b") as stream:
                stream.truncate(0)
        generation = _changed(directory)
        return {"text": "", "offset": 0, "size": 0, "generation": generation, "reset": True}
