"""Download and verify a pinned, separately published demo model bundle."""

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

MAX_BYTES = 100 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported demo asset manifest")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe demo asset")
        if sha256(path) != expected:
            raise ValueError(f"Demo asset checksum mismatch: {name}")
    required = {"hands.jsonl", "model/head.pt", "model/metadata.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Demo asset manifest is incomplete")
    return manifest  # type: ignore[no-any-return]


def unpack_assets(archive: Path, destination: Path, *, expected_sha256: str) -> None:
    if sha256(archive) != expected_sha256:
        raise ValueError("Downloaded demo bundle checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "assets"
        stage.mkdir()
        with tarfile.open(archive) as bundle:
            entries = bundle.getmembers()
            if sum(entry.size for entry in entries) > MAX_BYTES:
                raise ValueError("Demo bundle exceeds the size limit")
            for entry in entries:
                path = (stage / entry.name).resolve()
                if not path.is_relative_to(stage) or not entry.isfile():
                    raise ValueError("Unsafe demo bundle entry")
            bundle.extractall(stage, filter="data")
        verify_assets(stage)
        if destination.exists():
            raise ValueError("Refusing to replace an existing demo asset directory")
        shutil.move(str(stage), destination)


def download_assets(url: str, destination: Path, *, expected_sha256: str) -> None:
    if destination.exists():
        verify_assets(destination)
        return
    if not url.startswith("https://"):
        raise ValueError("Demo asset downloads require HTTPS")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "assets.tar.gz"
        size = 0
        with urlopen(url, timeout=60) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("Downloaded demo bundle exceeds the size limit")
                output.write(chunk)
        unpack_assets(archive, destination, expected_sha256=expected_sha256)
