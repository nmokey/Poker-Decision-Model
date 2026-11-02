import io
import json
import tarfile

import pytest

from poker_decision.demo_assets import sha256, unpack_assets, verify_assets


def bundle(tmp_path, *, unsafe=False):
    archive = tmp_path / "assets.tar.gz"
    files = {"hands.jsonl": b"{}\n", "model/head.pt": b"weights", "model/metadata.json": b"{}"}
    import hashlib

    manifest = {"schema": 1, "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    files["manifest.json"] = json.dumps(manifest).encode()
    if unsafe:
        files["../escape"] = b"no"
    with tarfile.open(archive, "w:gz") as output:
        for name, value in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            output.addfile(entry, io.BytesIO(value))
    return archive


def test_verified_bundle_and_cached_tampering(tmp_path):
    archive = bundle(tmp_path)
    destination = tmp_path / "restored"
    unpack_assets(archive, destination, expected_sha256=sha256(archive))
    assert verify_assets(destination)["schema"] == 1
    (destination / "model/head.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_assets(destination)


def test_rejects_changed_archive_without_installing(tmp_path):
    archive = bundle(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        unpack_assets(archive, tmp_path / "restored", expected_sha256="0" * 64)
    assert not (tmp_path / "restored").exists()


def test_rejects_archive_traversal(tmp_path):
    archive = bundle(tmp_path, unsafe=True)
    with pytest.raises(ValueError, match="Unsafe"):
        unpack_assets(archive, tmp_path / "restored", expected_sha256=sha256(archive))
    assert not (tmp_path / "escape").exists()
