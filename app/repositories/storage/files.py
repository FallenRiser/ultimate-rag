from pathlib import Path
from typing import List

from app.utils.config import get_settings


def _base_dir() -> Path:
    return Path(get_settings().ingestion.storage_dir).resolve()


def _safe_path(user_id: str, version: str, filename: str) -> Path:
    """Resolve {storage}/{user_id}/{version}/{filename}, stripping any path traversal in
    the components. Raises ValueError if the result escapes the storage root."""
    base = _base_dir()
    # Path(...).name drops directory parts → blocks "../" and absolute-path tricks.
    target = (base / Path(user_id).name / Path(version).name / Path(filename).name).resolve()
    if base != target and base not in target.parents:
        raise ValueError("Resolved path escapes storage root")
    return target


def save_file(user_id: str, version: str, filename: str, content: bytes) -> Path:
    path = _safe_path(user_id, version, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def read_file(user_id: str, version: str, filename: str) -> bytes:
    path = _safe_path(user_id, version, filename)
    if not path.exists():
        raise FileNotFoundError(filename)
    return path.read_bytes()


def list_files(user_id: str, version: str) -> List[str]:
    folder = _base_dir() / Path(user_id).name / Path(version).name
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file())


if __name__ == "__main__":
    import tempfile

    from app.utils.config import get_settings as _gs

    _gs().ingestion.storage_dir = tempfile.mkdtemp()
    p = save_file("alice", "v1", "doc.pdf", b"hello")
    assert read_file("alice", "v1", "doc.pdf") == b"hello"
    assert "doc.pdf" in list_files("alice", "v1")
    # traversal is neutralized, not escaped
    assert save_file("alice", "v1", "../../evil.txt", b"x").name == "evil.txt"
    print("OK")
