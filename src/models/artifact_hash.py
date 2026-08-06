"""SHA256 helpers for joblib model artifacts."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return hex SHA256 of a file (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: str | Path, digest: str | None = None) -> str:
    """Write ``<stem>.sha256`` next to ``path``; return the digest."""
    path = Path(path)
    digest = digest or sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def read_expected_sha256(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    manifest_key: str = "model_blend_sha256",
) -> str | None:
    """Resolve expected hash from sidecar or optional metrics/manifest JSON."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        text = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if text:
            return text

    if manifest_path is None:
        manifest_path = path.parent / "model_compare.json"
    manifest = Path(manifest_path)
    if not manifest.exists():
        return None
    try:
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get(manifest_key)
    return str(value) if value else None


def verify_artifact_sha256(
    path: str | Path,
    expected: str | None = None,
    *,
    label: str | None = None,
    manifest_path: str | Path | None = None,
    manifest_key: str = "model_blend_sha256",
) -> str:
    """Verify file hash; fail on mismatch, warn if no expected hash.

    Returns the computed SHA256.
    """
    path = Path(path)
    label = label or path.name
    actual = sha256_file(path)
    if expected is None:
        expected = read_expected_sha256(
            path, manifest_path=manifest_path, manifest_key=manifest_key
        )
    if not expected:
        warnings.warn(
            f"No SHA256 recorded for {label}; skipping integrity check",
            stacklevel=2,
        )
        return actual
    if actual.lower() != expected.lower().strip():
        raise ValueError(
            f"SHA256 mismatch for {label}: expected {expected}, got {actual}"
        )
    return actual
