"""Stamp every evaluation artifact with what it was computed from.

The retrieval, fusion, DAT, gated-rerank and topic-compass artifacts were
written without recording which corpus, which eval set, which models or which
revision of the code produced them. The consequence was concrete: when the
v1/v2 numbers were found to disagree, the only way to work out what had been
run was to compare file modification times. Every reviewer since has asked the
same question -- "which corpus is this?" -- and there was nothing to point at.

Nothing here makes a network call. The stamp is the code revision, a content
hash of the corpus files and the eval sets, the index metadata, the models the
caller says it used, and the time. A checksum cannot reproduce the corpus, but
it can tell you whether the corpus you have is the one the number came from,
which is the question that was unanswerable before.

The committed artifacts predate this module and carry no stamp. That is
stated in the docs rather than back-filled: a provenance record written after
the fact would be a guess dressed up as a record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_revision() -> dict[str, Any]:
    """Commit hash plus whether the tree had uncommitted changes.

    A dirty flag matters as much as the hash: a number computed from edited,
    uncommitted code is attributed to a commit it did not come from.
    """
    rev = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_rev": rev,
        "git_dirty": bool(status) if status is not None else None,
    }


def corpus_manifest(index_dir: Path) -> dict[str, Any]:
    """Content hashes of the index files plus the index's own metadata."""
    index_dir = Path(index_dir)
    meta_path = index_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "index_dir": str(index_dir),
        "chunks_sha256": _sha256(index_dir / "chunks.jsonl"),
        "vectors_sha256": _sha256(index_dir / "vectors.npy"),
        "meta": meta,
    }


def eval_set_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        p = Path(p)
        n = (sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
             if p.exists() else None)
        out.append({"path": str(p), "sha256": _sha256(p), "n_rows": n})
    return out


def provenance(index_dir: Path, *, eval_sets: Iterable[Path] = (),
               models: Mapping[str, Any] | None = None,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The stamp written under `"provenance"` in every artifact."""
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **code_revision(),
        "corpus": corpus_manifest(index_dir),
        "eval_sets": eval_set_manifest(eval_sets),
        "models": dict(models or {}),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **dict(extra or {}),
    }
