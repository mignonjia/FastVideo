from __future__ import annotations

import os
import sys
from pathlib import Path


REQUIRED_ROOT_FILES = (
    "ptlflow/__init__.py",
    "eval_flow_divergence.py",
    "synthetic_flow.py",
    "calibration.json",
)


def _is_ptlflow_root(path: Path) -> bool:
    return all((path / rel_path).exists() for rel_path in REQUIRED_ROOT_FILES)


def _iter_candidate_roots(start_dir: Path):
    seen: set[Path] = set()

    def emit(path: Path):
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved

    env_root = os.environ.get("PTLFLOW_ROOT")
    if env_root:
        yield from emit(Path(env_root))

    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        yield from emit(parent)

    for parent in (start_dir, *start_dir.parents):
        yield from emit(parent)
        if parent.parent != parent:
            yield from emit(parent.parent / "ptlflow")


def resolve_ptlflow_root(start_dir: Path | None = None) -> Path:
    base_dir = Path(__file__).resolve().parent if start_dir is None else start_dir
    for candidate in _iter_candidate_roots(base_dir):
        if _is_ptlflow_root(candidate):
            return candidate

    raise RuntimeError(
        "Unable to locate the PTLFlow repo root. "
        "Set PTLFLOW_ROOT to the directory containing ptlflow/, "
        "eval_flow_divergence.py, synthetic_flow.py, and calibration.json."
    )


PTLFLOW_ROOT = resolve_ptlflow_root(Path(__file__).resolve().parent)


def ensure_ptlflow_root_on_path() -> Path:
    if str(PTLFLOW_ROOT) not in sys.path:
        sys.path.insert(0, str(PTLFLOW_ROOT))
    return PTLFLOW_ROOT
