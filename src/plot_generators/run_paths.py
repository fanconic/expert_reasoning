"""Utilities for resolving plotting run directories under figures/* trees."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


DEFAULT_SEARCH_SUBDIRS: tuple[str, ...] = (
    "standard",
    "",
    "transferability",
    "ablations",
)


def _normalize_search_subdirs(
    search_subdirs: Optional[Iterable[str]],
) -> list[str]:
    if search_subdirs is None:
        return list(DEFAULT_SEARCH_SUBDIRS)

    normalized: list[str] = []
    seen = set()
    for sub in search_subdirs:
        key = (sub or "").strip()
        if key in {".", "./"}:
            key = ""
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _candidate_roots(
    dataset_dir: Path, search_subdirs: Optional[Iterable[str]] = None
) -> list[Path]:
    roots: list[Path] = []
    seen = set()

    for sub in _normalize_search_subdirs(search_subdirs):
        root = dataset_dir if sub == "" else dataset_dir / sub
        if not root.exists() or not root.is_dir():
            continue
        if root in seen:
            continue
        seen.add(root)
        roots.append(root)

    # Add other first-level subdirs as a fallback pool.
    for child in dataset_dir.iterdir():
        if not child.is_dir() or child in seen:
            continue
        seen.add(child)
        roots.append(child)

    return roots


def _is_valid_run_dir(path: Path, required_file: Optional[str] = None) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if required_file is None:
        return True
    return (path / required_file).exists()


def _rank_match(dataset_dir: Path, match: Path) -> tuple[int, int, str]:
    rel = match.relative_to(dataset_dir)
    parts = rel.parts
    top = parts[0] if parts else ""
    priority = {
        "standard": 0,
        "": 1,
        "transferability": 2,
        "ablations": 3,
    }.get(top, 4)
    return (priority, len(parts), str(rel))


def resolve_run_dir(
    dataset_dir: str | Path,
    run_name: str,
    required_file: Optional[str] = None,
    search_subdirs: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """
    Resolve a run directory for `run_name` under a dataset directory.

    Search order:
      1) preferred subfolders (default: standard, root, transferability, ablations)
      2) all first-level subfolders
      3) recursive fallback (best-ranked match)
    """
    ds = Path(dataset_dir)
    if not ds.exists() or not ds.is_dir():
        return None

    # Fast path: preferred roots / first-level roots.
    for root in _candidate_roots(ds, search_subdirs=search_subdirs):
        cand = root / run_name
        if _is_valid_run_dir(cand, required_file=required_file):
            return cand

    # Recursive fallback in case structure drifted.
    matches: list[Path] = []
    for cand in ds.rglob(run_name):
        if _is_valid_run_dir(cand, required_file=required_file):
            matches.append(cand)
    if not matches:
        return None

    matches.sort(key=lambda p: _rank_match(ds, p))
    return matches[0]


def collect_run_names(
    dataset_dir: str | Path,
    required_file: Optional[str] = None,
    search_subdirs: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Collect unique run directory names from a dataset directory.

    Primarily scans first-level run dirs under preferred subfolders.
    Falls back to recursive discovery only if nothing is found.
    """
    ds = Path(dataset_dir)
    if not ds.exists() or not ds.is_dir():
        return []

    names: list[str] = []
    seen = set()

    for root in _candidate_roots(ds, search_subdirs=search_subdirs):
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if not _is_valid_run_dir(child, required_file=required_file):
                continue
            if child.name in seen:
                continue
            seen.add(child.name)
            names.append(child.name)

    if names:
        return names

    # Recursive fallback.
    for child in ds.rglob("*"):
        if not child.is_dir():
            continue
        if not _is_valid_run_dir(child, required_file=required_file):
            continue
        if child.name in seen:
            continue
        seen.add(child.name)
        names.append(child.name)

    return names
