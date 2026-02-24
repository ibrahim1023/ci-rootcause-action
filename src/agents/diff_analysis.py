from __future__ import annotations

from dataclasses import asdict
from pathlib import PurePosixPath

from src.parsers.git_diff_parser import DiffFile, parse_git_diff

LOCKFILE_NAMES = {
    "poetry.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "uv.lock",
}

MANIFEST_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "Pipfile",
    "setup.py",
    "setup.cfg",
    "package.json",
}


def _changed_path(diff_file: DiffFile) -> str:
    if diff_file.change_type == "deleted":
        return diff_file.old_path
    return diff_file.new_path


def _to_module_name(path: str) -> str:
    p = PurePosixPath(path)
    if p.name in LOCKFILE_NAMES or p.name in MANIFEST_NAMES:
        return "dependency"

    parts = [part for part in p.parts if part not in {"src", "tests"}]
    if not parts:
        return "root"

    if p.suffix:
        parts[-1] = p.stem

    return ".".join(parts)


def _summarize(files: list[DiffFile]) -> dict:
    added = 0
    removed = 0
    by_type: dict[str, int] = {"added": 0, "deleted": 0, "modified": 0, "renamed": 0}

    for file_diff in files:
        by_type[file_diff.change_type] = by_type.get(file_diff.change_type, 0) + 1
        for hunk in file_diff.hunks:
            added += hunk.added_lines
            removed += hunk.removed_lines

    return {
        "file_count": len(files),
        "change_type_counts": by_type,
        "added_lines": added,
        "removed_lines": removed,
    }


def _dependency_flags(changed_files: list[str]) -> tuple[dict, list[str]]:
    lockfiles = sorted(
        [path for path in changed_files if PurePosixPath(path).name in LOCKFILE_NAMES]
    )
    manifests = sorted(
        [path for path in changed_files if PurePosixPath(path).name in MANIFEST_NAMES]
    )

    indicators: list[str] = []
    if lockfiles and not manifests:
        indicators.append("lockfile_only_change")
    if manifests and not lockfiles:
        indicators.append("manifest_without_lockfile")
    if lockfiles and manifests:
        indicators.append("manifest_and_lockfile_changed")

    return (
        {
            "has_lockfile_change": bool(lockfiles),
            "changed_lockfiles": lockfiles,
            "has_manifest_change": bool(manifests),
            "changed_manifests": manifests,
        },
        indicators,
    )


def run_diff_analysis(raw_diff: str) -> dict:
    parsed = parse_git_diff(raw_diff)

    changed_files = [_changed_path(file_diff) for file_diff in parsed.files]
    module_mapping = {path: _to_module_name(path) for path in changed_files}
    changed_modules = sorted(set(module_mapping.values()))

    dependency_change_flags, dependency_drift_indicators = _dependency_flags(changed_files)

    return {
        "diff_summary": _summarize(parsed.files),
        "changed_files": sorted(changed_files),
        "changed_modules": changed_modules,
        "module_mapping": module_mapping,
        "dependency_change_flags": dependency_change_flags,
        "dependency_drift_indicators": dependency_drift_indicators,
        "files": [asdict(file_diff) for file_diff in parsed.files],
    }
