from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK_HEADER = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


@dataclass(frozen=True)
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added_lines: int
    removed_lines: int


@dataclass(frozen=True)
class DiffFile:
    old_path: str
    new_path: str
    change_type: str
    hunks: list[DiffHunk] = field(default_factory=list)
    renamed_from: str | None = None
    renamed_to: str | None = None


@dataclass(frozen=True)
class ParsedDiff:
    files: list[DiffFile]


def _strip_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _finalize_file(record: dict) -> DiffFile:
    old_path = record.get("old_path", "")
    new_path = record.get("new_path", "")

    change_type = "modified"
    if old_path == "/dev/null":
        change_type = "added"
    elif new_path == "/dev/null":
        change_type = "deleted"
    elif record.get("renamed_from") and record.get("renamed_to"):
        change_type = "renamed"

    normalized_old = _strip_prefix(old_path)
    normalized_new = _strip_prefix(new_path)

    return DiffFile(
        old_path=normalized_old,
        new_path=normalized_new,
        change_type=change_type,
        hunks=record.get("hunks", []),
        renamed_from=record.get("renamed_from"),
        renamed_to=record.get("renamed_to"),
    )


def parse_git_diff(raw_diff: str) -> ParsedDiff:
    files: list[DiffFile] = []
    current: dict | None = None

    for line in raw_diff.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                files.append(_finalize_file(current))

            parts = line.split()
            current = {
                "old_path": parts[2],
                "new_path": parts[3],
                "renamed_from": None,
                "renamed_to": None,
                "hunks": [],
            }
            continue

        if current is None:
            continue

        if line.startswith("rename from "):
            current["renamed_from"] = line.replace("rename from ", "", 1).strip()
            continue

        if line.startswith("rename to "):
            current["renamed_to"] = line.replace("rename to ", "", 1).strip()
            continue

        if line.startswith("--- "):
            current["old_path"] = line.replace("--- ", "", 1).strip()
            continue

        if line.startswith("+++ "):
            current["new_path"] = line.replace("+++ ", "", 1).strip()
            continue

        hunk = HUNK_HEADER.match(line)
        if hunk:
            old_start = int(hunk.group(1))
            old_count = int(hunk.group(2) or "1")
            new_start = int(hunk.group(3))
            new_count = int(hunk.group(4) or "1")

            current["hunks"].append(
                DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    added_lines=0,
                    removed_lines=0,
                )
            )
            continue

        if line.startswith("+") and not line.startswith("+++") and current["hunks"]:
            latest = current["hunks"][-1]
            current["hunks"][-1] = DiffHunk(
                old_start=latest.old_start,
                old_count=latest.old_count,
                new_start=latest.new_start,
                new_count=latest.new_count,
                added_lines=latest.added_lines + 1,
                removed_lines=latest.removed_lines,
            )
            continue

        if line.startswith("-") and not line.startswith("---") and current["hunks"]:
            latest = current["hunks"][-1]
            current["hunks"][-1] = DiffHunk(
                old_start=latest.old_start,
                old_count=latest.old_count,
                new_start=latest.new_start,
                new_count=latest.new_count,
                added_lines=latest.added_lines,
                removed_lines=latest.removed_lines + 1,
            )

    if current is not None:
        files.append(_finalize_file(current))

    return ParsedDiff(files=files)
