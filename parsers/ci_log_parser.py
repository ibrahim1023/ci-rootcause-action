from __future__ import annotations

import re
from dataclasses import dataclass

ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(error|exception|traceback|failed|failure|assertionerror)\b",
        re.IGNORECASE,
    ),
)

STACK_FRAME_PATTERN = re.compile(
    r"\s*File \"(?P<file>[^\"]+)\", line (?P<line>\d+)|"
    r"\s*at (?P<js_file>[^:\s]+):(\d+):(\d+)"
)


@dataclass(frozen=True)
class ParsedFailureEvent:
    event_index: int
    stage: str
    timestamp: str
    error_signature: str
    file: str | None
    line: int | None
    stack_frames: list[str]
    log_excerpt: str


@dataclass(frozen=True)
class ParsedLog:
    stages: list[str]
    failure_events: list[ParsedFailureEvent]


def _extract_stage(raw_line: str, current_stage: str) -> str:
    line = raw_line.strip()
    if line.startswith("##[group]"):
        return line.replace("##[group]", "", 1).strip() or current_stage
    if line.startswith("##[endgroup]"):
        return "global"
    return current_stage


def _normalize_signature(line: str) -> str:
    normalized = re.sub(r"\s+", " ", line.strip())
    return normalized[:160]


def _find_stack_context(lines: list[str], idx: int) -> tuple[str | None, int | None, list[str]]:
    start = max(0, idx - 5)
    end = min(len(lines), idx + 6)

    file_path: str | None = None
    line_no: int | None = None
    frames: list[str] = []

    for probe in lines[start:end]:
        match = STACK_FRAME_PATTERN.search(probe)
        if not match:
            continue

        if match.group("file") is not None:
            file_path = match.group("file")
            line_no = int(match.group("line"))
            frames.append(f"{file_path}:{line_no}")
        elif match.group("js_file") is not None:
            js_file = match.group("js_file")
            js_line_match = re.search(r":(\d+):(\d+)", probe)
            if js_line_match:
                line_no = int(js_line_match.group(1))
                file_path = js_file
                frames.append(f"{file_path}:{line_no}")

    return file_path, line_no, frames


def parse_ci_log(raw_log: str, timestamp: str = "1970-01-01T00:00:00Z") -> ParsedLog:
    lines = raw_log.splitlines()
    current_stage = "global"
    stages: list[str] = [current_stage]
    events: list[ParsedFailureEvent] = []

    for idx, raw_line in enumerate(lines):
        current_stage = _extract_stage(raw_line, current_stage)
        if current_stage not in stages:
            stages.append(current_stage)

        if not any(pattern.search(raw_line) for pattern in ERROR_PATTERNS):
            continue

        file_path, line_no, frames = _find_stack_context(lines, idx)
        excerpt_start = max(0, idx - 1)
        excerpt_end = min(len(lines), idx + 2)
        excerpt = "\n".join(lines[excerpt_start:excerpt_end]).strip()

        events.append(
            ParsedFailureEvent(
                event_index=idx,
                stage=current_stage,
                timestamp=timestamp,
                error_signature=_normalize_signature(raw_line),
                file=file_path,
                line=line_no,
                stack_frames=frames,
                log_excerpt=excerpt,
            )
        )

    return ParsedLog(stages=stages, failure_events=events)
