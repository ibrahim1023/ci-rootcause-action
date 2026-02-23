from __future__ import annotations

from dataclasses import asdict

from src.parsers.ci_log_parser import parse_ci_log


def _event_sort_key(event: dict, stage_order: dict[str, int]) -> tuple:
    # Deterministic tie-break sequence:
    # 1) timestamp
    # 2) stage order in log
    # 3) line position in raw log (event_index)
    # 4) file path
    # 5) line number
    # 6) normalized error signature
    return (
        event["timestamp"],
        stage_order.get(event["stage"], 10_000),
        event["event_index"],
        event.get("file") or "",
        event.get("line") or 0,
        event["error_signature"],
    )


def _build_failure_graph(failure_events: list[dict], first_idx: int | None) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []

    for idx, event in enumerate(failure_events):
        node = {
            "stage": event["stage"],
            "timestamp": event["timestamp"],
            "file": event["file"],
            "line": event["line"],
            "error_signature": event["error_signature"],
            "stack_frames": event["stack_frames"],
            "log_excerpt": event["log_excerpt"],
            "is_first_failure": idx == first_idx,
        }
        nodes.append(node)

        if idx > 0:
            edges.append({"from": idx - 1, "to": idx, "type": "temporal"})

    return {"nodes": nodes, "edges": edges}


def run_log_ingest(raw_log: str, timestamp: str = "1970-01-01T00:00:00Z") -> dict:
    parsed = parse_ci_log(raw_log=raw_log, timestamp=timestamp)
    event_dicts = [asdict(event) for event in parsed.failure_events]
    stage_order = {stage: idx for idx, stage in enumerate(parsed.stages)}

    sorted_events = sorted(event_dicts, key=lambda event: _event_sort_key(event, stage_order))
    first_failure_event = sorted_events[0] if sorted_events else None
    first_idx = None
    if first_failure_event is not None:
        # Mark first failure on timeline-ordered events for graph output.
        for idx, event in enumerate(event_dicts):
            if event["event_index"] == first_failure_event["event_index"]:
                first_idx = idx
                break

    failure_graph = _build_failure_graph(event_dicts, first_idx)

    if first_failure_event is not None:
        first_failure_event = {k: v for k, v in first_failure_event.items() if k != "event_index"}

    timeline_events = [
        {k: v for k, v in event.items() if k != "event_index"} for event in event_dicts
    ]
    return {
        "stages": parsed.stages,
        "failure_events": timeline_events,
        "failure_graph": failure_graph,
        "first_failure_event": first_failure_event,
    }
