from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.orchestration import PipelineRequest, run_pipeline


class BenchmarkSuiteError(RuntimeError):
    """Raised when benchmark suite loading or execution fails."""


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    log_path: str
    diff_path: str
    timestamp: str
    commit: str
    run_id: str
    base_commit: str
    head_commit: str
    expected_classification: str | None = None
    expected_primary_root_cause_contains: str | None = None


def _require_text(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise BenchmarkSuiteError(f"Benchmark case field '{field_name}' is required")
    return text


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int((len(ordered) - 1) * 0.95))
    return ordered[index]


def _basic_log_baseline_classification(first_failure_event: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(first_failure_event.get("error_signature", "")),
            str(first_failure_event.get("log_excerpt", "")),
            str(first_failure_event.get("stage", "")),
        ]
    ).lower()

    if any(
        token in text
        for token in (
            "timed out",
            "timeout",
            "connection reset",
            "network is unreachable",
            "runner lost",
        )
    ):
        return "INFRA"
    if re.search(r"\bts\d{4}\b", text) or "typescript" in text or "type error" in text:
        return "TYPECHECK"
    if any(token in text for token in ("ruff", "flake8", "eslint", "lint")):
        return "LINT"
    if any(token in text for token in ("build failed", "cannot compile", "compilation failed")):
        return "BUILD"
    if any(
        token in text for token in ("assertionerror", "test failed", "pytest", "jest", "failed:")
    ):
        return "TEST"
    return "UNKNOWN"


def _basic_log_baseline_root_cause(first_failure_event: dict[str, Any]) -> str:
    signature = str(first_failure_event.get("error_signature", "")).strip()
    excerpt = str(first_failure_event.get("log_excerpt", "")).strip()
    if signature:
        return signature
    if excerpt:
        return excerpt
    return "unknown failure"


def load_benchmark_suite(suite_path: str) -> tuple[str, list[BenchmarkCase]]:
    path = Path(suite_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkSuiteError(f"Unable to read benchmark suite '{suite_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkSuiteError(f"Invalid JSON in benchmark suite '{suite_path}': {exc}") from exc

    suite_name = _require_text(payload.get("suite_name", ""), "suite_name")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BenchmarkSuiteError("Benchmark suite must include a non-empty 'cases' list")

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for item in raw_cases:
        if not isinstance(item, dict):
            raise BenchmarkSuiteError("Each benchmark case must be a JSON object")

        case = BenchmarkCase(
            case_id=_require_text(item.get("case_id", ""), "case_id"),
            description=_require_text(item.get("description", ""), "description"),
            log_path=_require_text(item.get("log_path", ""), "log_path"),
            diff_path=_require_text(item.get("diff_path", ""), "diff_path"),
            timestamp=_require_text(item.get("timestamp", ""), "timestamp"),
            commit=_require_text(item.get("commit", ""), "commit"),
            run_id=_require_text(item.get("run_id", ""), "run_id"),
            base_commit=_require_text(item.get("base_commit", ""), "base_commit"),
            head_commit=_require_text(item.get("head_commit", ""), "head_commit"),
            expected_classification=(
                str(item.get("expected_classification")).strip()
                if item.get("expected_classification") is not None
                else None
            ),
            expected_primary_root_cause_contains=(
                str(item.get("expected_primary_root_cause_contains")).strip()
                if item.get("expected_primary_root_cause_contains") is not None
                else None
            ),
        )

        if case.case_id in seen_ids:
            raise BenchmarkSuiteError(f"Duplicate benchmark case_id: {case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)

    cases.sort(key=lambda case: case.case_id)
    return suite_name, cases


def run_benchmark_suite(
    suite_path: str,
    output_root: str,
    *,
    use_adk_runtime: bool | None = None,
    repeat_runs: int = 2,
) -> dict[str, Any]:
    if repeat_runs <= 0:
        raise BenchmarkSuiteError("repeat_runs must be > 0")

    suite_name, cases = load_benchmark_suite(suite_path)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    confidence_reproducible_count = 0
    artifact_hash_reproducible_count = 0
    for case in cases:
        log_path = Path(case.log_path)
        diff_path = Path(case.diff_path)

        if not log_path.exists():
            raise BenchmarkSuiteError(f"Benchmark case '{case.case_id}' log_path does not exist")
        if not diff_path.exists():
            raise BenchmarkSuiteError(f"Benchmark case '{case.case_id}' diff_path does not exist")

        case_output_dir = output_root_path / case.case_id
        confidence_values: list[float] = []
        timing_values: list[float] = []
        status_values: list[str] = []
        json_hash_values: list[str] = []
        md_hash_values: list[str] = []
        first_state: Any = None
        for _ in range(repeat_runs):
            request = PipelineRequest(
                raw_log=log_path.read_text(encoding="utf-8"),
                raw_diff=diff_path.read_text(encoding="utf-8"),
                timestamp=case.timestamp,
                commit=case.commit,
                run_id=case.run_id,
                base_commit=case.base_commit,
                head_commit=case.head_commit,
                output_dir=str(case_output_dir),
                create_fix_pr=False,
                use_adk_runtime=use_adk_runtime,
            )
            state = run_pipeline(request=request)
            reporter_output = state.agent_outputs.get("reporter", {})
            ranker_output = state.agent_outputs.get("root_cause_ranker", {})
            json_path = Path(str(reporter_output.get("ci_rca_json_path", "")))
            md_path = Path(str(reporter_output.get("ci_rca_md_path", "")))
            confidence_values.append(float(ranker_output.get("confidence", 0.0)))
            timing_values.append(float(state.pipeline_timing_ms))
            status_values.append(str(state.pipeline_status))
            json_hash_values.append(_sha256_file(json_path) if json_path.exists() else "")
            md_hash_values.append(_sha256_file(md_path) if md_path.exists() else "")
            if first_state is None:
                first_state = state

        state = first_state
        log_output = state.agent_outputs.get("log_ingest", {})
        classification_output = state.agent_outputs.get("failure_classification", {})
        ranker_output = state.agent_outputs.get("root_cause_ranker", {})
        reporter_output = state.agent_outputs.get("reporter", {})
        first_failure_event = log_output.get("first_failure_event", {})
        if not isinstance(first_failure_event, dict):
            first_failure_event = {}

        actual_classification = str(classification_output.get("classification", "UNKNOWN"))
        primary_root_cause_title = str(
            (ranker_output.get("primary_root_cause") or {}).get("title", "")
        )
        baseline_classification = _basic_log_baseline_classification(first_failure_event)
        baseline_root_cause_title = _basic_log_baseline_root_cause(first_failure_event)
        expected = case.expected_classification
        classification_match = expected is None or expected == actual_classification
        baseline_classification_match = expected is None or expected == baseline_classification
        expected_primary_contains = case.expected_primary_root_cause_contains
        primary_root_cause_match = (
            expected_primary_contains is None
            or expected_primary_contains.lower() in primary_root_cause_title.lower()
        )
        baseline_primary_root_cause_match = (
            expected_primary_contains is None
            or expected_primary_contains.lower() in baseline_root_cause_title.lower()
        )

        json_path = Path(str(reporter_output.get("ci_rca_json_path", "")))
        md_path = Path(str(reporter_output.get("ci_rca_md_path", "")))
        confidence_is_reproducible = len(set(confidence_values)) == 1
        if confidence_is_reproducible:
            confidence_reproducible_count += 1
        artifact_hash_is_reproducible = (
            len(set(json_hash_values)) == 1
            and len(set(md_hash_values)) == 1
            and bool(json_hash_values[0])
            and bool(md_hash_values[0])
        )
        if artifact_hash_is_reproducible:
            artifact_hash_reproducible_count += 1

        case_results.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "pipeline_status": state.pipeline_status,
                "classification": actual_classification,
                "expected_classification": expected,
                "classification_match": classification_match,
                "baseline_classification": baseline_classification,
                "baseline_classification_match": baseline_classification_match,
                "expected_primary_root_cause_contains": expected_primary_contains,
                "primary_root_cause_match": primary_root_cause_match,
                "baseline_primary_root_cause_title": baseline_root_cause_title,
                "baseline_primary_root_cause_match": baseline_primary_root_cause_match,
                "confidence": float(ranker_output.get("confidence", 0.0)),
                "confidence_values": confidence_values,
                "confidence_is_reproducible": confidence_is_reproducible,
                "timing_values_ms": timing_values,
                "timing_spread_ms": (
                    round(max(timing_values) - min(timing_values), 3) if timing_values else 0.0
                ),
                "status_values": status_values,
                "artifact_json_hash_values": json_hash_values,
                "artifact_md_hash_values": md_hash_values,
                "artifact_hash_is_reproducible": artifact_hash_is_reproducible,
                "primary_root_cause_title": primary_root_cause_title,
                "trace_id": state.trace_id,
                "pipeline_timing_ms": state.pipeline_timing_ms,
                "ci_rca_json_sha256": _sha256_file(json_path) if json_path.exists() else "",
                "ci_rca_md_sha256": _sha256_file(md_path) if md_path.exists() else "",
            }
        )

    completed = sum(1 for item in case_results if item["pipeline_status"] == "completed")
    matched = sum(1 for item in case_results if item["classification_match"])
    baseline_matched = sum(1 for item in case_results if item["baseline_classification_match"])
    root_cause_matched = sum(1 for item in case_results if item["primary_root_cause_match"])
    baseline_root_cause_matched = sum(
        1 for item in case_results if item["baseline_primary_root_cause_match"]
    )
    total_cases = len(case_results)
    total_timing = sum(float(item["pipeline_timing_ms"]) for item in case_results)
    timing_samples = [float(item["pipeline_timing_ms"]) for item in case_results]
    mean_time_to_diagnosis_ms = round(total_timing / total_cases, 3) if total_cases else 0.0
    return {
        "suite_name": suite_name,
        "total_cases": total_cases,
        "completed_cases": completed,
        "completion_rate": round(completed / total_cases, 4) if total_cases else 0.0,
        "classification_matches": matched,
        "classification_match_rate": round(matched / total_cases, 4) if total_cases else 0.0,
        "baseline_classification_matches": baseline_matched,
        "baseline_classification_match_rate": (
            round(baseline_matched / total_cases, 4) if total_cases else 0.0
        ),
        "classification_match_lift": (
            round((matched - baseline_matched) / total_cases, 4) if total_cases else 0.0
        ),
        "primary_root_cause_matches": root_cause_matched,
        "primary_root_cause_accuracy": (
            round(root_cause_matched / total_cases, 4) if total_cases else 0.0
        ),
        "baseline_primary_root_cause_matches": baseline_root_cause_matched,
        "baseline_primary_root_cause_accuracy": (
            round(baseline_root_cause_matched / total_cases, 4) if total_cases else 0.0
        ),
        "primary_root_cause_accuracy_lift": (
            round((root_cause_matched - baseline_root_cause_matched) / total_cases, 4)
            if total_cases
            else 0.0
        ),
        "confidence_reproducible_cases": confidence_reproducible_count,
        "confidence_reproducibility": (
            round(confidence_reproducible_count / total_cases, 4) if total_cases else 0.0
        ),
        "artifact_hash_reproducible_cases": artifact_hash_reproducible_count,
        "artifact_hash_reproducibility": (
            round(artifact_hash_reproducible_count / total_cases, 4) if total_cases else 0.0
        ),
        "mean_time_to_diagnosis_ms": mean_time_to_diagnosis_ms,
        "median_time_to_diagnosis_ms": round(_median(timing_samples), 3),
        "p95_time_to_diagnosis_ms": round(_p95(timing_samples), 3),
        "cases": case_results,
    }
