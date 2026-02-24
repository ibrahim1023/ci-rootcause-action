from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.contracts.models import FailureClass


@dataclass(frozen=True)
class ClassificationResult:
    classification: FailureClass
    signals: list[str]


RULE_PRIORITY: list[FailureClass] = [
    FailureClass.INFRA,
    FailureClass.DEPENDENCY,
    FailureClass.SECURITY,
    FailureClass.TYPECHECK,
    FailureClass.LINT,
    FailureClass.TEST,
    FailureClass.BUILD,
]

RULE_PATTERNS: dict[FailureClass, tuple[str, ...]] = {
    FailureClass.INFRA: (
        "timed out",
        "timeout",
        "connection reset",
        "temporary failure in name resolution",
        "service unavailable",
        "runner lost",
        "network is unreachable",
    ),
    FailureClass.DEPENDENCY: (
        "modulenotfounderror",
        "cannot find module",
        "no matching distribution found",
        "dependency conflict",
        "version solving failed",
        "lockfile",
    ),
    FailureClass.SECURITY: (
        "vulnerability",
        "security",
        "denied by policy",
        "secret detected",
    ),
    FailureClass.TYPECHECK: (
        "mypy",
        "pyright",
        "type error",
        "incompatible types",
        "typescript error",
        "ts2322",
        "ts2345",
    ),
    FailureClass.LINT: (
        "flake8",
        "ruff",
        "eslint",
        "pylint",
        "lint",
        "style violation",
    ),
    FailureClass.TEST: (
        "assertionerror",
        "failed: ",
        "test failed",
        "expected",
        "pytest",
        "jest",
    ),
    FailureClass.BUILD: (
        "compilation failed",
        "build failed",
        "cannot compile",
        "c compiler",
        "linker error",
    ),
}

TEST_ID_PATTERN = re.compile(r"(?:[a-zA-Z_]\w*::)+test_[a-zA-Z0-9_\[\]-]+|test_[a-zA-Z0-9_\[\]-]+")


def _normalize_signature(value: str) -> str:
    return " ".join(value.lower().split())


def _extract_test_ids(text: str) -> set[str]:
    return {match.group(0).lower() for match in TEST_ID_PATTERN.finditer(text)}


def _collect_text(failure_events: Iterable[dict]) -> str:
    chunks: list[str] = []
    for event in failure_events:
        chunks.append(str(event.get("error_signature", "")))
        chunks.append(str(event.get("log_excerpt", "")))
    return "\n".join(chunks).lower()


def _detect_flaky_test_pattern(
    failure_events: list[dict],
    historical_runs: list[dict] | None,
) -> dict:
    history = list(historical_runs or [])
    current_text = _collect_text(failure_events)
    current_test_ids = _extract_test_ids(current_text)
    if not current_test_ids:
        return {
            "detected": False,
            "score": 0.0,
            "matched_test_ids": [],
            "matched_failure_runs": 0,
            "unique_failure_signatures": 0,
            "history_window_size": len(history),
        }

    matched_failure_runs = 0
    matched_ids: set[str] = set()
    signatures: set[str] = set()

    for run in history:
        if not isinstance(run, dict):
            continue
        run_failure_events = run.get("failure_events")
        if not isinstance(run_failure_events, list):
            continue
        run_text = _collect_text(run_failure_events)
        run_test_ids = _extract_test_ids(run_text)
        overlap = current_test_ids.intersection(run_test_ids)
        if not overlap:
            continue
        matched_ids.update(overlap)
        matched_failure_runs += 1
        for event in run_failure_events:
            signatures.add(_normalize_signature(str(event.get("error_signature", ""))))
            signatures.add(_normalize_signature(str(event.get("log_excerpt", ""))))

    unique_failure_signatures = len([item for item in signatures if item])
    score = 0.0
    if matched_failure_runs >= 2:
        score += 0.6
    if unique_failure_signatures >= 2:
        score += 0.4

    return {
        "detected": score >= 1.0,
        "score": round(score, 4),
        "matched_test_ids": sorted(matched_ids),
        "matched_failure_runs": matched_failure_runs,
        "unique_failure_signatures": unique_failure_signatures,
        "history_window_size": len(history),
    }


def run_failure_classification(
    failure_events: list[dict],
    dependency_change_flags: dict | None = None,
    historical_runs: list[dict] | None = None,
) -> dict:
    signals: list[str] = []
    text = _collect_text(failure_events)
    flaky_test_detection = _detect_flaky_test_pattern(
        failure_events=failure_events,
        historical_runs=historical_runs,
    )

    if dependency_change_flags:
        if dependency_change_flags.get("has_lockfile_change"):
            signals.append("dep:lockfile_changed")
        if dependency_change_flags.get("has_manifest_change"):
            signals.append("dep:manifest_changed")

    matched_by_class: dict[FailureClass, list[str]] = {
        failure_class: [] for failure_class in RULE_PRIORITY
    }

    for failure_class in RULE_PRIORITY:
        patterns = RULE_PATTERNS[failure_class]
        for pattern in patterns:
            if pattern in text:
                matched_by_class[failure_class].append(f"pattern:{pattern}")

    if dependency_change_flags and (
        dependency_change_flags.get("has_lockfile_change")
        or dependency_change_flags.get("has_manifest_change")
    ):
        matched_by_class[FailureClass.DEPENDENCY].append("flag:dependency_change")

    for failure_class in RULE_PRIORITY:
        class_signals = matched_by_class[failure_class]
        if class_signals:
            signals.extend(class_signals)
            if failure_class is FailureClass.TEST and flaky_test_detection["detected"]:
                signals.extend(
                    [
                        "flake:historical_pattern_detected",
                        f"flake:matched_failure_runs={flaky_test_detection['matched_failure_runs']}",
                        (
                            "flake:unique_failure_signatures="
                            f"{flaky_test_detection['unique_failure_signatures']}"
                        ),
                    ]
                )
            return {
                "classification": failure_class.value,
                "signals": signals,
                "flaky_test_detection": flaky_test_detection,
            }

    if flaky_test_detection["detected"]:
        return {
            "classification": FailureClass.TEST.value,
            "signals": [
                "flake:historical_pattern_detected",
                f"flake:matched_failure_runs={flaky_test_detection['matched_failure_runs']}",
                f"flake:unique_failure_signatures={flaky_test_detection['unique_failure_signatures']}",
            ],
            "flaky_test_detection": flaky_test_detection,
        }

    return {
        "classification": FailureClass.UNKNOWN.value,
        "signals": ["fallback:insufficient_classification_signals"],
        "flaky_test_detection": flaky_test_detection,
    }


def evaluate_classification_accuracy(cases: list[dict]) -> dict:
    total = len(cases)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "misclassification_rate": 0.0}

    correct = 0
    errors: list[dict] = []

    for case in cases:
        result = run_failure_classification(
            failure_events=case["failure_events"],
            dependency_change_flags=case.get("dependency_change_flags"),
            historical_runs=case.get("historical_runs"),
        )
        expected = case["expected_classification"]
        actual = result["classification"]
        if actual == expected:
            correct += 1
        else:
            errors.append(
                {"name": case.get("name", "unknown"), "expected": expected, "actual": actual}
            )

    accuracy = correct / total
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "misclassification_rate": round(1 - accuracy, 4),
        "errors": errors,
    }
