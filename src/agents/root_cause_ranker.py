from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class CandidateScore:
    first_failure_score: float
    diff_proximity_score: float
    module_proximity_score: float
    dependency_drift_score: float
    classification_alignment_score: float

    @property
    def confidence(self) -> float:
        return round(
            (0.40 * self.first_failure_score)
            + (0.30 * self.diff_proximity_score)
            + (0.15 * self.module_proximity_score)
            + (0.10 * self.dependency_drift_score)
            + (0.05 * self.classification_alignment_score),
            4,
        )


def _module_name(file_path: str | None) -> str:
    if not file_path:
        return ""
    p = PurePosixPath(file_path)
    return p.stem


def _classification_alignment(classification: str, signature: str) -> float:
    signature_l = signature.lower()

    if classification == "DEPENDENCY" and (
        "modulenotfound" in signature_l or "cannot find module" in signature_l
    ):
        return 1.0
    if classification == "TYPECHECK" and (
        "type" in signature_l or "ts" in signature_l or "mypy" in signature_l
    ):
        return 1.0
    if classification == "LINT" and (
        "ruff" in signature_l or "eslint" in signature_l or "lint" in signature_l
    ):
        return 1.0
    if classification == "TEST" and ("assertionerror" in signature_l or "failed" in signature_l):
        return 1.0
    if classification == "INFRA" and (
        "timeout" in signature_l or "network" in signature_l or "connection" in signature_l
    ):
        return 1.0

    if classification == "UNKNOWN":
        return 0.5
    return 0.2


def _score_candidate(
    node: dict,
    changed_files: list[str],
    changed_modules: list[str],
    dependency_change_flags: dict,
    classification: str,
) -> CandidateScore:
    file_path = node.get("file")
    signature = node.get("error_signature", "")

    first_failure_score = 1.0 if node.get("is_first_failure") else 0.3

    diff_proximity_score = 1.0 if file_path and file_path in changed_files else 0.2

    module = _module_name(file_path)
    module_proximity_score = 1.0 if module and module in changed_modules else 0.2

    dependency_drift_score = (
        1.0
        if dependency_change_flags.get("has_lockfile_change")
        or dependency_change_flags.get("has_manifest_change")
        else 0.0
    )

    classification_alignment_score = _classification_alignment(classification, signature)

    return CandidateScore(
        first_failure_score=first_failure_score,
        diff_proximity_score=diff_proximity_score,
        module_proximity_score=module_proximity_score,
        dependency_drift_score=dependency_drift_score,
        classification_alignment_score=classification_alignment_score,
    )


def _candidate_title(node: dict) -> str:
    file_path = node.get("file") or "unknown file"
    line = node.get("line")
    signature = node.get("error_signature", "unknown failure")
    if line:
        return f"{signature} at {file_path}:{line}"
    return f"{signature} at {file_path}"


def _candidate_evidence(node: dict, classification: str) -> list[dict]:
    file_path = node.get("file") or "unknown"
    return [
        {
            "file": file_path,
            "line": node.get("line"),
            "excerpt": node.get("log_excerpt") or "",
            "signal": f"classification:{classification.lower()}",
        }
    ]


def _sort_key(candidate: dict) -> tuple:
    # Deterministic tie-break sequence:
    # 1) confidence desc
    # 2) evidence count desc
    # 3) earliest timestamp asc
    # 4) stable_id asc
    return (
        -candidate["confidence"],
        -len(candidate["evidence"]),
        candidate["timestamp"],
        candidate["stable_id"],
    )


def run_root_cause_ranker(
    failure_graph: dict,
    changed_files: list[str],
    changed_modules: list[str],
    dependency_change_flags: dict,
    classification: str,
) -> dict:
    nodes = failure_graph.get("nodes", [])

    candidates: list[dict] = []
    for idx, node in enumerate(nodes):
        score = _score_candidate(
            node=node,
            changed_files=changed_files,
            changed_modules=changed_modules,
            dependency_change_flags=dependency_change_flags,
            classification=classification,
        )

        candidates.append(
            {
                "stable_id": f"candidate-{idx:04d}",
                "title": _candidate_title(node),
                "evidence": _candidate_evidence(node, classification),
                "confidence": score.confidence,
                "timestamp": node.get("timestamp", ""),
                "score_breakdown": {
                    "first_failure_score": score.first_failure_score,
                    "diff_proximity_score": score.diff_proximity_score,
                    "module_proximity_score": score.module_proximity_score,
                    "dependency_drift_score": score.dependency_drift_score,
                    "classification_alignment_score": score.classification_alignment_score,
                },
            }
        )

    ranked = sorted(candidates, key=_sort_key)

    ranked_causes = [
        {
            "title": candidate["title"],
            "evidence": candidate["evidence"],
            "score": candidate["confidence"],
            "score_breakdown": candidate["score_breakdown"],
        }
        for candidate in ranked
    ]

    primary = ranked_causes[0] if ranked_causes else None
    confidence = primary["score"] if primary else 0.0

    return {
        "ranked_causes": ranked_causes,
        "primary_root_cause": primary,
        "confidence": confidence,
    }
