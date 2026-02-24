from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    INFRA = "INFRA"
    DEPENDENCY = "DEPENDENCY"
    BUILD = "BUILD"
    TEST = "TEST"
    LINT = "LINT"
    TYPECHECK = "TYPECHECK"
    SECURITY = "SECURITY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Evidence:
    file: str
    line: int | None = None
    excerpt: str | None = None
    signal: str | None = None

    def validate(self) -> None:
        if not self.file:
            raise ValueError("Evidence.file is required")
        if self.line is not None and self.line <= 0:
            raise ValueError("Evidence.line must be > 0 when present")


@dataclass(frozen=True)
class RankedCause:
    title: str
    evidence: list[Evidence]
    score: float

    def validate(self) -> None:
        if not self.title:
            raise ValueError("RankedCause.title is required")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("RankedCause.score must be between 0 and 1")
        for item in self.evidence:
            item.validate()


@dataclass(frozen=True)
class FailureNode:
    stage: str
    timestamp: str
    error_signature: str
    file: str | None = None
    line: int | None = None
    stack_frames: list[str] = field(default_factory=list)
    log_excerpt: str | None = None
    is_first_failure: bool = False

    def validate(self) -> None:
        if not self.stage:
            raise ValueError("FailureNode.stage is required")
        if not self.timestamp:
            raise ValueError("FailureNode.timestamp is required")
        if not self.error_signature:
            raise ValueError("FailureNode.error_signature is required")
        if self.line is not None and self.line <= 0:
            raise ValueError("FailureNode.line must be > 0 when present")


@dataclass(frozen=True)
class FailureGraph:
    nodes: list[FailureNode]

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("FailureGraph.nodes must not be empty")

        first_failure_count = 0
        for node in self.nodes:
            node.validate()
            if node.is_first_failure:
                first_failure_count += 1

        if first_failure_count != 1:
            raise ValueError(
                "FailureGraph must contain exactly one node with is_first_failure=true"
            )


@dataclass(frozen=True)
class PrimaryRootCause:
    title: str
    evidence: list[Evidence]
    confidence: float

    def validate(self) -> None:
        if not self.title:
            raise ValueError("PrimaryRootCause.title is required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("PrimaryRootCause.confidence must be between 0 and 1")
        for item in self.evidence:
            item.validate()


@dataclass(frozen=True)
class RCAMeta:
    commit: str
    run_id: str

    def validate(self) -> None:
        if not self.commit:
            raise ValueError("RCAMeta.commit is required")
        if not self.run_id:
            raise ValueError("RCAMeta.run_id is required")


@dataclass(frozen=True)
class RCAOutput:
    summary: str
    classification: FailureClass
    primary_root_cause: PrimaryRootCause
    ranked_alternatives: list[RankedCause]
    suggested_fix: list[str]
    meta: RCAMeta

    def validate(self) -> None:
        if not self.summary:
            raise ValueError("RCAOutput.summary is required")
        self.primary_root_cause.validate()
        for cause in self.ranked_alternatives:
            cause.validate()
        self.meta.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["classification"] = self.classification.value
        return payload

    def to_json(self) -> str:
        # Deterministic serialization: stable key ordering and stable separators.
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PRCreationResult:
    pr_created: bool
    pr_url: str | None = None
    pr_number: int | None = None
    pr_branch: str | None = None
    failure_reason: str | None = None

    def validate(self) -> None:
        if self.pr_created:
            if not self.pr_url:
                raise ValueError("PRCreationResult.pr_url is required when pr_created=true")
            if self.pr_number is None or self.pr_number <= 0:
                raise ValueError("PRCreationResult.pr_number must be > 0 when pr_created=true")
            if not self.pr_branch:
                raise ValueError("PRCreationResult.pr_branch is required when pr_created=true")
        else:
            if not self.failure_reason:
                raise ValueError(
                    "PRCreationResult.failure_reason is required when pr_created=false"
                )
