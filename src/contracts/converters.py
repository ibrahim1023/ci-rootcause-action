from __future__ import annotations

from src.contracts.models import (
    Evidence,
    FailureClass,
    FailureGraph,
    FailureNode,
    PRCreationResult,
    PrimaryRootCause,
    RankedCause,
    RCAMeta,
    RCAOutput,
)


def failure_graph_from_log_ingest(payload: dict) -> FailureGraph:
    nodes: list[FailureNode] = []
    for item in payload.get("failure_graph", []):
        nodes.append(
            FailureNode(
                stage=item["stage"],
                timestamp=item["timestamp"],
                error_signature=item["error_signature"],
                file=item.get("file"),
                line=item.get("line"),
                stack_frames=item.get("stack_frames", []),
                log_excerpt=item.get("log_excerpt"),
                is_first_failure=item.get("is_first_failure", False),
            )
        )

    graph = FailureGraph(nodes=nodes)
    graph.validate()
    return graph


def rca_output_from_agent_outputs(payload: dict) -> RCAOutput:
    primary = payload["primary_root_cause"]
    primary_evidence = [
        Evidence(
            file=evidence["file"],
            line=evidence.get("line"),
            excerpt=evidence.get("excerpt"),
            signal=evidence.get("signal"),
        )
        for evidence in primary.get("evidence", [])
    ]

    alternatives: list[RankedCause] = []
    for cause in payload.get("ranked_alternatives", []):
        cause_evidence = [
            Evidence(
                file=evidence["file"],
                line=evidence.get("line"),
                excerpt=evidence.get("excerpt"),
                signal=evidence.get("signal"),
            )
            for evidence in cause.get("evidence", [])
        ]
        alternatives.append(
            RankedCause(
                title=cause["title"],
                evidence=cause_evidence,
                score=float(cause["score"]),
            )
        )

    output = RCAOutput(
        summary=payload["summary"],
        classification=FailureClass(payload["classification"]),
        primary_root_cause=PrimaryRootCause(
            title=primary["title"],
            evidence=primary_evidence,
            confidence=float(primary["confidence"]),
        ),
        ranked_alternatives=alternatives,
        suggested_fix=list(payload.get("suggested_fix", [])),
        meta=RCAMeta(
            commit=payload["meta"]["commit"],
            run_id=payload["meta"]["run_id"],
        ),
    )
    output.validate()
    return output


def pr_result_from_agent_output(payload: dict) -> PRCreationResult:
    pr_number = payload.get("pr_number")
    result = PRCreationResult(
        pr_created=bool(payload["pr_created"]),
        pr_url=payload.get("pr_url"),
        pr_number=int(pr_number) if pr_number is not None and pr_number != "" else None,
        pr_branch=payload.get("pr_branch"),
        failure_reason=payload.get("failure_reason"),
    )
    result.validate()
    return result
