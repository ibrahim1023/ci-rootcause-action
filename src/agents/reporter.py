from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.contracts.converters import rca_output_from_agent_outputs

PR_COMMENT_FORMAT_VERSION = "1"


class ArtifactWriteError(RuntimeError):
    """Raised when reporter artifacts cannot be written."""


def _build_summary(payload: dict[str, Any]) -> str:
    summary = str(payload.get("summary", "")).strip()
    if summary:
        return summary

    classification = str(payload["classification"]).strip()
    title = str(payload["primary_root_cause"]["title"]).strip()
    return f"{classification} failure: {title}"


def _normalize_primary_root_cause(payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload["primary_root_cause"]
    confidence = primary.get("confidence", payload.get("confidence", primary.get("score", 0.0)))
    return {
        "title": str(primary["title"]),
        "evidence": list(primary.get("evidence", [])),
        "confidence": float(confidence),
    }


def _normalize_ranked_alternatives(payload: dict[str, Any]) -> list[dict[str, Any]]:
    provided = payload.get("ranked_alternatives")
    if provided is not None:
        return [
            {
                "title": str(item["title"]),
                "evidence": list(item.get("evidence", [])),
                "score": float(item["score"]),
            }
            for item in provided
        ]

    ranked_causes = list(payload.get("ranked_causes", []))
    return [
        {
            "title": str(item["title"]),
            "evidence": list(item.get("evidence", [])),
            "score": float(item.get("score", item.get("confidence", 0.0))),
        }
        for item in ranked_causes[1:]
    ]


def _normalize_suggested_fix(payload: dict[str, Any]) -> list[str]:
    provided = payload.get("suggested_fix")
    if provided is not None:
        return [str(step).strip() for step in provided if str(step).strip()]

    fix_steps = list(payload.get("fix_steps", []))
    return [
        str(step.get("instruction", "")).strip()
        for step in fix_steps
        if str(step.get("instruction", "")).strip()
    ]


def _build_rca_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": _build_summary(payload),
        "classification": str(payload["classification"]),
        "primary_root_cause": _normalize_primary_root_cause(payload),
        "ranked_alternatives": _normalize_ranked_alternatives(payload),
        "suggested_fix": _normalize_suggested_fix(payload),
        "meta": {
            "commit": str(payload["meta"]["commit"]),
            "run_id": str(payload["meta"]["run_id"]),
        },
    }


def _render_markdown(rca_payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# CI Root Cause Analysis",
        "",
        rca_payload["summary"],
        "",
        "## Classification",
        f"- {rca_payload['classification']}",
        "",
        "## Primary Root Cause",
        f"- Title: {rca_payload['primary_root_cause']['title']}",
        f"- Confidence: {float(rca_payload['primary_root_cause']['confidence']):.4f}",
        "",
        "## Evidence",
    ]

    evidence = list(rca_payload["primary_root_cause"].get("evidence", []))
    if evidence:
        for item in evidence:
            file_path = str(item.get("file", "unknown"))
            line = item.get("line")
            location = f"{file_path}:{line}" if line else file_path
            signal = str(item.get("signal", "")).strip()
            if signal:
                lines.append(f"- {location} ({signal})")
            else:
                lines.append(f"- {location}")
    else:
        lines.append("- No evidence recorded")

    lines.extend(["", "## Ranked Alternatives"])
    alternatives = list(rca_payload.get("ranked_alternatives", []))
    if alternatives:
        for index, item in enumerate(alternatives, start=1):
            lines.append(f"{index}. {item['title']} (score: {float(item['score']):.4f})")
    else:
        lines.append("1. None")

    lines.extend(["", "## Suggested Fix"])
    suggested_fix = list(rca_payload.get("suggested_fix", []))
    if suggested_fix:
        for index, step in enumerate(suggested_fix, start=1):
            lines.append(f"{index}. {step}")
    else:
        lines.append("1. None")

    lines.extend(
        [
            "",
            "## Metadata",
            f"- Commit: {rca_payload['meta']['commit']}",
            f"- Run ID: {rca_payload['meta']['run_id']}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_pr_comment(markdown: str) -> dict[str, str]:
    return {
        "format": "markdown",
        "version": PR_COMMENT_FORMAT_VERSION,
        "body": "\n".join(
            [
                f"<!-- ci-rootcause:pr-comment:v{PR_COMMENT_FORMAT_VERSION} -->",
                markdown,
            ]
        ),
    }


def _build_artifact_upload_hooks(
    json_path: Path,
    md_path: Path,
) -> list[dict[str, str]]:
    return [
        {"name": "ci-rca-json", "path": str(json_path)},
        {"name": "ci-rca-md", "path": str(md_path)},
    ]


def run_reporter(payload: dict[str, Any], output_dir: str = ".") -> dict[str, Any]:
    rca_payload = _build_rca_payload(payload)
    # Validate and normalize against the typed contract before writing artifacts.
    rca_contract = rca_output_from_agent_outputs(rca_payload)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "ci-rca.json"
    md_path = output_path / "ci-rca.md"

    json_content = rca_contract.to_json()
    markdown_content = _render_markdown(rca_payload)

    try:
        json_path.write_text(json_content + "\n", encoding="utf-8")
        md_path.write_text(markdown_content, encoding="utf-8")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write reporter artifact: {exc}") from exc

    pr_comment = _build_pr_comment(markdown_content)
    return {
        "ci_rca_payload": json.loads(json_content),
        "ci_rca_json_path": str(json_path),
        "ci_rca_md_path": str(md_path),
        "pr_comment": pr_comment,
        "artifact_upload_hooks": _build_artifact_upload_hooks(json_path=json_path, md_path=md_path),
    }
