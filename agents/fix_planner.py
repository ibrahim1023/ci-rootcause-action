from __future__ import annotations

import re
from dataclasses import dataclass

from src.contracts.models import FailureClass

FILE_REF_PATTERN = re.compile(r"(?P<path>[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)")
CREATE_PATTERN = re.compile(r"\b(create|add)\b")
DELETE_PATTERN = re.compile(r"\b(delete|remove)\b")
RENAME_PATTERN = re.compile(r"\brename\b")
ALLOWED_PATCH_OPERATIONS = ("modify", "create", "delete", "rename")


PROMPT_TEMPLATE = (
    "You are a constrained fix planner. Use only evidence-backed files, provide minimal "
    "non-speculative steps, and return strict JSON output schema."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "fix_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "instruction": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["file", "instruction", "reason"],
            },
        },
        "patch_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "operation": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["file", "operation", "summary"],
            },
        },
    },
    "required": ["fix_steps"],
}


@dataclass(frozen=True)
class FixStep:
    file: str
    instruction: str
    reason: str


def _normalize_steps(steps: list[FixStep]) -> list[FixStep]:
    # Deterministic post-processing: stable sort + normalized whitespace.
    normalized: list[FixStep] = []
    for step in steps:
        normalized.append(
            FixStep(
                file=step.file.strip(),
                instruction=" ".join(step.instruction.strip().split()),
                reason=" ".join(step.reason.strip().split()),
            )
        )

    return sorted(normalized, key=lambda item: (item.file, item.instruction, item.reason))


def _extract_allowed_files(primary_root_cause: dict) -> set[str]:
    evidence = primary_root_cause.get("evidence", [])
    files = {str(item.get("file", "")).strip() for item in evidence if item.get("file")}
    return {file_path for file_path in files if file_path}


def _validate_contract(payload: dict) -> None:
    if "classification" not in payload:
        raise ValueError("Fix planner input missing 'classification'")
    if "primary_root_cause" not in payload:
        raise ValueError("Fix planner input missing 'primary_root_cause'")

    primary = payload["primary_root_cause"]
    if "title" not in primary or "evidence" not in primary:
        raise ValueError("primary_root_cause must include 'title' and 'evidence'")


def _validate_evidence_scope(primary_root_cause: dict, allowed_files: set[str]) -> None:
    evidence_files = _extract_allowed_files(primary_root_cause)
    if not evidence_files:
        raise ValueError("No evidence-backed files provided for fix planning")

    if not evidence_files.issubset(allowed_files):
        raise ValueError("Evidence contains files outside allowed fix scope")


def _validate_no_speculative_references(steps: list[FixStep], allowed_files: set[str]) -> None:
    for step in steps:
        if step.file not in allowed_files:
            raise ValueError(f"Speculative file reference rejected: {step.file}")

        # Guard against hidden speculative references embedded in instructions.
        for match in FILE_REF_PATTERN.finditer(step.instruction):
            if match.group("path") not in allowed_files:
                raise ValueError(
                    f"Speculative file reference rejected in instruction: {match.group('path')}"
                )


def _infer_patch_operation(instruction: str) -> str:
    lowered = instruction.lower()
    if RENAME_PATTERN.search(lowered):
        return "rename"
    if DELETE_PATTERN.search(lowered):
        return "delete"
    if CREATE_PATTERN.search(lowered):
        return "create"
    return "modify"


def _template_steps(classification: FailureClass, file_path: str, title: str) -> list[FixStep]:
    if classification == FailureClass.TYPECHECK:
        return [
            FixStep(
                file=file_path,
                instruction="Update type annotations and value flow to satisfy static type checks.",
                reason=f"Primary RCA indicates a type-check issue: {title}",
            )
        ]
    if classification == FailureClass.LINT:
        return [
            FixStep(
                file=file_path,
                instruction="Apply minimal lint-compliant formatting and naming adjustments.",
                reason=f"Primary RCA indicates a lint issue: {title}",
            )
        ]
    if classification == FailureClass.TEST:
        return [
            FixStep(
                file=file_path,
                instruction=(
                    "Adjust implementation or assertion so observed behavior "
                    "matches expected test contract."
                ),
                reason=f"Primary RCA indicates a test failure: {title}",
            )
        ]
    if classification == FailureClass.DEPENDENCY:
        return [
            FixStep(
                file=file_path,
                instruction="Align dependency references with lockfile and import expectations.",
                reason=f"Primary RCA indicates a dependency issue: {title}",
            )
        ]

    return [
        FixStep(
            file=file_path,
            instruction="Apply the smallest change that resolves the evidenced failure signature.",
            reason=f"Primary RCA to address: {title}",
        )
    ]


def _to_patch_plan(steps: list[FixStep]) -> list[dict]:
    grouped: dict[str, dict[str, list[str] | str]] = {}
    for step in steps:
        operation = _infer_patch_operation(step.instruction)
        entry = grouped.setdefault(
            step.file,
            {
                "operations": [],
                "summaries": [],
            },
        )
        operations = entry["operations"]
        summaries = entry["summaries"]
        if operation not in operations:
            operations.append(operation)
        if step.instruction not in summaries:
            summaries.append(step.instruction)

    patch_plan: list[dict] = []
    for file_path in sorted(grouped):
        operations = list(grouped[file_path]["operations"])
        summaries = list(grouped[file_path]["summaries"])
        if len(operations) > 1:
            raise ValueError(
                "Conflicting patch operations for file "
                f"{file_path}: {', '.join(sorted(operations))}"
            )
        operation = operations[0] if operations else "modify"
        if operation not in ALLOWED_PATCH_OPERATIONS:
            raise ValueError(f"Unsupported patch operation inferred: {operation}")
        patch_plan.append(
            {
                "file": file_path,
                "operation": operation,
                "summary": "; ".join(summaries),
            }
        )
    return patch_plan


def run_fix_planner(payload: dict) -> dict:
    _validate_contract(payload)

    classification = FailureClass(payload["classification"])
    primary = payload["primary_root_cause"]

    allowed_files = set(payload.get("allowed_files") or _extract_allowed_files(primary))
    _validate_evidence_scope(primary, allowed_files)

    raw_steps_payload = payload.get("candidate_fix_steps")
    if raw_steps_payload:
        raw_steps = [
            FixStep(
                file=str(step["file"]),
                instruction=str(step["instruction"]),
                reason=str(step.get("reason", "Evidence-backed fix step")),
            )
            for step in raw_steps_payload
        ]
    else:
        primary_file = sorted(_extract_allowed_files(primary))[0]
        raw_steps = _template_steps(
            classification=classification,
            file_path=primary_file,
            title=str(primary["title"]),
        )

    normalized_steps = _normalize_steps(raw_steps)
    _validate_no_speculative_references(normalized_steps, allowed_files)

    fix_steps = [
        {
            "file": step.file,
            "instruction": step.instruction,
            "reason": step.reason,
        }
        for step in normalized_steps
    ]

    output = {
        "prompt_template": PROMPT_TEMPLATE,
        "output_schema": OUTPUT_SCHEMA,
        "fix_steps": fix_steps,
        "patch_plan": _to_patch_plan(normalized_steps),
    }

    return output
