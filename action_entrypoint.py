from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from core.orchestration import PipelineRequest, run_pipeline


class ActionInputError(RuntimeError):
    """Raised when GitHub Action inputs are invalid."""


SAFE_ROLLOUT_PROFILE = "safe-github-rollout"


def _input_key(name: str) -> str:
    return f"INPUT_{name.upper()}"


def _get_input(name: str, *, required: bool = False, default: str | None = None) -> str:
    value = os.getenv(_input_key(name))
    if value is None or value == "":
        value = default
    if required and (value is None or value == ""):
        raise ActionInputError(f"Missing required action input: {name}")
    return value or ""


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ActionInputError(f"Invalid boolean value for {name}: '{value}'")


def _parse_positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ActionInputError(
            f"Invalid integer value for {name}: '{value}'") from exc
    if parsed <= 0:
        raise ActionInputError(f"{name} must be > 0")
    return parsed


def _parse_confidence_threshold(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ActionInputError(
            f"Invalid float value for {name}: '{value}'") from exc
    if not (0.0 <= parsed <= 1.0):
        raise ActionInputError(f"{name} must be between 0.0 and 1.0")
    return parsed


def _load_simple_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ActionInputError(
            f"Unable to read config_path '{path}': {exc}") from exc

    config: dict[str, str] = {}
    for index, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ActionInputError(
                f"Invalid config line {index} in '{path}': expected key: value")
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ActionInputError(
                f"Invalid config line {index} in '{path}': empty key")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        config[key] = value
    return config


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ActionInputError(f"Unable to read file '{path}': {exc}") from exc


def _load_validated_changes(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ActionInputError(
            f"Unable to read validated changes file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ActionInputError(
            f"Invalid JSON in validated changes file '{path}': {exc}") from exc

    if not isinstance(raw, list):
        raise ActionInputError(
            "validated_changes_path must point to a JSON list")

    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ActionInputError(
                "Each validated change must be a JSON object")
        file_path = str(item.get("file", "")).strip()
        content = item.get("content")
        if not file_path or not isinstance(content, str):
            raise ActionInputError(
                "Each validated change must include string fields: file, content"
            )
        normalized.append({"file": file_path, "content": content})
    return normalized


def _load_historical_runs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ActionInputError(
            f"Unable to read historical runs file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ActionInputError(
            f"Invalid JSON in historical runs file '{path}': {exc}") from exc

    if not isinstance(raw, list):
        raise ActionInputError(
            "historical_runs_path must point to a JSON list")

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ActionInputError("Each historical run must be a JSON object")
        failure_events = item.get("failure_events", [])
        if failure_events is not None and not isinstance(failure_events, list):
            raise ActionInputError(
                "Each historical run field 'failure_events' must be a JSON list")
        normalized.append(dict(item))
    return normalized


def _build_diff_from_git(base_ref: str, head_ref: str) -> str:
    if not base_ref or not head_ref:
        return ""
    try:
        result = subprocess.run(
            ["git", "diff", "--no-color", "--no-ext-diff",
                f"{base_ref}..{head_ref}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""
    return result.stdout


def _resolve_raw_log(config: dict[str, str]) -> str:
    if "raw_log" in config and config["raw_log"].strip():
        return config["raw_log"]
    env_log = os.getenv("CI_ROOTCAUSE_RAW_LOG", "").strip()
    if env_log:
        return env_log
    log_path = config.get("log_path") or os.getenv("CI_ROOTCAUSE_LOG_PATH", "")
    if log_path.strip():
        return _read_text(Path(log_path.strip()))
    return "ERROR: ci-rootcause synthetic fallback failure event"


def _resolve_raw_diff(config: dict[str, str], base_ref: str, head_ref: str) -> str:
    diff_path = config.get("diff_path") or os.getenv(
        "CI_ROOTCAUSE_DIFF_PATH", "")
    if diff_path.strip():
        return _read_text(Path(diff_path.strip()))
    return _build_diff_from_git(base_ref=base_ref, head_ref=head_ref)


def _build_summary(state: Any) -> dict[str, Any]:
    classification = "UNKNOWN"
    confidence = 0.0
    title = ""
    json_path = ""
    md_path = ""
    pr_created = False
    pr_url = ""
    pr_number = ""

    classification_output = state.agent_outputs.get(
        "failure_classification", {})
    ranker_output = state.agent_outputs.get("root_cause_ranker", {})
    reporter_output = state.agent_outputs.get("reporter", {})
    pr_output = state.agent_outputs.get("pr_creation", {})

    if isinstance(classification_output, dict):
        classification = str(classification_output.get(
            "classification", classification))
    if isinstance(ranker_output, dict):
        confidence = float(ranker_output.get("confidence", confidence))
        primary = ranker_output.get("primary_root_cause")
        if isinstance(primary, dict):
            title = str(primary.get("title", title))
    if isinstance(reporter_output, dict):
        json_path = str(reporter_output.get("ci_rca_json_path", json_path))
        md_path = str(reporter_output.get("ci_rca_md_path", md_path))
    if isinstance(pr_output, dict):
        pr_created = bool(pr_output.get("pr_created", pr_created))
        if pr_output.get("pr_url"):
            pr_url = str(pr_output["pr_url"])
        if pr_output.get("pr_number") is not None:
            pr_number = str(pr_output["pr_number"])

    return {
        "classification": classification,
        "confidence": f"{confidence:.4f}",
        "primary_root_cause_title": title,
        "rca_json_path": json_path,
        "rca_md_path": md_path,
        "pr_created": "true" if pr_created else "false",
        "pr_url": pr_url,
        "pr_number": pr_number,
    }


def _write_github_outputs(outputs: dict[str, str]) -> None:
    output_path = os.getenv("GITHUB_OUTPUT", "").strip()
    lines = [f"{key}={value}" for key, value in outputs.items()]
    payload = "\n".join(lines) + "\n"
    if output_path:
        Path(output_path).write_text(payload, encoding="utf-8")
        return
    print(payload, end="")


def _emit_failure_outputs() -> None:
    _write_github_outputs(
        {
            "classification": "UNKNOWN",
            "confidence": "0.0000",
            "primary_root_cause_title": "",
            "rca_json_path": "",
            "rca_md_path": "",
            "pr_created": "false",
            "pr_url": "",
            "pr_number": "",
        }
    )


def main() -> int:
    try:
        github_token = _get_input("github_token", required=True)
        create_fix_pr = _parse_bool(
            _get_input("create_fix_pr", default="false"),
            name="create_fix_pr",
        )
        offline_only = _parse_bool(
            _get_input("offline_only", default="false"),
            name="offline_only",
        )
        _parse_bool(_get_input("post_pr_comment", default="true"),
                    name="post_pr_comment")
        base_ref_input = _get_input("base_ref", default="")
        head_ref_input = _get_input("head_ref", default="")
        config_path_value = _get_input(
            "config_path", default=".ci-rootcause.yml")
        max_fix_files = _parse_positive_int(
            _get_input("max_fix_files", default="5"),
            name="max_fix_files",
        )
        min_pr_confidence = _parse_confidence_threshold(
            _get_input("min_pr_confidence", default="0.75"),
            name="min_pr_confidence",
        )

        config = _load_simple_config(Path(config_path_value))
        rollout_profile = (
            _get_input("rollout_profile", default="").strip(
            ) or config.get("profile", "").strip()
        )
        if rollout_profile:
            if rollout_profile != SAFE_ROLLOUT_PROFILE:
                raise ActionInputError(
                    "Unsupported rollout_profile "
                    f"'{rollout_profile}'. Expected '{SAFE_ROLLOUT_PROFILE}'."
                )
            if min_pr_confidence < 0.9:
                min_pr_confidence = 0.9

        output_dir = config.get("output_dir", "artifacts")

        base_ref = (
            base_ref_input or config.get(
                "base_commit", "") or os.getenv("GITHUB_BASE_REF", "")
        )
        head_ref = head_ref_input or config.get(
            "head_commit", "") or os.getenv("GITHUB_SHA", "")
        commit = config.get("commit", "") or head_ref or "unknown-commit"

        run_id = config.get("run_id", "") or os.getenv(
            "GITHUB_RUN_ID", "") or "unknown-run-id"
        timestamp = config.get("timestamp", "") or os.getenv(
            "CI_ROOTCAUSE_TIMESTAMP", "")
        if not timestamp:
            timestamp = "1970-01-01T00:00:00Z"

        repository = config.get("repository", "") or os.getenv(
            "GITHUB_REPOSITORY", "")
        target_branch = (
            config.get("target_branch", "")
            or os.getenv("GITHUB_BASE_REF", "")
            or os.getenv("GITHUB_REF_NAME", "")
            or "main"
        )

        validated_changes_path = config.get(
            "validated_changes_path", "").strip()
        validated_changes = _load_validated_changes(
            Path(validated_changes_path) if validated_changes_path else None
        )
        historical_runs_path = config.get("historical_runs_path", "").strip()
        historical_runs = _load_historical_runs(
            Path(historical_runs_path) if historical_runs_path else None
        )

        if create_fix_pr:
            file_count = len({change["file"] for change in validated_changes})
            if file_count > max_fix_files:
                create_fix_pr = False

        raw_log = _resolve_raw_log(config=config)
        raw_diff = _resolve_raw_diff(
            config=config, base_ref=base_ref, head_ref=head_ref)

        request = PipelineRequest(
            raw_log=raw_log,
            raw_diff=raw_diff,
            timestamp=timestamp,
            commit=commit,
            run_id=run_id,
            base_commit=base_ref or commit,
            head_commit=head_ref or commit,
            output_dir=output_dir,
            create_fix_pr=create_fix_pr,
            offline_only=offline_only,
            dry_run=False,
            github_token=github_token,
            repository=repository or None,
            target_branch=target_branch or None,
            validated_changes=validated_changes,
            fail_fast=False,
            historical_runs=historical_runs,
            min_pr_confidence=min_pr_confidence,
        )

        state = run_pipeline(request=request)
        outputs = _build_summary(state=state)
        _write_github_outputs(outputs)
        return 0 if state.pipeline_status in {"completed", "partial"} else 2
    except Exception as exc:
        print(f"ci-rootcause action error: {exc}")
        _emit_failure_outputs()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
