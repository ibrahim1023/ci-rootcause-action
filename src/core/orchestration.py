from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from src.agents.diff_analysis import run_diff_analysis
from src.agents.failure_classification import run_failure_classification
from src.agents.fix_planner import run_fix_planner
from src.agents.log_ingest import run_log_ingest
from src.agents.pr_creation import run_pr_creation
from src.agents.reporter import run_reporter
from src.agents.root_cause_ranker import run_root_cause_ranker
from src.core.provider_adapters import resolve_provider_defaults


class OrchestrationError(RuntimeError):
    """Raised when deterministic orchestration cannot be completed."""


AgentHandler = Callable[["PipelineState"], dict[str, Any]]

TYPECHECK_INT_ASSIGNMENT_PATTERN = re.compile(
    r"^(?P<prefix>\s*[A-Za-z_]\w*\s*:\s*int\s*=\s*)"
    r"(?P<quote>['\"])(?P<number>-?\d+)(?P=quote)"
    r"(?P<suffix>\s*(?:#.*)?)$"
)


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    depends_on: tuple[str, ...]
    handler: AgentHandler


@dataclass
class DeterministicAgentRegistry:
    _registrations: dict[str, AgentRegistration] = field(default_factory=dict)
    _registration_order: list[str] = field(default_factory=list)

    def register(self, registration: AgentRegistration) -> None:
        if registration.name in self._registrations:
            raise OrchestrationError(f"Duplicate agent registration: {registration.name}")

        for dependency in registration.depends_on:
            if dependency == registration.name:
                raise OrchestrationError(f"Agent cannot depend on itself: {registration.name}")

        self._registrations[registration.name] = registration
        self._registration_order.append(registration.name)

    def get(self, name: str) -> AgentRegistration:
        try:
            return self._registrations[name]
        except KeyError as exc:
            raise OrchestrationError(f"Unknown agent registration: {name}") from exc

    def resolve_order(self) -> list[str]:
        pending = {
            name: set(registration.depends_on) for name, registration in self._registrations.items()
        }

        resolved: list[str] = []
        while pending:
            ready = [name for name, deps in pending.items() if not deps]
            if not ready:
                unresolved = sorted(pending)
                raise OrchestrationError(f"Cyclic or missing dependencies detected: {unresolved}")

            ready.sort(key=self._registration_order.index)
            current = ready[0]
            resolved.append(current)
            del pending[current]
            for deps in pending.values():
                deps.discard(current)

        return resolved


@dataclass(frozen=True)
class ADKRuntimeScaffold:
    """ADK-facing runtime scaffold with deterministic agent registration."""

    registry: DeterministicAgentRegistry

    @property
    def backend(self) -> str:
        return "google-adk" if _module_exists("google.adk") else "google-adk-scaffold"

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "depends_on": list(self.registry.get(name).depends_on),
            }
            for name in self.registry.resolve_order()
        ]


@dataclass(frozen=True)
class RepoContext:
    repository: str
    target_branch: str = "main"


@dataclass(frozen=True)
class CommitContext:
    commit: str
    base_commit: str
    head_commit: str


@dataclass(frozen=True)
class RunContext:
    run_id: str
    timestamp: str
    job_id: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    ci_provider: str
    provider_adapter: str
    repo: RepoContext
    commit: CommitContext
    run: RunContext


@dataclass(frozen=True)
class PipelineRequest:
    raw_log: str
    raw_diff: str
    timestamp: str
    commit: str
    run_id: str
    base_commit: str
    head_commit: str
    output_dir: str
    create_fix_pr: bool = False
    dry_run: bool = False
    github_token: str | None = None
    repository: str | None = None
    target_branch: str | None = None
    validated_changes: list[dict[str, str]] = field(default_factory=list)
    fail_fast: bool = False
    historical_runs: list[dict[str, Any]] = field(default_factory=list)
    min_pr_confidence: float = 0.75
    offline_only: bool = False
    ci_provider: str | None = None
    provider_adapter: str | None = None
    config: PipelineConfig | None = None
    use_adk_runtime: bool | None = None


@dataclass
class PipelineState:
    request: PipelineRequest
    shared: dict[str, Any] = field(default_factory=dict)
    agent_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    agent_status: dict[str, str] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    trace_id: str = ""
    input_hashes: dict[str, str] = field(default_factory=dict)
    structured_logs: list[dict[str, Any]] = field(default_factory=list)
    agent_timing_ms: dict[str, float] = field(default_factory=dict)
    pipeline_timing_ms: float = 0.0
    nondeterministic_components: list[str] = field(default_factory=list)
    pipeline_status: str = "pending"
    config: PipelineConfig | None = None


def _run_log_ingest_agent(state: PipelineState) -> dict[str, Any]:
    return run_log_ingest(raw_log=state.request.raw_log, timestamp=state.config.run.timestamp)


def _run_diff_analysis_agent(state: PipelineState) -> dict[str, Any]:
    return run_diff_analysis(raw_diff=state.request.raw_diff)


def _run_failure_classification_agent(state: PipelineState) -> dict[str, Any]:
    log_output = state.agent_outputs["log_ingest"]
    diff_output = state.agent_outputs["diff_analysis"]
    return run_failure_classification(
        failure_events=log_output["failure_events"],
        dependency_change_flags=diff_output["dependency_change_flags"],
        historical_runs=state.request.historical_runs,
    )


def _run_root_cause_ranker_agent(state: PipelineState) -> dict[str, Any]:
    log_output = state.agent_outputs["log_ingest"]
    diff_output = state.agent_outputs["diff_analysis"]
    classification_output = state.agent_outputs["failure_classification"]
    return run_root_cause_ranker(
        failure_graph=log_output["failure_graph"],
        changed_files=diff_output["changed_files"],
        changed_modules=diff_output["changed_modules"],
        dependency_change_flags=diff_output["dependency_change_flags"],
        classification=classification_output["classification"],
    )


def _run_fix_planner_agent(state: PipelineState) -> dict[str, Any]:
    ranker_output = state.agent_outputs["root_cause_ranker"]
    classification_output = state.agent_outputs["failure_classification"]
    primary = ranker_output["primary_root_cause"]
    if not primary:
        raise OrchestrationError("Fix planner requires a primary_root_cause from ranker output")
    return run_fix_planner(
        {
            "classification": classification_output["classification"],
            "primary_root_cause": primary,
        }
    )


def _build_reporter_payload(state: PipelineState) -> dict[str, Any]:
    ranker_output = state.agent_outputs["root_cause_ranker"]
    classification_output = state.agent_outputs["failure_classification"]
    fix_output = state.agent_outputs["fix_planner"]

    primary = ranker_output["primary_root_cause"]
    if not primary:
        raise OrchestrationError("Reporter requires a primary_root_cause from ranker output")

    return {
        "summary": f"{classification_output['classification']} failure: {primary['title']}",
        "classification": classification_output["classification"],
        "confidence": ranker_output["confidence"],
        "primary_root_cause": primary,
        "ranked_causes": ranker_output["ranked_causes"],
        "fix_steps": fix_output["fix_steps"],
        "meta": {
            "commit": state.config.commit.commit,
            "run_id": state.config.run.run_id,
        },
    }


def _run_reporter_agent(state: PipelineState) -> dict[str, Any]:
    payload = _build_reporter_payload(state)
    return run_reporter(payload=payload, output_dir=state.request.output_dir)


def _normalize_repo_relative_path(file_path: str) -> str | None:
    candidate = Path(file_path.strip())
    if not str(candidate):
        return None
    if candidate.is_absolute():
        return None
    if ".." in candidate.parts:
        return None
    if candidate == Path("."):
        return None
    return candidate.as_posix()


def _synthesize_typecheck_change(
    *,
    file_path: str,
    evidence_line: int | None,
) -> str | None:
    try:
        original = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return None

    lines = original.splitlines(keepends=True)
    if not lines:
        return None

    line_index = max(0, (evidence_line or 1) - 1)
    if line_index >= len(lines):
        line_index = len(lines) - 1

    target = lines[line_index]
    if "type: ignore" in target:
        return None

    target_body = target.rstrip("\n")
    int_assignment = TYPECHECK_INT_ASSIGNMENT_PATTERN.match(target_body)
    if int_assignment is not None:
        replacement = (
            f"{int_assignment.group('prefix')}"
            f"{int_assignment.group('number')}"
            f"{int_assignment.group('suffix')}"
        )
        if target.endswith("\n"):
            lines[line_index] = f"{replacement}\n"
        else:
            lines[line_index] = replacement
        return "".join(lines)

    suffix = "  # type: ignore[assignment]"
    if target.endswith("\n"):
        lines[line_index] = f"{target.rstrip()}{suffix}\n"
    else:
        lines[line_index] = f"{target}{suffix}"
    return "".join(lines)


def _resolve_validated_changes_for_pr_creation(
    *,
    request_validated_changes: list[dict[str, str]],
    classification: str,
    primary_root_cause: dict[str, Any],
    fix_output: dict[str, Any],
) -> list[dict[str, str]]:
    if request_validated_changes:
        return request_validated_changes

    if classification != "TYPECHECK":
        return []

    evidence_by_file: dict[str, int | None] = {}
    for item in primary_root_cause.get("evidence", []):
        if not isinstance(item, dict):
            continue
        file_path = _normalize_repo_relative_path(str(item.get("file", "")))
        if not file_path:
            continue
        line_value = item.get("line")
        line_no = int(line_value) if isinstance(line_value, int) and line_value > 0 else None
        evidence_by_file[file_path] = line_no

    synthesized: list[dict[str, str]] = []
    for step in fix_output.get("fix_steps", []):
        if not isinstance(step, dict):
            continue
        file_path = _normalize_repo_relative_path(str(step.get("file", "")))
        if not file_path:
            continue
        content = _synthesize_typecheck_change(
            file_path=file_path,
            evidence_line=evidence_by_file.get(file_path),
        )
        if content is None:
            continue
        synthesized.append({"file": file_path, "content": content})

    deduped: dict[str, str] = {}
    for change in synthesized:
        deduped[str(change["file"])] = str(change["content"])
    return [{"file": key, "content": deduped[key]} for key in sorted(deduped)]


def _run_pr_creation_agent(state: PipelineState) -> dict[str, Any]:
    ranker_output = state.agent_outputs["root_cause_ranker"]
    classification_output = state.agent_outputs["failure_classification"]
    fix_output = state.agent_outputs["fix_planner"]
    primary = ranker_output["primary_root_cause"]

    allowed_files = sorted({step["file"] for step in fix_output["fix_steps"] if step.get("file")})
    validated_changes = _resolve_validated_changes_for_pr_creation(
        request_validated_changes=state.request.validated_changes,
        classification=str(classification_output["classification"]),
        primary_root_cause=primary,
        fix_output=fix_output,
    )
    payload = {
        "create_fix_pr": state.request.create_fix_pr,
        "min_pr_confidence": state.request.min_pr_confidence,
        "offline_only": state.request.offline_only,
        "dry_run": state.request.dry_run,
        "github_token": state.request.github_token or "",
        "repository": state.config.repo.repository,
        "target_branch": state.config.repo.target_branch,
        "summary": f"{classification_output['classification']} failure: {primary['title']}",
        "classification": classification_output["classification"],
        "confidence": ranker_output["confidence"],
        "primary_root_cause": primary,
        "meta": {
            "run_id": state.config.run.run_id,
            "base_commit": state.config.commit.base_commit,
            "head_commit": state.config.commit.head_commit,
        },
        "allowed_files": allowed_files,
        "validated_changes": validated_changes,
    }
    return run_pr_creation(payload=payload)


def build_default_registry() -> DeterministicAgentRegistry:
    registry = DeterministicAgentRegistry()
    registry.register(
        AgentRegistration(
            name="log_ingest",
            depends_on=(),
            handler=_run_log_ingest_agent,
        )
    )
    registry.register(
        AgentRegistration(
            name="diff_analysis",
            depends_on=(),
            handler=_run_diff_analysis_agent,
        )
    )
    registry.register(
        AgentRegistration(
            name="failure_classification",
            depends_on=("log_ingest", "diff_analysis"),
            handler=_run_failure_classification_agent,
        )
    )
    registry.register(
        AgentRegistration(
            name="root_cause_ranker",
            depends_on=("log_ingest", "diff_analysis", "failure_classification"),
            handler=_run_root_cause_ranker_agent,
        )
    )
    registry.register(
        AgentRegistration(
            name="fix_planner",
            depends_on=("root_cause_ranker", "failure_classification"),
            handler=_run_fix_planner_agent,
        )
    )
    registry.register(
        AgentRegistration(
            name="reporter",
            depends_on=("root_cause_ranker", "failure_classification", "fix_planner"),
            handler=_run_reporter_agent,
        )
    )
    registry.register(
        AgentRegistration(
            name="pr_creation",
            depends_on=("root_cause_ranker", "failure_classification", "fix_planner"),
            handler=_run_pr_creation_agent,
        )
    )
    return registry


def _blocked_dependencies(
    registration: AgentRegistration,
    state: PipelineState,
) -> list[str]:
    blocked: list[str] = []
    for name in registration.depends_on:
        if state.agent_status.get(name) in {"failed", "skipped"}:
            blocked.append(name)
    return blocked


def resolve_pipeline_config(request: PipelineRequest) -> PipelineConfig:
    if request.config is not None:
        return request.config

    provider_defaults = resolve_provider_defaults()
    ci_provider = str(request.ci_provider or provider_defaults.ci_provider).strip()
    provider_adapter = str(request.provider_adapter or provider_defaults.provider_adapter).strip()
    repository = str(request.repository or provider_defaults.repository).strip()
    target_branch = str(request.target_branch or provider_defaults.target_branch).strip() or "main"

    return PipelineConfig(
        ci_provider=ci_provider,
        provider_adapter=provider_adapter,
        repo=RepoContext(
            repository=repository,
            target_branch=target_branch,
        ),
        commit=CommitContext(
            commit=request.commit,
            base_commit=request.base_commit,
            head_commit=request.head_commit,
        ),
        run=RunContext(
            run_id=request.run_id,
            timestamp=request.timestamp,
            job_id=None,
        ),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config_hash(config: PipelineConfig) -> str:
    payload = {
        "ci_provider": config.ci_provider,
        "provider_adapter": config.provider_adapter,
        "repo": {
            "repository": config.repo.repository,
            "target_branch": config.repo.target_branch,
        },
        "commit": {
            "commit": config.commit.commit,
            "base_commit": config.commit.base_commit,
            "head_commit": config.commit.head_commit,
        },
        "run": {
            "run_id": config.run.run_id,
            "timestamp": config.run.timestamp,
            "job_id": config.run.job_id,
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_text(serialized)


def _compute_input_hashes(request: PipelineRequest, config: PipelineConfig) -> dict[str, str]:
    historical_runs_payload = json.dumps(
        request.historical_runs,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "raw_log_sha256": _sha256_text(request.raw_log),
        "raw_diff_sha256": _sha256_text(request.raw_diff),
        "historical_runs_sha256": _sha256_text(historical_runs_payload),
        "config_sha256": _config_hash(config),
    }


def _compute_trace_id(request: PipelineRequest, config: PipelineConfig) -> str:
    material = {
        "ci_provider": config.ci_provider,
        "repository": config.repo.repository,
        "base_commit": config.commit.base_commit,
        "head_commit": config.commit.head_commit,
        "run_id": config.run.run_id,
        "timestamp": config.run.timestamp,
        "job_id": config.run.job_id,
        "request_commit": request.commit,
    }
    serialized = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return _sha256_text(serialized)[:24]


def _append_structured_log(
    state: PipelineState,
    event: str,
    *,
    agent: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "sequence": len(state.structured_logs) + 1,
        "trace_id": state.trace_id,
        "run_id": state.config.run.run_id if state.config is not None else state.request.run_id,
        "event": event,
    }
    if agent is not None:
        entry["agent"] = agent
    if details:
        entry["details"] = details
    state.structured_logs.append(entry)


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return round((end_ns - start_ns) / 1_000_000.0, 3)


def _count_values(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _write_observability_artifact(state: PipelineState) -> None:
    try:
        output_root = Path(state.request.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        path = output_root / "ci-rca-observability.json"

        failure_agents = [str(item.get("agent", "unknown")) for item in state.failures]
        failure_error_types = [str(item.get("error_type", "unknown")) for item in state.failures]
        log_events = [str(item.get("event", "")) for item in state.structured_logs]
        status_values = list(state.agent_status.values())
        observability_payload = {
            "trace_id": state.trace_id,
            "run_id": state.config.run.run_id if state.config else state.request.run_id,
            "pipeline_status": state.pipeline_status,
            "ci_provider": state.config.ci_provider if state.config else "",
            "provider_adapter": state.config.provider_adapter if state.config else "",
            "repository": state.config.repo.repository if state.config else "",
            "agent_status_counts": _count_values(status_values),
            "failure_taxonomy": {
                "total_failures": len(state.failures),
                "by_agent": _count_values(failure_agents),
                "by_error_type": _count_values(failure_error_types),
                "failures": state.failures,
            },
            "timing_ms": {
                "pipeline": state.pipeline_timing_ms,
                "agents": {
                    name: state.agent_timing_ms[name] for name in sorted(state.agent_timing_ms)
                },
            },
            "structured_log_event_counts": _count_values(log_events),
            "nondeterministic_components": list(state.nondeterministic_components),
        }
        path.write_text(
            json.dumps(observability_payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        _append_structured_log(
            state,
            "observability_write_failed",
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )


def run_pipeline(
    request: PipelineRequest,
    registry: DeterministicAgentRegistry | None = None,
) -> PipelineState:
    use_adk_runtime = request.use_adk_runtime
    if use_adk_runtime is None:
        use_adk_runtime = _module_exists("google.adk")

    if use_adk_runtime and not request.fail_fast:
        try:
            return _run_pipeline_with_adk(request=request, registry=registry)
        except Exception:
            # Fall back to deterministic local orchestration if ADK runtime fails.
            pass

    return _run_pipeline_local(request=request, registry=registry)


def _run_pipeline_local(
    request: PipelineRequest,
    registry: DeterministicAgentRegistry | None = None,
) -> PipelineState:
    active_registry = registry or build_default_registry()
    config = resolve_pipeline_config(request)
    state = PipelineState(
        request=request,
        config=config,
        trace_id=_compute_trace_id(request=request, config=config),
        input_hashes=_compute_input_hashes(request=request, config=config),
        nondeterministic_components=["timing_metrics"],
    )
    pipeline_start_ns = time.perf_counter_ns()
    _append_structured_log(
        state,
        "pipeline_started",
        details={
            "runtime": "local",
            "agent_count": len(active_registry.resolve_order()),
            "input_hashes": state.input_hashes,
        },
    )

    for name in active_registry.resolve_order():
        agent_start_ns = time.perf_counter_ns()
        state.execution_order.append(name)
        _append_structured_log(state, "agent_started", agent=name)
        registration = active_registry.get(name)
        blocked_by = _blocked_dependencies(registration=registration, state=state)
        if blocked_by:
            output = {
                "status": "skipped",
                "reason": "dependency_failed",
                "blocked_by": blocked_by,
            }
            state.agent_outputs[name] = output
            state.shared[name] = output
            state.agent_status[name] = "skipped"
            duration_ms = _elapsed_ms(agent_start_ns, time.perf_counter_ns())
            state.agent_timing_ms[name] = duration_ms
            _append_structured_log(
                state,
                "agent_skipped",
                agent=name,
                details={"blocked_by": blocked_by, "duration_ms": duration_ms},
            )
            continue

        try:
            output = registration.handler(state)
        except Exception as exc:
            failure = {
                "agent": name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            state.failures.append(failure)
            output = {"status": "failed", "error": failure}
            state.agent_outputs[name] = output
            state.shared[name] = output
            state.agent_status[name] = "failed"
            duration_ms = _elapsed_ms(agent_start_ns, time.perf_counter_ns())
            state.agent_timing_ms[name] = duration_ms
            _append_structured_log(
                state,
                "agent_failed",
                agent=name,
                details={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "duration_ms": duration_ms,
                },
            )
            if request.fail_fast:
                raise OrchestrationError(
                    f"Pipeline failed in agent '{name}': {type(exc).__name__}: {exc}"
                ) from exc
            continue

        state.agent_outputs[name] = output
        state.shared[name] = output
        state.agent_status[name] = "completed"
        duration_ms = _elapsed_ms(agent_start_ns, time.perf_counter_ns())
        state.agent_timing_ms[name] = duration_ms
        _append_structured_log(
            state,
            "agent_completed",
            agent=name,
            details={"duration_ms": duration_ms},
        )

    if state.failures:
        completed_count = sum(1 for status in state.agent_status.values() if status == "completed")
        state.pipeline_status = "partial" if completed_count else "failed"
    else:
        state.pipeline_status = "completed"
    state.pipeline_timing_ms = _elapsed_ms(pipeline_start_ns, time.perf_counter_ns())
    _append_structured_log(
        state,
        "pipeline_completed",
        details={
            "pipeline_status": state.pipeline_status,
            "failure_count": len(state.failures),
            "duration_ms": state.pipeline_timing_ms,
        },
    )
    _write_observability_artifact(state)

    return state


STATE_AGENT_OUTPUTS = "__ci_rootcause_agent_outputs"
STATE_SHARED = "__ci_rootcause_shared"
STATE_EXECUTION_ORDER = "__ci_rootcause_execution_order"
STATE_AGENT_STATUS = "__ci_rootcause_agent_status"
STATE_FAILURES = "__ci_rootcause_failures"
STATE_TRACE_ID = "__ci_rootcause_trace_id"
STATE_INPUT_HASHES = "__ci_rootcause_input_hashes"
STATE_STRUCTURED_LOGS = "__ci_rootcause_structured_logs"
STATE_AGENT_TIMING_MS = "__ci_rootcause_agent_timing_ms"
STATE_PIPELINE_TIMING_MS = "__ci_rootcause_pipeline_timing_ms"
STATE_NONDETERMINISTIC_COMPONENTS = "__ci_rootcause_nondeterministic_components"


def _run_pipeline_with_adk(
    request: PipelineRequest,
    registry: DeterministicAgentRegistry | None = None,
) -> PipelineState:
    # Import ADK modules lazily to keep local deterministic runtime independent.
    import asyncio

    from google.adk import Runner
    from google.adk.agents import BaseAgent, InvocationContext, SequentialAgent
    from google.adk.events import Event
    from google.adk.events.event_actions import EventActions
    from google.adk.runners import InMemorySessionService
    from google.genai import types

    active_registry = registry or build_default_registry()
    config = resolve_pipeline_config(request)
    trace_id = _compute_trace_id(request=request, config=config)
    input_hashes = _compute_input_hashes(request=request, config=config)

    class DeterministicADKAgent(BaseAgent):
        registration: AgentRegistration
        request: PipelineRequest
        config: PipelineConfig

        async def _run_async_impl(self, ctx: InvocationContext):
            state = ctx.session.state
            agent_outputs = deepcopy(state.get(STATE_AGENT_OUTPUTS, {}))
            shared = deepcopy(state.get(STATE_SHARED, {}))
            execution_order = list(state.get(STATE_EXECUTION_ORDER, []))
            agent_status = deepcopy(state.get(STATE_AGENT_STATUS, {}))
            failures = deepcopy(state.get(STATE_FAILURES, []))
            structured_logs = deepcopy(state.get(STATE_STRUCTURED_LOGS, []))
            agent_timing_ms = deepcopy(state.get(STATE_AGENT_TIMING_MS, {}))
            current_trace_id = str(state.get(STATE_TRACE_ID, trace_id))
            current_input_hashes = deepcopy(state.get(STATE_INPUT_HASHES, input_hashes))
            nondeterministic_components = list(
                state.get(STATE_NONDETERMINISTIC_COMPONENTS, ["timing_metrics"])
            )

            name = self.registration.name
            agent_start_ns = time.perf_counter_ns()
            execution_order.append(name)
            structured_logs.append(
                {
                    "sequence": len(structured_logs) + 1,
                    "trace_id": current_trace_id,
                    "run_id": self.config.run.run_id,
                    "event": "agent_started",
                    "agent": name,
                }
            )

            blocked_by = _blocked_dependencies(
                registration=self.registration,
                state=PipelineState(
                    request=self.request,
                    shared=shared,
                    agent_outputs=agent_outputs,
                    execution_order=execution_order,
                    agent_status=agent_status,
                    failures=failures,
                    config=self.config,
                ),
            )

            if blocked_by:
                output = {
                    "status": "skipped",
                    "reason": "dependency_failed",
                    "blocked_by": blocked_by,
                }
                agent_outputs[name] = output
                shared[name] = output
                agent_status[name] = "skipped"
                duration_ms = _elapsed_ms(agent_start_ns, time.perf_counter_ns())
                agent_timing_ms[name] = duration_ms
                structured_logs.append(
                    {
                        "sequence": len(structured_logs) + 1,
                        "trace_id": current_trace_id,
                        "run_id": self.config.run.run_id,
                        "event": "agent_skipped",
                        "agent": name,
                        "details": {"blocked_by": blocked_by, "duration_ms": duration_ms},
                    }
                )
            else:
                try:
                    output = self.registration.handler(
                        PipelineState(
                            request=self.request,
                            shared=shared,
                            agent_outputs=agent_outputs,
                            execution_order=execution_order,
                            agent_status=agent_status,
                            failures=failures,
                            config=self.config,
                        )
                    )
                    agent_outputs[name] = output
                    shared[name] = output
                    agent_status[name] = "completed"
                    duration_ms = _elapsed_ms(agent_start_ns, time.perf_counter_ns())
                    agent_timing_ms[name] = duration_ms
                    structured_logs.append(
                        {
                            "sequence": len(structured_logs) + 1,
                            "trace_id": current_trace_id,
                            "run_id": self.config.run.run_id,
                            "event": "agent_completed",
                            "agent": name,
                            "details": {"duration_ms": duration_ms},
                        }
                    )
                except Exception as exc:
                    failure = {
                        "agent": name,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    failures.append(failure)
                    output = {"status": "failed", "error": failure}
                    agent_outputs[name] = output
                    shared[name] = output
                    agent_status[name] = "failed"
                    duration_ms = _elapsed_ms(agent_start_ns, time.perf_counter_ns())
                    agent_timing_ms[name] = duration_ms
                    structured_logs.append(
                        {
                            "sequence": len(structured_logs) + 1,
                            "trace_id": current_trace_id,
                            "run_id": self.config.run.run_id,
                            "event": "agent_failed",
                            "agent": name,
                            "details": {
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                                "duration_ms": duration_ms,
                            },
                        }
                    )

            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(
                    state_delta={
                        STATE_AGENT_OUTPUTS: agent_outputs,
                        STATE_SHARED: shared,
                        STATE_EXECUTION_ORDER: execution_order,
                        STATE_AGENT_STATUS: agent_status,
                        STATE_FAILURES: failures,
                        STATE_TRACE_ID: current_trace_id,
                        STATE_INPUT_HASHES: current_input_hashes,
                        STATE_STRUCTURED_LOGS: structured_logs,
                        STATE_AGENT_TIMING_MS: agent_timing_ms,
                        STATE_NONDETERMINISTIC_COMPONENTS: nondeterministic_components,
                    }
                ),
            )

    async def _execute() -> dict[str, Any]:
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="ci-rootcause",
            user_id="ci-rootcause",
            session_id=request.run_id or "run",
            state={
                STATE_TRACE_ID: trace_id,
                STATE_INPUT_HASHES: input_hashes,
                STATE_AGENT_TIMING_MS: {},
                STATE_PIPELINE_TIMING_MS: 0.0,
                STATE_NONDETERMINISTIC_COMPONENTS: ["timing_metrics"],
                STATE_STRUCTURED_LOGS: [
                    {
                        "sequence": 1,
                        "trace_id": trace_id,
                        "run_id": config.run.run_id,
                        "event": "pipeline_started",
                        "details": {
                            "runtime": "adk",
                            "agent_count": len(active_registry.resolve_order()),
                            "input_hashes": input_hashes,
                        },
                    }
                ],
            },
        )

        sub_agents = [
            DeterministicADKAgent(
                name=registration.name,
                registration=registration,
                request=request,
                config=config,
            )
            for registration in (
                active_registry.get(name) for name in active_registry.resolve_order()
            )
        ]

        root = SequentialAgent(
            name="ci_rootcause_pipeline",
            sub_agents=sub_agents,
        )
        runner = Runner(
            app_name="ci-rootcause",
            agent=root,
            session_service=session_service,
        )

        user_message = types.Content(
            role="user",
            parts=[types.Part(text="run ci-rootcause pipeline")],
        )
        pipeline_start_ns = time.perf_counter_ns()
        async for _ in runner.run_async(
            user_id="ci-rootcause",
            session_id=request.run_id or "run",
            new_message=user_message,
        ):
            pass
        pipeline_duration_ms = _elapsed_ms(pipeline_start_ns, time.perf_counter_ns())

        session = await session_service.get_session(
            app_name="ci-rootcause",
            user_id="ci-rootcause",
            session_id=request.run_id or "run",
        )
        session_state = dict(session.state)
        session_state[STATE_PIPELINE_TIMING_MS] = pipeline_duration_ms
        return session_state

    session_state = asyncio.run(_execute())

    state = PipelineState(
        request=request,
        shared=deepcopy(session_state.get(STATE_SHARED, {})),
        agent_outputs=deepcopy(session_state.get(STATE_AGENT_OUTPUTS, {})),
        execution_order=list(session_state.get(STATE_EXECUTION_ORDER, [])),
        agent_status=deepcopy(session_state.get(STATE_AGENT_STATUS, {})),
        failures=deepcopy(session_state.get(STATE_FAILURES, [])),
        trace_id=str(session_state.get(STATE_TRACE_ID, trace_id)),
        input_hashes=deepcopy(session_state.get(STATE_INPUT_HASHES, input_hashes)),
        structured_logs=deepcopy(session_state.get(STATE_STRUCTURED_LOGS, [])),
        agent_timing_ms=deepcopy(session_state.get(STATE_AGENT_TIMING_MS, {})),
        pipeline_timing_ms=float(session_state.get(STATE_PIPELINE_TIMING_MS, 0.0)),
        nondeterministic_components=list(
            session_state.get(STATE_NONDETERMINISTIC_COMPONENTS, ["timing_metrics"])
        ),
        config=config,
    )

    if state.failures:
        completed_count = sum(1 for status in state.agent_status.values() if status == "completed")
        state.pipeline_status = "partial" if completed_count else "failed"
    else:
        state.pipeline_status = "completed"
    _append_structured_log(
        state,
        "pipeline_completed",
        details={
            "pipeline_status": state.pipeline_status,
            "failure_count": len(state.failures),
            "duration_ms": state.pipeline_timing_ms,
        },
    )
    _write_observability_artifact(state)

    return state
