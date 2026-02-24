from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib import error, parse, request


class ProviderAdapterError(RuntimeError):
    """Raised when git provider command execution fails."""


class BranchCreationError(RuntimeError):
    """Raised when the fix branch cannot be created."""


class PatchApplicationError(RuntimeError):
    """Raised when validated file changes cannot be applied safely."""


class GuardrailViolationError(RuntimeError):
    """Raised when PR creation guardrails are violated."""


class GitHubAPIError(ProviderAdapterError):
    """Base class for typed GitHub API failures."""


class GitHubRateLimitError(GitHubAPIError):
    """Raised when GitHub API rate limits are encountered."""


class GitHubTransientError(GitHubAPIError):
    """Raised when a transient GitHub API/network error is encountered."""


class GitCommandRunner(Protocol):
    def run(self, args: list[str], cwd: Path) -> None:
        """Execute a git command."""


class GitHubClient(Protocol):
    def find_open_pull_request(
        self,
        *,
        owner: str,
        repo: str,
        head_branch: str,
        base_branch: str,
    ) -> dict[str, Any] | None:
        """Return an existing open PR for the branch pair, if present."""

    def create_pull_request(
        self,
        *,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
    ) -> dict[str, Any]:
        """Create a pull request and return response payload."""


@dataclass(frozen=True)
class BranchCreationPlan:
    base_ref: str
    head_ref: str
    pr_branch: str


@dataclass(frozen=True)
class ValidatedFileChange:
    file: str
    content: str


@dataclass(frozen=True)
class PullRequestRequest:
    owner: str
    repo: str
    title: str
    body: str
    head_branch: str
    base_branch: str


def _normalize_ref_segment(value: str) -> str:
    normalized = "".join(ch for ch in value.lower() if ch.isalnum())
    if not normalized:
        raise BranchCreationError("Branch segment must contain at least one alphanumeric character")
    return normalized[:12]


def build_fix_branch_name(base_ref: str, head_ref: str) -> str:
    base_segment = _normalize_ref_segment(base_ref)
    head_segment = _normalize_ref_segment(head_ref)
    return f"ci-rootcause/fix/{base_segment}-{head_segment}"


def build_branch_creation_plan(payload: dict[str, Any]) -> BranchCreationPlan:
    meta = payload.get("meta", {})
    base_ref = str(payload.get("base_ref") or meta.get("base_commit") or "").strip()
    head_ref = str(payload.get("head_ref") or meta.get("head_commit") or "").strip()

    if not base_ref:
        raise BranchCreationError("Missing base_ref or meta.base_commit for branch creation")
    if not head_ref:
        raise BranchCreationError("Missing head_ref or meta.head_commit for branch creation")

    return BranchCreationPlan(
        base_ref=base_ref,
        head_ref=head_ref,
        pr_branch=build_fix_branch_name(base_ref=base_ref, head_ref=head_ref),
    )


def _normalize_repo_relative_path(file_path: str) -> str:
    normalized_input = file_path.strip()
    if not normalized_input:
        raise PatchApplicationError("Change file path must not be empty")
    candidate = Path(normalized_input)
    if candidate == Path("."):
        raise PatchApplicationError("Change file path must not be empty")
    if candidate.is_absolute():
        raise PatchApplicationError(f"Absolute paths are not allowed: {file_path}")
    if ".." in candidate.parts:
        raise PatchApplicationError(f"Parent directory traversal is not allowed: {file_path}")
    return candidate.as_posix()


def _extract_evidence_files(payload: dict[str, Any]) -> set[str]:
    explicit = payload.get("allowed_files")
    if explicit is not None:
        return {_normalize_repo_relative_path(str(item)) for item in explicit}

    primary = payload.get("primary_root_cause", {})
    evidence = primary.get("evidence", [])
    files = set()
    for item in evidence:
        file_path = str(item.get("file", "")).strip()
        if file_path:
            files.add(_normalize_repo_relative_path(file_path))
    return files


def _extract_validated_changes(payload: dict[str, Any]) -> list[ValidatedFileChange]:
    raw_changes = payload.get("validated_changes", [])
    changes: list[ValidatedFileChange] = []
    for change in raw_changes:
        file_path = _normalize_repo_relative_path(str(change["file"]))
        content = str(change["content"])
        changes.append(ValidatedFileChange(file=file_path, content=content))
    return sorted(changes, key=lambda item: item.file)


def _validate_changes_against_evidence(
    changes: list[ValidatedFileChange],
    evidence_files: set[str],
) -> None:
    if not evidence_files:
        raise PatchApplicationError("No evidence-backed files available for PR creation")
    if not changes:
        raise PatchApplicationError("No validated_changes provided for PR creation")

    for change in changes:
        if change.file not in evidence_files:
            raise PatchApplicationError(
                f"Validated change file is outside evidence scope: {change.file}"
            )


def apply_validated_changes(
    changes: list[ValidatedFileChange],
    repo_path: str,
) -> list[str]:
    repo_root = Path(repo_path)
    applied_files: list[str] = []
    for change in changes:
        target = repo_root / change.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(change.content, encoding="utf-8")
        applied_files.append(change.file)
    return sorted(applied_files)


class SubprocessGitRunner:
    def run(self, args: list[str], cwd: Path) -> None:
        try:
            subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ProviderAdapterError(
                f"Git command failed ({' '.join(args)}): {stderr or exc.returncode}"
            ) from exc


def create_fix_branch(
    plan: BranchCreationPlan,
    repo_path: str,
    git_runner: GitCommandRunner | None = None,
) -> str:
    runner = git_runner or SubprocessGitRunner()
    cwd = Path(repo_path)

    try:
        runner.run(["git", "rev-parse", "--verify", plan.base_ref], cwd=cwd)
        runner.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{plan.pr_branch}"],
            cwd=cwd,
        )
    except ProviderAdapterError as exc:
        message = str(exc)
        # `show-ref --verify --quiet` exits non-zero when the branch is absent.
        if "show-ref" in message:
            pass
        else:
            raise BranchCreationError(f"Unable to verify base reference: {exc}") from exc
    else:
        return plan.pr_branch

    try:
        runner.run(["git", "branch", plan.pr_branch, plan.base_ref], cwd=cwd)
    except ProviderAdapterError as exc:
        raise BranchCreationError(f"Unable to create fix branch '{plan.pr_branch}': {exc}") from exc

    return plan.pr_branch


def _build_commit_message(payload: dict[str, Any], changed_files: Iterable[str]) -> str:
    run_id = str(payload.get("meta", {}).get("run_id", "")).strip() or "unknown-run"
    file_count = len(list(changed_files))
    return f"ci-rootcause: apply evidence-backed fix plan ({run_id}, files={file_count})"


def _build_pr_body(payload: dict[str, Any], changed_files: list[str]) -> str:
    summary = str(payload.get("summary", "")).strip() or "No summary provided"
    classification = str(payload.get("classification", "UNKNOWN")).strip() or "UNKNOWN"
    confidence = float(payload.get("confidence", 0.0))
    run_id = str(payload.get("meta", {}).get("run_id", "unknown-run")).strip() or "unknown-run"
    root_cause = str(payload.get("primary_root_cause", {}).get("title", "")).strip()

    lines = [
        f"ci-rootcause automated fix proposal for run `{run_id}`.",
        "",
        f"- Classification: `{classification}`",
        f"- Confidence: `{confidence:.4f}`",
        f"- Summary: {summary}",
    ]
    if root_cause:
        lines.append(f"- Primary root cause: {root_cause}")

    lines.extend(
        [
            "- Changed files:",
            *[f"  - `{path}`" for path in changed_files],
            "",
            "<!-- ci-rootcause:auto-merge=false -->",
            "<!-- ci-rootcause:respect-branch-protection=true -->",
            f"<!-- ci-rootcause:run-id={run_id} -->",
        ]
    )
    return "\n".join(lines)


def build_pull_request_request(
    payload: dict[str, Any],
    pr_branch: str,
    changed_files: list[str],
) -> PullRequestRequest:
    repository = str(payload.get("repository", "")).strip()
    if "/" not in repository:
        raise GuardrailViolationError("repository must be in 'owner/repo' format")
    owner, repo = repository.split("/", maxsplit=1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        raise GuardrailViolationError("repository must be in 'owner/repo' format")

    base_branch = str(payload.get("target_branch") or payload.get("base_branch") or "main").strip()
    if not base_branch:
        raise GuardrailViolationError("target base branch is required")

    run_id = str(payload.get("meta", {}).get("run_id", "unknown-run")).strip() or "unknown-run"
    title = f"ci-rootcause: suggested fix ({run_id})"
    body = _build_pr_body(payload=payload, changed_files=changed_files)

    return PullRequestRequest(
        owner=owner,
        repo=repo,
        title=title,
        body=body,
        head_branch=pr_branch,
        base_branch=base_branch,
    )


def _enforce_pr_guardrails(payload: dict[str, Any]) -> None:
    if bool(payload.get("auto_merge", False)):
        raise GuardrailViolationError("auto-merge is prohibited for ci-rootcause PRs")
    if bool(payload.get("bypass_branch_protections", False)):
        raise GuardrailViolationError("branch protection bypass is prohibited for ci-rootcause PRs")


def _resolve_min_pr_confidence(payload: dict[str, Any]) -> float:
    raw = payload.get("min_pr_confidence", 0.75)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise GuardrailViolationError(
            "min_pr_confidence must be a float between 0.0 and 1.0"
        ) from exc
    if not (0.0 <= value <= 1.0):
        raise GuardrailViolationError("min_pr_confidence must be between 0.0 and 1.0")
    return value


def checkout_fix_branch(
    plan: BranchCreationPlan,
    repo_path: str,
    git_runner: GitCommandRunner | None = None,
) -> None:
    runner = git_runner or SubprocessGitRunner()
    cwd = Path(repo_path)
    try:
        runner.run(["git", "checkout", plan.pr_branch], cwd=cwd)
    except ProviderAdapterError as exc:
        raise BranchCreationError(
            f"Unable to checkout fix branch '{plan.pr_branch}': {exc}"
        ) from exc


def push_fix_branch(
    plan: BranchCreationPlan,
    repo_path: str,
    git_runner: GitCommandRunner | None = None,
) -> None:
    runner = git_runner or SubprocessGitRunner()
    cwd = Path(repo_path)
    try:
        runner.run(["git", "push", "-u", "origin", plan.pr_branch], cwd=cwd)
    except ProviderAdapterError as exc:
        raise BranchCreationError(
            f"Unable to push fix branch '{plan.pr_branch}' to origin: {exc}"
        ) from exc


def commit_evidence_backed_changes(
    plan: BranchCreationPlan,
    payload: dict[str, Any],
    changed_files: list[str],
    repo_path: str,
    git_runner: GitCommandRunner | None = None,
) -> str:
    runner = git_runner or SubprocessGitRunner()
    cwd = Path(repo_path)
    commit_message = _build_commit_message(payload=payload, changed_files=changed_files)
    sorted_files = sorted(changed_files)

    try:
        runner.run(["git", "add", "--", *sorted_files], cwd=cwd)
        runner.run(["git", "commit", "-m", commit_message], cwd=cwd)
    except ProviderAdapterError as exc:
        raise BranchCreationError(
            f"Unable to commit evidence-backed changes on '{plan.pr_branch}': {exc}"
        ) from exc

    return commit_message


class GitHubRESTClient:
    def __init__(
        self,
        token: str,
        api_base: str = "https://api.github.com",
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        self._token = token.strip()
        if not self._token:
            raise GuardrailViolationError("github_token is required to create PRs")
        self._api_base = api_base.rstrip("/")
        if max_retries < 0:
            raise GuardrailViolationError("max_retries must be >= 0")
        if backoff_seconds < 0.0:
            raise GuardrailViolationError("backoff_seconds must be >= 0.0")
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def _compute_retry_delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after >= 0.0:
            return retry_after
        return self._backoff_seconds * float(2**attempt)

    def _rate_limit_retry_after(self, exc: error.HTTPError) -> float | None:
        retry_after_header = exc.headers.get("Retry-After")
        if retry_after_header:
            try:
                return max(0.0, float(retry_after_header))
            except ValueError:
                return 0.0

        reset_epoch = exc.headers.get("X-RateLimit-Reset")
        if not reset_epoch:
            return None
        try:
            reset_ts = float(reset_epoch)
        except ValueError:
            return 0.0
        return max(0.0, reset_ts - time.time())

    def _is_rate_limited(self, exc: error.HTTPError, message: str) -> bool:
        if exc.code == 429:
            return True
        lowered = message.lower()
        return exc.code == 403 and "rate limit" in lowered

    def _request_once(
        self,
        method: str,
        path: str,
        query: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._api_base}{path}"
        if query:
            url = f"{url}?{parse.urlencode(query)}"
        body = None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url=url, method=method, headers=headers, data=body)
        with request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request(
        self,
        method: str,
        path: str,
        query: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                return self._request_once(method=method, path=path, query=query, payload=payload)
            except error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")
                is_last_attempt = attempt >= self._max_retries

                if self._is_rate_limited(exc=exc, message=message):
                    retry_after = self._rate_limit_retry_after(exc)
                    if is_last_attempt:
                        raise GitHubRateLimitError(
                            f"GitHub API rate limit ({method} {path}): {message}"
                        ) from exc
                    time.sleep(self._compute_retry_delay(attempt=attempt, retry_after=retry_after))
                    continue

                if exc.code in {500, 502, 503, 504}:
                    if is_last_attempt:
                        raise GitHubTransientError(
                            f"GitHub API transient failure ({method} {path}): {message}"
                        ) from exc
                    time.sleep(self._compute_retry_delay(attempt=attempt))
                    continue

                raise GitHubAPIError(
                    f"GitHub API request failed ({method} {path}): {message}"
                ) from exc
            except error.URLError as exc:
                is_last_attempt = attempt >= self._max_retries
                if is_last_attempt:
                    raise GitHubTransientError(
                        f"GitHub API network error ({method} {path}): {exc}"
                    ) from exc
                time.sleep(self._compute_retry_delay(attempt=attempt))

        raise GitHubTransientError(
            f"GitHub API transient failure ({method} {path}) exceeded retries"
        )

    def find_open_pull_request(
        self,
        *,
        owner: str,
        repo: str,
        head_branch: str,
        base_branch: str,
    ) -> dict[str, Any] | None:
        payload = self._request(
            method="GET",
            path=f"/repos/{owner}/{repo}/pulls",
            query={
                "state": "open",
                "head": f"{owner}:{head_branch}",
                "base": base_branch,
                "per_page": "1",
            },
        )
        if payload:
            return dict(payload[0])
        return None

    def create_pull_request(
        self,
        *,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
    ) -> dict[str, Any]:
        return self._request(
            method="POST",
            path=f"/repos/{owner}/{repo}/pulls",
            payload={
                "title": title,
                "head": head_branch,
                "base": base_branch,
                "body": body,
                "draft": False,
                "maintainer_can_modify": False,
            },
        )


def create_or_reuse_pull_request(
    payload: dict[str, Any],
    pr_branch: str,
    changed_files: list[str],
    github_client: GitHubClient,
) -> dict[str, Any]:
    request_payload = build_pull_request_request(
        payload=payload,
        pr_branch=pr_branch,
        changed_files=changed_files,
    )
    existing = github_client.find_open_pull_request(
        owner=request_payload.owner,
        repo=request_payload.repo,
        head_branch=request_payload.head_branch,
        base_branch=request_payload.base_branch,
    )
    if existing is not None:
        return dict(existing)

    return github_client.create_pull_request(
        owner=request_payload.owner,
        repo=request_payload.repo,
        title=request_payload.title,
        body=request_payload.body,
        head_branch=request_payload.head_branch,
        base_branch=request_payload.base_branch,
    )


def find_existing_pull_request(
    payload: dict[str, Any],
    pr_branch: str,
    github_client: GitHubClient,
) -> dict[str, Any] | None:
    request_payload = build_pull_request_request(
        payload=payload,
        pr_branch=pr_branch,
        changed_files=[],
    )
    return github_client.find_open_pull_request(
        owner=request_payload.owner,
        repo=request_payload.repo,
        head_branch=request_payload.head_branch,
        base_branch=request_payload.base_branch,
    )


def run_pr_creation(
    payload: dict[str, Any],
    repo_path: str = ".",
    git_runner: GitCommandRunner | None = None,
    github_client: GitHubClient | None = None,
) -> dict[str, Any]:
    if not bool(payload.get("create_fix_pr", False)):
        return {
            "pr_created": False,
            "pr_url": None,
            "pr_number": None,
            "pr_branch": None,
            "failure_reason": "create_fix_pr=false",
        }
    if bool(payload.get("offline_only", False)):
        return {
            "pr_created": False,
            "pr_url": None,
            "pr_number": None,
            "pr_branch": None,
            "failure_reason": "offline_only=true",
        }

    _enforce_pr_guardrails(payload)
    min_pr_confidence = _resolve_min_pr_confidence(payload)
    confidence = float(payload.get("confidence", 0.0))
    if confidence < min_pr_confidence:
        return {
            "pr_created": False,
            "pr_url": None,
            "pr_number": None,
            "pr_branch": None,
            "failure_reason": (
                f"confidence_below_threshold:{confidence:.4f}<{min_pr_confidence:.4f}"
            ),
        }

    plan = build_branch_creation_plan(payload)

    evidence_files = _extract_evidence_files(payload)
    changes = _extract_validated_changes(payload)
    _validate_changes_against_evidence(changes=changes, evidence_files=evidence_files)

    client = github_client
    if not bool(payload.get("dry_run", False)):
        if client is None:
            token = str(payload.get("github_token", "")).strip()
            client = GitHubRESTClient(token=token)
        existing = find_existing_pull_request(
            payload=payload,
            pr_branch=plan.pr_branch,
            github_client=client,
        )
        if existing is not None:
            return {
                "pr_created": True,
                "pr_url": str(existing["html_url"]),
                "pr_number": int(existing["number"]),
                "pr_branch": plan.pr_branch,
                "failure_reason": None,
                "commit_message": None,
            }

    created_branch = create_fix_branch(plan=plan, repo_path=repo_path, git_runner=git_runner)
    checkout_fix_branch(plan=plan, repo_path=repo_path, git_runner=git_runner)
    changed_files = apply_validated_changes(changes=changes, repo_path=repo_path)
    commit_message = commit_evidence_backed_changes(
        plan=plan,
        payload=payload,
        changed_files=changed_files,
        repo_path=repo_path,
        git_runner=git_runner,
    )

    if bool(payload.get("dry_run", False)):
        return {
            "pr_created": False,
            "pr_url": None,
            "pr_number": None,
            "pr_branch": created_branch,
            "failure_reason": "dry_run=true",
            "commit_message": commit_message,
        }

    push_fix_branch(plan=plan, repo_path=repo_path, git_runner=git_runner)
    pr_payload = create_or_reuse_pull_request(
        payload=payload,
        pr_branch=created_branch,
        changed_files=changed_files,
        github_client=client,
    )

    return {
        "pr_created": True,
        "pr_url": str(pr_payload["html_url"]),
        "pr_number": int(pr_payload["number"]),
        "pr_branch": created_branch,
        "failure_reason": None,
        "commit_message": commit_message,
    }
