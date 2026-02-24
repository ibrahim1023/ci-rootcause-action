from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ProviderResolution:
    ci_provider: str
    provider_adapter: str
    repository: str
    target_branch: str


def _read_env(environ: Mapping[str, str], key: str) -> str:
    return str(environ.get(key, "")).strip()


def resolve_provider_defaults(
    environ: Mapping[str, str] | None = None,
) -> ProviderResolution:
    env = os.environ if environ is None else environ

    if _read_env(env, "GITLAB_CI").lower() == "true":
        return ProviderResolution(
            ci_provider="gitlab-ci",
            provider_adapter="gitlab",
            repository=_read_env(env, "CI_PROJECT_PATH"),
            target_branch=(
                _read_env(env, "CI_MERGE_REQUEST_TARGET_BRANCH_NAME")
                or _read_env(env, "CI_DEFAULT_BRANCH")
                or "main"
            ),
        )

    if _read_env(env, "GITHUB_ACTIONS").lower() == "true":
        return ProviderResolution(
            ci_provider="github-actions",
            provider_adapter="github",
            repository=_read_env(env, "GITHUB_REPOSITORY"),
            target_branch=(
                _read_env(env, "GITHUB_BASE_REF") or _read_env(env, "GITHUB_REF_NAME") or "main"
            ),
        )

    return ProviderResolution(
        ci_provider="github-actions",
        provider_adapter="github",
        repository="",
        target_branch="main",
    )
