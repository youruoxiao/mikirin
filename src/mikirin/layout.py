from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def agent_assets(self) -> Path:
        return self.root / "agent"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def raw_inbox(self) -> Path:
        return self.raw / "inbox"

    @property
    def staging(self) -> Path:
        return self.data / "staging"

    @property
    def rejected(self) -> Path:
        return self.data / "rejected"

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @property
    def manual(self) -> Path:
        return self.data / "manual"

    @property
    def llm_cache(self) -> Path:
        return self.data / "llm_cache"

    @property
    def sessions(self) -> Path:
        return self.data / "sessions"

    @property
    def user_data(self) -> Path:
        return self.data / "user"

    @property
    def user_db(self) -> Path:
        return self.db / "user.db"

    @property
    def db(self) -> Path:
        return self.root / "db"

    @property
    def pipeline_db(self) -> Path:
        return self.manifests / "pipeline.sqlite"

    @property
    def current_pointer(self) -> Path:
        return self.db / "current.json"

    @property
    def building_db(self) -> Path:
        return self.db / "content.building.db"

    def initialize(self) -> None:
        for path in (
            self.raw,
            self.agent_assets,
            self.raw_inbox,
            self.staging,
            self.rejected,
            self.manifests,
            self.manual,
            self.llm_cache,
            self.sessions,
            self.user_data,
            self.db,
        ):
            path.mkdir(parents=True, exist_ok=True)


def discover_root(start: Path | None = None) -> Path:
    """Use a Mikirin source checkout, or the starting directory for installed use.

    An unrelated project's pyproject.toml must not redirect user data. Discovery
    is read-only; installed packages never use site-packages as a data root.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if not (candidate / "src" / "mikirin" / "__init__.py").is_file():
            continue
        try:
            with (candidate / "pyproject.toml").open("rb") as handle:
                metadata = tomllib.load(handle)
        except (OSError, ValueError):
            continue
        project = metadata.get("project")
        if isinstance(project, dict) and project.get("name") == "mikirin":
            return candidate
    return current
