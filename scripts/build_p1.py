"""One-shot, side-build entry point for the currently implemented P1 passes.

The command deliberately keeps the published database immutable until the user
passes ``--publish`` to the ``publish`` or ``all`` command.  Progress is stored
as one atomically replaced JSON document so an interrupted build can be resumed
with the same build id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


# Resolve the checkout's src/ directory even when invoked from another directory
# without an editable install. This changes import resolution only.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SCRIPT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mikirin.atomic import atomic_write_json  # noqa: E402
from mikirin.deepseek import StructuredAPIClient  # noqa: E402
from mikirin.envfile import load_env_file  # noqa: E402
from mikirin.graph.implicit import ImplicitRelationArbitrator  # noqa: E402
from mikirin.graph.store import apply_implicit_results, load_implicit_candidates  # noqa: E402
from mikirin.graph.tasks import TASK_GRAPH_VERSION  # noqa: E402
from mikirin.model_config import ModelProfile  # noqa: E402
from mikirin.p1_build import (  # noqa: E402
    P1BuildConfig,
    P1_PROCESSOR_VERSION,
    BuildPaths,
    deterministic_edge_digest,
    materialize_derived_edges,
    materialize_node_pagerank,
    materialize_task_pagerank,
    p1_release_issues,
    prepare_side_build,
    publish_side_build,
    read_current_database,
    validate_database,
)
from mikirin.schema import CONTENT_SCHEMA_VERSION  # noqa: E402
from mikirin.slang.pipeline import (  # noqa: E402
    SlangPipelineConfig,
    discover_slang,
)
from mikirin.slang.arbitration import SlangArbitrator, parse_json_strict  # noqa: E402
from mikirin.slang.candidates import normalize_for_matching  # noqa: E402
from mikirin.slang.store import apply_arbitration_results, load_arbitration_candidates  # noqa: E402


PROGRESS_SCHEMA_VERSION = 1
STAGE_NAMES = ("prepare", "slang", "graph", "pagerank", "implicit", "validate", "publish")
COMMAND_NAMES = STAGE_NAMES + ("all",)
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _safe_id(value: str, label: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must contain only ASCII letters, digits, '.', '_' or '-' and be at most 128 characters"
        )
    return value


def _safe_run_id(value: str) -> str:
    if not value or any(ord(char) < 32 for char in value):
        raise ValueError("run-id must be a non-empty printable SQLite identifier")
    return value


def _db_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def _relative_name(root: Path, path: Path) -> str:
    """Return a safe database filename for progress metadata."""

    resolved = path.resolve()
    db_root = (root / "db").resolve()
    if resolved.parent != db_root:
        raise ValueError(f"database must be directly inside {db_root}: {path}")
    return resolved.name


def _read_runtime_embedding_contract(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Read only non-secret model metadata.

    The actual credential is intentionally never looked up.  ``key_env`` is an
    environment-variable *name*, not the credential value.
    """

    defaults: dict[str, Any] = {}
    config_path = root / "data" / "manifests" / "runtime_model_config.json"
    if config_path.is_file():
        try:
            raw = _json_load(config_path)
            roles = raw.get("roles", {})
            embedding = roles.get("embedding", {}) if isinstance(roles, dict) else {}
            if isinstance(embedding, dict):
                defaults = embedding
        except (OSError, ValueError, json.JSONDecodeError):
            # Model configuration is not needed by the deterministic passes;
            # implicit remains pending if its future adapter needs it.
            defaults = {}

    model = (
        args.qwen_model
        or os.environ.get("MIKIRIN_QWEN_MODEL")
        or "Qwen/Qwen3-8B"
    )
    base_url = (
        args.siliconflow_base_url
        or os.environ.get("MIKIRIN_SILICONFLOW_BASE_URL")
        or defaults.get("base_url")
        or "https://api.siliconflow.cn/v1"
    )
    key_env = (
        args.siliconflow_key_env
        or os.environ.get("MIKIRIN_SILICONFLOW_KEY_ENV")
        or defaults.get("key_env")
        or "SILICONFLOW_API_KEY"
    )
    return {
        "provider": "siliconflow",
        "model": str(model),
        "base_url": str(base_url),
        "key_env": str(key_env),
        "credential_value_loaded": False,
        "config_path": str(config_path),
    }


def _p1_config(args: argparse.Namespace) -> P1BuildConfig:
    defaults = P1BuildConfig()
    names = (
        "anime_subject_type",
        "staff_degree_cap",
        "cv_degree_cap",
        "staff_min_shared",
        "cv_min_shared",
        "tag_df_cap",
        "tag_min_similarity",
        "same_season_neighbors",
        "edge_top_k",
        "pagerank_damping",
        "pagerank_tolerance",
        "pagerank_iterations",
    )
    values = {
        name: getattr(args, name)
        if getattr(args, name) is not None
        else getattr(defaults, name)
        for name in names
    }
    config = P1BuildConfig(**values)
    config.validate()
    return config


def _slang_config(args: argparse.Namespace) -> SlangPipelineConfig:
    config = SlangPipelineConfig(
        max_subjects=args.slang_max_subjects,
        max_documents_per_subject=args.slang_max_documents_per_subject,
        candidate_limit=args.slang_candidate_limit,
        local_candidate_limit=args.slang_local_candidate_limit,
        incremental_base_run_id=args.incremental_base_run_id,
    )
    config.validate()
    return config


def _content_hash(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_neighbors(payload: Any) -> dict[str, list[dict[str, Any]]]:
    """Frozen JSON contract: normalized candidate -> [{entity_id, score, evidence?}]."""
    if not isinstance(payload, dict):
        raise ValueError("vector neighbors must be a JSON object keyed by candidate text")
    normalized = {}
    for candidate, neighbors in payload.items():
        key = normalize_for_matching(candidate) if isinstance(candidate, str) else ""
        if not key or key in normalized:
            raise ValueError("vector neighbors contain empty or duplicate normalized candidate keys")
        if not isinstance(neighbors, list):
            raise ValueError("each candidate's vector neighbors must be an array")
        seen = set()
        for neighbor in neighbors:
            if not isinstance(neighbor, dict) or not {"entity_id", "score"} <= neighbor.keys():
                raise ValueError("each vector neighbor requires entity_id and score")
            if neighbor.keys() - {"entity_id", "score", "evidence"}:
                raise ValueError("vector neighbor fields must be entity_id, score, and optional evidence")
            entity_id, score = neighbor["entity_id"], neighbor["score"]
            if not isinstance(entity_id, str) or not entity_id.strip() or entity_id != entity_id.strip():
                raise ValueError("vector neighbor entity_id must be nonempty, without surrounding whitespace")
            if entity_id in seen:
                raise ValueError("duplicate vector neighbor entity_id for candidate")
            seen.add(entity_id)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("vector neighbor score must be finite and in [0,1]")
            if "evidence" in neighbor and not isinstance(neighbor["evidence"], dict):
                raise ValueError("vector neighbor evidence must be an object")
        normalized[key] = neighbors
    return normalized


def _load_neighbors_file(path: str | Path) -> tuple[dict, dict]:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = _validate_neighbors(parse_json_strict(raw.decode("utf-8-sig")))
    return payload, {"format_version": 1, "path": str(source), "size_bytes": len(raw),
                     "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                     "content_sha256": _content_hash(payload)}


def _resolve_slang_inputs(args: argparse.Namespace, recorded: dict | None) -> tuple[SlangPipelineConfig, dict | None]:
    requested = _slang_config(args)
    if requested.incremental_base_run_id is not None:
        _safe_run_id(requested.incremental_base_run_id)
    if recorded is None:
        if args.vector_neighbors_file is None:
            return requested, None
        payload, metadata = _load_neighbors_file(args.vector_neighbors_file)
        return replace(requested, vector_neighbors_by_candidate=payload), metadata

    config = SlangPipelineConfig(**recorded["slang"])
    config.validate()
    if args.incremental_base_run_id is not None and args.incremental_base_run_id != config.incremental_base_run_id:
        raise ValueError("--incremental-base-run-id does not match the frozen build; use a new build-id")
    metadata = recorded.get("slang_neighbors_input")
    if metadata is None:
        if args.vector_neighbors_file is not None or config.vector_neighbors_by_candidate:
            raise ValueError("existing build has no frozen neighbors file contract; use a new build-id")
        return config, None
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        raise ValueError("invalid frozen neighbors file contract")
    frozen = _validate_neighbors(config.vector_neighbors_by_candidate)
    if _content_hash(frozen) != metadata.get("content_sha256"):
        raise ValueError("frozen vector neighbors content hash mismatch in progress.json")
    source = args.vector_neighbors_file or metadata.get("path")
    if not isinstance(source, (str, Path)) or not str(source):
        raise ValueError("frozen neighbors source path is missing")
    payload, current = _load_neighbors_file(source)
    if (current["sha256"] != metadata.get("sha256") or current["size_bytes"] != metadata.get("size_bytes")
            or current["content_sha256"] != metadata["content_sha256"] or payload != frozen):
        raise ValueError("frozen vector neighbors file hash/content mismatch; use a new build-id")
    return config, metadata


@dataclass
class BuildContext:
    root: Path
    build_id: str
    version: str
    run_id: str
    progress_path: Path
    progress: dict[str, Any]
    p1_config: P1BuildConfig
    slang_config: SlangPipelineConfig
    model_contract: dict[str, Any]

    @property
    def paths(self) -> BuildPaths | None:
        source_name = self.progress.get("source_database")
        target_name = self.progress.get("target_database")
        if not source_name or not target_name:
            return None
        source = self.root / "db" / str(source_name)
        target = self.root / "db" / str(target_name)
        _relative_name(self.root, source)
        _relative_name(self.root, target)
        return BuildPaths(
            source=source,
            target=target,
            version=self.version,
            build_id=self.build_id,
        )

    def save(self) -> None:
        self.progress["updated_at"] = utc_now()
        atomic_write_json(self.progress_path, self.progress)

    def set_paths(self, paths: BuildPaths) -> None:
        self.progress["source_database"] = _relative_name(self.root, paths.source)
        self.progress["target_database"] = _relative_name(self.root, paths.target)
        self.save()


def _new_progress(
    *,
    root: Path,
    build_id: str,
    version: str,
    run_id: str,
    p1_config: P1BuildConfig,
    slang_config: SlangPipelineConfig,
    model_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "progress_schema_version": PROGRESS_SCHEMA_VERSION,
        "processor_version": P1_PROCESSOR_VERSION,
        "content_schema_version": CONTENT_SCHEMA_VERSION,
        "build_id": build_id,
        "version": version,
        "run_id": run_id,
        "source_database": None,
        "target_database": f"content.v{version}.db",
        "config": {
            "p1": asdict(p1_config),
            "slang": asdict(slang_config),
            "embedding": model_contract,
        },
        "stages": {
            name: {"status": "pending", "attempts": 0} for name in STAGE_NAMES
        },
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "root": str(root),
    }


def _validate_progress(
    progress: dict[str, Any], *, build_id: str, version: str, run_id: str
) -> None:
    if progress.get("progress_schema_version") != PROGRESS_SCHEMA_VERSION:
        raise ValueError("unsupported P1 progress schema version")
    for key, expected in (("build_id", build_id), ("version", version), ("run_id", run_id)):
        if str(progress.get(key)) != expected:
            raise ValueError(f"progress {key} does not match the requested build")
    stages = progress.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("P1 progress has no stages object")
    for name in STAGE_NAMES:
        if not isinstance(stages.get(name), dict):
            stages[name] = {"status": "pending", "attempts": 0}


def _resolve_context(args: argparse.Namespace) -> BuildContext:
    root = Path(args.root).resolve()
    build_id_arg = args.build_id
    existing_progress: dict[str, Any] | None = None
    version_arg = args.version

    if build_id_arg:
        _safe_id(build_id_arg, "build-id")
        candidate = root / "data" / "manifests" / "p1" / build_id_arg / "progress.json"
        if candidate.is_file():
            existing_progress = _json_load(candidate)
            stored_version = str(existing_progress.get("version", ""))
            if version_arg is not None and version_arg != stored_version:
                raise ValueError("--version does not match the existing build progress")
            version_arg = stored_version

    if version_arg is None:
        _source, pointer = read_current_database(root)
        current_version = str(pointer.get("version", "")).strip()
        if not current_version:
            raise ValueError("db/current.json has no version")
        version_arg = f"{current_version}-p1"
    version = _safe_id(str(version_arg), "version")
    build_id = _safe_id(build_id_arg or f"p1-{version}", "build-id")
    progress_path = root / "data" / "manifests" / "p1" / build_id / "progress.json"

    if existing_progress is None and progress_path.is_file():
        existing_progress = _json_load(progress_path)
    p1_config = _p1_config(args)
    recorded_config = existing_progress.get("config") if existing_progress is not None else None
    slang_config, neighbors_input = _resolve_slang_inputs(args, recorded_config)
    model_contract = _read_runtime_embedding_contract(root, args)
    run_id = args.run_id
    if existing_progress is not None:
        stored_run_id = str(existing_progress.get("run_id", ""))
        if run_id is not None and run_id != stored_run_id:
            raise ValueError("--run-id does not match the existing build progress")
        run_id = run_id or stored_run_id
        _validate_progress(existing_progress, build_id=build_id, version=version, run_id=run_id)
    else:
        run_id = run_id or f"slang-{build_id}"
        _safe_run_id(run_id)
        if run_id == slang_config.incremental_base_run_id:
            raise ValueError("incremental base run-id must differ from the new run-id")
        existing_progress = _new_progress(
            root=root,
            build_id=build_id,
            version=version,
            run_id=run_id,
            p1_config=p1_config,
            slang_config=slang_config,
            model_contract=model_contract,
        )
        existing_progress["config"].update(
            {
                "slang_arbitration_limit": args.slang_arbitration_limit,
                "llm_workers": args.llm_workers,
                "implicit_pair_limit": args.implicit_pair_limit,
                "implicit_evidence_per_pair": args.implicit_evidence_per_pair,
                "slang_neighbors_input": neighbors_input,
            }
        )
        atomic_write_json(progress_path, existing_progress)

    if str(existing_progress.get("run_id")) != run_id:
        raise ValueError("progress run id is invalid")
    _safe_run_id(run_id)
    if run_id == slang_config.incremental_base_run_id:
        raise ValueError("incremental base run-id must differ from the new run-id")
    # CLI config is authoritative for a new invocation, but an existing build
    # keeps the recorded values so a resume cannot silently change its contract.
    if existing_progress.get("config"):
        recorded = existing_progress["config"]
        if isinstance(recorded, dict):
            if isinstance(recorded.get("p1"), dict):
                p1_config = P1BuildConfig(**recorded["p1"])
            if isinstance(recorded.get("slang"), dict):
                slang_config = SlangPipelineConfig(**recorded["slang"])
            if isinstance(recorded.get("embedding"), dict):
                model_contract = dict(recorded["embedding"])

    return BuildContext(
        root=root,
        build_id=build_id,
        version=version,
        run_id=run_id,
        progress_path=progress_path,
        progress=existing_progress,
        p1_config=p1_config,
        slang_config=slang_config,
        model_contract=model_contract,
    )


def _target_path(ctx: BuildContext) -> Path:
    target = ctx.root / "db" / f"content.v{ctx.version}.db"
    _relative_name(ctx.root, target)
    return target


def _existing_build_row(database: Path, build_id: str) -> tuple[str, str] | None:
    with closing(sqlite3.connect(_db_uri(database), uri=True)) as db:
        row = db.execute("SELECT id,status FROM build WHERE id=?", (build_id,)).fetchone()
    return None if row is None else (str(row[0]), str(row[1]))


def _prepare(ctx: BuildContext) -> dict[str, Any]:
    stage = ctx.progress["stages"]["prepare"]
    target = _target_path(ctx)
    if stage.get("status") == "completed":
        paths = ctx.paths
        if paths is None or not paths.target.is_file():
            raise FileNotFoundError("completed prepare stage has no side-build database")
        if _existing_build_row(paths.target, ctx.build_id) is None:
            raise RuntimeError("side-build database does not contain this build id")
        return {"resumed": True, "source": paths.source.name, "target": paths.target.name}

    if target.exists():
        row = _existing_build_row(target, ctx.build_id)
        if row is None:
            raise FileExistsError(
                f"side-build target already exists and is not owned by this build: {target}"
            )
        source, _pointer = read_current_database(ctx.root)
        if source.resolve() == target.resolve():
            raise RuntimeError("refusing to use current database as the side-build target")
        paths = BuildPaths(source=source, target=target, version=ctx.version, build_id=ctx.build_id)
        ctx.set_paths(paths)
        return {"resumed": True, "source": source.name, "target": target.name}

    source, _pointer = read_current_database(ctx.root)
    _relative_name(ctx.root, source)
    if source.resolve() == target.resolve():
        raise RuntimeError("refusing to copy current database onto itself")
    paths = prepare_side_build(ctx.root, version=ctx.version, build_id=ctx.build_id)
    ctx.set_paths(paths)
    return {"resumed": False, "source": paths.source.name, "target": paths.target.name}


def _require_prepared(ctx: BuildContext) -> BuildPaths:
    if ctx.progress["stages"]["prepare"].get("status") != "completed":
        raise RuntimeError("prepare stage is not completed; run prepare or all first")
    paths = ctx.paths
    if paths is None or not paths.target.is_file():
        raise FileNotFoundError("P1 side-build database is missing")
    if paths.source.resolve() == paths.target.resolve():
        raise RuntimeError("side-build target resolves to current/source database")
    return paths


def _run_slang(ctx: BuildContext) -> dict[str, Any]:
    paths = _require_prepared(ctx)
    discovery = discover_slang(
        paths.target,
        build_id=ctx.build_id,
        run_id=ctx.run_id,
        config=ctx.slang_config,
    )
    candidates = load_arbitration_candidates(
        paths.target,
        run_id=ctx.run_id,
        limit=int(ctx.progress["config"].get("slang_arbitration_limit", 5_000)),
    )
    if not candidates:
        return {
            "run_id": ctx.run_id,
            "discovery": discovery,
            "arbitrated": 0,
            "alias_application": {
                "confirmed": 0,
                "provisional": 0,
                "rejected": 0,
                "aliases_written": 0,
            },
            "model": ctx.model_contract,
        }
    # The dotenv loader only adds missing variables and never returns or logs
    # their values.  Load it immediately before the first credential lookup.
    profile = ModelProfile(
        provider="siliconflow",
        base_url=str(ctx.model_contract["base_url"]),
        model=str(ctx.model_contract["model"]),
        api_style="chat-completions",
        key_env=str(ctx.model_contract["key_env"]),
    )
    load_env_file(ctx.root / ".env")
    client = StructuredAPIClient(
        api_key=profile.key(),
        base_url=profile.base_url,
        model=profile.model,
        provider=profile.provider,
    )
    arbitrator = SlangArbitrator(
        client=client,
        cache=ctx.root / "data" / "llm_cache",
        model_profile=profile,
        max_tokens=1_024,
        reasoning_effort="low",
        batch_size=5,
        workers=int(ctx.progress["config"].get("llm_workers", 64)),
    )
    decisions = arbitrator.arbitrate_batches(candidates)["decisions"] if candidates else []
    applied = apply_arbitration_results(
        paths.target,
        run_id=ctx.run_id,
        decisions=decisions,
    ) if decisions else {"confirmed": 0, "provisional": 0, "rejected": 0, "aliases_written": 0}
    return {
        "run_id": ctx.run_id,
        "discovery": discovery,
        "arbitrated": len(decisions),
        "alias_application": applied,
        "model": ctx.model_contract,
    }


def _run_graph(ctx: BuildContext) -> dict[str, Any]:
    paths = _require_prepared(ctx)
    derived = materialize_derived_edges(
        paths.target,
        build_id=ctx.build_id,
        config=ctx.p1_config,
    )
    edge_digest = deterministic_edge_digest(paths.target, build_id=ctx.build_id)

    # Graph is one materialization stage: derived edges must be followed by
    # global PageRank and then task-personalized PageRank.  Keep the legacy
    # pagerank stage in progress.json as a compatibility marker so old resume
    # commands and the validation gate still have an explicit checkpoint.
    pagerank = ctx.progress["stages"]["pagerank"]
    pagerank["status"] = "running"
    pagerank["attempts"] = int(pagerank.get("attempts", 0)) + 1
    pagerank["started_at"] = utc_now()
    pagerank.pop("error", None)
    pagerank.pop("result", None)
    pagerank.pop("completed_at", None)
    ctx.save()
    try:
        global_counts = materialize_node_pagerank(
            paths.target,
            build_id=ctx.build_id,
            config=ctx.p1_config,
        )
        task_counts = materialize_task_pagerank(
            paths.target,
            build_id=ctx.build_id,
        )
    except Exception as exc:
        pagerank["status"] = "failed"
        pagerank["error"] = f"{type(exc).__name__}: {exc}"
        pagerank["failed_at"] = utc_now()
        ctx.save()
        raise
    pagerank["status"] = "completed"
    pagerank["result"] = {"global": global_counts, "task_personalized": task_counts,
                          "task_graph_version": TASK_GRAPH_VERSION}
    pagerank["completed_at"] = utc_now()
    ctx.save()
    return {
        "derived": derived,
        "edge_digest": edge_digest,
        "global": global_counts,
        "task_personalized": task_counts,
        "task_graph_version": TASK_GRAPH_VERSION,
    }


def _run_pagerank(ctx: BuildContext) -> dict[str, Any]:
    if ctx.progress["stages"]["graph"].get("status") != "completed":
        raise RuntimeError("graph stage is not completed; run graph or all first")
    paths = _require_prepared(ctx)
    global_counts = materialize_node_pagerank(
        paths.target,
        build_id=ctx.build_id,
        config=ctx.p1_config,
    )
    task_counts = materialize_task_pagerank(
        paths.target,
        build_id=ctx.build_id,
    )
    return {"global": global_counts, "task_personalized": task_counts,
            "task_graph_version": TASK_GRAPH_VERSION}


def _run_implicit(ctx: BuildContext) -> dict[str, Any]:
    paths = _require_prepared(ctx)
    if ctx.progress["stages"]["slang"].get("status") != "completed":
        raise RuntimeError("slang stage is not completed; run slang or all first")
    candidates = load_implicit_candidates(
        paths.target,
        run_id=ctx.run_id,
        pair_limit=int(ctx.progress["config"].get("implicit_pair_limit", 500)),
        evidence_per_pair=int(ctx.progress["config"].get("implicit_evidence_per_pair", 3)),
    )
    if not candidates:
        return {
            "candidates": 0,
            "decisions": 0,
            "materialized": {"accepted_evidence": 0, "rejected": 0, "implicit_edges": 0},
            "model": ctx.model_contract,
        }
    # Pass A is local candidate loading.  Only after it is complete do we load
    # the dotenv file and resolve the credential for Pass B.
    profile = ModelProfile(
        provider="siliconflow",
        base_url=str(ctx.model_contract["base_url"]),
        model=str(ctx.model_contract["model"]),
        api_style="chat-completions",
        key_env=str(ctx.model_contract["key_env"]),
    )
    load_env_file(ctx.root / ".env")
    client = StructuredAPIClient(
        api_key=profile.key(),
        base_url=profile.base_url,
        model=profile.model,
        provider=profile.provider,
    )
    arbitrator = ImplicitRelationArbitrator(
        client=client,
        cache=ctx.root / "data" / "llm_cache",
        model_profile=profile,
        max_tokens=512,
        reasoning_effort="low",
        workers=int(ctx.progress["config"].get("llm_workers", 64)),
    )
    decisions = arbitrator.arbitrate_many(candidates) if candidates else []
    materialized = apply_implicit_results(
        paths.target,
        run_id=ctx.run_id,
        decisions=decisions,
    ) if decisions else {"accepted_evidence": 0, "rejected": 0, "implicit_edges": 0}
    return {
        "candidates": len(candidates),
        "decisions": len(decisions),
        "materialized": materialized,
        "model": ctx.model_contract,
    }


def _stage_status(ctx: BuildContext, name: str) -> str:
    return str(ctx.progress["stages"][name].get("status", "pending"))


_DOWNSTREAM_STAGES: dict[str, tuple[str, ...]] = {
    "prepare": ("slang", "graph", "pagerank", "implicit", "validate", "publish"),
    "slang": ("graph", "pagerank", "implicit", "validate", "publish"),
    "graph": ("pagerank", "implicit", "validate", "publish"),
    "pagerank": ("validate", "publish"),
    "implicit": ("validate", "publish"),
    "validate": ("publish",),
    "publish": (),
}


def _invalidate_downstream(ctx: BuildContext, name: str) -> None:
    for downstream in _DOWNSTREAM_STAGES[name]:
        stage = ctx.progress["stages"][downstream]
        if stage.get("status") == "pending":
            continue
        stage["status"] = "pending"
        for key in ("result", "error", "started_at", "completed_at", "failed_at", "pending_at"):
            stage.pop(key, None)


def _release_report(ctx: BuildContext, validation: dict[str, Any] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stages": {"slang": {"run_id": ctx.run_id}},
        "user_acceptance": ctx.progress.get("user_acceptance", {}),
    }
    if validation is not None:
        report["validation"] = validation
    return report


def _run_release_gate(ctx: BuildContext, validation: dict[str, Any] | None = None) -> list[str]:
    paths = _require_prepared(ctx)
    issues = p1_release_issues(
        paths.target,
        build_id=ctx.build_id,
        report=_release_report(ctx, validation),
    )
    if issues:
        raise RuntimeError("P1 release gate failed: " + "; ".join(issues))
    return issues


def _validation(ctx: BuildContext) -> dict[str, Any]:
    paths = _require_prepared(ctx)
    database_report = validate_database(paths.target)
    checks: dict[str, Any] = {
        "integrity": database_report.get("integrity") == "ok",
        "foreign_keys": not database_report.get("foreign_key_errors"),
        "schema_version": int(database_report.get("schema_version") or 0)
        == CONTENT_SCHEMA_VERSION,
        "side_build": paths.source.resolve() != paths.target.resolve(),
        "build_row": False,
    }
    build_row = _existing_build_row(paths.target, ctx.build_id)
    checks["build_row"] = build_row is not None and build_row[0] == ctx.build_id

    required = ("prepare", "slang", "graph", "pagerank")
    missing = [name for name in required if _stage_status(ctx, name) != "completed"]
    implicit_status = _stage_status(ctx, "implicit")
    if implicit_status != "completed":
        missing.append("implicit")

    current_pointer: dict[str, Any] | None = None
    try:
        current_database, current_pointer = read_current_database(ctx.root)
        checks["current_pointer_readable"] = current_database.is_file()
        checks["target_not_current"] = (
            current_database.resolve() != paths.target.resolve()
            or (build_row is not None and build_row[1] == "published")
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks["current_pointer_readable"] = False
        checks["target_not_current"] = False

    database_ok = all(bool(value) for value in checks.values())
    release_issues = p1_release_issues(
        paths.target,
        build_id=ctx.build_id,
        report=_release_report(ctx),
    )
    publishable = database_ok and not missing and not release_issues
    return {
        "database": database_report,
        "checks": checks,
        "missing_stages": missing,
        "pending_stages": list(missing),
        "release_issues": release_issues,
        "publishable": publishable,
        "current_pointer": current_pointer,
    }


def _run_validation(ctx: BuildContext) -> dict[str, Any]:
    return _validation(ctx)


def _run_publish(ctx: BuildContext, *, requested: bool) -> dict[str, Any]:
    paths = _require_prepared(ctx)
    validation = ctx.progress["stages"]["validate"].get("result")
    if not isinstance(validation, dict) or validation.get("publishable") is not True:
        raise RuntimeError("publish requires a successful full validation")
    # Do not rely solely on the implementation detail that publish_side_build
    # currently calls the gate.  This command-level check keeps a mocked or
    # replaced publisher from ever bypassing acceptance requirements.
    _run_release_gate(ctx, validation)
    if not requested:
        return {
            "_stage_status": "pending",
            "published": False,
            "publishable": True,
            "reason": "explicit --publish is required to switch db/current.json",
        }

    published = publish_side_build(
        ctx.root,
        paths,
        report={
            "build_id": ctx.build_id,
            **_release_report(ctx, validation),
            "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "pending_stages": validation.get("pending_stages", []),
        },
    )
    current_database, pointer = read_current_database(ctx.root)
    if current_database.resolve() != published.resolve() or pointer.get("database") != published.name:
        raise RuntimeError("publish completed but db/current.json does not point to the side build")
    return {"published": True, "database": published.name, "current": pointer}


def _run_stage(
    ctx: BuildContext,
    name: str,
    operation: Callable[[], dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    stage = ctx.progress["stages"][name]
    if stage.get("status") == "completed" and not force:
        stage["last_action"] = "resumed_completed"
        ctx.save()
        result = stage.get("result")
        return result if isinstance(result, dict) else {}

    _invalidate_downstream(ctx, name)
    stage["status"] = "running"
    stage["attempts"] = int(stage.get("attempts", 0)) + 1
    stage["started_at"] = utc_now()
    stage.pop("error", None)
    stage.pop("result", None)
    stage.pop("completed_at", None)
    ctx.save()
    try:
        outcome = operation()
    except Exception as exc:
        stage["status"] = "failed"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        stage["failed_at"] = utc_now()
        ctx.save()
        raise
    status = str(outcome.get("_stage_status", "completed"))
    outcome = {key: value for key, value in outcome.items() if key != "_stage_status"}
    stage["status"] = status
    stage["result"] = outcome
    if status == "completed":
        stage["completed_at"] = utc_now()
    else:
        stage["pending_at"] = utc_now()
    ctx.save()
    return outcome


def _add_common_arguments(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--root", default=default, help="project root (default: repository root)")
    parser.add_argument("--version", default=default, help="side-build version")
    parser.add_argument("--build-id", default=default, help="stable progress/build identifier")
    parser.add_argument("--run-id", default=default, help="Slang run identifier")
    parser.add_argument("--incremental-base-run-id", "--slang-incremental-base-run-id",
                        dest="incremental_base_run_id", default=default,
                        help="completed compatible slang run in the cloned database; frozen on first invocation")
    parser.add_argument("--vector-neighbors-file", "--slang-vector-neighbors-file",
                        dest="vector_neighbors_file", default=default,
                        help="UTF-8 JSON candidate -> [{entity_id,score,evidence?}]; content and SHA-256 frozen in progress; checked on every resume")
    parser.add_argument(
        "--publish",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="allow publish/all to atomically switch db/current.json",
    )

    parser.add_argument("--qwen-model", default=default, help="Qwen model name")
    parser.add_argument("--siliconflow-base-url", default=default, help="SiliconFlow base URL")
    parser.add_argument(
        "--siliconflow-key-env",
        "--siliconflow-api-key-env",
        dest="siliconflow_key_env",
        default=default,
        help="environment variable name containing the SiliconFlow credential (never read here)",
    )

    p1_defaults = P1BuildConfig()
    parser.add_argument("--anime-subject-type", type=int, default=default, help="anime subject type")
    parser.add_argument("--staff-degree-cap", type=int, default=default, help=f"default: {p1_defaults.staff_degree_cap}")
    parser.add_argument("--cv-degree-cap", type=int, default=default, help=f"default: {p1_defaults.cv_degree_cap}")
    parser.add_argument("--staff-min-shared", type=int, default=default, help=f"default: {p1_defaults.staff_min_shared}")
    parser.add_argument("--cv-min-shared", type=int, default=default, help=f"default: {p1_defaults.cv_min_shared}")
    parser.add_argument("--tag-df-cap", type=int, default=default, help=f"default: {p1_defaults.tag_df_cap}")
    parser.add_argument("--tag-min-similarity", type=float, default=default, help=f"default: {p1_defaults.tag_min_similarity}")
    parser.add_argument("--same-season-neighbors", type=int, default=default, help=f"default: {p1_defaults.same_season_neighbors}")
    parser.add_argument("--edge-top-k", type=int, default=default, help=f"default: {p1_defaults.edge_top_k}")
    parser.add_argument("--pagerank-damping", type=float, default=default, help=f"default: {p1_defaults.pagerank_damping}")
    parser.add_argument("--pagerank-tolerance", type=float, default=default, help=f"default: {p1_defaults.pagerank_tolerance}")
    parser.add_argument("--pagerank-iterations", type=int, default=default, help=f"default: {p1_defaults.pagerank_iterations}")
    parser.add_argument("--slang-max-subjects", type=int, default=default, help="optional bounded Slang run")
    parser.add_argument("--slang-candidate-limit", type=int, default=argparse.SUPPRESS if suppress_defaults else 5000)
    parser.add_argument("--slang-local-candidate-limit", type=int, default=argparse.SUPPRESS if suppress_defaults else 5000)
    parser.add_argument("--slang-arbitration-limit", type=int, default=argparse.SUPPRESS if suppress_defaults else 5000)
    parser.add_argument("--llm-workers", type=int, default=argparse.SUPPRESS if suppress_defaults else 64)
    parser.add_argument("--implicit-pair-limit", type=int, default=argparse.SUPPRESS if suppress_defaults else 500)
    parser.add_argument("--implicit-evidence-per-pair", type=int, default=argparse.SUPPRESS if suppress_defaults else 3)
    parser.add_argument(
        "--slang-max-documents-per-subject",
        type=int,
        default=default,
        help="optional per-subject Slang document bound",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build implemented P1 passes in a side database with atomic progress.",
        epilog="Use the same --build-id to resume. Only publish/all --publish may switch db/current.json.",
    )
    _add_common_arguments(parser, suppress_defaults=False)
    parser.set_defaults(root=str(SCRIPT_ROOT))
    subparsers = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    for name in COMMAND_NAMES:
        subparser = subparsers.add_parser(name, help=f"run the {name} stage")
        _add_common_arguments(subparser, suppress_defaults=True)
    return parser


def _final_report(ctx: BuildContext, *, command: str, ok: bool, error: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": ok,
        "command": command,
        "build_id": ctx.build_id,
        "version": ctx.version,
        "run_id": ctx.run_id,
        "source_database": ctx.progress.get("source_database"),
        "target_database": ctx.progress.get("target_database"),
        "progress": str(ctx.progress_path),
        "stages": ctx.progress.get("stages", {}),
        "current_switch_allowed": command in ("publish", "all"),
    }
    if error is not None:
        report["error"] = error
    return report


def _task_graph_current(ctx: BuildContext) -> bool:
    stage = ctx.progress["stages"]["pagerank"]
    result = stage.get("result", {})
    return (stage.get("status") == "completed" and isinstance(result, dict)
            and result.get("task_graph_version") == TASK_GRAPH_VERSION)


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.publish and args.stage not in ("publish", "all"):
        raise ValueError("--publish is valid only with the publish or all stage")
    ctx = _resolve_context(args)
    setattr(args, "_p1_context", ctx)
    command = args.stage

    if command == "prepare":
        _run_stage(ctx, "prepare", lambda: _prepare(ctx))
    elif command == "slang":
        _run_stage(ctx, "slang", lambda: _run_slang(ctx))
    elif command == "graph":
        # A pre-existing build from the split-stage implementation may have a
        # completed graph marker but no completed pagerank marker.  Rebuild the
        # composite stage once so resume cannot validate incomplete graph data.
        _run_stage(
            ctx,
            "graph",
            lambda: _run_graph(ctx),
            force=not _task_graph_current(ctx),
        )
    elif command == "pagerank":
        _run_stage(ctx, "pagerank", lambda: _run_pagerank(ctx), force=not _task_graph_current(ctx))
    elif command == "implicit":
        _run_stage(ctx, "implicit", lambda: _run_implicit(ctx))
    elif command == "validate":
        validation = _run_stage(ctx, "validate", lambda: _run_validation(ctx), force=True)
        ok = bool(validation.get("publishable"))
        return (0 if ok else 1), _final_report(ctx, command=command, ok=ok)
    elif command == "publish":
        _run_stage(ctx, "validate", lambda: _run_validation(ctx), force=True)
        _run_stage(ctx, "publish", lambda: _run_publish(ctx, requested=args.publish))
    elif command == "all":
        _run_stage(ctx, "prepare", lambda: _prepare(ctx))
        _run_stage(ctx, "slang", lambda: _run_slang(ctx))
        _run_stage(
            ctx,
            "graph",
            lambda: _run_graph(ctx),
            force=not _task_graph_current(ctx),
        )
        _run_stage(ctx, "implicit", lambda: _run_implicit(ctx))
        _run_stage(ctx, "validate", lambda: _run_validation(ctx), force=True)
        _run_stage(ctx, "publish", lambda: _run_publish(ctx, requested=args.publish))
    else:  # argparse choices make this unreachable.
        raise ValueError(f"unknown stage: {command}")

    validation_result = ctx.progress["stages"]["validate"].get("result", {})
    ok = True
    if command in ("all", "publish") and _stage_status(ctx, "validate") == "completed":
        ok = bool(validation_result.get("publishable"))
    return (0 if ok else 1), _final_report(ctx, command=command, ok=ok)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ctx: BuildContext | None = None
    try:
        code, report = run(args)
    except Exception as exc:
        # Context may not exist if argument/path resolution failed.  Always emit
        # one JSON object rather than a traceback so shell callers can parse it.
        ctx = getattr(args, "_p1_context", None)
        if ctx is not None:
            report = _final_report(
                ctx,
                command=args.stage,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            report = {
                "ok": False,
                "command": getattr(args, "stage", None),
                "error": f"{type(exc).__name__}: {exc}",
            }
        code = 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
