from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import os
from contextlib import closing
from datetime import date
from pathlib import Path

from mikirin.assets import RawAssetStore, load_raw_asset_manifest
from mikirin.downloads import AssemblyPlan
from mikirin.ids import authoritative_entity_id, provisional_entity_id
from mikirin.layout import ProjectLayout, discover_root
from mikirin.pipeline import PipelineStore
from mikirin.publish import Publisher
from mikirin.quality import QualityInspector
from mikirin.atomic import atomic_write_json
from mikirin.range_fetch import HttpRangeFragmentFetcher, RangeSpec
from mikirin.search_index import SearchIndexer
from mikirin.query import KnowledgeQueries
from mikirin.chat import KnowledgeChatSession, RemoteAPIChatClient
from mikirin.tui import run_tui
from mikirin.envfile import load_env_file
from mikirin.model_config import ModelConfigStore
from mikirin.cache import ModelCache
from mikirin.deepseek import StructuredAPIClient
from mikirin.organize import ResolutionOrganizer, default_run_id
from mikirin.run_lock import exclusive_process_lock
from mikirin.sources.bgm_archive import ArchiveInventory
from mikirin.sources.bgm_loader import BgmStagingLoader
from mikirin.sources.bgm_staging import BgmArchiveStager
from mikirin.sources.atwiki import AtwikiCrawler
from mikirin.sources.atwiki_staging import AtwikiStager
from mikirin.sources.moegirl import MoegirlDatasetFetcher
from mikirin.sources.moegirl_staging import MoegirlStager
from mikirin.sources.supplementary_loader import SupplementaryStagingLoader
from mikirin.sources.community_insights_loader import CommunityInsightsLoader
from mikirin.sources.bgm_selection import BgmSampleDiscoverer
from mikirin.sources.bgm_crawl import BgmContentCrawler
from mikirin.sources.bgm_web_staging import BgmWebStager


def _layout(root: str | None) -> ProjectLayout:
    project_root = Path(root).resolve() if root else discover_root()
    return ProjectLayout(project_root)


def command_init(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    layout.initialize()
    PipelineStore(layout.pipeline_db).initialize()
    print(f"Mikirin 已初始化: {layout.root}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    pipeline = PipelineStore(layout.pipeline_db)
    pipeline.initialize()
    current = Publisher(layout).current()
    payload = {
        "project": str(layout.root),
        "current_database": str(current) if current else None,
        "pipeline_tasks": [
            {
                "task_key": task.task_key,
                "source": task.source,
                "state": task.state,
                "input_version": task.input_version,
                "processor_version": task.processor_version,
            }
            for task in pipeline.list_tasks()
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_demo(args: argparse.Namespace) -> int:
    from mikirin.demo import create_demo

    layout = _layout(args.root)
    database = create_demo(layout)
    print(f"演示库已就绪：{database}")
    print("内含三部虚构作品；无需下载或 API key。试试 search \"放映部\" 或 tui --browse-only。")
    return 0


def command_search(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    database = Path(args.database).resolve() if args.database else Publisher(layout).current()
    if database is None:
        raise FileNotFoundError("尚无知识库。先运行 mikirin demo 试用，或导入自己的数据。")
    hits = KnowledgeQueries.search_entities(database, args.query, limit=args.limit)
    print(json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False, indent=2))
    return 0


def command_tui(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    layout.initialize()
    load_env_file(layout.root / ".env")
    database = Publisher(layout).current()
    if database is None:
        raise FileNotFoundError("尚无知识库。先运行 mikirin demo 试用，或导入自己的数据。")
    if args.model or args.base_url or args.api_style:
        from mikirin.model_config import ModelProfile

        store = ModelConfigStore(layout.manifests / "runtime_model_config.json")
        profile = store.load()["chat"]
        custom_endpoint = bool(args.base_url and args.base_url.rstrip("/") != profile.base_url.rstrip("/"))
        store.set_custom("chat", ModelProfile(
            provider="custom" if custom_endpoint else profile.provider,
            base_url=args.base_url or profile.base_url,
            model=args.model or profile.model,
            api_style=args.api_style or profile.api_style,
            key_env="MIKIRIN_API_KEY" if custom_endpoint else profile.key_env,
        ))
    return run_tui(layout=layout, database=database, browse_only=args.browse_only,
                   retrieval_mode=getattr(args, 'retrieval_mode', None))


def _candidate_ids_from_file(path: str | None) -> list[str] | None:
    """从 resolve-rules 的 needs-llm.jsonl 或纯文本 ID 列表读取候选 ID。"""
    if path is None:
        return None
    candidate_ids: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                candidate_ids.append(json.loads(line)["candidate_id"])
            else:
                candidate_ids.append(line)
    return candidate_ids


def _discover_resolution_compiled(layout: ProjectLayout) -> Path | None:
    """自动发现最新一次 resolve-rules 产出的 resolution-compiled.jsonl。"""
    rules_root = layout.staging / "resolution_rules"
    if not rules_root.exists():
        return None
    candidates = sorted(rules_root.glob("rules-*/resolution-compiled.jsonl"))
    return candidates[-1] if candidates else None


def _run_entity_merge(
    database: Path,
    layout: ProjectLayout,
    *,
    build_id: str,
    resolution_compiled: str | None,
    manual_resolution: str | None,
) -> dict[str, object] | None:
    """装载完成后、FTS 重建前执行 A2 实体合并（§1.6）。

    无裁决输入（显式或自动发现）则跳过，保持非合并构建路径可用。
    黄金集硬门禁失败会抛异常中止发布。
    """
    from mikirin.merge_entities import apply_merges, load_merge_map, verify_golden

    compiled_path = (
        Path(resolution_compiled).resolve()
        if resolution_compiled
        else _discover_resolution_compiled(layout)
    )
    if compiled_path is None or not compiled_path.exists():
        return None
    manual_path = (
        Path(manual_resolution).resolve()
        if manual_resolution
        else layout.manual / "entity_resolution.jsonl"
    )
    plan = load_merge_map(
        database, resolution_compiled=compiled_path, manual_resolution=manual_path
    )
    report = apply_merges(
        database,
        plan,
        build_id=build_id,
        rejected_dir=layout.rejected,
        quality_dir=layout.manifests / "quality",
    )
    golden = verify_golden(database, manual_resolution=manual_path, plan=plan)
    payload = report.to_dict()
    payload["golden"] = golden.to_dict()
    payload["awareness"] = list(plan.awareness)
    payload["unresolved_candidates"] = list(plan.unresolved_candidates)
    payload["resolution_compiled"] = str(compiled_path)
    payload["manual_resolution"] = str(manual_path) if manual_path.exists() else None
    return payload


def command_organize_resolution(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    layout.initialize()
    load_env_file(layout.root / ".env")
    database = Publisher(layout).current()
    if database is None:
        raise FileNotFoundError("No published content database")
    profile = ModelConfigStore(layout.manifests / "runtime_model_config.json").load()[
        "organizer"
    ]
    client = None
    if not args.dry_run:
        client = StructuredAPIClient(
            api_key=profile.key(),
            base_url=profile.base_url,
            model=profile.model,
            provider=profile.provider,
        )
    with exclusive_process_lock(layout.manifests / "organizer.lock"):
        candidate_ids = _candidate_ids_from_file(args.candidate_file)
        limit = args.limit
        if candidate_ids is not None:
            limit = max(limit, len(candidate_ids))
        report = ResolutionOrganizer(
            database=database,
            staging_root=layout.staging,
            cache=ModelCache(layout.llm_cache),
            client=client,
            model_name=profile.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            workers=args.workers,
        ).run(
            limit=limit,
            run_id=args.run_id or default_run_id(),
            dry_run=args.dry_run,
            candidate_ids=candidate_ids,
            persons_per_batch=args.persons_per_batch,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_compile_resolution(args: argparse.Namespace) -> int:
    from mikirin.resolution_apply import (
        compile_resolution,
        load_llm_decisions,
        load_rule_decisions,
    )

    layout = _layout(args.root)
    layout.initialize()
    rule_path = (
        layout.staging / "resolution_rules" / args.rules_run / "rule-decisions.jsonl"
    )
    llm_dir = layout.staging / "deepseek_resolution" / args.llm_run
    compiled, review_queue, summary = compile_resolution(
        load_rule_decisions(rule_path), load_llm_decisions(llm_dir)
    )
    out_dir = layout.staging / "resolution_rules" / args.rules_run
    with (out_dir / "resolution-compiled.jsonl").open("w", encoding="utf-8") as handle:
        for item in compiled:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out_dir / "review-queue.jsonl").open("w", encoding="utf-8") as handle:
        for item in review_queue:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary["rules_run"] = args.rules_run
    summary["llm_run"] = args.llm_run
    summary["canonical_database_modified"] = False
    atomic_write_json(out_dir / "resolution-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_resolve_rules(args: argparse.Namespace) -> int:
    from mikirin.resolve_rules import (
        decide_all,
        evaluate,
        load_candidate_features,
        load_labeled_decisions,
    )

    layout = _layout(args.root)
    layout.initialize()
    database = Publisher(layout).current()
    if database is None:
        raise FileNotFoundError("No published content database")
    run_id = args.run_id or default_run_id()
    out_dir = layout.staging / "resolution_rules" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    features = load_candidate_features(database)
    decisions = decide_all(features)
    with (out_dir / "rule-decisions.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")
    needs_llm = [d for d in decisions if d.decision == "needs_llm"]
    with (out_dir / "needs-llm.jsonl").open("w", encoding="utf-8") as handle:
        for decision in needs_llm:
            handle.write(
                json.dumps(
                    {
                        "candidate_id": decision.candidate_id,
                        "matched_evidence": decision.matched_evidence,
                        "conflicting_evidence": decision.conflicting_evidence,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    labels = load_labeled_decisions(layout.staging)
    report = evaluate(decisions, labels)
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.decision] = counts.get(decision.decision, 0) + 1
    report.update(
        {
            "run_id": run_id,
            "candidate_total": len(decisions),
            "decision_counts": counts,
            "needs_llm_file": str(out_dir / "needs-llm.jsonl"),
            "canonical_database_modified": False,
        }
    )
    atomic_write_json(out_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_escalate_resolution(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    layout.initialize()
    load_env_file(layout.root / ".env")
    database = Publisher(layout).current()
    if database is None:
        raise FileNotFoundError("No published content database")
    if Path(args.source_run).name != args.source_run:
        raise ValueError("source_run must be a run ID, not a path")
    source_path = (
        layout.staging / "deepseek_resolution" / args.source_run / "decisions.json"
    )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    candidate_ids = [
        item["candidate_id"]
        for item in payload.get("decisions", [])
        if item.get("review_required")
    ]
    if args.include_failed:
        batches_dir = (
            layout.staging / "deepseek_resolution" / args.source_run / "batches"
        )
        seen = set(candidate_ids)
        for error_path in sorted(batches_dir.glob("*.error.json")):
            error = json.loads(error_path.read_text(encoding="utf-8"))
            for candidate_id in error.get("expected_candidate_ids", []):
                if candidate_id not in seen:
                    candidate_ids.append(candidate_id)
                    seen.add(candidate_id)
    candidate_ids = candidate_ids[: args.limit]
    if not candidate_ids:
        raise RuntimeError("source run contains no review_required candidates")
    profile = ModelConfigStore(layout.manifests / "runtime_model_config.json").load()[
        "organizer"
    ]
    client = None
    if not args.dry_run:
        client = StructuredAPIClient(
            api_key=profile.key(),
            base_url=profile.base_url,
            model=args.model,
            provider=profile.provider,
        )
    run_id = args.run_id or f"pro-review-{args.source_run}"[:80]
    with exclusive_process_lock(layout.manifests / "organizer.lock"):
        report = ResolutionOrganizer(
            database=database,
            staging_root=layout.staging,
            cache=ModelCache(layout.llm_cache),
            client=client,
            model_name=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            workers=args.workers,
        ).run(
            limit=len(candidate_ids),
            run_id=run_id,
            dry_run=args.dry_run,
            candidate_ids=candidate_ids,
            persons_per_batch=args.persons_per_batch,
        )
    report["source_run"] = args.source_run
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_review_resolution(args: argparse.Namespace) -> int:
    from mikirin.manual_review import run_review

    layout = _layout(args.root)
    layout.initialize()
    queue_path = Path(args.queue or (
        layout.staging / "resolution_rules" / "rules-20260727" / "human-review-queue.json"
    ))
    return run_review(layout, queue_path, args.min_collects)


def command_settle_session(args: argparse.Namespace) -> int:
    from mikirin.agent.types import AssistantMessage, UserMessage
    from mikirin.settlement import MemorySettlement
    from mikirin.session import SessionStore

    layout = _layout(args.root)
    layout.initialize()
    load_env_file(layout.root / ".env")
    store = SessionStore(layout.sessions)
    session_id = args.session_id or store.latest()
    if session_id is None:
        raise RuntimeError("没有可结算的 session")
    messages: list[dict[str, str]] = []
    for message in store.replay_messages(session_id):
        if isinstance(message, UserMessage):
            messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage) and message.text:
            messages.append({"role": "assistant", "content": message.text})
    profiles = ModelConfigStore(
        layout.manifests / "runtime_model_config.json"
    ).load()
    flash_profile = profiles["organizer"]
    pro_profile = profiles["chat"]
    from mikirin.structured_role import client_for_profile

    settlement = MemorySettlement(
        layout.user_db,
        client_for_profile(flash_profile, flash_profile.key()),
        client_for_profile(pro_profile, pro_profile.key()),
    )
    report = settlement.settle(session_id, messages, dry_run=args.dry_run)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def command_publish_empty(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    publisher = Publisher(layout)
    publisher.create_empty_build(build_id=f"build-{args.version}", code_version=args.code_version)
    target = publisher.publish(version=args.version)
    print(f"已发布空白验证库: {target}")
    return 0


def command_ingest_file(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    asset = RawAssetStore(layout).ingest(
        Path(args.path),
        source=args.source,
        source_version=args.source_version,
        acquired_on=date.fromisoformat(args.acquired_on) if args.acquired_on else None,
        license_id=args.license_id,
        redistributable=args.redistributable,
        attribution_required=args.attribution_required,
        commercial_use_allowed=args.commercial_use_allowed,
    )
    pipeline = PipelineStore(layout.pipeline_db)
    pipeline.initialize()
    task_key = f"asset:{asset.id}"
    created = pipeline.discover(
        task_key=task_key,
        source=asset.source,
        input_version=asset.source_version,
        processor_version="raw-ingest-v1",
        metadata={"asset_id": asset.id, "relative_path": asset.relative_path},
    )
    if created:
        from mikirin.pipeline import TaskState

        pipeline.transition(task_key, TaskState.FETCHED)
    print(json.dumps({"asset": asset.id, "path": asset.relative_path}, ensure_ascii=False))
    return 0


def command_inspect_bgm_archive(args: argparse.Namespace) -> int:
    inventory = ArchiveInventory.inspect(Path(args.path))
    payload: dict[str, object] = {
        "archive": str(inventory.archive_path),
        "members": inventory.members,
        "missing_required": inventory.missing_required,
    }
    if args.count:
        payload["record_counts"] = inventory.count_records()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if inventory.missing_required else 0


def command_stage_bgm_archive(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    report = BgmArchiveStager(layout).stage(
        Path(args.path),
        source_version=args.source_version,
        source_asset_id=args.source_asset_id,
    )
    print(
        json.dumps(
            {
                "source_version": report.source_version,
                "source_asset_id": report.source_asset_id,
                "counts": report.counts,
                "rejected": report.rejected,
                "output_directory": report.output_directory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_build_bgm_content(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    layout = _layout(args.root)
    asset = load_raw_asset_manifest(Path(args.asset_manifest))
    publisher = Publisher(layout)
    build_id = f"build-{args.version}"
    database = publisher.create_empty_build(build_id=build_id, code_version=args.code_version)
    try:
        report = BgmStagingLoader().load(
            staging_directory=layout.staging / "bgm_archive" / args.source_version,
            database_path=database,
            build_id=build_id,
            asset=asset,
            workers=args.workers,
        )
        counts: dict[str, object] = {"bgm_archive": report.counts}
        supplementary = SupplementaryStagingLoader()
        if args.moegirl_version:
            moegirl_report = supplementary.load_moegirl(
                staging_directory=layout.staging
                / "moegirl_dataset"
                / args.moegirl_version,
                database_path=database,
                source_version=args.moegirl_version,
                build_id=build_id,
            )
            counts["moegirl_dataset"] = moegirl_report.counts
        if args.atwiki_version:
            atwiki_report = supplementary.load_atwiki(
                staging_directory=layout.staging / "atwiki" / args.atwiki_version,
                database_path=database,
                source_version=args.atwiki_version,
            )
            counts["atwiki_sakuga"] = atwiki_report.counts
        if args.bgm_web_version:
            bgm_web_report = supplementary.load_bgm_web(
                staging_directory=layout.staging / "bgm_web" / args.bgm_web_version,
                database_path=database,
                source_version=args.bgm_web_version,
                workers=args.workers,
            )
            counts["bangumi_web"] = bgm_web_report.counts
        insights = CommunityInsightsLoader()
        community_briefs_run = getattr(args, "community_briefs_run", None)
        if community_briefs_run:
            brief_report = insights.load_community_briefs(
                staging_directory=layout.staging / "community_briefs" / community_briefs_run,
                database_path=database,
                build_id=build_id,
            )
            counts["community_reports"] = brief_report.counts
        discussion_maps_run = getattr(args, "discussion_maps_run", None)
        if discussion_maps_run:
            theme_report = insights.load_discussion_maps(
                staging_directory=layout.staging / "discussion_maps" / discussion_maps_run,
                database_path=database,
                build_id=build_id,
            )
            counts["discussion_themes"] = theme_report.counts
        merge_report = _run_entity_merge(
            database,
            layout,
            build_id=build_id,
            resolution_compiled=args.resolution_compiled,
            manual_resolution=args.manual_resolution,
        )
        if merge_report is not None:
            counts["entity_merge"] = merge_report
        if args.defer_publish:
            print(
                json.dumps(
                    {
                        "database": str(database),
                        "build_id": build_id,
                        "counts": counts,
                        "status": "deferred",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        search_report = SearchIndexer.rebuild(database)
        counts["search_index"] = {
            "entities": search_report.entities,
            "documents": search_report.documents,
            "chunks": search_report.chunks,
        }
        Publisher.validate(database)
        quality = QualityInspector.inspect(database)
        QualityInspector.assert_publishable(quality)
        quality_path = layout.manifests / "quality" / f"{args.version}.json"
        atomic_write_json(quality_path, quality.to_dict())
        target = publisher.publish(
            version=args.version, keep=getattr(args, "keep_versions", 3)
        )
        published_quality = quality.to_dict()
        published_quality["database"] = str(target)
        atomic_write_json(quality_path, published_quality)
    except BaseException:
        database.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "database": str(target),
                "build_id": build_id,
                "counts": counts,
                "quality_report": str(quality_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_finalize_content(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    layout = _layout(args.root)
    database = layout.building_db
    if not database.exists():
        raise FileNotFoundError(f"没有待完成的旁路构建: {database}")
    with closing(sqlite3.connect(database)) as db:
        row = db.execute("SELECT id FROM build LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("旁路构建缺少 build 记录")
    build_id = str(row[0])
    counts: dict[str, object] = {}
    supplementary = SupplementaryStagingLoader()
    if args.atwiki_version:
        report = supplementary.load_atwiki(
            staging_directory=layout.staging / "atwiki" / args.atwiki_version,
            database_path=database,
            source_version=args.atwiki_version,
        )
        counts["atwiki_sakuga"] = report.counts
    if args.bgm_web_version:
        report = supplementary.load_bgm_web(
            staging_directory=layout.staging / "bgm_web" / args.bgm_web_version,
            database_path=database,
            source_version=args.bgm_web_version,
            workers=args.workers,
        )
        counts["bangumi_web"] = report.counts
    insights = CommunityInsightsLoader()
    community_briefs_run = getattr(args, "community_briefs_run", None)
    if community_briefs_run:
        report = insights.load_community_briefs(
            staging_directory=layout.staging / "community_briefs" / community_briefs_run,
            database_path=database,
            build_id=build_id,
        )
        counts["community_reports"] = report.counts
    discussion_maps_run = getattr(args, "discussion_maps_run", None)
    if discussion_maps_run:
        report = insights.load_discussion_maps(
            staging_directory=layout.staging / "discussion_maps" / discussion_maps_run,
            database_path=database,
            build_id=build_id,
        )
        counts["discussion_themes"] = report.counts
    merge_report = _run_entity_merge(
        database,
        layout,
        build_id=build_id,
        resolution_compiled=args.resolution_compiled,
        manual_resolution=args.manual_resolution,
    )
    if merge_report is not None:
        counts["entity_merge"] = merge_report
    search = SearchIndexer.rebuild(database)
    counts["search_index"] = {
        "entities": search.entities,
        "documents": search.documents,
        "chunks": search.chunks,
    }
    Publisher.validate(database)
    quality = QualityInspector.inspect(database)
    QualityInspector.assert_publishable(quality)
    quality_path = layout.manifests / "quality" / f"{args.version}.json"
    atomic_write_json(quality_path, quality.to_dict())
    target = Publisher(layout).publish(
        version=args.version, keep=getattr(args, "keep_versions", 3)
    )
    published_quality = quality.to_dict()
    published_quality["database"] = str(target)
    atomic_write_json(quality_path, published_quality)
    print(
        json.dumps(
            {
                "database": str(target),
                "build_id": build_id,
                "counts": counts,
                "quality_report": str(quality_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_assemble_download(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).resolve()
    plan = AssemblyPlan.from_json(plan_path, root=plan_path.parent)
    destination = plan.assemble(Path(args.output).resolve())
    print(f"分段已校验并合并: {destination}")
    return 0


def command_fetch_range(args: argparse.Namespace) -> int:
    destination = HttpRangeFragmentFetcher(
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout,
    ).fetch(
        url=args.url,
        spec=RangeSpec(args.start, args.end),
        destination=Path(args.output),
    )
    print(
        json.dumps(
            {
                "path": str(destination),
                "start": args.start,
                "end": args.end,
                "bytes": destination.stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    return 0


def command_crawl_atwiki(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    crawler = AtwikiCrawler(layout, delay_seconds=args.delay, max_attempts=args.max_attempts)
    pages = crawler.discover() if args.refresh_queue or not crawler.queue_path.exists() else crawler.load_queue()
    if args.discover_only:
        print(json.dumps({"discovered": len(pages), "queue": str(crawler.queue_path)}, ensure_ascii=False))
        return 0
    report = crawler.crawl(pages, limit=args.limit)
    print(json.dumps({"discovered": len(pages), **report}, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


def command_stage_atwiki(args: argparse.Namespace) -> int:
    report = AtwikiStager(_layout(args.root)).stage(source_version=args.source_version)
    print(
        json.dumps(
            {
                "source_version": report.source_version,
                "pages": report.pages,
                "empty_bodies": report.empty_bodies,
                "output_directory": report.output_directory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_fetch_moegirl(args: argparse.Namespace) -> int:
    report = MoegirlDatasetFetcher(
        _layout(args.root), max_attempts=args.max_attempts
    ).fetch_all(limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


def command_stage_moegirl(args: argparse.Namespace) -> int:
    report = MoegirlStager(_layout(args.root)).stage(source_version=args.source_version)
    print(
        json.dumps(
            {
                "source_version": report.source_version,
                "characters": report.characters,
                "mapping_candidates": report.mapping_candidates,
                "output_directory": report.output_directory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_discover_bgm_sample(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    output = BgmSampleDiscoverer(
        layout,
        root_url=args.base_url,
        delay_seconds=args.delay,
        max_attempts=args.max_attempts,
    ).discover(
        start_year=args.start_year,
        end_year=args.end_year,
        yearly_limit=args.yearly_limit,
        rank_limit=args.rank_limit,
        user_ids=args.user,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({
        "selection": str(output),
        "subjects": payload["subject_count"],
        "assets": payload["asset_count"],
    }, ensure_ascii=False, indent=2))
    return 0


def _selection_path(layout: ProjectLayout, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    pointer = json.loads(
        (layout.manifests / "bgm_web" / "current-selection.json").read_text(encoding="utf-8")
    )
    return layout.root / pointer["path"]


def command_seed_bgm_content(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    crawler = BgmContentCrawler(layout, delay_seconds=args.delay)
    with exclusive_process_lock(layout.manifests / "bgm-crawler.lock"):
        crawler.seed(_selection_path(layout, args.selection))
        reconciled = crawler.reconcile_archived_review_pages()
        report = crawler.status()
        report["reconciled_review_page_ones"] = reconciled
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_crawl_bgm_content(args: argparse.Namespace) -> int:
    layout = _layout(args.root)
    crawler = BgmContentCrawler(
        layout,
        delay_seconds=args.delay,
        max_attempts=args.max_attempts,
        short_comment_page_cap=args.short_comment_pages,
    )
    with exclusive_process_lock(layout.manifests / "bgm-crawler.lock"):
        if args.workers > 1:
            report = crawler.crawl_parallel(workers=args.workers, limit=args.limit)
        else:
            report = crawler.crawl(limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_bgm_content_status(args: argparse.Namespace) -> int:
    report = BgmContentCrawler(_layout(args.root)).status()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_stage_bgm_web(args: argparse.Namespace) -> int:
    report = BgmWebStager(_layout(args.root)).stage(
        source_version=args.source_version,
        require_complete=args.require_complete,
    )
    print(json.dumps({
        "source_version": report.source_version,
        "documents": report.documents,
        "documents_hash": report.documents_hash,
        "documents_bytes": report.documents_bytes,
        "document_kinds": report.document_kinds,
        "missing_author_hashes": report.missing_author_hashes,
        "source_pages": report.source_pages,
        "empty_pages": report.empty_pages,
        "duplicates": report.duplicates,
        "pending_pages": report.pending_pages,
        "partial": report.partial,
        "output_directory": report.output_directory,
    }, ensure_ascii=False, indent=2))
    return 0


def command_id_authoritative(args: argparse.Namespace) -> int:
    print(authoritative_entity_id(args.source, args.entity_type, args.source_id))
    return 0


def command_id_provisional(args: argparse.Namespace) -> int:
    print(
        provisional_entity_id(
            source=args.source,
            entity_type=args.entity_type,
            source_record_id=args.source_record_id,
            record_locator=args.record_locator,
            raw_name=args.raw_name,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mikirin", description="Mikirin 数据平台 CLI")
    parser.add_argument("--root", help="数据工作目录；默认使用当前 Mikirin 项目或当前目录")
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init", help="初始化项目本地目录和状态库")
    init_parser.set_defaults(handler=command_init)

    demo_parser = commands.add_parser("demo", help="创建可离线试用的虚构演示知识库")
    demo_parser.set_defaults(handler=command_demo)

    status_parser = commands.add_parser("status", help="查看发布版本和管线任务")
    status_parser.set_defaults(handler=command_status)

    search_parser = commands.add_parser("search", help="查询已发布知识库的实体名和别名")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--database", help="显式指定content.db；默认使用current.json")
    search_parser.set_defaults(handler=command_search)

    tui_parser = commands.add_parser("tui", help="启动知识库增强的 API 对话终端")
    tui_parser.add_argument("--model", help="覆盖并保存 chat 模型名；默认沿用 /models 配置")
    tui_parser.add_argument("--base-url", help="覆盖 chat API 地址；新地址使用 MIKIRIN_API_KEY")
    tui_parser.add_argument(
        "--api-style",
        choices=("responses", "chat-completions"),
        help="覆盖 chat 接口类型；默认沿用已保存配置",
    )
    tui_parser.add_argument(
        "--browse-only", action="store_true", help="不调用 API，仅浏览本地数据库"
    )
    tui_parser.set_defaults(handler=command_tui)
    tui_parser.add_argument('--retrieval-mode', choices=('fts', 'anchored', 'evidence', 'hybrid'),
                            help='仅本次会话启用实验检索；未设置时保持稳定路径')

    organize_parser = commands.add_parser(
        "organize-resolution", help="Run a cached, reviewable character-resolution batch"
    )
    organize_parser.add_argument("--limit", type=int, default=20)
    organize_parser.add_argument("--batch-size", type=int, default=5)
    organize_parser.add_argument("--max-tokens", type=int, default=4096)
    organize_parser.add_argument(
        "--candidate-file",
        help="只裁决文件列出的候选（resolve-rules 的 needs-llm.jsonl 或每行一个 ID）",
    )
    organize_parser.add_argument(
        "--persons-per-batch",
        type=int,
        help="按萌百角色分组后每批人数（同一人候选同批裁决，避免跨批矛盾）",
    )
    organize_parser.add_argument(
        "--workers", type=int, default=1, help="并发批次数（DeepSeek 并发提速）"
    )
    organize_parser.add_argument("--run-id")
    organize_parser.add_argument("--dry-run", action="store_true")
    organize_parser.set_defaults(handler=command_organize_resolution)

    rules_parser = commands.add_parser(
        "resolve-rules", help="确定性预裁决全部角色候选，输出灰色地带清单"
    )
    rules_parser.add_argument("--run-id")
    rules_parser.set_defaults(handler=command_resolve_rules)

    compile_parser = commands.add_parser(
        "compile-resolution", help="合并规则层与 LLM 裁决为统一构建输入"
    )
    compile_parser.add_argument("rules_run", help="resolve-rules 的 run-id")
    compile_parser.add_argument("llm_run", help="organize-resolution 的 run-id")
    compile_parser.set_defaults(handler=command_compile_resolution)

    escalate_parser = commands.add_parser(
        "escalate-resolution", help="Send only review_required decisions to a stronger model"
    )
    escalate_parser.add_argument("source_run")
    escalate_parser.add_argument("--model", default="deepseek-v4-pro")
    escalate_parser.add_argument("--limit", type=int, default=500)
    escalate_parser.add_argument("--batch-size", type=int, default=5)
    escalate_parser.add_argument("--max-tokens", type=int, default=4096)
    escalate_parser.add_argument("--workers", type=int, default=1)
    escalate_parser.add_argument("--persons-per-batch", type=int)
    escalate_parser.add_argument(
        "--include-failed",
        action="store_true",
        help="把 source_run 失败批次里无裁决的候选一并升级",
    )
    escalate_parser.add_argument("--run-id")
    escalate_parser.add_argument("--dry-run", action="store_true")
    escalate_parser.set_defaults(handler=command_escalate_resolution)

    review_parser = commands.add_parser("review-resolution", help="人工复核消歧队列")
    review_parser.add_argument("--queue", help="队列 JSON 路径（默认 human-review-queue.json）")
    review_parser.add_argument("--min-collects", type=int, default=20)
    review_parser.set_defaults(handler=command_review_resolution)

    settle_parser = commands.add_parser("settle-session", help="结算一个 session：抽取记忆卡片并裁决冲突")
    settle_parser.add_argument("--session-id", help="默认最近一次 session")
    settle_parser.add_argument("--dry-run", action="store_true")
    settle_parser.set_defaults(handler=command_settle_session)

    publish_parser = commands.add_parser("publish-empty", help="构建并发布空白验证数据库")
    publish_parser.add_argument("--version", required=True)
    publish_parser.add_argument("--code-version", default="working-tree")
    publish_parser.set_defaults(handler=command_publish_empty)

    ingest_parser = commands.add_parser("ingest-file", help="归档一个已经取得的原始文件")
    ingest_parser.add_argument("source")
    ingest_parser.add_argument("source_version")
    ingest_parser.add_argument("path")
    ingest_parser.add_argument("--acquired-on", help="YYYY-MM-DD；默认今天（UTC）")
    ingest_parser.add_argument("--license-id", default="unknown")
    ingest_parser.add_argument("--redistributable", action="store_true")
    ingest_parser.add_argument("--attribution-required", action="store_true")
    ingest_parser.add_argument("--commercial-use-allowed", action="store_true")
    ingest_parser.set_defaults(handler=command_ingest_file)

    inspect_parser = commands.add_parser(
        "inspect-bgm-archive", help="识别并校验 BGM Archive 压缩包"
    )
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--count", action="store_true", help="逐行解析并统计全部记录")
    inspect_parser.set_defaults(handler=command_inspect_bgm_archive)

    stage_parser = commands.add_parser(
        "stage-bgm-archive", help="把 BGM Archive 规范化为 staging JSONL"
    )
    stage_parser.add_argument("path")
    stage_parser.add_argument("source_version")
    stage_parser.add_argument("source_asset_id")
    stage_parser.set_defaults(handler=command_stage_bgm_archive)

    build_bgm_parser = commands.add_parser(
        "build-bgm-content", help="从 BGM staging 构建并原子发布 content.db"
    )
    build_bgm_parser.add_argument("source_version")
    build_bgm_parser.add_argument("asset_manifest")
    build_bgm_parser.add_argument("--version", required=True)
    build_bgm_parser.add_argument("--code-version", default="working-tree")
    build_bgm_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行预处理进程数；SQLite 写入始终保持单进程（默认 1）",
    )
    build_bgm_parser.add_argument(
        "--keep-versions", type=int, default=3, help="发布后保留的 content.db 版本数"
    )
    build_bgm_parser.add_argument(
        "--defer-publish", action="store_true", help="保留旁路构建，稍后补源并发布"
    )
    build_bgm_parser.add_argument(
        "--moegirl-version", help="同时装载已完成的萌娘 staging 提交版本"
    )
    build_bgm_parser.add_argument(
        "--atwiki-version", help="同时装载已完成的作画 Wiki staging 日期版本"
    )
    build_bgm_parser.add_argument(
        "--bgm-web-version", help="同时装载已完成的 Bangumi 社区文本 staging 版本"
    )
    build_bgm_parser.add_argument(
        "--community-briefs-run", help="物化 A3.3 community_briefs staging run-id"
    )
    build_bgm_parser.add_argument(
        "--discussion-maps-run", help="物化 A3.6 discussion_maps staging run-id"
    )
    build_bgm_parser.add_argument(
        "--resolution-compiled",
        help="A2 实体合并裁决输入；默认自动发现最新 rules-*/resolution-compiled.jsonl",
    )
    build_bgm_parser.add_argument(
        "--manual-resolution",
        help="人工裁决 JSONL（最高优先级）；默认 data/manual/entity_resolution.jsonl",
    )
    build_bgm_parser.set_defaults(handler=command_build_bgm_content)

    finalize_parser = commands.add_parser(
        "finalize-content", help="给旁路构建补充来源、重建检索索引并原子发布"
    )
    finalize_parser.add_argument("--version", required=True)
    finalize_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行预处理进程数；SQLite 写入始终保持单进程（默认 1）",
    )
    finalize_parser.add_argument(
        "--keep-versions", type=int, default=3, help="发布后保留的 content.db 版本数"
    )
    finalize_parser.add_argument("--atwiki-version")
    finalize_parser.add_argument("--bgm-web-version")
    finalize_parser.add_argument(
        "--community-briefs-run", help="物化 A3.3 community_briefs staging run-id"
    )
    finalize_parser.add_argument(
        "--discussion-maps-run", help="物化 A3.6 discussion_maps staging run-id"
    )
    finalize_parser.add_argument(
        "--resolution-compiled",
        help="A2 实体合并裁决输入；默认自动发现最新 rules-*/resolution-compiled.jsonl",
    )
    finalize_parser.add_argument(
        "--manual-resolution",
        help="人工裁决 JSONL（最高优先级）；默认 data/manual/entity_resolution.jsonl",
    )
    finalize_parser.set_defaults(handler=command_finalize_content)

    assemble_parser = commands.add_parser(
        "assemble-download", help="校验连续字节分段并按官方 SHA-256 合并"
    )
    assemble_parser.add_argument("plan")
    assemble_parser.add_argument("output")
    assemble_parser.set_defaults(handler=command_assemble_download)

    range_parser = commands.add_parser(
        "fetch-range", help="断线后从精确字节边界自动续传单个大文件分片"
    )
    range_parser.add_argument("url")
    range_parser.add_argument("start", type=int)
    range_parser.add_argument("end", type=int)
    range_parser.add_argument("output")
    range_parser.add_argument("--max-attempts", type=int, default=100)
    range_parser.add_argument("--timeout", type=float, default=90.0)
    range_parser.set_defaults(handler=command_fetch_range)

    atwiki_parser = commands.add_parser(
        "crawl-atwiki", help="发现并断点归档作画 Wiki 页面"
    )
    atwiki_parser.add_argument("--delay", type=float, default=2.0)
    atwiki_parser.add_argument("--max-attempts", type=int, default=5)
    atwiki_parser.add_argument("--limit", type=int, help="仅抓取本轮前 N 个未完成页面")
    atwiki_parser.add_argument("--refresh-queue", action="store_true")
    atwiki_parser.add_argument("--discover-only", action="store_true")
    atwiki_parser.set_defaults(handler=command_crawl_atwiki)

    stage_atwiki_parser = commands.add_parser(
        "stage-atwiki", help="把完整的作画 Wiki raw 快照转为规范 JSONL"
    )
    stage_atwiki_parser.add_argument("source_version")
    stage_atwiki_parser.set_defaults(handler=command_stage_atwiki)

    moegirl_parser = commands.add_parser(
        "fetch-moegirl", help="按固定Git提交逐文件下载并校验萌娘数据集"
    )
    moegirl_parser.add_argument("--limit", type=int)
    moegirl_parser.add_argument("--max-attempts", type=int, default=8)
    moegirl_parser.set_defaults(handler=command_fetch_moegirl)

    bgm_sample_parser = commands.add_parser(
        "discover-bgm-sample", help="Archive Bangumi ranking pages and build a fixed subject sample"
    )
    bgm_sample_parser.add_argument("--start-year", type=int, default=2011)
    bgm_sample_parser.add_argument("--end-year", type=int, default=2025)
    bgm_sample_parser.add_argument("--yearly-limit", type=int, default=50)
    bgm_sample_parser.add_argument("--rank-limit", type=int, default=250)
    bgm_sample_parser.add_argument("--user", action="append", default=[])
    bgm_sample_parser.add_argument("--base-url", default="https://bangumi.tv")
    bgm_sample_parser.add_argument("--delay", type=float, default=1.0)
    bgm_sample_parser.add_argument("--max-attempts", type=int, default=5)
    bgm_sample_parser.set_defaults(handler=command_discover_bgm_sample)

    seed_bgm_parser = commands.add_parser(
        "seed-bgm-content", help="Seed resumable episode/comment/review queues from a fixed sample"
    )
    seed_bgm_parser.add_argument("--selection")
    seed_bgm_parser.add_argument("--delay", type=float, default=1.0)
    seed_bgm_parser.set_defaults(handler=command_seed_bgm_content)

    crawl_bgm_parser = commands.add_parser(
        "crawl-bgm-content", help="Run the resumable single-writer Bangumi content crawler"
    )
    crawl_bgm_parser.add_argument("--limit", type=int)
    crawl_bgm_parser.add_argument("--delay", type=float, default=1.0)
    crawl_bgm_parser.add_argument("--max-attempts", type=int, default=5)
    crawl_bgm_parser.add_argument("--short-comment-pages", type=int, default=8)
    crawl_bgm_parser.add_argument("--workers", "-w", type=int, default=1,
                                  help="Number of parallel workers (default 1, use 3-4 for faster episode crawl)")
    crawl_bgm_parser.set_defaults(handler=command_crawl_bgm_content)

    bgm_status_parser = commands.add_parser(
        "bgm-content-status", help="Show Bangumi content crawl queue counts"
    )
    bgm_status_parser.set_defaults(handler=command_bgm_content_status)

    stage_bgm_web_parser = commands.add_parser(
        "stage-bgm-web",
        help="Normalize archived Bangumi comments and reviews into immutable JSONL",
    )
    stage_bgm_web_parser.add_argument("source_version")
    stage_bgm_web_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Refuse to create a staging snapshot until all content pages finish",
    )
    stage_bgm_web_parser.set_defaults(handler=command_stage_bgm_web)

    stage_moegirl_parser = commands.add_parser(
        "stage-moegirl", help="生成萌娘角色、别名、属性和BGM匹配候选JSONL"
    )
    stage_moegirl_parser.add_argument("source_version")
    stage_moegirl_parser.set_defaults(handler=command_stage_moegirl)

    ids_parser = commands.add_parser("id", help="生成稳定实体 ID")
    id_commands = ids_parser.add_subparsers(dest="id_kind", required=True)

    auth_parser = id_commands.add_parser("authoritative")
    auth_parser.add_argument("source")
    auth_parser.add_argument("entity_type")
    auth_parser.add_argument("source_id")
    auth_parser.set_defaults(handler=command_id_authoritative)

    provisional_parser = id_commands.add_parser("provisional")
    provisional_parser.add_argument("source")
    provisional_parser.add_argument("entity_type")
    provisional_parser.add_argument("source_record_id")
    provisional_parser.add_argument("record_locator")
    provisional_parser.add_argument("raw_name")
    provisional_parser.set_defaults(handler=command_id_provisional)
    from mikirin.feature_cli import register_commands

    register_commands(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"mikirin: {error}", file=sys.stderr)
        return 1
