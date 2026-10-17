"""Opt-in research and companion commands, separate from ingestion plumbing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from mikirin.envfile import load_env_file
from mikirin.layout import ProjectLayout, discover_root
from mikirin.publish import Publisher
from mikirin.query import KnowledgeQueries


def layout_for(args: argparse.Namespace) -> ProjectLayout:
    return ProjectLayout(Path(args.root).resolve() if args.root else discover_root())


def database_for(args: argparse.Namespace) -> Path:
    path = Path(args.database).resolve() if getattr(args, 'database', None) else Publisher(layout_for(args)).current()
    if path is None or not path.is_file():
        raise FileNotFoundError('尚无知识库，请先运行 demo 或指定 --database。')
    return path


def show(value: object) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def read_records(path: str) -> object:
    source = Path(path)
    if source.suffix.lower() == '.csv':
        with source.open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))
    return json.loads(source.read_text(encoding='utf-8'))


def output_report(value: object, path: str | None) -> int:
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Never silently replace human-labelled review artifacts.
        with destination.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        return show({'output': str(destination.resolve())})
    return show(value)


def p1_review(args: argparse.Namespace) -> int:
    from mikirin import p1_evaluation as evaluation

    action = args.review_action
    if action == 'graph-prepare':
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            raise FileExistsError('请指定新的审核目录，避免覆盖已填写的标签。')
        database = database_for(args)
        queries = evaluation.load_graph_queries(args.queries)
        for row in queries:
            hits = KnowledgeQueries.search_entities(database, row['query'], limit=8)
            if not row.get('seed_entity_ids'):
                row['seed_entity_ids'] = [hit.entity_id for hit in hits]
            if 'lexical_candidates' not in row:
                row['lexical_candidates'] = [
                    {'entity_id': hit.entity_id, 'score': hit.retrieval_weight / (i + 1)}
                    for i, hit in enumerate(hits)
                ]
        fingerprint = None
        if args.vectors:
            from mikirin.semantic_search import SemanticEpisodeSearch

            layout = layout_for(args)
            load_env_file(layout.root / '.env')
            semantic = SemanticEpisodeSearch(layout.root, database)
            try:
                if not semantic.available:
                    raise RuntimeError(f'向量索引不可用：{semantic.status()}')
                fingerprint = semantic.index.index_fingerprint
                for row in queries:
                    row['vector_candidates'] = [
                        {'entity_id': hit.subject_entity_id, 'score': hit.score}
                        for hit in semantic.search(row['query'], top_k=8)
                    ]
            finally:
                semantic.close()
        result = evaluation.run_graph_comparison(
            database, queries, build_id=args.build_id, output_dir=output_dir,
            seed=args.seed, index_fingerprint=fingerprint,
        )
        return show({'output_dir': str(output_dir.resolve()), 'metrics': result['metrics']})
    if action == 'graph-score':
        result = evaluation.evaluate_graph_ratings(read_records(args.blind), read_records(args.private))
    elif action == 'slang-calibrate':
        result = evaluation.calibrate_slang_thresholds(read_records(args.input), threshold_version=args.version)
    elif action == 'slang-score':
        database = database_for(args)
        result = evaluation.evaluate_slang_golden(
            read_records(args.input), search=lambda text: KnowledgeQueries.search_entities(database, text, limit=8),
        )
    elif action == 'implicit-prepare':
        result = evaluation.prepare_implicit_review(database_for(args), build_id=args.build_id, seed=args.seed)
    elif action == 'implicit-score':
        result = evaluation.evaluate_implicit_review(read_records(args.input))
    else:
        result = evaluation.prepare_monthly_alias_review(database_for(args), month=args.month, seed=args.seed)
    return output_report(result, args.output)


def rss(args: argparse.Namespace) -> int:
    from mikirin.proactive import ProactiveService, fetch_http_feed

    service = ProactiveService(layout_for(args).user_db,
                               fetcher=fetch_http_feed if args.rss_action == 'poll' else None)
    action = args.rss_action
    if action == 'subscribe':
        # Resolve a unique work before saving consent; ambiguous names need an ID.
        database = database_for(args)
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
            row = db.execute("SELECT id FROM entity WHERE id=? AND entity_type='subject'", (args.subject,)).fetchone()
        ids = [row[0]] if row else [hit.entity_id for hit in KnowledgeQueries.search_entities(database, args.subject)
                                   if hit.entity_type == 'subject' and hit.match_kind in {'exact', 'canonical'}]
        if len(ids) != 1:
            raise ValueError('订阅对象不唯一，请先搜索，再提供作品实体 ID。')
        return show(asdict(service.subscribe(ids[0], args.url)))
    if action == 'unsubscribe':
        return show({'unsubscribed': service.unsubscribe(args.subscription_id)})
    if action == 'configure':
        return show(asdict(service.configure(enabled=None if args.enabled is None else args.enabled == 'on',
                                             probability=args.probability)))
    if action == 'poll':
        result = service.poll()
        show(asdict(result))
        return int(bool(result.errors))
    if action == 'deliver':
        printed = False
        def display(topic):
            nonlocal printed
            # One machine-readable result, flushed before acknowledging.
            print(json.dumps({'delivered': [asdict(topic)]}, ensure_ascii=False), flush=True)
            printed = True
        result = service.deliver(display)
        if not printed:
            show(asdict(result))
        elif result.failed:
            print('RSS 显示后的确认失败，队列将保留重试。', file=sys.stderr)
        return int(bool(result.failed))
    return show({'settings': asdict(service.status()),
                 'subscriptions': [asdict(item) for item in service.subscriptions()]})


def memory_queue(args: argparse.Namespace) -> int:
    from mikirin.settlement import MemorySettlement

    layout = layout_for(args)
    settlement = MemorySettlement(layout.user_db)
    if args.queue_action == 'status':
        return show(settlement.queue_status())
    if args.queue_action == 'enqueue':
        from mikirin.agent.types import UserMessage, AssistantMessage
        from mikirin.session import SessionStore

        store = SessionStore(layout.sessions)
        session_id = args.session_id or store.latest()
        if session_id is None:
            raise ValueError('没有可结算的会话。')
        messages = []
        for message in store.replay_messages(session_id):
            if isinstance(message, UserMessage):
                messages.append({'role': 'user', 'content': message.content})
            elif isinstance(message, AssistantMessage) and message.text:
                messages.append({'role': 'assistant', 'content': message.text})
        if not any(message['role'] == 'user' for message in messages):
            raise ValueError('会话没有用户消息，不入队。')
        return show({'job_id': settlement.enqueue(session_id, messages)})
    from mikirin.model_config import ModelConfigStore
    from mikirin.deepseek import StructuredAPIClient

    if not 1 <= args.limit <= 100:
        raise ValueError('limit must be between 1 and 100')
    if not any(job['status'] != 'done' for job in settlement.queue_status()):
        return show({'reports': [], 'queue': settlement.queue_status()})
    load_env_file(layout.root / '.env')
    profiles = ModelConfigStore(layout.manifests / 'runtime_model_config.json').load()
    from mikirin.structured_role import client_for_profile

    clients = [client_for_profile(profiles[role], profiles[role].key()) for role in ('organizer', 'chat')]
    settlement = MemorySettlement(layout.user_db, *clients)
    reports = settlement.run_pending(limit=args.limit)
    show({'reports': [report.to_dict() for report in reports], 'queue': settlement.queue_status()})
    return int(any(job['status'] == 'failed' for job in settlement.queue_status()))


def memory_review(args: argparse.Namespace) -> int:
    from mikirin.memory_review import export_weekly_review, review_report

    if args.memory_review_action == 'prepare':
        output = Path(args.output)
        report = export_weekly_review(layout_for(args).user_db, output, week=args.week)
        return show({'output': str(output.resolve()), 'sample_size': report['sample_size']})
    return output_report(review_report(Path(args.input)), args.output)


def retrieval(args: argparse.Namespace) -> int:
    from mikirin.hybrid_retrieval import HybridRetriever, HybridSemanticSearch

    if getattr(args, 'output', None) and Path(args.output).exists():
        raise FileExistsError('报告已存在，请指定新的 --output。')
    database = database_for(args)
    semantic = None
    if args.vectors:
        layout = layout_for(args)
        load_env_file(layout.root / '.env')
        semantic = HybridSemanticSearch(layout.root, database)
    try:
        disambiguator = None
        classifier = None
        if args.disambiguate or args.stance_classify:
            from mikirin.disambiguation import ClosedCandidateDisambiguator, ClosedStanceClassifier
            from mikirin.cache import ModelCache
            from mikirin.deepseek import StructuredAPIClient
            from mikirin.model_config import ModelConfigStore

            layout = layout_for(args)
            load_env_file(layout.root / '.env')
            profile = ModelConfigStore(layout.manifests / 'runtime_model_config.json').load()['organizer']
            if profile.api_style != 'chat-completions':
                raise ValueError('候选消歧需要 organizer 配置为 chat-completions。')
            client = StructuredAPIClient(api_key=profile.key(), base_url=profile.base_url,
                                         model=profile.model, provider=profile.provider)
            options = dict(model=profile.model, provider=profile.provider, base_url=profile.base_url)
            if args.disambiguate:
                disambiguator = ClosedCandidateDisambiguator(client, ModelCache(layout.llm_cache), **options)
            if args.stance_classify:
                classifier = ClosedStanceClassifier(client, ModelCache(layout.llm_cache), **options)
        retriever = HybridRetriever(database, semantic=semantic, disambiguator=disambiguator,
                                    stance_classifier=classifier)
        if args.command == 'research-search':
            return show(retriever.search(args.query, limit=args.limit, mode=args.mode))
        from mikirin.retrieval_evaluation import evaluate_retrieval

        report = evaluate_retrieval(retriever, args.benchmark, modes=tuple(args.modes), limit=args.limit)
        output_report(report, args.output)
        return int(any(row['error'] for row in report['results']))
    finally:
        if semantic is not None:
            semantic.close()


def retrieval_review(args: argparse.Namespace) -> int:
    from mikirin.retrieval_evaluation import prepare_blind_review, score_blind_review

    if args.retrieval_review_action == 'prepare':
        destination = Path(args.output_dir)
        if destination.exists():
            raise FileExistsError('请使用新的盲评目录，避免覆盖已有评分。')
        package = prepare_blind_review(read_records(args.report), seed=args.seed)
        destination.mkdir(parents=True)
        for name, payload in package.items():
            output_report(payload, str(destination / f'retrieval-{name}.json'))
        return 0
    return output_report(score_blind_review(read_records(args.blind), read_records(args.private)), args.output)


def graph_search(args: argparse.Namespace) -> int:
    from mikirin.graph.search import OfflineGraphSearch

    if not 1 <= args.limit <= 50:
        raise ValueError('limit must be between 1 and 50')
    if args.mode != 'simple' and not args.build_id:
        raise ValueError('graph/fusion 需要显式 --build-id；实验结果不会改变默认检索。')
    database = database_for(args)
    hits = KnowledgeQueries.search_entities(database, args.query, limit=args.limit)
    lexical = [{'entity_id': hit.entity_id, 'score': hit.retrieval_weight / (i + 1)} for i, hit in enumerate(hits)]
    vectors = []
    vector_status = {'enabled': False}
    if args.vectors:
        from mikirin.semantic_search import SemanticEpisodeSearch

        layout = layout_for(args)
        load_env_file(layout.root / '.env')
        semantic = SemanticEpisodeSearch(layout.root, database)
        try:
            vector_status = semantic.status()
            vectors = [{'entity_id': hit.subject_entity_id, 'score': hit.score}
                       for hit in semantic.search(args.query, top_k=args.limit)]
        finally:
            semantic.close()
    results = OfflineGraphSearch(database).search(
        args.seed or [hit.entity_id for hit in hits], mode=args.mode,
        task=args.task, build_id=args.build_id, lexical_candidates=lexical,
        vector_candidates=vectors, top_k=args.limit,
    )
    return show({'experimental': args.mode != 'simple', 'mode': args.mode,
                 'vector_status': vector_status, 'results': [r.to_dict() for r in results]})


def register_commands(commands: argparse._SubParsersAction) -> None:
    memory_audit = commands.add_parser('memory-review', help='每周20张记忆卡片的人工抽查')
    memory_actions = memory_audit.add_subparsers(dest='memory_review_action', required=True)
    prepare_memory = memory_actions.add_parser('prepare')
    prepare_memory.add_argument('--week', help='ISO week，如2026-W42')
    prepare_memory.add_argument('--output', required=True)
    prepare_memory.set_defaults(handler=memory_review)
    score_memory = memory_actions.add_parser('score')
    score_memory.add_argument('--input', required=True)
    score_memory.add_argument('--output')
    score_memory.set_defaults(handler=memory_review)
    for name in ('research-search', 'retrieval-eval'):
        experiment = commands.add_parser(name, help='A3.5 四档实验检索/60问评测，不改变默认检索')
        experiment.add_argument('--database')
        experiment.add_argument('--limit', type=int, default=8)
        experiment.add_argument('--vectors', action='store_true', help='显式允许双路 embedding API')
        experiment.add_argument('--disambiguate', action='store_true', help='显式允许 organizer 闭集候选消歧API；仅hybrid使用')
        experiment.add_argument('--stance-classify', action='store_true', help='显式允许 organizer 为原文标注观点；模型标签不等于人工验收')
        if name == 'research-search':
            experiment.add_argument('query')
            experiment.add_argument('--mode', choices=('fts', 'anchored', 'evidence', 'hybrid'), default='evidence')
        else:
            experiment.add_argument('--benchmark', help='默认使用随包60问；缺人工gold的指标保持null')
            experiment.add_argument('--modes', nargs='+', choices=('fts', 'anchored', 'evidence', 'hybrid'),
                                    default=['fts', 'anchored', 'evidence', 'hybrid'])
            experiment.add_argument('--output', required=True)
        experiment.set_defaults(handler=retrieval)
    blind = commands.add_parser('retrieval-review', help='60问检索报告的匿名化与人工评分汇总')
    blind_actions = blind.add_subparsers(dest='retrieval_review_action', required=True)
    prepare_blind = blind_actions.add_parser('prepare')
    prepare_blind.add_argument('--report', required=True)
    prepare_blind.add_argument('--seed', type=int, default=0)
    prepare_blind.add_argument('--output-dir', required=True)
    prepare_blind.set_defaults(handler=retrieval_review)
    score_blind = blind_actions.add_parser('score')
    score_blind.add_argument('--blind', required=True)
    score_blind.add_argument('--private', required=True)
    score_blind.add_argument('--output')
    score_blind.set_defaults(handler=retrieval_review)
    feeds = commands.add_parser('rss', help='明确订阅的 RSS 更新与主动话题；默认关闭，不下载媒体')
    feed_actions = feeds.add_subparsers(dest='rss_action', required=True)
    for name in ('status', 'poll', 'deliver'):
        feed_actions.add_parser(name).set_defaults(handler=rss)
    sub = feed_actions.add_parser('subscribe')
    sub.add_argument('subject', help='作品名/别名或实体ID')
    sub.add_argument('url', help='用户明确提供的 RSS/Atom 地址')
    sub.add_argument('--database')
    sub.set_defaults(handler=rss)
    unsub = feed_actions.add_parser('unsubscribe')
    unsub.add_argument('subscription_id')
    unsub.set_defaults(handler=rss)
    settings = feed_actions.add_parser('configure')
    settings.add_argument('--enabled', choices=('on', 'off'))
    settings.add_argument('--probability', type=float, help='0~1，默认0.30')
    settings.set_defaults(handler=rss)
    queue = commands.add_parser('memory-queue', help='可靠的会话结算队列；仅 run 调用模型')
    queue_actions = queue.add_subparsers(dest='queue_action', required=True)
    queue_actions.add_parser('status').set_defaults(handler=memory_queue)
    enqueue = queue_actions.add_parser('enqueue')
    enqueue.add_argument('--session-id')
    enqueue.set_defaults(handler=memory_queue)
    worker = queue_actions.add_parser('run')
    worker.add_argument('--limit', type=int, default=10)
    worker.set_defaults(handler=memory_queue)
    graph = commands.add_parser('graph-search', help='显式运行简单/图/融合检索实验，不改默认路径')
    graph.add_argument('query')
    graph.add_argument('--database')
    graph.add_argument('--mode', choices=('simple', 'graph', 'fusion'), default='simple')
    graph.add_argument('--build-id')
    graph.add_argument('--task', default='similar_work')
    graph.add_argument('--seed', action='append', help='显式种子实体 ID，可重复')
    graph.add_argument('--limit', type=int, default=8)
    graph.add_argument('--vectors', action='store_true', help='允许调用已配置 embedding API（可能计费）')
    graph.set_defaults(handler=graph_search)

    review = commands.add_parser('p1-review', help='黑话/隐式边/图检索人工审核与评测；不代填标签')
    actions = review.add_subparsers(dest='review_action', required=True)
    prepare = actions.add_parser('graph-prepare')
    prepare.add_argument('--database')
    prepare.add_argument('--build-id', required=True)
    prepare.add_argument('--queries', help='默认使用随包的30问模板')
    prepare.add_argument('--output-dir', required=True)
    prepare.add_argument('--seed', type=int, default=0)
    prepare.add_argument('--vectors', action='store_true', help='显式允许 embedding API')
    prepare.set_defaults(handler=p1_review)
    score = actions.add_parser('graph-score')
    score.add_argument('--blind', required=True)
    score.add_argument('--private', required=True)
    score.add_argument('--output')
    score.set_defaults(handler=p1_review)
    for name in ('slang-calibrate', 'slang-score', 'implicit-score'):
        action = actions.add_parser(name)
        action.add_argument('--input', required=True, help='人工填写的 JSON 或 CSV')
        action.add_argument('--output')
        if name == 'slang-calibrate':
            action.add_argument('--version', required=True)
        if name == 'slang-score':
            action.add_argument('--database')
        action.set_defaults(handler=p1_review)
    for name in ('implicit-prepare', 'aliases-prepare'):
        action = actions.add_parser(name)
        action.add_argument('--database')
        action.add_argument('--seed', type=int, default=0)
        action.add_argument('--output', required=True)
        if name == 'implicit-prepare':
            action.add_argument('--build-id', required=True)
        else:
            action.add_argument('--month', required=True, help='YYYY-MM')
        action.set_defaults(handler=p1_review)
