from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from mikirin.assets import RawAssetStore
from mikirin.cache import ModelCache
from mikirin.agent.remote import RemoteAgentProvider
from mikirin.agent.tools import MemoryWriteTool, ProgressUpdateTool, RecallMemoryTool
from mikirin.agent.types import AgentEvent
from mikirin.chat import KnowledgeAgentSession
from mikirin.context import CompactionRecord
from mikirin.context import message_text
from mikirin.deepseek import StructuredAPIClient
from mikirin.layout import ProjectLayout
from mikirin.model_config import EmbeddingProfile, ModelConfigStore, ModelProfile, ModelRole
from mikirin.organize import ResolutionOrganizer, default_run_id
from mikirin.publish import Publisher
from mikirin.prompts import PromptBuilder, PromptLayers
from mikirin.query import KnowledgeQueries
from mikirin.run_lock import exclusive_process_lock
from mikirin.session import SessionStore, message_from_dict
from mikirin.user_store import UserStore


HELP = """Commands:
  /models                         show configured API profiles (never keys)
  /use <role> <provider> [model]  role=organizer|chat, provider=deepseek|openai
  /set <role> <url> <model> <style> <key_env>  configure any compatible API
    embedding: /set embedding <url> <model> openai-embeddings <key_env> <dimensions> [query_prefix] [document_prefix]
  /ingest <path> <source> <version>  archive an original file into data/raw
  /organize mappings [limit]      run a reviewable entity-resolution pilot
  /plan mappings [limit]          write organizer requests without calling an API
  /search <query>                 search the published knowledge base only
  /status                         show current database and raw inbox
  /memory                         show local memory files and watch progress
  /rss <status|subscribe|unsubscribe|configure|poll|deliver>  manage opt-in RSS topics
  /settle                         settle this session's memory using APIs
  /session                        show current append-only session
  /new                            start a new session
  /fork                           fork the current session and continue there
  /help                           show this help
  /quit                           exit, optionally settling memory online
  /exit                           alias for /quit

Any other input is sent to the chat model with local knowledge retrieval.
With --browse-only, input searches local knowledge; /organize and /settle are
disabled and exit never calls APIs. /plan and model configuration remain local.
EOF or Ctrl+C exits without settlement."""


@dataclass(slots=True)
class AgentConsole:
    layout: ProjectLayout
    database: Path
    browse_only: bool = False
    retrieval_mode: str | None = None
    profiles: dict[ModelRole, ModelProfile] = field(init=False)
    model_store: ModelConfigStore = field(init=False)
    chat_session: KnowledgeAgentSession | None = field(default=None, init=False)
    semantic_search: object | None = field(default=None, init=False)
    session_store: SessionStore = field(init=False)
    session_id: str = field(init=False)
    user_store: UserStore = field(init=False)
    streamed_delta: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.model_store = ModelConfigStore(
            self.layout.manifests / "runtime_model_config.json"
        )
        self.profiles = self.model_store.load()
        self.session_store = SessionStore(self.layout.sessions)
        self.user_store = UserStore(self.layout.user_db)
        self.session_id = self.session_store.latest() or self.session_store.create(
            metadata={"database": str(self.database)}
        )
        self.user_store.register_session(
            self.session_id, jsonl_path=self.session_store.path(self.session_id)
        )
        self._sync_session_mirror()

    def run(self) -> int:
        print("Mikirin（ミキリン）——你的动画搭子")
        if self.browse_only:
            print("离线浏览（--browse-only）：输入内容仅搜索本地知识库，不调用 API。")
            print("命令：/help 查看全部命令，/quit 退出；/plan 和模型配置仍可使用。")
        else:
            print("随便聊：作品、角色、声优、作画、某一集、社区评价都可以问。")
            print("命令：/help 查看全部命令，/quit 退出（退出时可结算本次记忆）")
        print(f"session> {self.session_id}")
        if self.retrieval_mode:
            print(f"实验检索：{self.retrieval_mode}（本次会话选择，不代表盲评已通过）")
        # Delivery reads the local queue only. Fetching feeds is an explicit
        # /rss poll operation and is never performed in browse-only mode.
        try:
            from mikirin.proactive import ProactiveService

            ProactiveService(self.layout.user_db).deliver(self._display_proactive)
        except Exception as error:
            print(f"主动话题暂不可用：{type(error).__name__}；普通对话不受影响。")
        while True:
            try:
                text = input("you> ").strip()
                if not text:
                    continue
                if text.startswith("/"):
                    if self.handle_command(text):
                        return 0
                else:
                    self.chat(text)
            except (EOFError, KeyboardInterrupt):
                # A closed input stream cannot answer a settlement prompt.
                print()
                return 0
            except Exception as error:
                print(f"error> {type(error).__name__}: {error}")

    def handle_command(self, text: str) -> bool:
        """Handle one command; return True when the console should exit."""
        lexer = shlex.shlex(text, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""  # Preserve Windows paths, including quoted paths.
        parts = list(lexer)
        if not parts:
            return False
        command = parts[0]
        if command in {
            "/help", "/models", "/status", "/memory", "/settle", "/session",
            "/new", "/fork", "/quit", "/exit",
        } and len(parts) != 1:
            raise ValueError(f"usage: {command}")
        if command == "/help":
            print(HELP)
        elif command == "/models":
            print(json.dumps(
                {role: profile.public_dict() for role, profile in self.profiles.items()},
                ensure_ascii=False, indent=2,
            ))
        elif command == "/use":
            self._use(parts)
        elif command == "/set":
            self._set_custom(parts)
        elif command == "/ingest":
            self._ingest(parts)
        elif command in {"/organize", "/plan"}:
            self._organize(parts, dry_run=command == "/plan")
        elif command == "/search":
            query = " ".join(parts[1:]).strip()
            if not query:
                raise ValueError("usage: /search <query>")
            hits = KnowledgeQueries.search_entities(self.database, query, limit=10)
            print(json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False, indent=2))
        elif command == "/status":
            current = Publisher(self.layout).current()
            inbox_count = sum(1 for path in self.layout.raw_inbox.rglob("*") if path.is_file())
            print(json.dumps({
                "current_database": str(current) if current else None,
                "raw_inbox": str(self.layout.raw_inbox),
                "raw_inbox_files": inbox_count,
                "model_config": str(self.model_store.path),
            }, ensure_ascii=False, indent=2))
        elif command == "/memory":
            self._show_memory()
        elif command == "/rss":
            self._rss(parts[1:] or ["status"])
        elif command == "/settle":
            self._settle()
        elif command == "/session":
            print(json.dumps({
                "session_id": self.session_id,
                "path": str(self.session_store.path(self.session_id)),
                "entries": len(self.session_store.entries(self.session_id)),
            }, ensure_ascii=False, indent=2))
        elif command == "/new":
            self.session_id = self.session_store.create(metadata={"database": str(self.database)})
            self.user_store.register_session(
                self.session_id, jsonl_path=self.session_store.path(self.session_id)
            )
            self._sync_session_mirror()
            self.chat_session = None
            print(f"session> {self.session_id}")
        elif command == "/fork":
            parent_session_id = self.session_id
            self.session_id = self.session_store.fork(parent_session_id)
            self.user_store.register_session(
                self.session_id,
                jsonl_path=self.session_store.path(self.session_id),
                parent_session_id=parent_session_id,
            )
            self._sync_session_mirror()
            self.chat_session = None
            print(f"session> {self.session_id} (fork)")
        elif command in {"/quit", "/exit"}:
            self._offer_settlement()
            return True
        else:
            raise ValueError(f"unknown command {command!r}; use /help")
        return False

    def _api_key(self, role: ModelRole) -> str | None:
        profile = self.profiles[role]
        try:
            return profile.key()
        except RuntimeError:
            print(
                f"缺少 {role} API key：请设置环境变量 {profile.key_env}，"
                "或在项目 .env 中配置后重启。"
                "可用 /models 检查配置、/search 本地搜索，"
                "或使用 --browse-only 离线浏览。"
            )
            return None

    def _rss(self, arguments: list[str]) -> None:
        if self.browse_only and arguments[0] == "poll":
            print("--browse-only 不访问 RSS 网络；退出离线模式后再 poll。")
            return
        if arguments == ["deliver"]:
            from mikirin.proactive import ProactiveService

            result = ProactiveService(self.layout.user_db).deliver(self._display_proactive)
            print(f"主动话题：已投递 {len(result.delivered)}，失败 {len(result.failed)}，跳过 {result.skipped or '无'}。")
            return
        if arguments[0] == "subscribe" and "--database" not in arguments:
            arguments = [*arguments, "--database", str(self.database)]
        from mikirin.cli import build_parser

        try:
            args = build_parser().parse_args(["--root", str(self.layout.root), "rss", *arguments])
        except SystemExit as error:
            if error.code == 0:
                return
            raise ValueError("/rss 参数错误；使用 /rss --help 查看用法。") from None
        args.handler(args)

    def _display_proactive(self, topic) -> None:
        from mikirin.agent.types import AssistantMessage, TextContent
        from mikirin.session import message_to_dict

        message = AssistantMessage((TextContent(topic.content),))
        seen = any(entry.get('payload', {}).get('proactive_topic_id') == topic.id
                   for entry in self.session_store.entries(self.session_id))
        if not seen:
            entry_id = self.session_store.append(self.session_id, 'assistant',
                {**message_to_dict(message), 'proactive_topic_id': topic.id})
            self.user_store.mirror_message(session_id=self.session_id, entry_id=entry_id,
                                          role='assistant', content=topic.content)
            if self.chat_session is not None:
                self.chat_session.history.append(message)
        print(f"Mikirin> {topic.content}", flush=True)

    def _use(self, parts: list[str]) -> None:
        if len(parts) not in {3, 4}:
            raise ValueError("usage: /use <organizer|chat> <deepseek|openai> [model]")
        role = parts[1]
        if role not in {"organizer", "chat"}:
            raise ValueError("role must be organizer or chat")
        profile = self.model_store.set_provider(
            role, parts[2], model=parts[3] if len(parts) == 4 else None  # type: ignore[arg-type]
        )
        self.profiles = self.model_store.load()
        if role == "chat":
            self.chat_session = None
        self.session_store.append(
            self.session_id, "model_change", {"role": role, **profile.public_dict()}
        )
        print(json.dumps(profile.public_dict(), ensure_ascii=False, indent=2))

    def _set_custom(self, parts: list[str]) -> None:
        if len(parts) > 1 and parts[1] == "embedding":
            if len(parts) not in {7, 8, 9}:
                raise ValueError("usage: /set embedding <url> <model> openai-embeddings <key_env> <dimensions> [query_prefix] [document_prefix]")
            profile = self.model_store.set_custom("embedding", EmbeddingProfile(
                provider="custom", base_url=parts[2], model=parts[3],
                api_style=parts[4], key_env=parts[5], dimensions=int(parts[6]),
                query_prefix=parts[7] if len(parts) >= 8 else "",
                document_prefix=parts[8] if len(parts) >= 9 else "",
            ))
            self.profiles = self.model_store.load()
            if self.semantic_search is not None:
                self.semantic_search.close()
            self.semantic_search = None
            self.chat_session = None
            print(json.dumps(profile.public_dict(), ensure_ascii=False, indent=2))
            print("embedding 配置已保存；指纹不匹配的旧索引会停用，需另行重建。")
            return
        if len(parts) != 6:
            raise ValueError(
                "usage: /set <organizer|chat> <https-url> <model> "
                "<responses|chat-completions> <key_env>"
            )
        role = parts[1]
        if role not in {"organizer", "chat"}:
            raise ValueError("role must be organizer or chat")
        if role == "organizer" and parts[4] != "chat-completions":
            raise ValueError("organizer currently requires a Chat Completions JSON-mode API")
        profile = self.model_store.set_custom(
            role,  # type: ignore[arg-type]
            ModelProfile(
                provider="custom",
                base_url=parts[2],
                model=parts[3],
                api_style=parts[4],  # type: ignore[arg-type]
                key_env=parts[5],
            ),
        )
        self.profiles = self.model_store.load()
        if role == "chat":
            self.chat_session = None
        self.session_store.append(
            self.session_id, "model_change", {"role": role, **profile.public_dict()}
        )
        print(json.dumps(profile.public_dict(), ensure_ascii=False, indent=2))

    def _ingest(self, parts: list[str]) -> None:
        if len(parts) != 4:
            raise ValueError("usage: /ingest <path> <source> <version>")
        asset = RawAssetStore(self.layout).ingest(
            Path(parts[1]), source=parts[2], source_version=parts[3], acquired_on=date.today()
        )
        print(json.dumps({"asset_id": asset.id, "raw_path": asset.relative_path}, indent=2))

    def _organize(self, parts: list[str], *, dry_run: bool) -> None:
        if self.browse_only and not dry_run:
            print("--browse-only 离线模式禁用 /organize；可用 /plan mappings [limit]。")
            return
        if len(parts) > 3 or len(parts) < 2 or parts[1] != "mappings":
            raise ValueError("usage: /organize mappings [limit] or /plan mappings [limit]")
        limit = int(parts[2]) if len(parts) == 3 else 20
        if limit < 1 or limit > 500:
            raise ValueError("TUI pilot limit must be between 1 and 500")
        profile = self.profiles["organizer"]
        client = None
        if not dry_run:
            api_key = self._api_key("organizer")
            if api_key is None:
                return
            client = StructuredAPIClient(
                api_key=api_key, base_url=profile.base_url,
                model=profile.model, provider=profile.provider,
            )
        with exclusive_process_lock(self.layout.manifests / "organizer.lock"):
            report = ResolutionOrganizer(
                database=self.database,
                staging_root=self.layout.staging,
                cache=ModelCache(self.layout.llm_cache),
                client=client,
                model_name=profile.model,
            ).run(limit=limit, run_id=default_run_id(), dry_run=dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not dry_run:
            print("Results are staged for review; the published database was not modified.")

    def chat(self, text: str) -> None:
        if self.browse_only:
            hits = KnowledgeQueries.search_entities(self.database, text, limit=10)
            print(json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False, indent=2))
            return
        if self.chat_session is None:
            profile = self.profiles["chat"]
            api_key = self._api_key("chat")
            if api_key is None:
                return
            provider = RemoteAgentProvider(
                base_url=profile.base_url, model=profile.model,
                api_key=api_key, api_style=profile.api_style,
            )
            replayed = self.session_store.replay_messages(self.session_id)
            compaction = self.session_store.latest_entry(self.session_id, "compaction")
            summary = None
            compacted_count = 0
            if compaction is not None:
                summary = str(compaction["payload"].get("summary") or "") or None
                compacted_count = int(compaction["payload"].get("compacted_message_count") or 0)
            if self.semantic_search is None and self.retrieval_mode in {None, "hybrid"}:
                from mikirin.semantic_search import SemanticEpisodeSearch
                if self.retrieval_mode == "hybrid":
                    from mikirin.hybrid_retrieval import HybridSemanticSearch

                    self.semantic_search = HybridSemanticSearch(self.layout.root, self.database)
                else:
                    self.semantic_search = SemanticEpisodeSearch(self.layout.root, self.database)
            retriever = None
            if self.retrieval_mode:
                from mikirin.hybrid_retrieval import HybridRetriever

                retriever = HybridRetriever(self.database, semantic=self.semantic_search)
            self.chat_session = KnowledgeAgentSession(
                database=self.database,
                provider=provider,
                history=replayed[compacted_count:],
                semantic=self.semantic_search,
                retriever=retriever,
                retrieval_mode=self.retrieval_mode or "evidence",
                system_prompt=self._session_prompt(summary).rendered,
                conversation_summary=summary,
                compacted_message_count=compacted_count,
                compaction_sink=self._persist_compaction,
                prompt_rebuilder=self._rebuild_prompt_after_compaction,
                additional_tools=(
                    MemoryWriteTool(
                        PromptBuilder(self.layout).user_path,
                        PromptBuilder(self.layout).memory_path,
                    ),
                    ProgressUpdateTool(
                        self.database,
                        self.user_store,
                        self.session_id,
                    ),
                    RecallMemoryTool(self.user_store),
                ),
            )
        # Progress and explicit memory tools commit synchronously. Rebuild the
        # volatile layer every turn rather than reusing a session-start snapshot.
        summary = self.chat_session.conversation_summary
        layers = PromptBuilder(self.layout).build(
            database=self.database,
            conversation_summary=summary if isinstance(summary, str) else None,
        )
        self.chat_session.system_prompt = layers.rendered
        self.chat_session.respond(text, emit=self._show_agent_event)

    def _show_agent_event(self, event: AgentEvent) -> None:
        from mikirin.agent.types import AssistantMessage

        is_assistant = isinstance(event.message, AssistantMessage)
        if event.type == "message_start":
            self.streamed_delta = False
            if is_assistant:
                print("Mikirin> ", end="", flush=True)
        elif event.type == "message_update" and event.delta and is_assistant:
            self.streamed_delta = True
            print(event.delta, end="", flush=True)
        elif event.type == "message_end" and event.message is not None:
            if is_assistant:
                if not self.streamed_delta:
                    print(event.message.text, end="", flush=True)
                print()
                if event.message.error_message:
                    print(f"error> {event.message.error_message}")
            entry_id = self.session_store.append_message(self.session_id, event.message)
            self.user_store.mirror_message(
                session_id=self.session_id,
                entry_id=entry_id,
                role=event.message.role,
                content=message_text(event.message),
            )
        elif event.type == "tool_execution_start" and event.tool_call is not None:
            print(f"\n[工具] {event.tool_call.name} ...", flush=True)
        elif event.type == "tool_execution_end" and event.tool_call is not None:
            summary = self._tool_summary(event)
            print(f"[工具完成] {event.tool_call.name}{summary}", flush=True)

    @staticmethod
    def _tool_summary(event: AgentEvent) -> str:
        if event.tool_result is None:
            return ""
        if event.data.get("is_error") or event.tool_result.is_error:
            return "（失败，继续作答）"
        content = getattr(event.tool_result, "content", "")
        if not isinstance(content, str) or not content.startswith("{"):
            return ""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return ""
        if not isinstance(payload, dict):
            return ""
        parts = []
        if isinstance(payload.get("entities"), list):
            parts.append(f"实体 {len(payload['entities'])}")
        if isinstance(payload.get("documents"), list):
            parts.append(f"文档 {len(payload['documents'])}")
        if isinstance(payload.get("semantic_episodes"), list) and payload["semantic_episodes"]:
            parts.append(f"语义候选 {len(payload['semantic_episodes'])}")
        return f"（{'，'.join(parts)}）" if parts else "（已返回）"

    def _session_prompt(self, summary: str | None = None) -> PromptLayers:
        existing = self.session_store.latest_entry(self.session_id, "prompt_snapshot")
        compaction = self.session_store.latest_entry(self.session_id, "compaction")
        if existing is not None and (
            compaction is None or existing["created_at"] >= compaction["created_at"]
        ):
            return PromptLayers.from_dict(existing["payload"])
        layers = PromptBuilder(self.layout).build(
            database=self.database, conversation_summary=summary
        )
        self.session_store.append(self.session_id, "prompt_snapshot", layers.to_dict())
        return layers

    def _persist_compaction(self, record: CompactionRecord) -> None:
        self.session_store.append(self.session_id, "compaction", asdict(record))

    def _rebuild_prompt_after_compaction(self, summary: str) -> str:
        layers = PromptBuilder(self.layout).build(
            database=self.database, conversation_summary=summary
        )
        self.session_store.append(self.session_id, "prompt_snapshot", layers.to_dict())
        return layers.rendered

    def _show_memory(self) -> None:
        user_path = PromptBuilder(self.layout).user_path
        memory_path = PromptBuilder(self.layout).memory_path
        print("=== USER.md ===")
        print(user_path.read_text(encoding="utf-8") if user_path.exists() else "（空）")
        print("=== MEMORY.md ===")
        print(memory_path.read_text(encoding="utf-8") if memory_path.exists() else "（空）")
        progress = self.user_store.all_progress()
        if progress:
            print("=== 追番进度 ===")
            for record in progress:
                print(f"  {record.subject_entity_id}: {record.progress_value} ({record.status})")

    def _offer_settlement(self) -> None:
        if self.browse_only:
            return
        entries = self.session_store.entries(self.session_id)
        if not any(e.get("type") == "user" for e in entries):
            return
        try:
            answer = input("退出前结算这次对话的记忆吗？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer not in {"y", "yes"}:
            return
        try:
            self._settle()
        except (Exception, KeyboardInterrupt) as error:
            print(f"记忆结算未完成：{error}。会话已保留，可稍后使用 /settle 重试。")

    def _settle(self) -> None:
        if self.browse_only:
            print("--browse-only 离线模式禁用 /settle，不调用 API。")
            return
        from mikirin.agent.types import AssistantMessage, UserMessage
        from mikirin.settlement import MemorySettlement

        messages: list[dict[str, str]] = []
        for message in self.session_store.replay_messages(self.session_id):
            if isinstance(message, UserMessage):
                messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AssistantMessage) and message.text:
                messages.append({"role": "assistant", "content": message.text})
        if not any(message["role"] == "user" for message in messages):
            print("当前会话没有用户对话，无需结算。")
            return
        # Validate both keys before constructing either remote client.
        flash_key = self._api_key("organizer")
        if flash_key is None:
            return
        pro_key = self._api_key("chat")
        if pro_key is None:
            return
        flash = self.profiles["organizer"]
        pro = self.profiles["chat"]
        from mikirin.structured_role import client_for_profile

        report = MemorySettlement(
            self.layout.user_db,
            client_for_profile(flash, flash_key, chat_factory=StructuredAPIClient),
            client_for_profile(pro, pro_key, chat_factory=StructuredAPIClient),
        ).settle(self.session_id, messages)
        print(
            f"记忆结算：抽取 {report.facts_extracted} 条，"
            f"新增 {report.notes_added}，失效旧卡 {report.notes_expired}，"
            f"跳过 {report.skipped}"
            + ("（本 session 已结算过）" if report.already_settled else "")
        )

    def _sync_session_mirror(self) -> None:
        for entry in self.session_store.entries(self.session_id):
            if entry.get("type") not in {"user", "assistant", "tool_result"}:
                continue
            message = message_from_dict(entry["payload"])
            self.user_store.mirror_message(
                session_id=self.session_id,
                entry_id=str(entry["entry_id"]),
                role=message.role,
                content=message_text(message),
            )


def run_tui(*, layout: ProjectLayout, database: Path, browse_only: bool = False,
            retrieval_mode: str | None = None) -> int:
    console = AgentConsole(layout=layout, database=database, browse_only=browse_only,
                           retrieval_mode=retrieval_mode)
    try:
        return console.run()
    finally:
        if console.semantic_search is not None:
            console.semantic_search.close()
