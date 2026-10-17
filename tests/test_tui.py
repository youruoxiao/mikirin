import ast
import inspect
import io
import json
import os
import re
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mikirin.agent.types import (
    AgentEvent,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from mikirin.layout import ProjectLayout
from mikirin.model_config import ModelProfile
from mikirin.organize import ResolutionEvidence
from mikirin.publish import Publisher
from mikirin.search_index import SearchIndexer
from mikirin.tui import HELP, AgentConsole, run_tui


class UnexpectedInput(BaseException):
    """Escape the UI's exception handler if a regression keeps reading input."""


class ConsoleTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.layout = ProjectLayout(Path(directory))
        self.database = Publisher(self.layout).create_empty_build(
            build_id="tui-test", code_version="test"
        )
        SearchIndexer.rebuild(self.database)
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        # Block both API construction and transport. Assert calls as well because
        # the interactive loop intentionally catches exceptions and keeps going.
        for target in (
            "urllib.request.urlopen",
            "socket.create_connection",
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.getaddrinfo",
            "mikirin.tui.RemoteAgentProvider",
            "mikirin.tui.StructuredAPIClient",
            "mikirin.semantic_search.SemanticEpisodeSearch",
            "mikirin.settlement.MemorySettlement",
        ):
            guard = self.enterContext(patch(
                target, side_effect=AssertionError(f"unexpected network path: {target}")
            ))
            self.addCleanup(guard.assert_not_called)
        self.console = AgentConsole(self.layout, self.database)

    def seed_conversation(self) -> None:
        for message in (
            UserMessage("我喜欢动画"),
            AssistantMessage((TextContent("记住了。"),)),
        ):
            self.console.session_store.append_message(self.console.session_id, message)

    def scripted_input(self, *inputs) -> Mock:
        replies = iter(inputs)

        def read(_prompt):
            try:
                reply = next(replies)
            except StopIteration:
                raise UnexpectedInput("console requested unexpected extra input") from None
            if isinstance(reply, type) and issubclass(reply, BaseException):
                raise reply()
            return reply

        return self.enterContext(patch("builtins.input", side_effect=read))

    def run_inputs(self, *inputs) -> Mock:
        reader = self.scripted_input(*inputs)
        self.assertEqual(self.console.run(), 0)
        return reader

    def test_browse_input_searches_locally_even_with_existing_chat_session(self) -> None:
        self.console.browse_only = True
        self.console.chat_session = Mock()
        with patch("mikirin.tui.KnowledgeQueries.search_entities", return_value=[]) as search:
            self.run_inputs("孤独摇滚", "/quit")
        search.assert_called_once_with(self.database, "孤独摇滚", limit=10)
        self.console.chat_session.respond.assert_not_called()
        self.assertIn("离线浏览", self.output.getvalue())
        self.assertIn("[]", self.output.getvalue())

    def test_browse_blocks_organize_and_settle_with_or_without_keys(self) -> None:
        self.console.browse_only = True
        self.seed_conversation()
        for key in ("", "present-but-must-not-be-used"):
            with self.subTest(key_available=bool(key)), patch.dict(
                os.environ, {"DEEPSEEK_API_KEY": key}
            ), patch.object(ModelProfile, "key") as key_lookup, patch(
                "mikirin.tui.ResolutionOrganizer"
            ) as organizer:
                self.console.handle_command("/organize mappings 1")
                self.console.handle_command("/settle")
                # Guard the underlying entry points, not just command dispatch.
                self.console._organize(["/organize", "mappings"], dry_run=False)
                self.console._settle()
                self.console._offer_settlement()
                key_lookup.assert_not_called()
                organizer.assert_not_called()
        self.assertIn("离线模式禁用 /organize", self.output.getvalue())
        self.assertIn("离线模式禁用 /settle", self.output.getvalue())

    def test_browse_exit_never_prompts_or_settles_resumed_history(self) -> None:
        self.seed_conversation()
        self.console = AgentConsole(self.layout, self.database, browse_only=True)
        for exit_command in ("/quit", "/exit"):
            with self.subTest(command=exit_command):
                reader = self.run_inputs(exit_command)
                self.assertEqual(reader.call_count, 1)

    def test_browse_plan_writes_requests_without_keys_or_clients(self) -> None:
        self.console.browse_only = True
        evidence = ResolutionEvidence(
            candidate_id="candidate:test", left_entity_id="left", right_entity_id="right",
            source_confidence=0.5, moegirl_name="角色", moegirl_aliases=[],
            moegirl_facts={}, bangumi_name="角色", bangumi_infobox="",
            bangumi_summary="", bangumi_subjects=[],
        )
        before = self.database.read_bytes()
        with patch("mikirin.organize.load_resolution_evidence", return_value=[evidence]), patch(
            "mikirin.tui.default_run_id", return_value="offline-plan"
        ), patch.object(ModelProfile, "key") as key_lookup:
            self.console.handle_command("/plan mappings 1")
        key_lookup.assert_not_called()
        run_dir = self.layout.staging / "deepseek_resolution" / "offline-plan"
        request = json.loads((run_dir / "batches/00001.request.json").read_text("utf-8"))
        self.assertEqual(request["expected_candidate_ids"], ["candidate:test"])
        manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
        self.assertTrue(manifest["dry_run"])
        self.assertEqual(manifest["planned_batches"], 1)
        self.assertEqual(self.database.read_bytes(), before)

    def test_browse_local_commands_and_configuration_remain_usable(self) -> None:
        self.console.browse_only = True
        original_session = self.console.session_id
        commands = (
            "/help", "/models", "/status", "/memory", "/session", "/search 动画",
            "/use chat openai test-model",
            "/set organizer https://example.test/v1 local-model chat-completions TEST_KEY",
            "/fork", "/new",
        )
        with patch.object(ModelProfile, "key") as key_lookup:
            for command in commands:
                with self.subTest(command=command):
                    self.assertFalse(self.console.handle_command(command))
        key_lookup.assert_not_called()
        self.assertEqual(self.console.profiles["chat"].model, "test-model")
        self.assertEqual(self.console.profiles["organizer"].key_env, "TEST_KEY")
        self.assertNotEqual(self.console.session_id, original_session)

    def test_ingest_preserves_quoted_and_unquoted_windows_paths(self) -> None:
        self.console.browse_only = True
        paths = (r"C:\anime\input.json", r"C:\anime files\input.json")
        with patch("mikirin.tui.RawAssetStore") as assets:
            assets.return_value.ingest.return_value = SimpleNamespace(
                id="asset:test", relative_path="raw/test.json"
            )
            for path in paths:
                with self.subTest(path=path):
                    quoted = f'"{path}"' if " " in path else path
                    self.console.handle_command(f"/ingest {quoted} test v1")
                    self.assertEqual(assets.return_value.ingest.call_args.args[0], Path(path))

    def test_missing_chat_key_is_actionable_and_does_not_start_a_session(self) -> None:
        before = self.console.session_store.entries(self.console.session_id)
        self.run_inputs("你好", "/models", "/quit")
        output = self.output.getvalue()
        self.assertIn("缺少 chat API key", output)
        self.assertIn("DEEPSEEK_API_KEY", output)
        self.assertIn("/models", output)
        self.assertIn("--browse-only", output)
        self.assertNotIn("Traceback", output)
        self.assertNotIn("RuntimeError", output)
        self.assertIsNone(self.console.chat_session)
        self.assertEqual(self.console.session_store.entries(self.console.session_id), before)

    def test_missing_organizer_key_does_not_start_organizer(self) -> None:
        with patch("mikirin.tui.ResolutionOrganizer") as organizer:
            self.console.handle_command("/organize mappings")
        organizer.assert_not_called()
        self.assertIn("缺少 organizer API key", self.output.getvalue())
        self.assertFalse((self.layout.manifests / "organizer.lock").exists())

    def test_settle_preflights_both_keys_before_constructing_clients(self) -> None:
        self.seed_conversation()
        self.console.profiles["chat"] = ModelProfile(
            provider="custom", base_url="https://example.test/v1", model="chat",
            api_style="chat-completions", key_env="CHAT_KEY",
        )
        self.console.handle_command("/settle")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-key"}):
            self.console.handle_command("/settle")
        self.assertIn("缺少 organizer API key", self.output.getvalue())
        self.assertIn("缺少 chat API key", self.output.getvalue())
        self.assertIn("CHAT_KEY", self.output.getvalue())
        self.assertNotIn("fake-key", self.output.getvalue())

    def test_missing_key_during_exit_settlement_still_exits(self) -> None:
        self.seed_conversation()
        reader = self.run_inputs("/quit", "yes")
        self.assertEqual(reader.call_count, 2)
        self.assertIn("缺少 organizer API key", self.output.getvalue())
        self.assertEqual(len(self.console.session_store.replay_messages(self.console.session_id)), 2)

    def test_eof_and_keyboard_interrupt_exit_without_followup_input(self) -> None:
        self.seed_conversation()
        for browse_only in (False, True):
            self.console.browse_only = browse_only
            for signal in (EOFError, KeyboardInterrupt):
                with self.subTest(browse_only=browse_only, signal=signal):
                    reader = self.run_inputs(signal)
                    self.assertEqual(reader.call_count, 1)

    def test_settlement_prompt_eof_interrupt_or_decline_exits(self) -> None:
        self.seed_conversation()
        for answer in (EOFError, KeyboardInterrupt, "", "n"):
            with self.subTest(answer=answer):
                reader = self.run_inputs("/quit", answer)
                self.assertEqual(reader.call_count, 2)

    def test_exit_settlement_failure_does_not_resume_the_input_loop(self) -> None:
        self.seed_conversation()
        for error in (RuntimeError("mock failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error)), patch.object(
                AgentConsole, "_settle", side_effect=error
            ):
                reader = self.run_inputs("/exit", "y")
                self.assertEqual(reader.call_count, 2)
        self.assertIn("会话已保留", self.output.getvalue())
        self.assertIn("/settle 重试", self.output.getvalue())

    def test_empty_session_settlement_needs_no_api(self) -> None:
        self.console.handle_command("/settle")
        self.assertIn("无需结算", self.output.getvalue())
        reader = self.run_inputs("/quit")
        self.assertEqual(reader.call_count, 1)

    def test_online_settle_dispatches_once_with_only_conversation_messages(self) -> None:
        self.seed_conversation()
        self.console.session_store.append_message(
            self.console.session_id, ToolResultMessage("call", "search", "tool output")
        )
        report = SimpleNamespace(
            facts_extracted=1, notes_added=1, notes_expired=0, skipped=0,
            already_settled=False,
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-key"}), patch(
            "mikirin.tui.StructuredAPIClient"
        ) as client, patch("mikirin.settlement.MemorySettlement") as settlement, patch(
            "builtins.input"
        ) as reader:
            settlement.return_value.settle.return_value = report
            self.assertFalse(self.console.handle_command("/settle"))
        reader.assert_not_called()
        self.assertEqual(client.call_count, 2)
        settlement.return_value.settle.assert_called_once_with(
            self.console.session_id,
            [{"role": "user", "content": "我喜欢动画"},
             {"role": "assistant", "content": "记住了。"}],
        )
        self.assertIn("记忆结算：抽取 1 条", self.output.getvalue())

    def test_online_chat_still_uses_configured_provider(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-key"}), patch(
            "mikirin.tui.RemoteAgentProvider"
        ) as provider, patch("mikirin.tui.KnowledgeAgentSession") as session, patch(
            "mikirin.semantic_search.SemanticEpisodeSearch"
        ):
            self.console.chat("你好")
            self.console.chat("继续")
        provider.assert_called_once()
        self.assertEqual(provider.call_args.kwargs["api_key"], "fake-key")
        session.assert_called_once()
        self.assertEqual(session.return_value.respond.call_count, 2)
        self.assertEqual(session.return_value.respond.call_args.args, ("继续",))
        self.assertNotIn("fake-key", self.output.getvalue())

    def test_online_organize_still_runs_with_client(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake-key"}), patch(
            "mikirin.tui.StructuredAPIClient"
        ) as client, patch("mikirin.tui.ResolutionOrganizer") as organizer:
            organizer.return_value.run.return_value = {"dry_run": False}
            self.console.handle_command("/organize mappings 3")
        client.assert_called_once()
        self.assertIs(organizer.call_args.kwargs["client"], client.return_value)
        self.assertFalse(organizer.return_value.run.call_args.kwargs["dry_run"])
        self.assertEqual(organizer.return_value.run.call_args.kwargs["limit"], 3)

    def test_command_errors_are_reported_and_loop_remains_usable(self) -> None:
        self.console.browse_only = True
        reader = self.run_inputs("", "/unknown", '/use "unfinished', "/search", "/help", "/quit")
        self.assertEqual(reader.call_count, 6)
        self.assertIn("unknown command", self.output.getvalue())
        self.assertIn("No closing quotation", self.output.getvalue())
        self.assertIn("usage: /search", self.output.getvalue())
        self.assertIn(HELP, self.output.getvalue())

    def test_command_arguments_are_validated(self) -> None:
        invalid = (
            "/quit extra", "/exit extra", "/settle extra", "/memory extra",
            "/new extra", "/fork extra", "/search", '/search ""',
            "/use", "/use embedding openai", "/use chat unknown", "/set chat",
            "/set organizer https://example.test/v1 model responses KEY", "/ingest",
            "/plan", "/plan invalid", "/plan mappings 0", "/plan mappings 501",
            "/plan mappings invalid", "/plan mappings 1 extra",
        )
        for command in invalid:
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.console.handle_command(command)
        self.assertFalse(self.console.handle_command("  "))

    def test_help_lists_exactly_the_dispatched_commands(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(AgentConsole.handle_command)))
        handled = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and re.fullmatch(r"/[a-z]+", node.value)
        }
        documented = set(re.findall(r"^  (/[a-z]+)\b", HELP, re.MULTILINE))
        self.assertEqual(documented, handled)
        self.assertIn("--browse-only", HELP)

    def test_tool_completion_does_not_duplicate_assistant_prompt(self) -> None:
        call = ToolCall("call", "search_anime", {})
        result = ToolResultMessage("call", "search_anime", '{"entities": []}')
        self.console._show_agent_event(AgentEvent("tool_execution_end", tool_call=call,
                                                 tool_result=result))
        message = AssistantMessage((TextContent("找到了"),))
        self.console._show_agent_event(AgentEvent("message_start", message=message))
        self.console._show_agent_event(AgentEvent("message_end", message=message))
        self.assertEqual(self.output.getvalue().count("Mikirin> "), 1)
        self.assertIn("Mikirin> 找到了\n", self.output.getvalue())

    def test_streamed_answer_is_not_printed_twice_and_errors_are_visible(self) -> None:
        message = AssistantMessage((TextContent("回答"),))
        self.console._show_agent_event(AgentEvent("message_start", message=message))
        self.console._show_agent_event(AgentEvent("message_update", message=message, delta="回答"))
        self.console._show_agent_event(AgentEvent("message_end", message=message))
        self.assertEqual(self.output.getvalue().count("回答"), 1)
        failure = AssistantMessage((), stop_reason="error", error_message="mock model error")
        self.console._show_agent_event(AgentEvent("message_start", message=failure))
        self.console._show_agent_event(AgentEvent("message_end", message=failure))
        self.assertIn("error> mock model error", self.output.getvalue())

    def test_tool_summary_tolerates_nonstandard_payloads_and_reports_failure(self) -> None:
        for content in ("plain text", "{broken", '{"entities":null,"documents":3}', "[]"):
            with self.subTest(content=content):
                event = AgentEvent("tool_execution_end", tool_result=ToolResultMessage(
                    "call", "tool", content
                ))
                self.assertIsInstance(AgentConsole._tool_summary(event), str)
        event = AgentEvent("tool_execution_end", tool_result=ToolResultMessage(
            "call", "tool", "failure", is_error=True
        ))
        self.assertIn("失败", AgentConsole._tool_summary(event))

    def test_run_tui_forwards_browse_only(self) -> None:
        self.seed_conversation()
        reader = self.scripted_input("/quit")
        self.assertEqual(run_tui(
            layout=self.layout, database=self.database, browse_only=True
        ), 0)
        self.assertEqual(reader.call_count, 1)
        self.assertIn("离线浏览", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
