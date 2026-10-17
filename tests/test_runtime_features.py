import asyncio
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from mikirin.demo import create_demo
from mikirin.agent.tools import KnowledgeSearchTool, ProgressUpdateTool
from mikirin.layout import ProjectLayout
from mikirin.prompts import PromptBuilder
from mikirin.tui import AgentConsole


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.layout = ProjectLayout(Path(self.enterContext(tempfile.TemporaryDirectory())))
        self.database = create_demo(self.layout)
        self.console = AgentConsole(self.layout, self.database)

    def test_each_turn_uses_fresh_memory_even_when_session_is_reused(self):
        PromptBuilder(self.layout).initialize_user_files()
        self.console.chat_session = Mock(conversation_summary=None)
        with patch('urllib.request.urlopen', side_effect=AssertionError('offline')):
            self.console.chat('你好')
            PromptBuilder(self.layout).memory_path.write_text('我不喜欢恐怖片', encoding='utf-8')
            self.console.chat('继续')
        self.assertIn('我不喜欢恐怖片', self.console.chat_session.system_prompt)
        self.assertEqual(self.console.chat_session.respond.call_count, 2)

    def test_embedding_configuration_is_local_and_invalidates_loaded_index(self):
        old = self.console.semantic_search = Mock()
        with redirect_stdout(io.StringIO()), patch('urllib.request.urlopen', side_effect=AssertionError('offline')):
            self.console.handle_command('/set embedding https://example.test/v1 embed-model openai-embeddings EMBED_KEY 64 "query: " "document: "')
        old.close.assert_called_once()
        self.assertIsNone(self.console.semantic_search)
        profile = self.console.model_store.load()['embedding']
        self.assertEqual(profile.dimensions, 64)
        self.assertEqual(profile.query_prefix, 'query: ')

    def test_progress_correction_is_injected_on_next_chat_turn(self):
        self.console.chat_session = Mock(conversation_summary=None)
        tool = ProgressUpdateTool(self.database, self.console.user_store, self.console.session_id)

        async def update(_):
            pass

        asyncio.run(tool.execute('watch-1', {'subject': '放映部', 'episode': 7}, update))
        self.console.chat('记住了吗')
        self.assertIn('7', self.console.chat_session.system_prompt)
        asyncio.run(tool.execute('watch-2', {'subject': '放映部', 'episode': 3}, update))
        self.console.chat('刚才说错了')
        prompt = self.console.chat_session.system_prompt
        self.assertIn('<watch-progress>', prompt)
        block = prompt.split('<watch-progress>')[1].split('</watch-progress>')[0]
        self.assertIn('3', block)
        self.assertNotIn('7', block)

    def test_browse_mode_blocks_feed_fetch_but_allows_local_settings(self):
        self.console.browse_only = True
        with redirect_stdout(io.StringIO()) as output, patch('mikirin.proactive.fetch_http_feed') as fetcher:
            self.console.handle_command('/rss configure --enabled on --probability 0')
            self.console.handle_command('/rss poll')
            self.console.handle_command('/rss status')
        fetcher.assert_not_called()
        self.assertIn('--browse-only', output.getvalue())

    def test_experimental_search_forwards_current_focus_and_mode(self):
        retriever = Mock()
        retriever.search.return_value = {'entities': [], 'documents': []}
        focus = {'subject_entity_id': 'demo:subject:1'}
        tool = KnowledgeSearchTool(self.database, retriever=retriever,
                                   retrieval_mode='hybrid', focus=focus)

        async def update(_):
            pass

        asyncio.run(tool.execute('search-1', {'query': '这集争议', 'limit': 5}, update))
        retriever.search.assert_called_once_with('这集争议', limit=5,
                                                mode='hybrid', focus=focus)


if __name__ == '__main__':
    unittest.main()
