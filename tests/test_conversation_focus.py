import tempfile
import unittest
from pathlib import Path

from mikirin.conversation_focus import (
    EMPTY_FOCUS,
    ConversationFocus,
    refresh_focus,
    resolve_pronoun,
    tick_focus,
)


class FocusTests(unittest.TestCase):
    def test_refresh_sets_subject_and_episode(self):
        focus = refresh_focus(
            EMPTY_FOCUS, subject_id="s1", subject_name="孤独摇滚",
            episode_id="e8", episode_sort=8.0,
        )
        self.assertEqual(focus.active_subject_id, "s1")
        self.assertEqual(focus.active_episode_sort, 8.0)

    def test_subject_switch_clears_episode(self):
        focus = refresh_focus(
            EMPTY_FOCUS, subject_id="s1", subject_name="a",
            episode_id="e8", episode_sort=8.0,
        )
        focus = refresh_focus(focus, subject_id="s2", subject_name="b")
        self.assertEqual(focus.active_subject_id, "s2")
        self.assertIsNone(focus.active_episode_id)

    def test_decay_clears_after_threshold(self):
        focus = refresh_focus(EMPTY_FOCUS, subject_id="s1", subject_name="a",
                              confidence=0.5)
        for _ in range(20):
            focus = tick_focus(focus, decay_turns=3, decay_step=0.1)
        self.assertTrue(focus.is_empty)

    def test_pronoun_resolution(self):
        focus = refresh_focus(
            EMPTY_FOCUS, subject_id="s1", subject_name="孤独摇滚",
            episode_id="e8", episode_sort=8.0,
        )
        marker, subject_id, episode = resolve_pronoun(focus, "那集的分镜谁画的")
        self.assertEqual(marker, "pronoun")
        self.assertEqual(subject_id, "s1")
        self.assertEqual(episode, 8.0)
        self.assertEqual(resolve_pronoun(EMPTY_FOCUS, "那集"), ("", None, None))


class TopicLayerTests(unittest.TestCase):
    def test_empty_focus_gives_empty_fence(self):
        from mikirin.topic import EMPTY_TOPIC, build_topic_layer

        self.assertEqual(build_topic_layer(Path("nonexistent.db"), EMPTY_FOCUS),
                         EMPTY_TOPIC)


if __name__ == "__main__":
    unittest.main()
