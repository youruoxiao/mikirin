import unittest

from scripts.extract_discussion_maps import extract_map, is_reply_echo, split_comment


class ExtractDiscussionMapsTests(unittest.TestCase):
    def test_split_clauses_remain_verbatim(self) -> None:
        source = "开场的镜头切换很突然。第二段仍然是原文！"
        clauses = split_comment("c1", source, 0)
        self.assertGreaterEqual(len(clauses), 2)
        self.assertTrue(all(clause.text in source for clause in clauses))

    def test_repeated_specific_phrase_becomes_supported_theme(self) -> None:
        payload = {
            "episode_entity_id": "bgm:episode:test",
            "comments": [
                {"comment_id": "c1", "content": "五分钟毕业的时间跳切让我非常意外。"},
                {"comment_id": "c2", "content": "五分钟毕业之后直接进入动画公司，这个跳切很大胆。"},
                {"comment_id": "c3", "content": "配乐还不错，但没有留下太深印象。"},
            ],
        }
        result = extract_map(payload)
        self.assertTrue(result["themes"])
        theme = result["themes"][0]
        self.assertIn("五分钟毕业", "".join(q["quote"] for q in theme["quotes"]))
        comments = {row["comment_id"]: row["content"] for row in payload["comments"]}
        for quote in theme["quotes"]:
            self.assertIn(quote["quote"], comments[quote["comment_id"]])

    def test_url_clause_is_not_used(self) -> None:
        payload = {
            "episode_entity_id": "bgm:episode:test",
            "comments": [
                {"comment_id": "c1", "content": "详细解说见 https://example.com/a"},
                {"comment_id": "c2", "content": "详细解说见 https://example.com/b"},
                {"comment_id": "c3", "content": "结尾的纸箱机器人很有趣。"},
                {"comment_id": "c4", "content": "纸箱机器人在结尾再次出现。"},
            ],
        }
        result = extract_map(payload)
        rendered = str(result)
        self.assertNotIn("http", rendered)
        self.assertIn("纸箱机器人", rendered)

    def test_nested_reply_copy_is_not_independent_support(self) -> None:
        self.assertTrue(
            is_reply_echo(
                "甲 说: SF青春群像戏还行，所以说肥宅会有对象吗",
                "乙 说: SF青春群像戏还行，所以说肥宅会有对象吗error object null",
            )
        )

    def test_related_clauses_in_one_comment_are_a_sparse_fallback(self) -> None:
        payload = {
            "episode_entity_id": "bgm:episode:sparse",
            "comments": [
                {
                    "comment_id": "c1",
                    "content": "客人来做客却被批评失礼。客人来做客却只能吃不优雅的食物。",
                },
                {"comment_id": "c2", "content": "だが帰る！"},
            ],
        }
        result = extract_map(payload)
        self.assertTrue(result["themes"])
        quotes = result["themes"][0]["quotes"]
        self.assertEqual(quotes[0]["comment_id"], quotes[1]["comment_id"])
        self.assertIn("客人来做客", result["themes"][0]["theme"])

    def test_client_signature_and_reaction_only_text_are_ignored(self) -> None:
        payload = {
            "episode_entity_id": "bgm:episode:noise",
            "comments": [
                {"comment_id": "c1", "content": "[来自Bangumi for android] 获取"},
                {"comment_id": "c2", "content": "哈哈哈哈哈哈哈哈哈哈"},
                {"comment_id": "c5", "content": "Yippee!"},
                {"comment_id": "c6", "content": "Yippee!"},
                {"comment_id": "c3", "content": "纸箱机器人在结尾出现得很突兀。"},
                {"comment_id": "c4", "content": "结尾的纸箱机器人破坏了严肃气氛。"},
            ],
        }
        rendered = str(extract_map(payload))
        self.assertNotIn("来自Bangumi", rendered)
        self.assertNotIn("哈哈哈哈", rendered)
        self.assertNotIn("Yippee", rendered)
        self.assertIn("纸箱机器人", rendered)

    def test_username_with_id_metadata_is_removed_from_label(self) -> None:
        payload = {
            "episode_entity_id": "bgm:episode:reply",
            "comments": [
                {"comment_id": "c1", "content": "甲(id: alpha) 说: 纸箱机器人破坏了严肃气氛。"},
                {"comment_id": "c2", "content": "乙 说: 纸箱机器人在结尾显得突兀。"},
            ],
        }
        result = extract_map(payload)
        self.assertTrue(result["themes"])
        self.assertNotIn("说:", result["themes"][0]["theme"])
        self.assertIn("纸箱机器人", result["themes"][0]["theme"])

    def test_two_character_name_requires_three_comment_support(self) -> None:
        payload = {
            "episode_entity_id": "bgm:episode:short-name",
            "comments": [
                {"comment_id": "c1", "content": "狂三初登场很有压迫感。"},
                {"comment_id": "c2", "content": "这集狂三的异色瞳让人印象很深。"},
                {"comment_id": "c3", "content": "狂三作为病娇角色终于登场了。"},
            ],
        }
        result = extract_map(payload)
        self.assertTrue(result["themes"])
        self.assertIn("狂三", str(result["themes"]))

    def test_identical_short_quotes_do_not_make_a_colon_duplicate(self) -> None:
        payload = {
            "episode_entity_id": "bgm:episode:short-repeat",
            "comments": [
                {"comment_id": "c1", "content": "小麦终于出场了！"},
                {"comment_id": "c2", "content": "小麦终于出场了！"},
            ],
        }
        result = extract_map(payload)
        self.assertTrue(result["themes"])
        self.assertEqual(result["themes"][0]["theme"], "小麦终于出场了！")


if __name__ == "__main__":
    unittest.main()
