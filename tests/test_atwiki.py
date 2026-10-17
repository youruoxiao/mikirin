import unittest

from mikirin.sources.atwiki import _PageLinkParser
from mikirin.sources.atwiki_parse import extract_wikibody_text, split_role_sections


class AtwikiParserTests(unittest.TestCase):
    def test_only_canonical_sakuga_pages_are_discovered(self) -> None:
        parser = _PageLinkParser()
        parser.feed(
            """
            <a href="//w.atwiki.jp/sakuga/pages/123.html">作品 A</a>
            <a href="https://www18.atwiki.jp/sakuga/pages/456.html">人物 B</a>
            <a href="/sakuga/editx/123.html">編集</a>
            <a href="https://example.com/sakuga/pages/789.html">外部</a>
            """
        )
        self.assertEqual(parser.pages, {123: "作品 A", 456: "人物 B"})

    def test_wikibody_excludes_controls_and_extracts_role_sections(self) -> None:
        html = """
        <html><div id="wikibody" class="box">
        <p>監督　りんたろう<br>脚本　渡辺麻実</p>
        <p>原画<br>山田太郎<br>佐藤花子</p>
        <div id="atwiki-page-tags"><p>タグ編集</p></div>
        <div>広告</div></div></html>
        """
        text = extract_wikibody_text(html)
        self.assertNotIn("タグ編集", text)
        self.assertNotIn("広告", text)
        sections = split_role_sections(text)
        self.assertEqual(sections[0].role, "監督")
        self.assertEqual(sections[0].lines, ("りんたろう",))
        self.assertEqual(sections[2].role, "原画")
        self.assertEqual(sections[2].lines, ("山田太郎", "佐藤花子"))


if __name__ == "__main__":
    unittest.main()
