import unittest

from mikirin.sources.bgm_selection import SubjectListParser


class BgmSelectionTests(unittest.TestCase):
    def test_only_subject_title_links_are_selected(self) -> None:
        parser = SubjectListParser()
        parser.feed("""
        <a class="subjectCover" href="/subject/123">cover</a>
        <h3><a href="/subject/123" class="l">作品 A</a></h3>
        <a class="l" href="https://example.com/subject/9">external</a>
        <h3><a class="l extra" href="/subject/456">作品 B</a></h3>
        <a href="?page=16">16</a>
        """)
        self.assertEqual(
            [(item.subject_id, item.title) for item in parser.subjects],
            [(123, "作品 A"), (456, "作品 B")],
        )
        self.assertEqual(max(parser.page_numbers), 16)


if __name__ == "__main__":
    unittest.main()
