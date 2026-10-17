import unittest

from mikirin.sources.bgm_aliases import extract_infobox_aliases, normalize_alias


class BgmAliasTests(unittest.TestCase):
    def test_scalar_and_block_aliases_are_extracted(self) -> None:
        infobox = """{{Infobox animanga/TVAnime
|中文名= 轻音少女
|别名={
[K-ON!]
[けいおん!]
}
|website= https://example.invalid/
}}"""
        self.assertEqual(
            extract_infobox_aliases(infobox), ["轻音少女", "K-ON!", "けいおん!"]
        )

    def test_alias_normalization_is_nfkc_and_casefolded(self) -> None:
        self.assertEqual(normalize_alias(" Ｋ－ＯＮ！ "), "k-on!")


if __name__ == "__main__":
    unittest.main()
