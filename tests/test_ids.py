import unittest

from mikirin.ids import authoritative_entity_id, provisional_entity_id


class StableIdTests(unittest.TestCase):
    def test_authoritative_id_matches_contract(self) -> None:
        self.assertEqual(
            authoritative_entity_id("bgm", "subject", 328651),
            "bgm:subject:328651",
        )

    def test_same_name_different_source_record_does_not_merge(self) -> None:
        first = provisional_entity_id(
            source="atwiki",
            entity_type="person",
            source_record_id="work-a",
            record_locator="episode-1:row-3",
            raw_name="山田太郎",
        )
        second = provisional_entity_id(
            source="atwiki",
            entity_type="person",
            source_record_id="work-b",
            record_locator="episode-4:row-2",
            raw_name="山田太郎",
        )
        self.assertNotEqual(first, second)

    def test_provisional_id_is_deterministic(self) -> None:
        arguments = {
            "source": "atwiki",
            "entity_type": "person",
            "source_record_id": "work-a",
            "record_locator": "episode-1:row-3",
            "raw_name": "山田太郎",
        }
        self.assertEqual(provisional_entity_id(**arguments), provisional_entity_id(**arguments))


if __name__ == "__main__":
    unittest.main()

