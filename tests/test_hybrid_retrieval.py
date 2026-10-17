import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from collections import Counter
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from mikirin.conversation_focus import ConversationFocus
from mikirin.episode_card import CARD_SCHEMA_VERSION
from mikirin.hybrid_retrieval import HybridRetriever, HybridSemanticSearch, RETRIEVAL_MODES
from mikirin.schema import CONTENT_SCHEMA
from mikirin.vector_bundle import VectorBundlePublisher
from mikirin.vector_index import EmbeddingConfig, VectorIndexWriter
from mikirin.vector_sources import COMMUNITY_CHUNK_SCHEMA_VERSION


class OfflineSemantic:
    available = True

    def __init__(self):
        self.calls = []

    def status(self):
        return {"available": True, "embedding_fingerprint": "offline-test-only"}

    def search_dual(self, query, **kwargs):
        self.calls.append(kwargs)
        return {"cards": [{"episode_entity_id": "bgm:episode:1", "score": .8},
                          {"episode_entity_id": "bgm:episode:2", "score": .6},
                          {"episode_entity_id": "invented", "score": 1}],
                "chunks": [{"chunk_id": "c1", "score": .9}, {"chunk_id": "c2", "score": .6},
                           {"chunk_id": "c3", "score": .3}, {"chunk_id": "c1", "score": .9},
                           {"chunk_id": "invented", "score": 1}]}


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "synthetic.db"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript(CONTENT_SCHEMA)
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("INSERT INTO build VALUES ('test-build', '', '', 'test', 'ready', '')")
            for i, name in ((1, "合成动画甲"), (2, "合成动画乙")):
                sid = f"bgm:subject:{i}"
                db.execute("INSERT INTO entity VALUES (?, 'subject', ?, 'active')", (sid, name))
                db.execute("INSERT INTO subjects(entity_id,type,name,source_asset_id) VALUES (?,2,?,'asset')", (sid, name))
                db.execute("INSERT INTO entity_fts VALUES (?, 'subject', ?, ?)", (sid, name, name))
                db.execute("INSERT INTO source_entity VALUES (?, 'bgm','subject',?,?,?,?,'resolved',1,NULL)",
                           (f"source{i}", str(i), sid, name, f"https://example.test/subject/{i}"))
            for i, subject, sort, kind in ((1, 1, 1, 0), (2, 2, 1, 0), (3, 1, 2, 0), (4, 1, 8.5, 1)):
                eid = f"bgm:episode:{i}"
                db.execute("INSERT INTO entity VALUES (?, 'episode', ?, 'active')", (eid, f"章节{i}"))
                db.execute("INSERT INTO episodes(entity_id,subject_entity_id,sort,type,name,description,source_asset_id) VALUES (?,?,?,?,?,'合成章节描述','asset')", (eid, f"bgm:subject:{subject}", sort, kind, f"章节{i}"))
            for alias, status, confidence in (("放映部", "confirmed", 1), ("小映部", "provisional", 1)):
                db.execute("INSERT INTO entity_alias(id,entity_id,alias_text,normalized_alias,source_kind,status,confidence) VALUES (?,'bgm:subject:1',?,?,'mined',?,?)", (alias, alias, alias, status, confidence))
            for i in range(1, 10):
                content = f"舞台演出测试{i}，合成评论原文。"
                subject = 2 if i == 9 else 1
                episode = 2 if i == 9 else 1
                self.add_document(db, f"d{i}", content, f"a{i}", subject, episode,
                                  rating=3 if i == 3 else 8, kind="bgm_review" if i in (7, 8) else "bgm_ep_comment")
                db.execute("INSERT INTO document_chunk VALUES (?,?,0,0,?,?,?,'asset')",
                           (f"c{i}", f"d{i}", len(content), content, "hash"))
                db.execute("INSERT INTO document_chunk_fts VALUES (?,?,?)", (f"c{i}", f"d{i}", content))

    @staticmethod
    def add_document(db, key, content, author, subject=1, episode=1, rating=8, kind="bgm_ep_comment"):
        db.execute("INSERT INTO source_document(id,source,source_record_id,title,url,content,content_hash,source_asset_id) VALUES (?,'bgm',?,?,?,?,'hash','asset')",
                   (key, key, key, f"https://example.test/comment/{key}", content))
        db.execute("INSERT INTO community_document VALUES (?,?,?,?,?,?,NULL,'test',1,1,1)",
                   (key, kind, f"bgm:subject:{subject}", f"bgm:episode:{episode}" if episode else None, author, rating))
        db.execute("INSERT INTO document_fts VALUES (?,?,?)", (key, key, content))

    def test_modes_and_read_only_database(self):
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        semantic = OfflineSemantic()
        for mode in RETRIEVAL_MODES:
            result = HybridRetriever(self.database, semantic).search("舞台演出", mode=mode)
            self.assertTrue(result["documents"])
            self.assertEqual(result["mode"], mode)
            self.assertFalse(result["human_approved"])
            self.assertEqual(result["evidence_pack"] is None, mode in ("fts", "anchored"))
        self.assertEqual(len(semantic.calls), 1)
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())

    def test_dual_formula_unique_chunks_and_subject_aggregate(self):
        result = HybridRetriever(self.database, OfflineSemantic()).search("无法词面匹配的记忆")
        first = result["episodes"][0]
        self.assertAlmostEqual(first["score"], .5 * .8 + .3 * .9 + .1 * .6)
        self.assertAlmostEqual(first["components"]["top3_chunk_mean"], .6)
        self.assertEqual(result["subjects"][0]["score"], first["score"])
        self.assertTrue(first["supporting"])
        self.assertEqual(result["provenance"]["build"]["id"], "test-build")

    def test_provenance_keeps_inherited_builds_without_claiming_current_build(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("INSERT INTO build VALUES ('a-inherited-build', 'old', 'old', 'old-code', 'ready', 'inherited')")
            db.execute("INSERT INTO build VALUES ('z-p1-side-build', 'new', 'new', 'new-code', 'ready', 'side-build')")
            db.execute("INSERT INTO schema_metadata VALUES ('content_schema_version', '10')")
            db.execute("PRAGMA user_version=10")
        result = HybridRetriever(self.database).search("舞台演出", mode="fts")
        provenance = result["provenance"]
        self.assertEqual([b["id"] for b in provenance["builds"]],
                         ["a-inherited-build", "test-build", "z-p1-side-build"])
        self.assertEqual(provenance["builds"][2]["notes"], "side-build")
        self.assertIsNone(provenance["build"])
        self.assertIsNone(provenance["current_build_id"])
        self.assertIn("not inferred", provenance["build_identity_policy"])
        self.assertEqual(provenance["schema_metadata"], {"content_schema_version": "10"})
        self.assertEqual(provenance["sqlite_user_version"], 10)
        self.assertGreater(provenance["sqlite_schema_version"], 0)

    def test_provenance_single_or_missing_record_is_not_publication_proof(self):
        retriever = HybridRetriever(self.database)
        provenance = retriever.search("舞台演出", mode="fts")["provenance"]
        self.assertEqual(provenance["builds"], [provenance["build"]])
        self.assertIsNone(provenance["current_build_id"])
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("DROP TABLE build")
            db.execute("DROP TABLE schema_metadata")
        provenance = retriever.search("舞台演出", mode="fts")["provenance"]
        self.assertEqual(provenance["builds"], [])
        self.assertIsNone(provenance["build"])
        self.assertIsNone(provenance["current_build_id"])
        self.assertEqual(provenance["schema_metadata"], {})

    def test_explicit_episode_beats_focus_and_closed_scope(self):
        semantic = OfflineSemantic()
        focus = ConversationFocus(active_subject_id="bgm:subject:2", active_episode_id="bgm:episode:2", confidence=.9)
        result = HybridRetriever(self.database, semantic).search("bgm:episode:1 舞台演出", focus=focus)
        self.assertEqual([e["episode_entity_id"] for e in result["episodes"]], ["bgm:episode:1"])
        self.assertFalse(result["trace"][1]["used"])
        self.assertEqual([r["stage"] for r in result["trace"]], ["explicit", "focus", "alias_fts", "semantic", "disambiguation"])

    def test_focus_pronoun_and_expired_focus(self):
        focus = {"active_subject_id": "bgm:subject:1", "active_episode_id": "bgm:episode:1", "confidence": .9}
        result = HybridRetriever(self.database).search("那集的舞台演出如何", focus=focus, mode="anchored")
        self.assertTrue(result["trace"][1]["used"])
        self.assertEqual({d["episode_entity_id"] for d in result["documents"]}, {"bgm:episode:1"})
        result = HybridRetriever(self.database).search("那集", focus={**focus, "turns_since_refresh": 11}, mode="anchored")
        self.assertFalse(result["trace"][1]["used"])

    def test_canonical_episode_number_and_invalid_number(self):
        result = HybridRetriever(self.database).search("合成动画甲第2集", mode="anchored")
        self.assertEqual(result["episodes"][0]["episode_entity_id"], "bgm:episode:3")
        result = HybridRetriever(self.database).search("合成动画甲第8集", mode="hybrid")
        self.assertIn("episode_scope_unresolved", result["missing"])
        self.assertFalse(result["documents"])

    def test_windows_alias_fallback_and_provisional_weight(self):
        good = HybridRetriever(self.database).search("放映部舞台演出", mode="hybrid")
        weak = HybridRetriever(self.database).search("小映部舞台演出", mode="hybrid")
        self.assertEqual(good["entities"][0]["entity_id"], "bgm:subject:1")
        self.assertAlmostEqual(weak["episodes"][0]["components"]["lexical"],
                               good["episodes"][0]["components"]["lexical"] * .5)

    def test_named_alias_episode_does_not_use_stale_other_subject_focus(self):
        focus = {"active_subject_id": "bgm:subject:2", "confidence": .9}
        result = HybridRetriever(self.database).search("放映部第1集", focus=focus, mode="anchored")
        self.assertFalse(result["trace"][1]["used"])
        self.assertEqual(result["episodes"][0]["episode_entity_id"], "bgm:episode:1")
        result = HybridRetriever(self.database).search("第1集的标题", focus=focus, mode="anchored")
        self.assertTrue(result["trace"][1]["used"])
        self.assertEqual(result["episodes"][0]["episode_entity_id"], "bgm:episode:2")

    def test_explicit_facts_skip_model_retrieval(self):
        semantic = OfflineSemantic()
        result = HybridRetriever(self.database, semantic).search("合成动画甲第1集的标题是什么")
        self.assertFalse(semantic.calls)
        self.assertEqual(result["trace"][3]["skip_reason"], "anchored_factual_query")
        self.assertNotIn("semantic_unavailable", result["missing"])

    def test_closed_disambiguation_rejects_forged_and_mutated_candidates(self):
        def bad(query, candidates):
            candidates[0]["episode_entity_id"] = "invented"
            return "invented"
        result = HybridRetriever(self.database, OfflineSemantic(), bad).search("陌生记忆")
        self.assertIn("disambiguator_outside_closed_candidates", result["missing"])
        self.assertNotIn("invented", [e["episode_entity_id"] for e in result["episodes"]])
        result = HybridRetriever(self.database, OfflineSemantic(), lambda q, c: "bgm:episode:2").search("陌生记忆")
        self.assertEqual(result["episodes"][0]["episode_entity_id"], "bgm:episode:2")
        self.assertEqual(result["subjects"][0]["subject_entity_id"], "bgm:subject:2")

    def test_pack_citations_caps_dissent_and_no_fabrication(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("UPDATE community_document SET author_hash='same' WHERE document_id IN ('d1','d2','d4','d5')")
            db.execute("UPDATE community_document SET author_hash='' WHERE document_id='d6'")
        result = HybridRetriever(self.database).search("合成动画甲的舞台演出如何评价", mode="evidence")
        documents = result["documents"]
        self.assertLessEqual(max(Counter(d["author_hash"] for d in documents).values()), 2)
        for count in Counter(d["document_id"] for d in documents if d["source_kind"] == "bgm_review").values():
            self.assertLessEqual(count / len(documents), .4)
        self.assertFalse(result["evidence_pack"]["dissenting"])
        self.assertIn("claim_level_dissent_missing", result["evidence_pack"]["missing"])
        with closing(sqlite3.connect(self.database)) as db:
            for d in documents:
                original = db.execute("SELECT content FROM source_document WHERE id=?", (d["document_id"],)).fetchone()[0]
                self.assertEqual(d["content"], original[d["char_start"]:d["char_end"]])
                self.assertTrue(d["url"] and d["author_hash"])

    def test_one_review_cannot_be_entire_pack(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("UPDATE community_document SET source_kind='bgm_review', author_hash='same'")
        result = HybridRetriever(self.database).search("bgm:subject:1", mode="evidence")
        self.assertEqual(result["documents"], [])
        self.assertIn("insufficient_diverse_evidence_for_author_or_review_caps", result["evidence_pack"]["missing"])

    def test_corrupt_chunk_and_out_of_scope_semantic_are_ignored(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("UPDATE document_chunk SET content='fabricated' WHERE id='c1'")
        result = HybridRetriever(self.database, OfflineSemantic()).search("bgm:episode:1")
        self.assertAlmostEqual(result["episodes"][0]["components"]["best_chunk"], .6)
        self.assertEqual(len(result["episodes"]), 1)

    def test_report_is_non_citable_and_theme_references_selected_source_only(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("INSERT INTO community_report VALUES ('bgm:subject:1','抽样摘要','[]','[]','[]','[]','[]','{}','digest','v1','v1','offline','run','test-build')")
            db.execute("INSERT INTO discussion_theme VALUES ('theme','bgm:subject:1','bgm:episode:1','演出结构','演出',?,'{}','offline','v1','test-build')", (json.dumps([{"document_id": "d1", "quote": "舞台演出测试1"}, {"document_id": "d1", "quote": "不存在的句子"}]),))
        result = HybridRetriever(self.database).search("合成动画甲第1集舞台演出", mode="evidence")
        pack = result["evidence_pack"]
        self.assertFalse(pack["report_signposts"][0]["citable"])
        self.assertEqual(len(pack["scope-summary"]), 1)
        self.assertTrue(pack["scope-summary"][0]["evidence_reference_only"])

    def test_semantic_errors_do_not_leak_response_or_key(self):
        semantic = OfflineSemantic()
        def fail(*args, **kwargs):
            raise RuntimeError("SECRET_API_KEY response body")
        semantic.search_dual = fail
        result = HybridRetriever(self.database, semantic).search("舞台演出")
        self.assertIn("semantic_failed:RuntimeError", result["missing"])
        self.assertNotIn("SECRET_API_KEY", json.dumps(result))

    def test_stance_closed_quotes_and_provenance(self):
        def classifier(query, candidates):
            return [{"candidate_id": d["candidate_id"], "stance": "dissenting",
                     "claim_quote": d["content"][0:6], "source": "human", "human_verified": True}
                    for d in candidates if d["document_id"] == "d5"]
        result = HybridRetriever(self.database, stance_classifier=classifier).search("合成动画甲如何评价", mode="evidence")
        pack = result["evidence_pack"]
        self.assertEqual(pack["dissenting"][0]["document_id"], "d5")
        self.assertEqual(pack["dissenting"][0]["label_source"], "model")
        self.assertFalse(pack["dissenting"][0]["human_verified"])
        self.assertNotIn("claim_level_dissent_missing", pack["missing"])
        self.assertEqual(result["documents"][0]["document_id"], "d5")

    def test_stance_rejects_invented_quotes_ids_and_conflicts(self):
        def classifier(query, candidates):
            key = candidates[0]["candidate_id"]
            quote = candidates[0]["content"]
            return [{"candidate_id": "invented", "stance": "dissenting", "claim_quote": quote},
                    {"candidate_id": key, "stance": "dissenting", "claim_quote": "invented"},
                    {"candidate_id": key, "stance": "dissenting", "claim_quote": quote},
                    {"candidate_id": key, "stance": "supporting", "claim_quote": quote}]
        result = HybridRetriever(self.database, stance_classifier=classifier).search("合成动画甲如何评价", mode="evidence")
        self.assertFalse(result["evidence_pack"]["dissenting"])
        self.assertIn("invalid_stance_label", result["evidence_pack"]["missing"])
        self.assertIn("conflicting_stance_labels", result["evidence_pack"]["missing"])

    def test_stance_failure_and_impossible_review_dissent(self):
        def classifier(query, candidates):
            return [{"candidate_id": d["candidate_id"], "stance": "dissenting", "claim_quote": d["content"]}
                    for d in candidates if d["document_id"] == "d7"]
        result = HybridRetriever(self.database, stance_classifier=classifier).search("合成动画甲如何评价", mode="evidence", limit=2)
        self.assertFalse(result["evidence_pack"]["dissenting"])
        self.assertIn("dissent_unavailable_under_evidence_caps", result["evidence_pack"]["missing"])
        def failing(query, candidates):
            raise RuntimeError("secret-token")
        result = HybridRetriever(self.database, stance_classifier=failing).search("合成动画甲如何评价", mode="evidence")
        self.assertIn("stance_classifier_failed:RuntimeError", result["evidence_pack"]["missing"])
        self.assertNotIn("secret-token", json.dumps(result))

    def test_arguments_and_missing_db_never_create_database(self):
        with self.assertRaises(ValueError):
            HybridRetriever(self.database).search("x", mode="default")
        with self.assertRaises(ValueError):
            HybridRetriever(self.database).search("x", limit=0)
        missing = self.database.parent / "missing.db"
        with self.assertRaises(sqlite3.OperationalError):
            HybridRetriever(missing).search("x")
        self.assertFalse(missing.exists())

    def test_evaluation_and_blind_review_real_sqlite_end_to_end(self):
        from mikirin.retrieval_evaluation import evaluate_retrieval, prepare_blind_review, score_blind_review
        benchmark = {"version": "offline-integration-v1", "questions": [
            {"id": "1", "query": "放映部第1集如何评价", "gold": None},
            {"id": "2", "query": "舞台演出哪个番", "gold": None}]}
        report = evaluate_retrieval(HybridRetriever(self.database, OfflineSemantic()), benchmark)
        self.assertEqual(len(report["results"]), 8)
        self.assertTrue(all(row["error"] is None for row in report["results"]))
        self.assertEqual(report["results"][-1]["result"]["provenance"]["build"]["id"], "test-build")
        scored = score_blind_review(**prepare_blind_review(report))
        self.assertFalse(scored["complete"])
        self.assertFalse(scored["human_approved"])


class DualIndexTests(unittest.TestCase):
    def test_real_local_indexes_one_mock_embedding_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            database = root / "content.db"
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("CREATE TABLE build(id TEXT)")
                db.execute("INSERT INTO build VALUES ('build')")
            config = EmbeddingConfig(provider="test", base_url="https://example.test/v1", model="test", dimensions=8)
            manifests = root / "data/manifests"
            manifests.mkdir(parents=True)
            (manifests / "runtime_model_config.json").write_text(json.dumps({"roles": {"embedding": {"provider": "test", "base_url": "https://example.test/v1", "model": "test", "dimensions": 8, "key_env": "OFFLINE_TEST_KEY"}}}), encoding="utf-8")
            publisher = VectorBundlePublisher(root / "data/vector_index")
            staging = publisher.create_staging()
            indexes = {}
            for name, kind, version in (("episode_cards", "episode_card", CARD_SCHEMA_VERSION), ("community_chunks", "community_chunk", COMMUNITY_CHUNK_SCHEMA_VERSION)):
                directory = staging / name
                with VectorIndexWriter(directory, dimensions=8, bits=16, embedding_fingerprint=config.fingerprint(), schema_versions={kind: version}) as writer:
                    writer.add(kind, "object1", [1.] * 8, subject_entity_id="subject1", episode_entity_id="episode1")
                    writer.add(kind, "object2", [1.] * 8, subject_entity_id="subject2", episode_entity_id="episode2")
                indexes[name] = directory
            publisher.write_manifest(staging, embedding_fingerprint=config.fingerprint(), index_directories=indexes, source_database={"build_id": "build"})
            publisher.publish(staging)
            with patch.dict(os.environ, {"OFFLINE_TEST_KEY": "offline-not-a-real-key"}):
                semantic = HybridSemanticSearch(root, database)
            try:
                self.assertTrue(semantic.available, semantic.status())
                with patch.object(semantic.client, "embed_query", return_value=[1.] * 8) as embed:
                    result = semantic.search_dual("query", subject_entity_id="subject1")
                embed.assert_called_once_with("query")
                self.assertEqual(len(result["cards"]), 1)
                self.assertEqual(len(result["chunks"]), 1)
                self.assertTrue(semantic.status()["bundle_fingerprint"])
            finally:
                semantic.close()


if __name__ == "__main__":
    unittest.main()
