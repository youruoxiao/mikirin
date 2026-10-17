from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchIndexReport:
    entities: int
    documents: int
    chunks: int


class SearchIndexer:
    @staticmethod
    def rebuild(database: Path) -> SearchIndexReport:
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("DELETE FROM entity_fts")
            db.execute("DELETE FROM document_fts")
            db.execute("DELETE FROM document_chunk_fts")
            db.execute(
                """
                INSERT INTO entity_fts(entity_id, entity_type, canonical_name, aliases)
                SELECT e.id, e.entity_type, e.canonical_name,
                       coalesce(group_concat(a.alias_text, ' '), '')
                FROM entity e
                LEFT JOIN entity_alias a ON a.entity_id = e.id
                  AND a.status NOT IN ('rejected', 'expired', 'invalidated')
                  AND (a.source_kind!='mined' OR a.status IN ('confirmed', 'provisional'))
                GROUP BY e.id, e.entity_type, e.canonical_name
                """
            )
            db.execute(
                """
                INSERT INTO document_fts(document_id, title, content)
                SELECT id, title, content FROM source_document
                """
            )
            db.execute(
                """
                INSERT INTO document_chunk_fts(chunk_id, document_id, content)
                SELECT id, document_id, content FROM document_chunk
                """
            )
            entities = int(db.execute("SELECT count(*) FROM entity_fts").fetchone()[0])
            documents = int(db.execute("SELECT count(*) FROM document_fts").fetchone()[0])
            chunks = int(db.execute("SELECT count(*) FROM document_chunk_fts").fetchone()[0])
        return SearchIndexReport(entities=entities, documents=documents, chunks=chunks)
