CONTENT_SCHEMA_VERSION = 10

CONTENT_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE build (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    code_version TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE ingestion_run (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    source_version TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE source_asset (
    id TEXT PRIMARY KEY,
    ingestion_run_id TEXT NOT NULL REFERENCES ingestion_run(id),
    url TEXT,
    local_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    http_status INTEGER,
    UNIQUE(ingestion_run_id, content_hash, local_path)
);

CREATE TABLE entity (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE source_entity (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_entity_type TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entity(id),
    raw_name TEXT NOT NULL,
    source_url TEXT,
    mapping_status TEXT NOT NULL,
    mapping_confidence REAL NOT NULL CHECK(mapping_confidence BETWEEN 0 AND 1),
    source_asset_id TEXT REFERENCES source_asset(id),
    UNIQUE(source, source_entity_type, source_entity_id)
);

CREATE TABLE entity_redirect (
    old_entity_id TEXT PRIMARY KEY,
    canonical_entity_id TEXT NOT NULL REFERENCES entity(id),
    reason TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    CHECK(old_entity_id <> canonical_entity_id)
);

CREATE TABLE entity_alias (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entity(id),
    alias_text TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    lang TEXT,
    source_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    source_asset_id TEXT REFERENCES source_asset(id),
    build_id TEXT REFERENCES build(id),
    license_id TEXT NOT NULL DEFAULT 'unknown',
    UNIQUE(entity_id, normalized_alias, source_kind, build_id)
);

CREATE TABLE resolution_decision (
    id TEXT PRIMARY KEY,
    left_source_entity_id TEXT NOT NULL REFERENCES source_entity(id),
    right_source_entity_id TEXT NOT NULL REFERENCES source_entity(id),
    decision TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    decision_source TEXT NOT NULL,
    decided_at TEXT NOT NULL
);

CREATE TABLE resolution_candidate (
    id TEXT PRIMARY KEY,
    left_entity_id TEXT NOT NULL,
    right_entity_id TEXT NOT NULL,
    relation_kind TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    evidence_json TEXT NOT NULL DEFAULT '{}',
    decision_source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    source_asset_id TEXT REFERENCES source_asset(id),
    UNIQUE(left_entity_id, right_entity_id, relation_kind, decision_source)
);

CREATE TABLE entity_fact (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entity(id),
    fact_kind TEXT NOT NULL,
    value_json TEXT NOT NULL,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    UNIQUE(entity_id, fact_kind, source_asset_id)
);

CREATE TABLE source_document (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    content TEXT NOT NULL,
    structured_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    fetched_at TEXT,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    UNIQUE(source, source_record_id, content_hash)
);

CREATE TABLE community_document (
    document_id TEXT PRIMARY KEY REFERENCES source_document(id),
    source_kind TEXT NOT NULL,
    subject_entity_id TEXT NOT NULL REFERENCES subjects(entity_id),
    episode_entity_id TEXT REFERENCES episodes(entity_id),
    author_hash TEXT NOT NULL DEFAULT '',
    rating INTEGER CHECK(rating BETWEEN 1 AND 10),
    posted_at_raw TEXT,
    license_id TEXT NOT NULL,
    redistributable INTEGER NOT NULL CHECK(redistributable IN (0, 1)),
    attribution_required INTEGER NOT NULL CHECK(attribution_required IN (0, 1)),
    commercial_use_allowed INTEGER NOT NULL CHECK(commercial_use_allowed IN (0, 1))
);

CREATE TABLE document_chunk (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES source_document(id),
    chunk_index INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE document_role_section (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES source_document(id),
    section_order INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    UNIQUE(document_id, section_order)
);

CREATE TABLE subjects (
    entity_id TEXT PRIMARY KEY REFERENCES entity(id),
    type INTEGER NOT NULL,
    name TEXT NOT NULL,
    name_cn TEXT,
    infobox_raw TEXT,
    summary TEXT,
    date TEXT,
    platform TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    meta_tags_json TEXT NOT NULL DEFAULT '[]',
    nsfw INTEGER NOT NULL DEFAULT 0 CHECK(nsfw IN (0, 1)),
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id)
);

CREATE TABLE characters (
    entity_id TEXT PRIMARY KEY REFERENCES entity(id),
    name TEXT NOT NULL,
    role INTEGER,
    infobox_raw TEXT,
    summary TEXT,
    comments INTEGER,
    collects INTEGER,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id)
);

CREATE TABLE persons (
    entity_id TEXT PRIMARY KEY REFERENCES entity(id),
    name TEXT NOT NULL,
    type INTEGER,
    career_json TEXT NOT NULL DEFAULT '[]',
    infobox_raw TEXT,
    summary TEXT,
    comments INTEGER,
    collects INTEGER,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id)
);

CREATE TABLE episodes (
    entity_id TEXT PRIMARY KEY REFERENCES entity(id),
    subject_entity_id TEXT NOT NULL REFERENCES subjects(entity_id),
    sort REAL,
    type INTEGER,
    name TEXT,
    name_cn TEXT,
    airdate TEXT,
    description TEXT,
    disc INTEGER,
    duration TEXT,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id)
);

CREATE TABLE community_report (
    subject_entity_id TEXT PRIMARY KEY REFERENCES subjects(entity_id),
    reception_summary TEXT NOT NULL,
    strengths_json TEXT NOT NULL,
    weaknesses_json TEXT NOT NULL,
    discussion_points_json TEXT NOT NULL,
    key_episodes_json TEXT NOT NULL,
    notable_disagreements_json TEXT NOT NULL,
    coverage_json TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    sampling_policy_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    run_id TEXT NOT NULL,
    build_id TEXT NOT NULL REFERENCES build(id)
);

CREATE TABLE discussion_theme (
    id TEXT PRIMARY KEY,
    subject_entity_id TEXT NOT NULL REFERENCES subjects(entity_id),
    episode_entity_id TEXT REFERENCES episodes(entity_id),
    theme_text TEXT NOT NULL,
    aspect_kind TEXT NOT NULL,
    quotes_json TEXT NOT NULL,
    coverage_json TEXT NOT NULL,
    extraction_model TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    build_id TEXT NOT NULL REFERENCES build(id),
    UNIQUE(subject_entity_id, episode_entity_id, theme_text, extraction_version)
);

CREATE TABLE slang_run (
    id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL REFERENCES build(id),
    processor_version TEXT NOT NULL,
    threshold_version TEXT NOT NULL,
    source_watermark TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    output_digest TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL
);

CREATE TABLE slang_candidate (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES slang_run(id),
    candidate_text TEXT NOT NULL,
    normalized_candidate TEXT NOT NULL,
    bucket TEXT NOT NULL CHECK(bucket IN ('self_reference', 'cross_work', 'single_char')),
    occurrence_count INTEGER NOT NULL CHECK(occurrence_count >= 0),
    document_count INTEGER NOT NULL CHECK(document_count >= 0),
    author_count INTEGER NOT NULL CHECK(author_count >= 0),
    feature_json TEXT NOT NULL DEFAULT '{}',
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(run_id, normalized_candidate)
);

CREATE TABLE slang_candidate_subject_stat (
    run_id TEXT NOT NULL REFERENCES slang_run(id),
    subject_entity_id TEXT NOT NULL REFERENCES subjects(entity_id),
    normalized_candidate TEXT NOT NULL,
    candidate_text TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL CHECK(occurrence_count >= 0),
    document_count INTEGER NOT NULL CHECK(document_count >= 0),
    author_count INTEGER NOT NULL CHECK(author_count >= 0),
    stats_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(run_id, subject_entity_id, normalized_candidate)
);

CREATE TABLE slang_scan_checkpoint (
    run_id TEXT NOT NULL REFERENCES slang_run(id),
    subject_entity_id TEXT NOT NULL REFERENCES subjects(entity_id),
    last_document_id TEXT NOT NULL,
    document_count INTEGER NOT NULL CHECK(document_count >= 0),
    character_count INTEGER NOT NULL CHECK(character_count >= 0),
    ngram_stats_json TEXT NOT NULL DEFAULT '{}',
    completed_at TEXT NOT NULL,
    PRIMARY KEY(run_id, subject_entity_id)
);

CREATE TABLE slang_evidence (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES slang_candidate(id),
    document_id TEXT NOT NULL REFERENCES source_document(id),
    char_start INTEGER NOT NULL CHECK(char_start >= 0),
    char_end INTEGER NOT NULL CHECK(char_end >= char_start),
    context_text TEXT NOT NULL,
    author_hash TEXT NOT NULL DEFAULT '',
    slot_kind TEXT,
    UNIQUE(candidate_id, document_id, char_start, char_end)
);

CREATE TABLE slang_link_candidate (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES slang_candidate(id),
    entity_id TEXT NOT NULL REFERENCES entity(id),
    rank INTEGER NOT NULL CHECK(rank >= 1),
    link_score REAL NOT NULL,
    link_bucket TEXT NOT NULL CHECK(link_bucket IN ('self_reference', 'cross_work', 'single_char')),
    evidence_json TEXT NOT NULL DEFAULT '{}',
    decision TEXT NOT NULL DEFAULT 'pending',
    decision_confidence REAL CHECK(decision_confidence BETWEEN 0 AND 1),
    arbitration_json TEXT,
    UNIQUE(candidate_id, entity_id)
);

CREATE TABLE cooccurrence (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES slang_run(id),
    document_id TEXT NOT NULL REFERENCES source_document(id),
    subject_entity_id TEXT NOT NULL REFERENCES subjects(entity_id),
    src_entity_id TEXT NOT NULL REFERENCES entity(id),
    dst_entity_id TEXT NOT NULL REFERENCES entity(id),
    author_hash TEXT NOT NULL DEFAULT '',
    src_mentions_json TEXT NOT NULL DEFAULT '[]',
    dst_mentions_json TEXT NOT NULL DEFAULT '[]',
    evidence_text TEXT NOT NULL,
    CHECK(src_entity_id < dst_entity_id),
    UNIQUE(run_id, document_id, src_entity_id, dst_entity_id)
);

CREATE TABLE derived_edge (
    src_entity_id TEXT NOT NULL REFERENCES entity(id),
    dst_entity_id TEXT NOT NULL REFERENCES entity(id),
    kind TEXT NOT NULL CHECK(kind IN ('staff_overlap', 'cv_overlap', 'tag_similarity', 'same_season')),
    weight REAL NOT NULL CHECK(weight >= 0),
    evidence_json TEXT NOT NULL DEFAULT '{}',
    build_id TEXT NOT NULL REFERENCES build(id),
    CHECK(src_entity_id < dst_entity_id),
    PRIMARY KEY(src_entity_id, dst_entity_id, kind, build_id)
);

CREATE TABLE node_pagerank (
    node_entity_id TEXT NOT NULL REFERENCES entity(id),
    graph_kind TEXT NOT NULL,
    score REAL NOT NULL CHECK(score >= 0),
    build_id TEXT NOT NULL REFERENCES build(id),
    PRIMARY KEY(node_entity_id, graph_kind, build_id)
);

CREATE TABLE task_pagerank (
    seed_entity_id TEXT NOT NULL REFERENCES entity(id),
    node_entity_id TEXT NOT NULL REFERENCES entity(id),
    graph_kind TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK(rank >= 1),
    score REAL NOT NULL CHECK(score >= 0),
    evidence_json TEXT NOT NULL DEFAULT '{}',
    build_id TEXT NOT NULL REFERENCES build(id),
    CHECK(seed_entity_id <> node_entity_id),
    PRIMARY KEY(seed_entity_id, node_entity_id, graph_kind, build_id)
);

CREATE TABLE implicit_edge_evidence (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES slang_run(id),
    document_id TEXT NOT NULL REFERENCES source_document(id),
    src_entity_id TEXT NOT NULL REFERENCES entity(id),
    dst_entity_id TEXT NOT NULL REFERENCES entity(id),
    relation TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    claim_quote TEXT NOT NULL,
    char_start INTEGER NOT NULL CHECK(char_start >= 0),
    char_end INTEGER NOT NULL CHECK(char_end >= char_start),
    author_hash TEXT NOT NULL DEFAULT '',
    model_fingerprint TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    CHECK(src_entity_id < dst_entity_id),
    UNIQUE(run_id, document_id, src_entity_id, dst_entity_id, relation, claim_quote)
);

CREATE TABLE implicit_edge (
    src_entity_id TEXT NOT NULL REFERENCES entity(id),
    dst_entity_id TEXT NOT NULL REFERENCES entity(id),
    relation TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    weight INTEGER NOT NULL CHECK(weight >= 0),
    evidence_count INTEGER NOT NULL CHECK(evidence_count >= 0),
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    provenance_json TEXT NOT NULL DEFAULT '{}',
    build_id TEXT NOT NULL REFERENCES build(id),
    CHECK(src_entity_id < dst_entity_id),
    PRIMARY KEY(src_entity_id, dst_entity_id, relation, sentiment, build_id)
);

CREATE TABLE score_snapshot (
    subject_entity_id TEXT NOT NULL REFERENCES subjects(entity_id),
    build_id TEXT NOT NULL REFERENCES build(id),
    score REAL,
    rank INTEGER,
    score_details_json TEXT NOT NULL DEFAULT '{}',
    favorite_counts_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(subject_entity_id, build_id)
);

CREATE TABLE rel_subject_subject (
    subject_id TEXT NOT NULL REFERENCES subjects(entity_id),
    relation_type INTEGER NOT NULL,
    related_subject_id TEXT NOT NULL REFERENCES subjects(entity_id),
    rel_order INTEGER,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    PRIMARY KEY(subject_id, relation_type, related_subject_id, source_asset_id)
);

CREATE TABLE rel_subject_character (
    subject_id TEXT NOT NULL REFERENCES subjects(entity_id),
    character_id TEXT NOT NULL REFERENCES characters(entity_id),
    role_type INTEGER NOT NULL,
    rel_order INTEGER,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    PRIMARY KEY(subject_id, character_id, role_type, source_asset_id)
);

CREATE TABLE rel_subject_person (
    subject_id TEXT NOT NULL REFERENCES subjects(entity_id),
    person_id TEXT NOT NULL REFERENCES persons(entity_id),
    position TEXT NOT NULL,
    appear_eps_json TEXT,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    PRIMARY KEY(subject_id, person_id, position, source_asset_id)
);

CREATE TABLE rel_person_character (
    person_id TEXT NOT NULL REFERENCES persons(entity_id),
    subject_id TEXT NOT NULL REFERENCES subjects(entity_id),
    character_id TEXT NOT NULL REFERENCES characters(entity_id),
    summary TEXT,
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    PRIMARY KEY(person_id, subject_id, character_id, source_asset_id)
);

CREATE TABLE rel_person_person (
    person_type TEXT NOT NULL,
    person_id TEXT NOT NULL REFERENCES entity(id),
    related_person_id TEXT NOT NULL REFERENCES entity(id),
    relation_type INTEGER NOT NULL,
    spoiler INTEGER NOT NULL DEFAULT 0 CHECK(spoiler IN (0, 1)),
    ended INTEGER NOT NULL DEFAULT 0 CHECK(ended IN (0, 1)),
    source_asset_id TEXT NOT NULL REFERENCES source_asset(id),
    PRIMARY KEY(person_type, person_id, related_person_id, relation_type, source_asset_id)
);

CREATE INDEX idx_source_entity_entity ON source_entity(entity_id);
CREATE INDEX idx_source_entity_raw_name ON source_entity(raw_name);
CREATE INDEX idx_alias_normalized ON entity_alias(normalized_alias);
CREATE INDEX idx_resolution_candidate_left ON resolution_candidate(left_entity_id);
CREATE INDEX idx_resolution_candidate_right ON resolution_candidate(right_entity_id);
CREATE INDEX idx_entity_fact_entity_kind ON entity_fact(entity_id, fact_kind);
CREATE INDEX idx_source_document_source_record ON source_document(source, source_record_id);
CREATE INDEX idx_community_document_subject_kind
ON community_document(subject_entity_id, source_kind);
CREATE INDEX idx_community_document_episode_kind
ON community_document(episode_entity_id, source_kind);
CREATE INDEX idx_document_chunk_document ON document_chunk(document_id, chunk_index);
CREATE INDEX idx_document_role_section_role ON document_role_section(role);
CREATE INDEX idx_episode_subject_sort ON episodes(subject_entity_id, sort);
CREATE INDEX idx_discussion_theme_subject ON discussion_theme(subject_entity_id);
CREATE INDEX idx_discussion_theme_episode ON discussion_theme(episode_entity_id);
CREATE INDEX idx_slang_candidate_run_score ON slang_candidate(run_id, score DESC);
CREATE INDEX idx_slang_subject_stat_candidate
ON slang_candidate_subject_stat(run_id, normalized_candidate);
CREATE INDEX idx_slang_checkpoint_completed
ON slang_scan_checkpoint(run_id, completed_at);
CREATE INDEX idx_slang_evidence_candidate ON slang_evidence(candidate_id);
CREATE INDEX idx_slang_link_candidate_status ON slang_link_candidate(decision, link_score DESC);
CREATE INDEX idx_cooccurrence_pair ON cooccurrence(src_entity_id, dst_entity_id);
CREATE INDEX idx_cooccurrence_document ON cooccurrence(document_id);
CREATE INDEX idx_derived_edge_src_kind ON derived_edge(src_entity_id, kind, weight DESC);
CREATE INDEX idx_derived_edge_dst_kind ON derived_edge(dst_entity_id, kind, weight DESC);
CREATE INDEX idx_node_pagerank_graph_score ON node_pagerank(graph_kind, score DESC);
CREATE INDEX idx_task_pagerank_seed
ON task_pagerank(seed_entity_id, graph_kind, rank, build_id);
CREATE INDEX idx_implicit_edge_src ON implicit_edge(src_entity_id, weight DESC);
CREATE INDEX idx_implicit_edge_dst ON implicit_edge(dst_entity_id, weight DESC);
CREATE INDEX idx_implicit_evidence_pair ON implicit_edge_evidence(src_entity_id, dst_entity_id);
CREATE INDEX idx_subject_person_person ON rel_subject_person(person_id);
CREATE INDEX idx_person_character_character ON rel_person_character(character_id);

CREATE VIRTUAL TABLE entity_fts USING fts5(
    entity_id UNINDEXED,
    entity_type UNINDEXED,
    canonical_name,
    aliases,
    tokenize='trigram'
);

CREATE VIRTUAL TABLE document_fts USING fts5(
    document_id UNINDEXED,
    title,
    content,
    tokenize='trigram'
);

CREATE VIRTUAL TABLE document_chunk_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    content,
    tokenize='trigram'
);

CREATE VIRTUAL TABLE community_report_fts USING fts5(
    subject_entity_id UNINDEXED,
    reception_summary,
    strengths_json,
    weaknesses_json,
    discussion_points_json,
    key_episodes_json,
    notable_disagreements_json,
    tokenize='trigram'
);

CREATE VIRTUAL TABLE discussion_theme_fts USING fts5(
    id UNINDEXED,
    subject_entity_id UNINDEXED,
    episode_entity_id UNINDEXED,
    theme_text,
    quotes_json,
    tokenize='trigram'
);
"""
