import sqlite3

from retrieval.reranker import HybridRanker


def test_geometry_table_and_reranker_geometry_score_exist():
    conn = sqlite3.connect("data/database/bim.db")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()

    assert ("geometry",) in tables, "geometry table should exist for geometry-aware reranking"

    guid = conn.execute("SELECT guid FROM elements LIMIT 1").fetchone()
    assert guid is not None, "database must contain at least one element for geometry-based tests"

    ranker = HybridRanker()
    candidates = {
        "ids": [guid[0]],
        "distances": [0.0],
        "metadatas": [{"ifc_class": "IfcWall", "name": "Sample Wall"}],
    }
    result = ranker.rank_candidates(candidates, "wall in room 101", (0.0, 0.0, 0.0, 1.0, 1.0, 3.0))

    assert result and isinstance(result[0]["st_raw"], float)
    assert 0.0 <= result[0]["st_raw"] <= 1.0
