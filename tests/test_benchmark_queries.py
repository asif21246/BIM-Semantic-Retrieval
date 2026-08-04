import importlib.util


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_mod", "scripts/06_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_queries_use_real_room_and_storey_context():
    module = load_eval_module()
    engine = module.RigorousBIMEvaluationEngine200()

    assert engine.benchmark_queries, "benchmark should contain at least one query"

    query = engine.benchmark_queries[0]["query"].lower()
    assert "room" in query or "level" in query or "storey" in query, (
        "benchmark queries should use the real room/storey context from BIM metadata"
    )
    assert "current facility context" not in query, (
        "benchmark queries should not be generic leakage templates"
    )
    assert any(term in query for term in ["issue", "leak", "repair", "defect", "maintenance", "fault", "inspection", "door", "wall", "pipe", "ceiling"]), (
        "benchmark queries should resemble realistic facility-management requests"
    )
