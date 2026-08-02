import sys
from loguru import logger
from retrieval.semantic_search import BIMSemanticSearchEngine
from retrieval.reranker.py import BIMPropertyReranker
from parser.relationship_extractor import build_bim_knowledge_graph

def run_central_retrieval_pipeline(site_log_query: str):
    logger.info("======================================================================")
    logger.info(f"STARTING HYBRID RETRIEVAL PIPELINE FOR QUERY: '{site_log_query}'")
    logger.info("======================================================================")
    
    # 1. Traversal Step: Extract structural constraints from Knowledge Graph
    knowledge_graph = build_bim_knowledge_graph()
    
    # 2. Dense Stage: Query ChromaDB Vector Store for top-5 candidates
    try:
        search_engine = BIMSemanticSearchEngine()
        vector_hits = search_engine.execute_top_k_search(site_log_query, k=5)
    except Exception as e:
        logger.error(f"Vector search execution interrupted: {e}")
        return

    # 3. Rerank Stage: Apply cross-encoder attribute re-scoring
    reranker = BIMPropertyReranker()
    final_matches = reranker.rerank_candidates(site_log_query, vector_hits)
    
    # 4. Result Presentation Layer
    print("\n?? FINAL SEMANTIC MATCHING RESULTS (HIGH-CONFIDENCE HIGHLIGHTS):")
    print("-----------------------------------------------------------------------------------------")
    for idx, match in enumerate(final_matches[:3], 1):
        guid = match["guid"]
        score = match["rerank_score"]
        meta = match["metadata"]
        print(f" [{idx}] MATCH CONFIDENCE WEIGHT: {score:.4f}")
        print(f"     -> Structural GUID:    {guid}")
        print(f"     -> IFC Data Class:     {meta.get('ifc_class', 'Unknown')}")
        print(f"     -> Component Name:     {meta.get('name', 'Unnamed')}")
        print(f"     -> Material Property:  {meta.get('material', 'Generic')}")
        print("-----------------------------------------------------------------------------------------")
    
    logger.success("Hybrid Semantic Search pipeline completed successfully.")

if __name__ == '__main__':
    # Test query mimicking your real construction log anomalies
    sample_query = "Wall crack noticed in main structural concrete element"
    run_central_retrieval_pipeline(sample_query)
