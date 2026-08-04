import os
import sys
from loguru import logger

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from retrieval.semantic_search import BIMSemanticSearchEngine
from retrieval.reranker import HybridRanker


def run_scientific_query_engine(query: str = None):
    PROJECT_WEIGHTS = {"ws": 0.50, "wt": 0.20, "wp": 0.15, "wc": 0.15}

    query = query or "Find the most relevant facility component for maintenance in the current area."
    logger.info("[STAGE 5] Launching retrieval orchestration for query: %s", query)

    search_engine = BIMSemanticSearchEngine()
    reranker = HybridRanker(weights=PROJECT_WEIGHTS)

    candidates = search_engine.fetch_top_candidates(query, k=20)
    scored_pool = reranker.rank_candidates(candidates, query, reference_bbox=(0.0, 0.0, 0.0, 4.0, 0.3, 3.0))

    if not scored_pool:
        logger.warning("No candidates returned for the supplied query.")
        return []

    semantic_only_sorted = sorted(scored_pool, key=lambda x: x["ss_raw"], reverse=True)
    hybrid_sorted = sorted(scored_pool, key=lambda x: x["final_score"], reverse=True)

    print("\n=========================================================================================")
    print("   PRIORITY 3: STRUCTURAL RETRIEVAL DISPLACEMENT VERIFICATION (BEFORE / AFTER COMPARISON) ")
    print("=========================================================================================")
    print(" STAGE 1: RAW SEMANTIC SEARCH RANKS             STAGE 2: PROPOSED HYBRID RE-RANKED RANKS")
    print(" ----------------------------------------       -----------------------------------------")
    for r in range(min(4, len(scored_pool))):
        sem_el = semantic_only_sorted[r]
        hyb_el = hybrid_sorted[r]
        print(f"  Rank {r+1}: {sem_el['guid'][:10]}... ({sem_el['ifc_class']})  -->   Rank {r+1}: {hyb_el['guid'][:10]}... ({hyb_el['ifc_class']})")
    print("=========================================================================================\n")

    print("=========================================================================================")
    print("   PRIORITY 7: EXPLAINABLE AI (XAI) MATHEMATICAL EVALUATION COMPLIANCE REPORT             ")
    print("=========================================================================================")
    for idx, match in enumerate(hybrid_sorted[:3], 1):
        print(f" [{idx}] Identified Target Component Unique Hash GUID: {match['guid']}")
        print(f"     -> Structural STEP Entity Class: {match['ifc_class']}")
        print(f"     -> Component Parameter Model Name: {match['name']}")

        c_ss = match['ss_raw'] * match['ws']
        c_st = match['st_raw'] * match['wt']
        c_sp = match['sp_raw'] * match['wp']
        c_sc = match['sc_raw'] * match['wc']

        print(f"     -> Semantic Similarity (Ss):    {match['ss_raw']:.4f}  | Weight (ws): {match['ws']:.2f} | Contribution: {c_ss:.4f}")
        print(f"     -> Spatial Proximity (St):       {match['st_raw']:.4f}  | Weight (wt): {match['wt']:.2f} | Contribution: {c_st:.4f} ( Clearance: {match['distance']:.2f}m)")
        print(f"     -> Property Similarity (Sp):     {match['sp_raw']:.4f}  | Weight (wp): {match['wp']:.2f} | Contribution: {c_sp:.4f}")
        print(f"     -> Class Compatibility (Sc):     {match['sc_raw']:.4f}  | Weight (wc): {match['wc']:.2f} | Contribution: {c_sc:.4f}")
        print(f"     => AGGREGATED MANUSCRIPT COMPOSITE WEIGHT VERDICT: {match['final_score']:.4f} ({match['final_score']*100:.1f}%)")
        print("-----------------------------------------------------------------------------------------")
    print("=========================================================================================\n")
    logger.success("[STAGE 5] Multi-criteria spatial-semantic query loops executed successfully.")
    return hybrid_sorted

if __name__ == '__main__':
    run_scientific_query_engine()
