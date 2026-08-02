import re
from loguru import logger

class BIMPropertyReranker:
    def __init__(self):
        logger.info("Initializing Heuristic Property-Based Cross-Reranker Engine...")

    def compute_similarity_score(self, log_text: str, element_context: str) -> float:
        """
        Computes a sharp matching weight by checking intersection keywords 
        between the raw construction site note and parsed building component contexts.
        """
        # Convert text blocks to lowercase tokens for uniform lookup normalization
        log_words = set(re.findall(r'\w+', log_text.lower()))
        context_words = set(re.findall(r'\w+', element_context.lower()))
        
        # Intersect matching strings
        overlapping_keywords = log_words.intersection(context_words)
        
        # Calculate a baseline intersection score
        base_score = len(overlapping_keywords) / max(len(log_words), 1)
        return float(base_score)

    def rerank_candidates(self, log_text: str, chroma_results: dict) -> list:
        """
        Processes vector search result hits and re-sorts them based on property accuracy.
        """
        logger.info("Executing Stage-2 Cross-Property attribute matching and reranking...")
        
        reranked_pool = []
        
        # Unpack ChromaDB vector result rows
        if not chroma_results or "documents" not in chroma_results or not chroma_results["documents"]:
            return []
            
        documents = chroma_results["documents"][0]
        ids = chroma_results["ids"][0]
        metadatas = chroma_results["metadatas"][0]
        
        for idx in range(len(documents)):
            doc_text = documents[idx]
            element_id = ids[idx]
            meta = metadatas[idx]
            
            # Extract score variations
            property_weight = self.compute_similarity_score(log_text, doc_text)
            
            reranked_pool.append({
                "guid": element_id,
                "context": doc_text,
                "metadata": meta,
                "rerank_score": property_weight
            })
            
        # Re-sort descending: highest attribute overlap score jumps to top-1 priority position
        reranked_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_pool

if __name__ == '__main__':
    reranker = BIMPropertyReranker()
    # Dummy mock run calculation test to verify system rules
    test_context = "Class: IfcWall. Name: Exterior Wall Part 2. Material: Concrete."
    score = reranker.compute_similarity_score("Wall crack noticed", test_context)
    logger.success(f"Reranker scoring module active. Test score calculation output: {score}")
