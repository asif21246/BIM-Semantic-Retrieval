import chromadb
from sentence_transformers import SentenceTransformer
from loguru import logger

class BIMSemanticSearchEngine:
    def __init__(self, chroma_path="data/database/chroma_db", model_name="all-mpnet-base-v2"):
        logger.info("Initializing Object-Oriented Semantic Search Engine...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection("bim_semantic_knowledge_base")
        self.encoder = SentenceTransformer(model_name)

    def execute_top_k_search(self, query_text: str, k: int = 3) -> dict:
        """
        Executes a dense vector proximity query against the 21,021 indexed elements.
        """
        logger.info(f"Executing Vector Proximity Search for query: '{query_text}'")
        
        # 1. Transform raw search query text into a vector tensor
        query_vector = self.encoder.encode(query_text, convert_to_numpy=True).tolist()
        
        # 2. Match nearest embedding profiles inside ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        return results

    def execute_filtered_search(self, query_text: str, target_class: str, k: int = 3) -> dict:
        """
        Executes a composite search that filters by specific engineering attribute classes.
        """
        logger.info(f"Executing Attribute-Filtered Search targeting class: {target_class}")
        query_vector = self.encoder.encode(query_text, convert_to_numpy=True).tolist()
        
        # Apply standard relational property metadata filtering constraints
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where={"ifc_class": target_class}
        )
        return results

if __name__ == '__main__':
    # Initialize the search engine entity definition
    try:
        search_engine = BIMSemanticSearchEngine()
        logger.success("Search engine successfully linked to active data directories.")
    except Exception as e:
        logger.warning(f"Vector collection index not initialized yet: {e}")
