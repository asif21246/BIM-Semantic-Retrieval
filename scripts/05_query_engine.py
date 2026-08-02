import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from loguru import logger

def query_semantic_engine():
    chroma_client = chromadb.PersistentClient(path="data/database/chroma_db")
    collection = chroma_client.get_collection("bim_semantic_knowledge_base")
    encoder = SentenceTransformer("all-mpnet-base-v2")
    
    logs_df = pd.read_csv("data/logs/site_logs.csv")
    test_query = logs_df["log_text"].iloc[0]
    logger.info(f"[STAGE 5] Simulating site query match for: '{test_query}'")
    
    query_vec = encoder.encode(test_query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=3)
    
    print("\n----------------- SEMANTIC RESULTS -----------------")
    for doc, match_id in zip(results["documents"][0], results["ids"][0]):
        print(f" -> GUID Match: {match_id} | Context: {doc}")
    print("----------------------------------------------------\n")

if __name__ == '__main__':
    query_semantic_engine()
