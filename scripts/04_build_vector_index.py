import os
import json
import chromadb
from loguru import logger
from tqdm import tqdm

def build_vector_index():
    emb_path, vector_dir = "data/processed/embeddings.json", "data/database/chroma_db"
    if not os.path.exists(emb_path):
        logger.error("No embed records found. Execute stage 3 first.")
        return
        
    logger.info("[STAGE 4] Launching ChromaDB persistence indexing arrays...")
    with open(emb_path, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    chroma_client = chromadb.PersistentClient(path=vector_dir)
    try: chroma_client.delete_collection("bim_semantic_knowledge_base")
    except: pass
    collection = chroma_client.get_or_create_collection("bim_semantic_knowledge_base")

    batch_size = 1000
    for i in tqdm(range(0, len(payloads), batch_size), desc="Uploading Index Batches"):
        chunk = payloads[i:i+batch_size]
        collection.add(
            embeddings=[x["vector"] for x in chunk],
            documents=[x["context"] for x in chunk],
            metadatas=[x["meta"] for x in chunk],
            ids=[x["guid"] for x in chunk]
        )
    logger.success(f"[STAGE 4] Indexed {collection.count()} elements inside search space.")

if __name__ == '__main__':
    build_vector_index()
