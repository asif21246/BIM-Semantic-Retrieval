import os
import json
import chromadb
import shutil
from loguru import logger
from tqdm import tqdm

def build_vector_index():
    emb_path = "data/processed/embeddings.json"
    vector_dir = "data/database/chroma_db"
    
    if not os.path.exists(emb_path):
        logger.error("No embedding cache discovered. Execute stage 3 first.")
        return
        
    if os.path.exists(vector_dir):
        logger.info("Hard-wiping stale sparse vector indexing folders...")
        shutil.rmtree(vector_dir, ignore_errors=True)
        
    os.makedirs(vector_dir, exist_ok=True)
    logger.info("[STAGE 4] Compiling fresh high-density persistent ChromaDB indices...")
    
    with open(emb_path, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    chroma_client = chromadb.PersistentClient(path=vector_dir)
    collection = chroma_client.get_or_create_collection(name="bim_semantic_knowledge_base")

    batch_size = 1000
    for i in tqdm(range(0, len(payloads), batch_size), desc="Indexing Batches"):
        chunk = payloads[i:i+batch_size]
        
        collection.add(
            embeddings=[x["vector"] for x in chunk],
            documents=[x["context"] for x in chunk],
            metadatas=[x["meta"] for x in chunk],
            ids=[x["guid"] for x in chunk]
        )
    logger.success(f"[STAGE 4] Complete! Populated {collection.count()} high-density element clusters.")

if __name__ == '__main__':
    build_vector_index()
