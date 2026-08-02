import os
import json
import chromadb
from loguru import logger
from tqdm import tqdm

def build_vector_index():
    emb_path, vector_dir = "data/processed/embeddings.json", "data/database/chroma_db"
    if not os.path.exists(emb_path):
        logger.error("No embedded records discovered. Execute stage 3 first.")
        return
        
    logger.info("[STAGE 4] Launching safe data-cleansing ChromaDB indexing array...")
    with open(emb_path, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    chroma_client = chromadb.PersistentClient(path=vector_dir)
    
    # Clean rebuild to clear old corrupted indices
    try: 
        chroma_client.delete_collection("bim_semantic_knowledge_base")
    except: 
        pass
    collection = chroma_client.get_or_create_collection("bim_semantic_knowledge_base")

    batch_size = 1000
    for i in tqdm(range(0, len(payloads), batch_size), desc="Uploading Index Batches"):
        chunk = payloads[i:i+batch_size]
        
        cleaned_embeddings = []
        cleaned_documents = []
        cleaned_metadatas = []
        cleaned_ids = []
        
        for x in chunk:
            raw_meta = x.get("meta", {})
            
            # Strict validation enforcement: force cast fields and swap None with strings
            safe_meta = {
                "ifc_class": str(raw_meta.get("ifc_class")) if raw_meta.get("ifc_class") is not None else "UnknownClass",
                "name": str(raw_meta.get("name")) if raw_meta.get("name") is not None else "Unnamed"
            }
            
            cleaned_embeddings.append(x["vector"])
            cleaned_documents.append(x["context"])
            cleaned_metadatas.append(safe_meta)
            cleaned_ids.append(str(x["guid"]))
            
        try:
            collection.add(
                embeddings=cleaned_embeddings,
                documents=cleaned_documents,
                metadatas=cleaned_metadatas,
                ids=cleaned_ids
            )
        except Exception as e:
            logger.error(f"Error loading index batch partition at offset {i}: {e}")
            continue
            
    logger.success(f"[STAGE 4] Complete! Securely indexed {collection.count()} elements inside search space.")

if __name__ == '__main__':
    build_vector_index()
