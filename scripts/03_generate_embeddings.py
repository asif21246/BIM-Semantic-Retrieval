import sqlite3
import json
import os
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm

def generate_vector_embeddings():
    db_path, out_dir = "data/database/bim.db", "data/processed"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT guid, ifc_class, name FROM elements")
    rows = cursor.fetchall()
    
    logger.info(f"[STAGE 3] Encoding {len(rows)} elements using all-mpnet-base-v2...")
    encoder = SentenceTransformer("all-mpnet-base-v2")
    
    payloads = []
    for row in tqdm(rows, desc="Vectorization Loop"):
        guid, cls, name = row
        context = f"Class: {cls}. Name: {name}."
        embedding = encoder.encode(context, convert_to_numpy=True).tolist()
        payloads.append({"guid": guid, "context": context, "vector": embedding, "meta": {"ifc_class": cls, "name": name}})
        
    with open(os.path.join(out_dir, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(payloads, f)
    conn.close()
    logger.success("[STAGE 3] Text embeddings generated and cached safely.")

if __name__ == '__main__':
    generate_vector_embeddings()
