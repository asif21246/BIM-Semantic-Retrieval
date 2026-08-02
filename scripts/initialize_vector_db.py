import os
import sqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm

def build_vector_knowledge_base():
    db_path = "data/database/bim.db"
    vector_dir = "data/database/chroma_db"
    os.makedirs(vector_dir, exist_ok=True)

    # 1. Establish connections to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT guid, ifc_class, name, storey, room, material FROM elements")
    rows = cursor.fetchall()
    
    if not rows:
        logger.error("Relational data empty! Run populate_db.py before building vector space.")
        return
        
    logger.info(f"Loaded {len(rows)} elements from SQLite. Bootstrapping AI Vector Index...")

    # 2. Initialize dense embedding transformer engine (MPNet Base V2)
    model_name = "all-mpnet-base-v2"
    logger.info(f"Loading embedding transformer framework: {model_name}")
    encoder = SentenceTransformer(model_name)

    # 3. Initialize ChromaDB persistent storage client layer
    chroma_client = chromadb.PersistentClient(path=vector_dir)
    collection = chroma_client.get_or_create_collection(name="bim_semantic_knowledge_base")

    # 4. Process assets into rich textual semantic contexts
    documents = []
    metadatas = []
    ids = []

    logger.info("Generating text embedding vector tensors...")
    for row in tqdm(rows, desc="Vectorization Loop"):
        guid, ifc_class, name, storey, room, material = row
        
        # Build multi-disciplinary context strings for spatial retrieval optimization
        semantic_context = f"Element Class: {ifc_class}. Asset Name: {name}. Located at Storey Level: {storey}, Room/Zone: {room}. Construction Material: {material}."
        
        documents.append(semantic_context)
        metadatas.append({
            "ifc_class": ifc_class,
            "name": name,
            "storey": storey,
            "room": room,
            "material": material
        })
        ids.append(guid)

    # 5. Batch insert vector payloads into ChromaDB collections (processing in chunks of 5000)
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        chunk_docs = documents[i:i+batch_size]
        chunk_meta = metadatas[i:i+batch_size]
        chunk_ids = ids[i:i+batch_size]
        
        logger.info(f"Encoding and inserting batch chunk: {i} to {i+len(chunk_docs)}")
        chunk_embeddings = encoder.encode(chunk_docs, convert_to_numpy=True).tolist()
        
        collection.add(
            embeddings=chunk_embeddings,
            documents=chunk_docs,
            metadatas=chunk_meta,
            ids=chunk_ids
        )

    conn.close()
    logger.success(f"SUCCESS: Vector Space Complete! Loaded {collection.count()} vectors into ChromaDB.")

if __name__ == '__main__':
    build_vector_knowledge_base()
