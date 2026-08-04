import sqlite3
import json
import os
import re
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm


def _normalize_name_for_semantics(name: str) -> str:
    if not name:
        return "generic component"

    text = str(name)
    text = text.replace("M_", "").replace("_", " ")
    text = re.sub(r"[:\-\/\\]+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in text.split() if len(token) > 2 and token.lower() not in {"type", "types", "public", "generic", "standard"}]
    if not tokens:
        return "generic component"
    return " ".join(tokens[:10])


def generate_enriched_vector_embeddings():
    db_path = "data/database/bim.db"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT guid, ifc_class, name, storey, room, material FROM elements")
    rows = cursor.fetchall()

    if not rows:
        logger.error("Relational data empty! Run stage 2 before generating embeddings.")
        conn.close()
        return

    logger.info(f"[STAGE 3] Encoding {len(rows)} elements using real BIM product-family semantics.")
    encoder = SentenceTransformer("all-mpnet-base-v2")

    payloads = []
    for row in tqdm(rows, desc="Semantic Vectorization Loop"):
        guid, cls, name, storey, room, material = row

        s_guid = str(guid).strip()
        s_cls = str(cls) if cls else "IfcWall"
        s_name = str(name) if name else "Unnamed"
        s_storey = str(storey) if storey else "Level 1"
        s_room = str(room) if room else "Room 101"
        s_mat = str(material) if material else "Generic"
        s_type = _normalize_name_for_semantics(s_name)

        context = f"Class: {s_cls}. Product family: {s_type}. Material: {s_mat}. Storey: {s_storey}. Room: {s_room}."

        embedding = encoder.encode(context, convert_to_numpy=True).tolist()

        payloads.append({
            "guid": s_guid,
            "context": context,
            "vector": embedding,
            "meta": {
                "ifc_class": s_cls,
                "name": s_name,
                "storey": s_storey,
                "room": s_room,
                "material": s_mat,
                "product_family": s_type,
            }
        })

    with open(os.path.join(out_dir, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(payloads, f)
    conn.close()
    logger.success("[STAGE 3] Semantic BIM product-family embeddings generated and cached.")

if __name__ == '__main__':
    generate_enriched_vector_embeddings()
