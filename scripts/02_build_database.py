import os
import json
import sqlite3
from loguru import logger

def build_relational_database():
    json_dir, db_path = "data/processed", "data/database/bim.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM elements")
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    logger.info(f"[STAGE 2] Building relational database rows from {len(json_files)} datasets...")

    inserted = 0
    for filename in json_files:
        with open(os.path.join(json_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                cursor.execute("INSERT OR REPLACE INTO elements (guid, ifc_class, name, storey, room, material) VALUES (?, ?, ?, ?, ?, ?)",
                               (item["guid"], item["class"], item["name"], "UnknownStorey", "UnknownRoom", "Generic"))
                inserted += 1
    conn.commit()
    conn.close()
    logger.success(f"[STAGE 2] Complete. Relational tables holding {inserted} elements.")

if __name__ == '__main__':
    build_relational_database()
