import re
import chromadb
from sentence_transformers import SentenceTransformer
from loguru import logger


class BIMSemanticSearchEngine:
    def __init__(self, chroma_path="data/database/chroma_db", model_name="all-mpnet-base-v2"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection("bim_semantic_knowledge_base")
        self.encoder = SentenceTransformer(model_name)
        logger.info("Modular Semantic Search Core initialized.")

    def _expand_query(self, query_text: str) -> str:
        if not query_text:
            return " facility component "

        q = str(query_text).lower().strip()
        expansions = [q]

        room_match = re.search(r"room\s*\d+[a-zA-Z0-9-]*", q)
        level_match = re.search(r"level\s*\d+[a-zA-Z0-9-]*|storey\s*\d+[a-zA-Z0-9-]*|floor\s*\d+[a-zA-Z0-9-]*", q)
        if room_match:
            expansions.append(f"room {room_match.group(0).replace('room ', '').strip()}")
        if level_match:
            expansions.append(level_match.group(0))

        lexical_map = {
            "sprinkler": ["sprinkler", "fire sprinkler", "pendent sprinkler", "spray sprinkler", "head"],
            "fixture": ["fixture", "water closet", "sink", "lavatory", "toilet", "equipment"],
            "fitting": ["fitting", "pipe fitting", "elbow", "tee", "union", "connector", "joint"],
            "pipe": ["pipe", "pipeline", "segment", "conduit", "tube", "duct"],
            "wall": ["wall", "partition wall", "outer wall", "crack", "moisture", "seal", "plaster"],
            "door": ["door", "entry door", "fire door", "does not close", "hinge", "frame", "leaf"],
            "window": ["window", "glazing", "opening", "seal leak", "ventilation", "frame"],
            "ceiling": ["ceiling", "ceiling panel", "acoustic panel", "water stain", "drop panel", "suspended ceiling"],
            "leak": ["leak", "leaking", "water damage", "drip", "seepage", "moisture"],
            "repair": ["repair", "maintenance", "inspection", "fault", "malfunction", "defect"],
            "inspection": ["inspection", "check", "condition", "service", "diagnostic"],
            "noise": ["noise", "vibration", "rattle", "hum"],
            "flow": ["flow", "airflow", "ventilation", "pressure", "supply"],
            "fault": ["fault", "malfunction", "issue", "problem", "failure"],
            "damage": ["damage", "crack", "deterioration", "defect", "breakage"],
        }

        for key, variants in lexical_map.items():
            if key in q:
                expansions.extend(variants)

        expanded = " ".join(dict.fromkeys(expansions))
        return expanded if expanded else q

    def _metadata_priority_score(self, query_text: str, metadata: dict) -> float:
        q = str(query_text).lower()
        q_tokens = {token for token in re.findall(r"[a-zA-Z0-9]+", q) if len(token) > 2}
        score = 0.0

        room = str(metadata.get("room", "")).lower()
        storey = str(metadata.get("storey", "")).lower()
        if room and room in q:
            score += 1.5
        if storey and storey in q:
            score += 1.5

        name = str(metadata.get("name", "")).lower()
        for token in q_tokens:
            if token in name:
                score += 0.5

        ifc_class = str(metadata.get("ifc_class", "")).lower()
        for token in q_tokens:
            if token in ifc_class:
                score += 0.3

        for token in q_tokens:
            if token in {"leak", "water", "defect", "repair", "inspection", "fault", "crack", "door", "wall", "pipe", "ceiling", "ventilation", "noise"}:
                if token in name or token in ifc_class:
                    score += 0.2

        material = str(metadata.get("material", "")).lower()
        if material and material in q:
            score += 0.2

        return score

    def fetch_top_candidates(self, query_text: str, k: int = 20) -> dict:
        """Fetches the top-K semantic candidates using dense similarity plus metadata-aware re-ranking."""
        if k <= 0:
            raise ValueError("k must be positive.")

        if self.collection.count() == 0:
            raise RuntimeError("Chroma collection is empty; build the vector index first.")

        expanded_query = self._expand_query(query_text)
        query_vector = self.encoder.encode(expanded_query, convert_to_numpy=True).astype("float32")
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(k, self.collection.count()),
            include=["distances", "metadatas", "documents"]
        )

        ids = results.get("ids", [])
        distances = results.get("distances", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])

        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if distances and isinstance(distances[0], list):
            distances = distances[0]
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if documents and isinstance(documents[0], list):
            documents = documents[0]

        seen = set()
        ranked = []
        for idx, candidate_id in enumerate(ids):
            sid = str(candidate_id).strip()
            if sid in seen:
                continue
            seen.add(sid)

            meta = metadatas[idx] if idx < len(metadatas) else {}
            relevance_bonus = self._metadata_priority_score(query_text, meta)
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            score = float((1.0 / (1.0 + distance)) + (0.15 * relevance_bonus))
            ranked.append({
                "id": sid,
                "distance": distance,
                "meta": meta,
                "document": documents[idx] if idx < len(documents) else "",
                "score": score,
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)

        unique_ids = [item["id"] for item in ranked]
        unique_distances = [item["distance"] for item in ranked]
        unique_metas = [item["meta"] for item in ranked]
        unique_docs = [item["document"] for item in ranked]

        return {
            "ids": unique_ids,
            "distances": unique_distances,
            "metadatas": unique_metas,
            "documents": unique_docs,
        }
