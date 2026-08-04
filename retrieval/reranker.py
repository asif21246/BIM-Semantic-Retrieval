import re
import numpy as np
import sqlite3
from loguru import logger

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape as shape_util


class HybridRanker:
    def __init__(self, weights: dict = None):
        self.weights = weights if weights else {
            "ws": 0.50,
            "wt": 0.20,
            "wp": 0.15,
            "wc": 0.15,
        }
        self.spatial_mode = self.inspect_spatial_information()
        logger.info(f"Published MCDM Hybrid Decision Ranker Core Activated: {self.weights}")
        logger.info(f"Spatial evidence mode: {self.spatial_mode['mode']} | buildings={self.spatial_mode['building_count']} storeys={self.spatial_mode['storey_count']} spaces={self.spatial_mode['space_count']} geometry_available={self.spatial_mode['geometry_available']}")

    def inspect_spatial_information(self):
        """Inspect the real IFC model and determine whether geometry or only hierarchy is available."""
        info = {
            "mode": "topology",
            "building_count": 0,
            "storey_count": 0,
            "space_count": 0,
            "geometry_available": False,
            "geometry_sample_count": 0,
        }

        try:
            model = ifcopenshell.open("data/ifc/Building-Architecture.ifc")
            info["building_count"] = len(model.by_type("IfcBuilding"))
            info["storey_count"] = len(model.by_type("IfcBuildingStorey"))
            info["space_count"] = len(model.by_type("IfcSpace"))

            settings = ifcopenshell.geom.settings()
            for product in model.by_type("IfcProduct")[:50]:
                try:
                    ifcopenshell.geom.create_shape(settings, product)
                    info["geometry_sample_count"] += 1
                except Exception:
                    continue
            info["geometry_available"] = info["geometry_sample_count"] > 0
            info["mode"] = "geometry" if info["geometry_available"] else "topology"
        except Exception:
            info["mode"] = "topology"

        return info

    def _load_ifc_product(self, guid: str):
        model = ifcopenshell.open("data/ifc/Building-Architecture.ifc")
        for product in model.by_type("IfcProduct"):
            if getattr(product, "GlobalId", None) == guid:
                return product
        return None

    def _get_product_context(self, guid: str):
        product = self._load_ifc_product(guid)
        context = {"building": None, "storey": None, "space": None}
        if product is None:
            return context

        try:
            for rel in getattr(product, "ContainedInStructure", []) or []:
                struct = rel.RelatingStructure
                if struct.is_a("IfcBuildingStorey"):
                    context["storey"] = getattr(struct, "Name", None)
                elif struct.is_a("IfcSpace"):
                    context["space"] = getattr(struct, "Name", None)
                if hasattr(struct, "Decomposes") and struct.Decomposes:
                    parent = struct.Decomposes[0].RelatingObject
                    if parent and parent.is_a("IfcBuilding"):
                        context["building"] = getattr(parent, "Name", None)
        except Exception:
            pass

        if context["building"] is None:
            try:
                model = ifcopenshell.open("data/ifc/Building-Architecture.ifc")
                for element in model.by_type("IfcBuilding"):
                    context["building"] = getattr(element, "Name", None)
                    break
            except Exception:
                pass

        return context

    def _extract_query_tokens(self, query_text: str):
        q_lower = query_text.lower()
        room_terms = set(re.findall(r"room\s*[a-z0-9-]+|space\s*[a-z0-9-]+|zone\s*[a-z0-9-]+", q_lower))
        level_terms = set(re.findall(r"level\s*[a-z0-9-]+|storey\s*[a-z0-9-]+|floor\s*[a-z0-9-]+", q_lower))
        return q_lower, room_terms, level_terms

    def compute_topology_spatial_score(self, guid: str, query_text: str) -> float:
        """Topology-based spatial reasoning using actual IFC building/storey/space hierarchy only."""
        q_lower, room_terms, level_terms = self._extract_query_tokens(query_text)
        context = self._get_product_context(guid)
        score = 0.0

        if context.get("space"):
            space_name = str(context["space"]).lower()
            if any(term in space_name for term in ["room", "hall", "bath", "lobby", "office", "living", "entry"]):
                score += 0.15
            for term in room_terms:
                if term.replace("room ", "").replace("space ", "").strip() in space_name:
                    score += 0.45

        if context.get("storey"):
            storey_name = str(context["storey"]).lower()
            for term in level_terms:
                if term.replace("level ", "").replace("storey ", "").replace("floor ", "").strip() in storey_name:
                    score += 0.30
            if "level" in q_lower and "level" in storey_name:
                score += 0.20

        if context.get("building") and "building" in q_lower:
            score += 0.10

        if score > 0:
            return min(1.0, score)
        return 0.0

    def compute_geometry_spatial_score(self, guid: str, reference_bbox: tuple = None) -> tuple:
        """Compute geometry-derived spatial score only when the IFC element truly has geometry available."""
        try:
            product = self._load_ifc_product(guid)
            if product is None:
                return (0.0, "topology")

            settings = ifcopenshell.geom.settings()
            shape = ifcopenshell.geom.create_shape(settings, product)
            centroid = np.asarray(shape_util.get_element_bbox_centroid(product, shape.geometry), dtype=float)

            if reference_bbox is None:
                return (1.0, "geometry")

            ref_min = np.asarray(reference_bbox[:3], dtype=float)
            ref_max = np.asarray(reference_bbox[3:], dtype=float)
            ref_center = (ref_min + ref_max) / 2.0
            distance_m = float(np.linalg.norm(centroid - ref_center))
            score = float(np.exp(-distance_m / 5.0))
            return (score, "geometry", distance_m)
        except Exception:
            return (0.0, "topology")

    def compute_mathematical_spatial_decay(self, distance_m: float, lambda_factor: float = 5.0) -> float:
        """Continuous decay function used when a valid geometric distance is available."""
        return float(np.exp(-distance_m / lambda_factor))

    def calculate_intent_class_score(self, ifc_class: str, query_text: str) -> float:
        """Class compatibility based on the query intent and IFC entity class."""
        q_lower = query_text.lower()
        c_lower = str(ifc_class).lower()

        if "door" in q_lower:
            if "ifcdoor" == c_lower:
                return 1.0
            if "ifcdoortype" in c_lower:
                return 0.9
            if "ifcopeningelement" in c_lower:
                return 0.8
            if "ifcwall" in c_lower:
                return 0.3
            return 0.1

        if "wall" in q_lower:
            if "ifcwall" in c_lower:
                return 1.0
            if "ifcopeningelement" in c_lower:
                return 0.7
            return 0.1

        if "beam" in q_lower:
            if "ifcbeam" in c_lower:
                return 1.0
            if "ifcmember" in c_lower:
                return 0.8
            return 0.1

        if "slab" in q_lower or "floor" in q_lower:
            if "ifcslab" in c_lower:
                return 1.0
            return 0.1

        if "column" in q_lower:
            if "ifccolumn" in c_lower:
                return 1.0
            return 0.1

        return 0.5

    def calculate_granular_property_score(self, guid: str, query_text: str) -> float:
        """Property similarity from actual BIM metadata only; no constant fallback values."""
        q_lower = query_text.lower()
        q_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", q_lower) if len(t) > 2}

        conn = sqlite3.connect("data/database/bim.db")
        cur = conn.cursor()
        cur.execute("""
            SELECT material, fire_rating, load_bearing, manufacturer, phase, ifc_class, name, storey, room
            FROM elements WHERE guid = ?
        """, (guid,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return 0.0

        material, fire_rating, load_bearing, manufacturer, phase, ifc_class, name, storey, room = row
        field_text = " ".join(str(v) for v in [material, fire_rating, load_bearing, manufacturer, phase, ifc_class, name, storey, room] if v is not None)
        field_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", field_text.lower()) if len(t) > 2}

        overlap = len(q_tokens & field_tokens)
        score = min(0.5, overlap * 0.1)

        q_synonyms = {
            "leak": {"leak", "leaking", "water", "drip", "seepage", "moisture"},
            "repair": {"repair", "maintenance", "fix", "service", "fault"},
            "inspection": {"inspection", "check", "condition", "diagnostic"},
            "damage": {"damage", "crack", "defect", "breakage"},
            "door": {"door", "frame", "hinge", "leaf"},
            "wall": {"wall", "partition", "plaster", "masonry"},
            "pipe": {"pipe", "tube", "conduit", "segment"},
            "ceiling": {"ceiling", "panel", "suspended"},
        }

        for token in q_tokens:
            for key, values in q_synonyms.items():
                if token in values and any(token in field_tokens or key in field_tokens for _ in [0]):
                    score += 0.12

        if "room" in q_lower and room is not None:
            score += 0.20 if str(room).lower() in q_lower else 0.0
        if "level" in q_lower or "storey" in q_lower:
            score += 0.15 if storey is not None and str(storey).lower() in q_lower else 0.0
        if material is not None and str(material).lower() in q_lower:
            score += 0.20
        if "fire" in q_lower and fire_rating is not None:
            score += 0.10 if "fire" in str(fire_rating).lower() or "hr" in q_lower else 0.0
        if "load" in q_lower and load_bearing is not None:
            score += 0.10 if "load" in str(load_bearing).lower() or "structural" in q_lower else 0.0

        return max(0.0, min(1.0, score))

    def calculate_name_similarity(self, candidate_name: str, query_text: str) -> float:
        candidate_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", str(candidate_name).lower()) if len(t) > 2}
        query_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", str(query_text).lower()) if len(t) > 2}
        if not candidate_tokens or not query_tokens:
            return 0.0
        overlap = len(candidate_tokens & query_tokens)
        return min(1.0, overlap / max(1, len(query_tokens)))

    def rank_candidates(self, candidates: dict, query_text: str, reference_bbox: tuple) -> list:
        """Hybrid reranking with real IFC geometry when available and topology-derived spatial logic otherwise."""
        ids = candidates.get("ids", [])
        distances = candidates.get("distances", [])
        metadatas = candidates.get("metadatas", [])

        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if distances and isinstance(distances[0], list):
            distances = distances[0]
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]

        seen = set()
        unique_ids, unique_distances, unique_metas = [], [], []
        for idx, guid in enumerate(ids):
            sid = str(guid).strip()
            if sid in seen:
                continue
            seen.add(sid)
            unique_ids.append(sid)
            unique_distances.append(float(distances[idx]) if idx < len(distances) else 1.0)
            unique_metas.append(metadatas[idx] if idx < len(metadatas) else {})

        ws, wt, wp, wc = self.weights["ws"], self.weights["wt"], self.weights["wp"], self.weights["wc"]
        ranked_pool = []

        for i, guid in enumerate(unique_ids):
            meta = unique_metas[i] if i < len(unique_metas) else {}
            ifc_class = str(meta.get("ifc_class", "UnknownClass")) if isinstance(meta, dict) else "UnknownClass"
            meta_name = str(meta.get("name", "Unnamed")) if isinstance(meta, dict) else "Unnamed"
            sem_distance = unique_distances[i]
            S_s = max(0.0, min(1.0, 1.0 - sem_distance))

            if self.spatial_mode["geometry_available"]:
                score_result = self.compute_geometry_spatial_score(guid, reference_bbox)
                if len(score_result) == 3:
                    S_t, spatial_source, distance_m = score_result
                else:
                    S_t, spatial_source = score_result
                    distance_m = None
            else:
                S_t = self.compute_topology_spatial_score(guid, query_text)
                spatial_source = "topology"
                distance_m = None

            S_p = self.calculate_granular_property_score(guid, query_text)
            S_c = self.calculate_intent_class_score(ifc_class, query_text)
            final_score = (ws * S_s) + (wt * S_t) + (wp * S_p) + (wc * S_c)

            ranked_pool.append({
                "guid": guid,
                "ifc_class": ifc_class,
                "name": meta_name,
                "ss_raw": S_s,
                "st_raw": S_t,
                "sp_raw": S_p,
                "sc_raw": S_c,
                "ws": ws,
                "wt": wt,
                "wp": wp,
                "wc": wc,
                "distance": float(sem_distance),
                "spatial_source": spatial_source,
                "spatial_distance_m": distance_m,
                "final_score": final_score,
            })

        ranked_pool.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked_pool

