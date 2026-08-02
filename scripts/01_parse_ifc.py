import os
import json
import ifcopenshell
from loguru import logger

def parse_ifc_to_raw():
    ifc_dir, out_dir = "data/ifc", "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    ifc_files = [f for f in os.listdir(ifc_dir) if f.endswith(".ifc")]
    logger.info(f"[STAGE 1] Parsing {len(ifc_files)} IFC files...")

    for filename in ifc_files:
        elements_data = []
        try:
            model = ifcopenshell.open(os.path.join(ifc_dir, filename))
            for el in model.by_type("IfcProduct"):
                elements_data.append({
                    "guid": el.GlobalId, "class": el.is_a(), "name": getattr(el, 'Name', 'Unnamed')
                })
            out_file = os.path.join(out_dir, f"{filename}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(elements_data, f, indent=4)
            logger.success(f" -> Dumped {len(elements_data)} raw records for {filename}")
        except Exception as e:
            logger.error(f"Error parsing {filename}: {e}")

if __name__ == '__main__':
    parse_ifc_to_raw()
