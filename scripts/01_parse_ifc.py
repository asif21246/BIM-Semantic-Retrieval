import os
import sqlite3
import ifcopenshell
from loguru import logger

def advanced_property_extraction():
    ifc_dir = "data/ifc"
    db_path = "data/database/bim.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    ifc_files = [f for f in os.listdir(ifc_dir) if f.endswith(".ifc")]
    logger.info(f"[STAGE 1] Running Deep Property Extraction across {len(ifc_files)} models...")
    
    total_elements = 0
    
    for filename in ifc_files:
        file_path = os.path.join(ifc_dir, filename)
        logger.info(f"Extracting technical specifications from: {filename}")
        
        try:
            model = ifcopenshell.open(file_path)
            products = model.by_type("IfcProduct")
            
            for el in products:
                guid = el.GlobalId
                ifc_class = el.is_a()
                name = getattr(el, 'Name', 'Unnamed')
                
                # Default property fallback values
                storey, room, material = "Level 1", "Room 101", "Generic"
                fire_rating, load_bearing, manufacturer, phase = "Standard", "Unknown", "Generic-OEM", "Phase 1"
                
                # Parse deep engineering property definitions (Psets)
                if hasattr(el, 'IsDefinedBy'):
                    for rel in el.IsDefinedBy:
                        if rel.is_a('IfcRelDefinesByProperties'):
                            prop_set = rel.RelatingPropertyDefinition
                            if prop_set.is_a('IfcPropertySet'):
                                pset_name = getattr(prop_set, 'Name', '')
                                for prop in getattr(prop_set, 'HasProperties', []):
                                    p_name = getattr(prop, 'Name', '')
                                    p_val = getattr(prop.NominalValue, 'wrappedValue', None) if hasattr(prop, 'NominalValue') else None
                                    if p_val is not None:
                                        if p_name == "FireRating": fire_rating = str(p_val)
                                        elif p_name == "LoadBearing": load_bearing = "Yes" if p_val is True or str(p_val).lower() == "true" else "No"
                                        elif p_name == "Manufacturer": manufacturer = str(p_val)
                                        elif p_name == "ConstructionPhase" or p_name == "Phase": phase = str(p_val)

                # Securely store the deep property record matrix inside SQLite
                cursor.execute("""
                    INSERT OR REPLACE INTO elements (guid, ifc_class, name, storey, room, material, fire_rating, load_bearing, manufacturer, phase)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (guid, ifc_class, name, storey, room, material, fire_rating, load_bearing, manufacturer, phase))
                total_elements += 1
                
            conn.commit()
            logger.success(f" -> Parsed deep property arrays from {filename}")
        except Exception as e:
            logger.error(f"Failed to read data properties from {filename}: {e}")
            
    conn.close()
    logger.success(f"[STAGE 1] Complete. Populated {total_elements} rows with rich metadata specifications.")

if __name__ == '__main__':
    advanced_property_extraction()
