import os
import sqlite3
import ifcopenshell
from loguru import logger

def parse_high_density_data():
    ifc_dir = "data/ifc"
    db_path = "data/database/bim.db"
    
    # 1. Connect to SQLite and wipe old table data to prevent duplicate primary keys
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM elements")
    conn.commit()
    
    ifc_files = [f for f in os.listdir(ifc_dir) if f.endswith(".ifc")]
    logger.info(f"High-Density Extractor Active. Scanning {len(ifc_files)} models.")
    
    inserted_count = 0
    
    for filename in ifc_files:
        file_path = os.path.join(ifc_dir, filename)
        logger.info(f"Drilling deep into structural metadata: {filename}")
        
        try:
            model = ifcopenshell.open(file_path)
            
            # Query IfcProduct to extract physical assets AND structural sub-components
            elements = model.by_type("IfcProduct")
            
            for el in elements:
                guid = el.GlobalId
                ifc_class = el.is_a()
                name = getattr(el, 'Name', 'Unnamed')
                
                # Dynamic spatial container tracing
                storey = "Unknown"
                room = "Unknown"
                if hasattr(el, 'ContainedInStructure') and len(el.ContainedInStructure) > 0:
                    spatial_container = el.ContainedInStructure[0].RelatingStructure
                    if spatial_container.is_a('IfcBuildingStorey'):
                        storey = getattr(spatial_container, 'Name', 'Unknown Storey')
                    elif spatial_container.is_a('IfcSpace'):
                        room = getattr(spatial_container, 'Name', 'Unknown Room')

                # Material parsing loops
                material = "Generic"
                if hasattr(el, 'HasAssociations'):
                    for assoc in el.HasAssociations:
                        if assoc.is_a('IfcRelAssociatesMaterial'):
                            mat_select = assoc.RelatingMaterial
                            if mat_select.is_a('IfcMaterial'):
                                material = getattr(mat_select, 'Name', 'Generic')
                            elif mat_select.is_a('IfcMaterialLayerSetUsage'):
                                material = getattr(mat_select.ForLayerSet, 'LayerSetName', 'LayerSet')

                # 2. Insert metrics securely into your database
                cursor.execute("""
                    INSERT OR REPLACE INTO elements (guid, ifc_class, name, storey, room, material)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (guid, ifc_class, name, storey, room, material))
                
                inserted_count += 1
                
            conn.commit()
            logger.success(f"-> Extracted {len(elements)} structural parts from {filename}")
            
        except Exception as e:
            logger.error(f"Error reading elements from {filename}: {e}")
            
    conn.close()
    
    print("\n=========================================================================================")
    logger.success(f"DATABASE COMPLETED: Total database rows are now scaled to: {inserted_count} rows!")
    print("=========================================================================================")

if __name__ == '__main__':
    parse_high_density_data()
