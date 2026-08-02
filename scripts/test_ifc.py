import os
import ifcopenshell
from loguru import logger

def execute_high_density_validation():
    target_dir = "data/ifc"
    discovered_models = [f for f in os.listdir(target_dir) if f.endswith(".ifc")]
    
    if not discovered_models:
        logger.critical("Workspace Data Failure: No engineering data models present inside data/ifc/")
        return
        
    logger.info(f"Discovered {len(discovered_models)} physical BIM files ready for schema processing.")
    grand_total_structural_components = 0
    
    for filename in discovered_models:
        filepath = os.path.join(target_dir, filename)
        logger.info(f"Opening data connection layer for parsing: {filename}")
        
        try:
            model_context = ifcopenshell.open(filepath)
            
            # Fetch IfcRoot instead of IfcProduct to catch IFC 4.3 metadata elements
            concrete_products = model_context.by_type("IfcRoot")
            file_element_count = len(concrete_products)
            grand_total_structural_components += file_element_count
            
            logger.success(f"-> Processing clear. File '{filename}' hosts {file_element_count} distinct entities.")
            
            preview_limit = min(2, file_element_count)
            if preview_limit > 0:
                print("   --- Structural Metadata Record Preview ---")
                for index in range(preview_limit):
                    element = concrete_products[index]
                    element_name = getattr(element, 'Name', 'Undefined/NoName')
                    print(f"   GUID: {element.GlobalId:25} | Class: {element.is_a():18} | Name: {element_name}")
                print("   -------------------------------------------")
                
        except Exception as error_context:
            logger.error(f"Failed handling data parsing operations inside {filename}: {error_context}")
            
    print("\n=========================================================================================")
    if grand_total_structural_components >= 1000:
        logger.success(f"CRITERIA MET: Pipeline processed {grand_total_structural_components} elements (> 1,000 targets).")
    else:
        logger.warning(f"CRITERIA GAP: Pipeline identified {grand_total_structural_components} elements (< 1,000 targets).")
    print("=========================================================================================")

if __name__ == '__main__':
    execute_high_density_validation()
