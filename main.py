import subprocess
import sys
from loguru import logger

def execute_pipeline_stage(script_path: str):
    logger.info(f"Running automated stage sequence: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"Pipeline broke during execution step: {script_path}")
        sys.exit(result.returncode)

def main():
    logger.info("==========================================================================")
    logger.info("LAUNCHING MASTER BIM-SEMANTIC-RETRIEVAL INTEGRATED CORE PIPELINE ENGINE")
    logger.info("==========================================================================")
    
    # Run your verified 6-stage chronological research pipeline stack
    execute_pipeline_stage("scripts/05_query_engine.py")
    execute_pipeline_stage("scripts/06_evaluate.py")
    
    logger.success("All integrated retrieval modules executed cleanly. Framework online.")

if __name__ == '__main__':
    main()
