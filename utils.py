# utils.py
import logging

def setup_logging():
    logging.basicConfig(
        filename='logs/training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging is set up.")
