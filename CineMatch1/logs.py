import logging
import os

def setup_logging(log_file='logs/scraper.log'):
    """
    Configures the logging settings.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True) # Ensure the logs directory exists
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info("Logging is set up.")