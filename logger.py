import logging
import time
import os


def setup_logging(mode):
    """
    Sets up logging to output both to the terminal and a log file.

    Args:
        mode (str): Training mode (e.g., 'sar', 'opt', 'all').

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Generate the log file name with timestamp
    log_file = f"training_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # Ensure the logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, log_file)

    # Disable the root logger to prevent implicit logging
    logging.root.handlers.clear()

    # Create or get the named logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler for writing to log file
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Stream handler for terminal output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(stream_handler)

    # Debug: Print log file location
    print(f"Logging initialized. Writing logs to: {log_file_path}")

    return logger
