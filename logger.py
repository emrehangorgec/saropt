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
    log_file = f"training_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, log_file)

    logging.root.handlers.clear()

    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(stream_handler)

    print(f"Logging initialized. Writing logs to: {log_file_path}")

    return logger
