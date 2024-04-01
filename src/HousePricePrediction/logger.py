import logging
import os

logger = logging.getLogger(__name__)


def setup_logging(logger_name, log_level, log_path, console_log=True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if log_path:
        log_directory, log_file = os.path.split(log_path)
        if not log_directory:
            log_directory = "."
        os.makedirs(log_directory, exist_ok=True)
        if not log_file:
            log_file = "logfile.log"
        log_file_path = os.path.join(log_directory, log_file)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
