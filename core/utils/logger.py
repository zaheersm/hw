import sys
import logging


def setup_logger(log_file, level=logging.INFO, stdout=False):
    logger = logging.getLogger(log_file)
    formatter = logging.Formatter('%(asctime)s | %(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stdout is True:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(level)
    return logger

