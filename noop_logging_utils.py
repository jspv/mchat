# noop_logging_utils.py
import logging

# Suppress all logging from litellm
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)  # Suppress litellm logs


def get_logger(name=None, level=None):
    return logging.getLogger(name)  # Return your logger or a dummy one


# mimic expected constants
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
