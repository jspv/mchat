import logging

from mchat.logging_config import LoggerConfigurator
from mchat.mchatweb import WebChatApp

# Configure the logger
log_config = LoggerConfigurator(
    log_to_console=True,
    log_to_file=True,
    file_log_level=logging.WARNING,
    console_log_level=logging.WARNING,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger("mchat.mchatweb").setLevel(logging.DEBUG)

app = WebChatApp()

# Override other loggers
logging.getLogger("watchfiles").setLevel(logging.WARNING)

if __name__ in {"__main__", "__mp_main__"}:
    try:
        # Run the WebChat application
        app.run(
            port=8882,
            title="MChat - Multi-Model Chat Framework",
            favicon="static/favicon-32x32.png",
            dark=True,
            log_config=log_config,
        )
    except Exception as e:
        logger.critical(f"Critical failure in WebChatApp: {e}", exc_info=True)
        import sys

        sys.exit(1)  # Exit with error status
