import argparse
import logging
import sys

from mchat.logging_config import LoggerConfigurator
from mchat.mchatweb import WebChatApp

log_config: LoggerConfigurator | None = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup_logging(verbose: bool = False) -> LoggerConfigurator:
    global log_config
    if log_config is not None:
        return log_config

    log_config = LoggerConfigurator(
        log_to_console=True,
        log_to_file=True,
        file_log_level=logging.WARNING,
        console_log_level=logging.WARNING,
    )

    log_config.add_console_filter(
        "mchat", logging.DEBUG if verbose else logging.WARNING
    )
    log_config.add_file_filter("mchat", logging.DEBUG)
    log_config.add_console_and_file_filters("watchfiles", logging.WARNING)
    log_config.add_console_and_file_filters("mchat.mchatweb", logging.DEBUG)
    # log_config.add_console_and_file_filters("autogen_agentchat", logging.DEBUG)

    return log_config


def main():
    parser = argparse.ArgumentParser(description="Run the MChat web chat app.")
    parser.add_argument(
        "--port", type=int, default=8882, help="Port to run the web chat app on."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.debug(f"Starting main() with port={args.port}, verbose={args.verbose}")

    app = WebChatApp()
    try:
        app.run(
            port=args.port,
            title="MChat - Multi-Model Chat Framework",
            favicon="static/favicon-32x32.png",
            dark=True,
            log_level=logging.DEBUG if args.verbose else logging.WARNING,
        )
    except Exception as e:
        logger.critical(f"Critical failure in WebChatApp: {e}", exc_info=True)
        sys.exit(1)


if __name__ in {"__main__", "__mp_main__"}:
    main()
