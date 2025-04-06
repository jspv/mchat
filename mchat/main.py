import argparse
import logging
import sys

from mchat.logging_config import LoggerConfigurator
from mchat.mchatweb import WebChatApp

log_config: LoggerConfigurator | None = None

logger = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> LoggerConfigurator:
    global log_config
    if log_config is not None:
        return log_config

    # Default to WARNING for both file and console logging
    console_log_level = logging.WARNING
    file_log_level = logging.WARNING
    mchat_console_level = logging.WARNING

    # Adjust log levels based on verbosity
    if verbosity >= 1:
        mchat_console_level = logging.INFO
    if verbosity >= 2:
        mchat_console_level = logging.DEBUG
    if verbosity >= 3:
        file_log_level = logging.DEBUG
    if verbosity >= 4:
        file_log_level = logging.TRACE

    print(f"file_log_level={file_log_level}, console_log_level={console_log_level}, ")

    log_config = LoggerConfigurator(
        log_to_console=True,
        log_to_file=True,
        file_log_level=file_log_level,
        console_log_level=console_log_level,
    )

    # customize the log levels for specific modules
    log_config.add_console_filter("mchat", mchat_console_level)
    log_config.add_file_filter("mchat", logging.TRACE)
    log_config.add_console_and_file_filters(__name__, logging.TRACE)
    log_config.add_console_and_file_filters("watchfiles", logging.WARNING)
    log_config.add_console_and_file_filters("mchat.mchatweb", logging.DEBUG)
    log_config.add_console_and_file_filters("markdown_it.rules_block", logging.WARNING)

    return log_config


def main():
    parser = argparse.ArgumentParser(description="Run the MChat web chat app.")
    parser.add_argument(
        "--port", type=int, default=8882, help="Port to run MChat on (default: 8882)."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase verbosity. Use -v for INFO, -vv for DEBUG, "
            "-vvv for really extended debugging in debug.log."
        ),
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable automatic reload."
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.trace("Trace logging is enabled.")

    logger.debug(
        f"Starting main() with port={args.port}, "
        f"verbose={args.verbose}, reload={args.reload}"
    )

    app = WebChatApp()
    try:
        app.run(
            port=args.port,
            title="MChat - Multi-Model Chat Framework",
            favicon="static/favicon-32x32.png",
            dark=True,
            log_config=log_config,
            reload=args.reload,
        )
    except Exception as e:
        logger.critical(f"Critical failure in WebChatApp: {e}", exc_info=True)
        sys.exit(1)


if __name__ in {"__main__", "__mp_main__"}:
    main()
