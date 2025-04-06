import asyncio
import functools
import inspect
import itertools
import logging
import logging.handlers
import time

from rich.console import Console
from rich.traceback import Traceback

TRACE_LEVEL_NUM = 5


# A quick placeholder for the trace_logging function, any module that include this
# one will get the tracelogging installed
def trace_logging(level: int = TRACE_LEVEL_NUM) -> None:
    """Add a TRACE level to the logging module."""
    if hasattr(logging.Logger, "trace"):
        return  # Already set up
    logging.addLevelName(level, "TRACE")
    logging.TRACE = level

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    logging.Logger.trace = trace


trace_logging(level=TRACE_LEVEL_NUM)


# Global call ID generator for async functions (optional if using task ID)
_async_call_counter = itertools.count()


def trace(logger, *, show_duration: bool = False):
    """
    Decorator factory for trace logging.
    :param logger: The logger instance to use.
    :param show_duration: Whether to include execution time in the log.
    """

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                call_id = next(_async_call_counter)
                task = asyncio.current_task()
                task_name = task.get_name() if task else "unnamed"
                task_id = id(task) if task else "<no-task>"
                start = time.perf_counter() if show_duration else None

                logger.trace(
                    f"[#{call_id}|{task_name}@{task_id}] Calling {func.__name__} (async) with args={args}, kwargs={kwargs}"
                )
                result = await func(*args, **kwargs)
                duration = (
                    f" in {time.perf_counter() - start:.3f}s" if show_duration else ""
                )
                logger.trace(
                    f"[#{call_id}|{task_name}@{task_id}] {func.__name__} (async) returned: {result}{duration}"
                )
                return result

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                call_id = next(_async_call_counter)
                start = time.perf_counter() if show_duration else None
                logger.trace(
                    f"[#{call_id}] Calling {func.__name__} (sync) with args={args}, kwargs={kwargs}"
                )
                result = func(*args, **kwargs)
                duration = (
                    f" in {time.perf_counter() - start:.3f}s" if show_duration else ""
                )
                logger.trace(
                    f"[#{call_id}] {func.__name__} (sync) returned: {result}{duration}"
                )
                return result

            return sync_wrapper

    return decorator


class RichFormatter(logging.Formatter):
    """
    A logging formatter that uses Rich to render:
      - Main log lines with bracket markup (if use_color=True).
      - Rich-styled traceback.
      - Optional stack info appended (stack_info=True).
    """

    def __init__(self, use_color=True, datefmt="%Y-%m-%d %H:%M:%S", **kwargs):
        super().__init__(datefmt=datefmt, **kwargs)
        self.use_color = use_color
        self.console = self.make_console(use_color)
        self.standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }

    def make_console(self, use_color: bool) -> Console:
        return Console(color_system="auto" if use_color else None, stderr=True)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record, including Rich-styled traceback if exc_info
        is present, while controlling newlines so we don't get extra blank lines.
        """
        try:
            # Color mappings for bracket-markup log line
            log_level_colors = {
                "TRACE": "cyan",
                "DEBUG": "blue",
                "INFO": "green",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold red",
            }

            # --- Build main log line (with or without bracket markup) ---
            timestamp = self.formatTime(record)
            level = record.levelname.upper()
            # module = record.module
            func = record.funcName
            lineno = record.lineno
            event = record.getMessage()

            timestamp_width = 19
            level_width = 8
            caller_width = 25

            caller_display = f"{record.name}:{func}:{lineno}"

            if self.use_color:
                level_color = log_level_colors.get(level, "white")
                timestamp_display = (
                    f"[bold green]{timestamp:<{timestamp_width}}[/bold green][reset]"
                )
                level_display = (
                    f"[{level_color}]{level:<{level_width}}[/{level_color}][reset]"
                )
                caller_display_formatted = (
                    f"[cyan]{caller_display:<{caller_width}}[/cyan][reset]"
                )
                event_display = f"[{level_color}]{event}[/{level_color}][reset]"
            else:
                # no-color fallback
                timestamp_display = f"{timestamp:<{timestamp_width}}"
                level_display = f"{level:<{level_width}}"
                caller_display_formatted = f"{caller_display:<{caller_width}}"
                event_display = event

            # Gather any extra fields
            extra_fields = {
                k: v for k, v in record.__dict__.items() if k not in self.standard_attrs
            }
            extra = " | ".join(f"{k}={v}" for k, v in extra_fields.items())

            main_line = (
                f"{timestamp_display} | {level_display} | "
                f"{caller_display_formatted} - {event_display}"
            )
            if extra:
                main_line += f" | {extra}"

            # --- Render everything in a capture context ---
            with self.console.capture() as capture:
                self.console.print(main_line, markup=self.use_color, end="")

                if record.exc_info:
                    exc_type, exc_value, exc_tb = record.exc_info
                    if exc_type and exc_value and exc_tb:
                        tb = Traceback.from_exception(
                            exc_type,
                            exc_value,
                            exc_tb,
                            show_locals=False,
                            width=100,
                        )
                        self.console.print()
                        self.console.print(tb, end="")

                if record.stack_info:
                    self.console.print(record.stack_info, style="dim", end="")

            return capture.get()

        except Exception as e:
            # Fallback if something goes awry
            return f"Formatting error: {e}"


class LevelOverrideFilter(logging.Filter):
    """
    Filter that enforces a default minimum level, but if the logger name
    starts with one of the override 'prefixes', it uses that override level.
    """

    def __init__(self, default_level: int, overrides: dict[str, int]):
        """
        :param default_level: The default logging threshold (e.g. WARNING).
        :param overrides: dict of logger_prefix -> min_level.
                         e.g. {"mchat": logging.DEBUG}
                         will apply to "mchat" itself AND "mchat.*"
        """
        super().__init__()
        self.default_level = default_level
        self.overrides = overrides

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Return True if this record should be logged, based on either:
          - The override for that prefix (if any)
          - Or the default_level.
        """
        for override_logger, override_level in self.overrides.items():
            if record.name == override_logger or record.name.startswith(
                override_logger + "."
            ):
                return record.levelno >= override_level

        return record.levelno >= self.default_level


class LoggerConfigurator:
    def __init__(
        self,
        log_to_console: bool = True,
        log_to_file: bool = False,
        file_path: str = "debug.log",
        console_log_level: int = logging.WARNING,
        file_log_level: int = logging.WARNING,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """
        Initializes the logger configurator with optional parameters.

        :param log_to_console: Whether to log to the console. Default is True.
        :param log_to_file: Whether to log to a file. Default is False.
        :param file_path: The file path for the log file. Default is 'debug.log'.
        :param console_log_level: The "base" log level for console output
                                  (default: WARNING).
        :param file_log_level: The "base" log level for file output (default: WARNING).
        :param max_bytes: The maximum size of the log file before rotation.
                          Default is 5 MB.
        :param backup_count: The number of backup files to keep. Default is 5.
        """
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.file_path = file_path

        self._console_log_level = console_log_level
        self._file_log_level = file_log_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Keep references to the console/file handlers we create
        self._my_handlers: list[logging.Handler] = []

        # Dictionaries to hold logger-specific overrides
        # e.g. { "my.special.logger": logging.DEBUG }
        self.console_filters: dict[str, int] = {}
        self.file_filters: dict[str, int] = {}

        self._configure()

    @property
    def console_log_level(self) -> int:
        """Return the current 'base' console log level."""
        return self._console_log_level

    @console_log_level.setter
    def console_log_level(self, level: int) -> None:
        """Set a new base console level and reconfigure logging."""
        self._console_log_level = level
        self._configure()

    @property
    def file_log_level(self) -> int:
        """Return the current 'base' file log level."""
        return self._file_log_level

    @file_log_level.setter
    def file_log_level(self, level: int) -> None:
        """Set a new base file level and reconfigure logging."""
        self._file_log_level = level
        self._configure()

    def _create_rich_formatter(self, use_color: bool = True) -> logging.Formatter:
        """Factory to create a RichFormatter with optional color."""
        return RichFormatter(use_color=use_color)

    def _configure(self) -> None:
        """
        Configures the standard Python logging system
        based on the instance's properties, while preserving any
        other (non-our-own) handlers already attached to the root logger.
        """
        root_logger = logging.getLogger()

        # Copy existing handlers
        existing_handlers = root_logger.handlers[:]

        # Remove OUR old console/file handlers from that list
        for h in self._my_handlers:
            if h in existing_handlers:
                existing_handlers.remove(h)

        self._my_handlers.clear()

        # Build console/file handlers if requested
        if self.log_to_console:
            console_formatter = self._create_rich_formatter(use_color=True)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            # Set the handler's level to NOTSET so that filter logic decides:
            console_handler.setLevel(logging.NOTSET)

            # Add our combined filter:
            console_handler.addFilter(
                LevelOverrideFilter(
                    default_level=self._console_log_level,
                    overrides=self.console_filters,
                )
            )
            self._my_handlers.append(console_handler)

        if self.log_to_file:
            file_formatter = self._create_rich_formatter(use_color=False)
            file_handler = logging.handlers.RotatingFileHandler(
                self.file_path,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )
            file_handler.setFormatter(file_formatter)
            # Set the handler's level to NOTSET so that filter logic decides:
            file_handler.setLevel(logging.NOTSET)

            file_handler.addFilter(
                LevelOverrideFilter(
                    default_level=self._file_log_level,
                    overrides=self.file_filters,
                )
            )
            self._my_handlers.append(file_handler)

        # Combine what's left of the old handlers + our new ones
        all_handlers = existing_handlers + self._my_handlers

        # Re-apply them all to the root logger (with force=True to wipe old)
        if all_handlers:
            # The filters manage the content, so
            logging.basicConfig(
                level=logging.NOTSET,
                handlers=all_handlers,
                force=True,
            )
        else:
            logging.disable(logging.CRITICAL)

    # ---- Methods to manage overrides for console handler ----

    def add_console_filter(self, logger_name: str, level: int = logging.DEBUG) -> None:
        """
        Add or update an override for the console output.
        :param logger_name: The name of the logger to allow at `level`.
        :param level: The minimum level for that logger (default: DEBUG).
        """
        self.console_filters[logger_name] = level
        self._configure()

    def remove_console_filter(self, logger_name: str) -> None:
        """Remove an override filter from the console handler."""
        if logger_name in self.console_filters:
            del self.console_filters[logger_name]
            self._configure()

    def get_console_filters(self) -> dict[str, int]:
        """Return current console logger-specific overrides."""
        return dict(self.console_filters)

    # ---- Methods to manage overrides for file handler ----

    def add_file_filter(self, logger_name: str, level: int = logging.DEBUG) -> None:
        """
        Add or update an override for the file output.
        :param logger_name: The name of the logger to allow at `level`.
        :param level: The minimum level for that logger (default: DEBUG).
        """
        self.file_filters[logger_name] = level
        self._configure()

    def remove_file_filter(self, logger_name: str) -> None:
        """Remove an override filter from the file handler."""
        if logger_name in self.file_filters:
            del self.file_filters[logger_name]
            self._configure()

    def get_file_filters(self) -> dict[str, int]:
        """Return current file logger-specific overrides."""
        return dict(self.file_filters)

    def add_console_and_file_filters(
        self, logger_name: str, level: int = logging.DEBUG
    ) -> None:
        """
        Add or update an override for both console and file output.
        :param logger_name: The name of the logger to allow at `level`.
        :param level: The minimum level for that logger (default: DEBUG).
        """
        self.add_console_filter(logger_name, level)
        self.add_file_filter(logger_name, level)
        self._configure()

    def remove_console_and_file_filters(self, logger_name: str) -> None:
        """Remove an override filter from both console and file handlers."""
        self.remove_console_filter(logger_name)
        self.remove_file_filter(logger_name)
        self._configure()
