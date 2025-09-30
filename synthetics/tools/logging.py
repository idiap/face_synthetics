#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal
from pathlib import Path
from datetime import datetime


_DEFAULT_LOGGER_NAME = "synthetics"
_DEFAULT_LOGGING_FMT = (
    "%(name)s@%(filename)s:%(lineno)d: %(levelname)s : %(message)s"
)


class ANSIFormatter:
    """Add color support on unix terminal"""

    GRAY = "\033[0;37m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[38;5;196m"
    BOLD_RED = "\033[31;1m"
    BOLD_PURPLE = "\033[1;35m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def _add_color_format(fmt, style):
    if "levelname" in fmt:
        if style == "%":
            fmt = fmt.replace("%(levelname)", "%(color)s%(levelname)s%(reset)")
        elif style == "{":
            fmt = fmt.replace("{levelname}", "{color}{levelname}{reset}")
        elif style == "$":
            fmt = fmt.replace("$levelname", "$color$levelname$reset")
        else:
            raise ValueError(f"Un-supported style type: {style}")
    return fmt


class ColorFormatter(logging.Formatter):
    """
    Logging colored formatter
    See: https://stackoverflow.com/a/56944256/3638629
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        validate: bool = True,
        *,
        defaults=None,
    ):
        """
        Initialize the formatter with specified format strings.

        Initialize the formatter either with the specified format string, or a
        default as described above. Allow for specialized date formatting with
        the optional datefmt argument. If datefmt is omitted, you get an
        ISO8601-like (or RFC 3339-like) format.

        Use a style parameter of '%', '{' or '$' to specify that you want to
        use one of %-formatting, :meth:`str.format` (``{}``) formatting or
        :class:`string.Template` formatting in your format string.
        """
        fmt = _add_color_format(fmt, style)
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        # Add color entry + reset
        self.colors = {
            logging.DEBUG: ANSIFormatter.GRAY,
            logging.INFO: ANSIFormatter.GREEN,
            logging.WARNING: ANSIFormatter.YELLOW,
            logging.ERROR: ANSIFormatter.RED,
            logging.FATAL: ANSIFormatter.BOLD_RED,
            logging.CRITICAL: ANSIFormatter.BOLD_PURPLE,
        }
        self.reset = ANSIFormatter.RESET

    def format(self, record):
        """Update format to add color support on `levelno`"""
        record.color = self.colors[record.levelno]
        record.reset = self.reset
        return super().format(record)


def set_logging_level(logger: logging.Logger, level: int):
    """
    Set the logging level for a given logger and its childs, following the
    rules:

        0: Warning + Error
        1: Info
        >1: Debug

    :param logger: Logger to set level for
    :param level: Which level to set to
    """
    _lut = {0: "WARNING", 1: "INFO"}
    ilevel = _lut.get(level, "DEBUG")
    logger.setLevel(ilevel)


def add_file_handler(
    logger: logging.Logger,
    filename: str | Path,
    fmt: str | None = None,
) -> None:
    """Add file handler, i.e. logging into file"""
    # Format
    if fmt is None:
        fmt = _DEFAULT_LOGGING_FMT
    formatter = logging.Formatter(fmt)
    # Handler
    handler = logging.FileHandler(filename, encoding="utf-8")
    handler.setFormatter(formatter)
    # Add to logger
    logger.addHandler(handler)


def setup_default_log_folder(
    logger: logging.Logger, folder: str | Path | None = None
) -> Path:
    """
    Add logging into file. If folder is not provided, default to current
    working directory
    """
    if folder is None:
        folder = Path(Path.cwd(), "logs")
    # Setup filename
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = Path(folder, now.strftime("%Y_%m_%d_%H_%M_%S.log"))
    add_file_handler(logger, filename)
    return folder


def setup(
    name: str | None = None,
    fmt: str | None = None,
    with_color: bool = True,
    level: int = logging.INFO,
    propagate: bool = True,
    handlers: Iterable[logging.Handler] | None = None,
) -> logging.Logger:
    """
    Create logger

    :param name: Name of the logger to create. If None use root logger
    :param fmt: Format of the output message, defaults to None
    :param level: Logging level
    :param propagate: If `True`, events logged to this logger will be passed to
        the handlers of higher level (ancestor) loggers, in addition to any
        handlers attached to this logger.
    :param with_color: If set to true, message levels will be colored, defaults
        to True
    :return: logger instance
    """
    from sys import stdout

    if name is None:
        name = _DEFAULT_LOGGER_NAME
    logger = logging.getLogger(name=name)
    logger.propagate = propagate
    # Add handler if needed
    if len(logger.handlers) == 0:
        if handlers is None:
            # No handler given log to console
            handlers = [logging.StreamHandler(stream=stdout)]
        # Setup handlers to logger
        if fmt is None:
            fmt = _DEFAULT_LOGGING_FMT
        for h in handlers:
            if h.formatter is None:
                if with_color and not isinstance(h, logging.FileHandler):
                    h.setFormatter(ColorFormatter(fmt))
                else:
                    h.setFormatter(logging.Formatter(fmt))
            logger.addHandler(h)
    # Set level
    logger.setLevel(level=level)
    return logger
