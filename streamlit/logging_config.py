import logging
import logging.config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors per log level."""

    COLORS = {
        logging.DEBUG: "\033[36m",       # Cyan
        logging.INFO: "\033[32m",        # Green
        logging.WARNING: "\033[33m",     # Yellow
        logging.ERROR: "\033[31m",       # Red
        logging.CRITICAL: "\033[35m",    # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        bold = self.BOLD if record.levelno >= logging.WARNING else ""
        levelname = f"{bold}{color}{record.levelname}{self.RESET}"
        msg = f"{color}{record.getMessage()}{self.RESET}"
        record.levelname = levelname
        record.msg = msg
        return super().format(record)


config_dict = {
    "version": 1,
    "formatters": {
        "colored": {
            "()": ColoredFormatter,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(config_dict)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*."""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    return log
