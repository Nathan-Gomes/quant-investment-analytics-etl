"""Log configuration that keeps every line attributable to a single run.

A log file that cannot say which run produced a line is not evidence. Every
record is stamped with the run currently in progress, so the question "what
happened on the run that produced this report" is answered by a grep rather
than by reading timestamps and guessing where one run ended and the next began.
"""

import logging
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s %(levelname)-8s [run=%(run_id)s] %(name)s: %(message)s"


class RunIdFilter(logging.Filter):
    """Stamps records with the run in progress, or ``-`` before one has started."""

    def __init__(self):
        super().__init__()
        self.run_id = "-"

    def filter(self, record):
        record.run_id = self.run_id
        return True


_run_filter = RunIdFilter()


def set_run_id(run_id):
    """Attribute every subsequent log line to ``run_id``."""
    _run_filter.run_id = run_id


def configure(log_path=None, level=logging.INFO, max_bytes=5_000_000, backups=5):
    """Send logs to stderr and, when ``log_path`` is given, to a rotating file.

    The rotation matters more than it looks: the previous handler appended to one
    file forever, so the record of an incident could be evicted by disk pressure
    long before anyone came looking for it.
    """
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_path, maxBytes=max_bytes,
                                            backupCount=backups, encoding="utf-8"))
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(_run_filter)
    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
