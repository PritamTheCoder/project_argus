import logging
import os
import tempfile

import src.utils.logging_setup as logging_setup
from src.utils.logging_setup import configure_logging


def test_configure_logging_writes_to_file():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_configured = logging_setup._configured

    tmp_dir = tempfile.mkdtemp()
    log_path = os.path.join(tmp_dir, "test.log")
    try:
        configure_logging(log_path=log_path, force=True)
        logging.getLogger("test.logger").info("hello from the test")

        for h in root.handlers:
            h.flush()

        with open(log_path, encoding="utf-8") as f:
            content = f.read()
        assert "hello from the test" in content
        assert "test.logger" in content
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        logging_setup._configured = saved_configured
        try:
            os.remove(log_path)
            os.rmdir(tmp_dir)
        except (PermissionError, FileNotFoundError):
            pass


def test_configure_logging_is_idempotent_by_default():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_configured = logging_setup._configured

    tmp_dir = tempfile.mkdtemp()
    log_path = os.path.join(tmp_dir, "test.log")
    try:
        configure_logging(log_path=log_path, force=True)
        count_after_first = len(root.handlers)

        configure_logging(log_path=log_path)  # no force → should be a no-op
        assert len(root.handlers) == count_after_first
    finally:
        for h in list(root.handlers):
            if h not in saved_handlers:
                root.removeHandler(h)
                h.close()
        logging_setup._configured = saved_configured
        try:
            os.remove(log_path)
            os.rmdir(tmp_dir)
        except (PermissionError, FileNotFoundError):
            pass
