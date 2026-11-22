import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LoggingSetupTests(unittest.TestCase):
    def test_imports_share_one_log_file_under_repository_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            project_root = temporary_root / "project"
            healthapp_root = project_root / "healthapp"
            logging_package = healthapp_root / "logging"
            compatibility_package = healthapp_root / "app_logging"
            logging_package.mkdir(parents=True)
            compatibility_package.mkdir(parents=True)

            for package in (healthapp_root, logging_package, compatibility_package):
                (package / "__init__.py").write_text("", encoding="utf-8")

            shutil.copyfile(
                REPOSITORY_ROOT / "healthapp" / "logging" / "logger.py",
                logging_package / "logger.py",
            )
            shutil.copyfile(
                REPOSITORY_ROOT / "healthapp" / "app_logging" / "logger.py",
                compatibility_package / "logger.py",
            )

            working_directory = temporary_root / "outside"
            working_directory.mkdir()
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root)
            script = """
import logging as stdlib_logging
from healthapp.logging.logger import logging as primary_logging
from healthapp.app_logging.logger import logging as compatibility_logging

assert primary_logging is stdlib_logging
assert compatibility_logging is primary_logging
primary_logging.getLogger("logger-smoke-test").info("logger-smoke-test-message")
"""
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=working_directory,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            log_files = list((project_root / "logs").glob("*.log"))
            self.assertEqual(len(log_files), 1)
            self.assertTrue(log_files[0].is_file())
            self.assertIn(
                "logger-smoke-test-message",
                log_files[0].read_text(encoding="utf-8"),
            )
            self.assertFalse((working_directory / "logs").exists())


if __name__ == "__main__":
    unittest.main()
