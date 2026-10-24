import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_local import MODEL_PATHS, available_port, configure_environment, startup_report


class PublicLauncherTests(unittest.TestCase):
    def test_default_model_paths_are_inside_this_project_not_parent(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory) / "clean-copy"
            root.mkdir()
            configure_environment(root)
            for name, relative in MODEL_PATHS.items():
                self.assertEqual(Path(os.environ[name]), (root / relative).resolve())
            self.assertEqual(os.environ["TRANSLATION_BACKEND"], "qwen")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["SHERPA_COMMIT_MIN_CHARS"], "14")

    def test_explicit_paths_override_local_config_and_resolve_from_root(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"QWEN_MODEL_DIR": "mine"}, clear=True):
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/local_settings.json").write_text(json.dumps({"QWEN_MODEL_DIR": "other", "SHERPA_MODEL_DIR": "my-asr"}), encoding="utf-8")
            configure_environment(root)
            self.assertEqual(Path(os.environ["QWEN_MODEL_DIR"]), (root / "mine").resolve())
            self.assertEqual(Path(os.environ["SHERPA_MODEL_DIR"]), (root / "my-asr").resolve())

    def test_unsupported_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/local_settings.json").write_text('{"UNKNOWN": "value"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                configure_environment(root)

    def test_missing_models_are_reported_without_importing_model_frameworks(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            configure_environment(root)
            checks = startup_report(root)
            self.assertEqual(next(c for c in checks if c["name"] == "Qwen translator")["status"], "error")
            self.assertEqual(next(c for c in checks if c["name"] == "Runtime directory")["status"], "ok")

    def test_invalid_port_is_rejected(self):
        for value in (0, 65536, -1):
            with self.assertRaises(ValueError):
                available_port(value)


if __name__ == "__main__":
    unittest.main()
