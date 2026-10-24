import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.text_stabilizer import TextStabilizer, normalize_text


class TextStabilizerTests(unittest.TestCase):
    def test_traditional_to_simplified_and_fillers(self):
        self.assertEqual(normalize_text("嗯 這個 測試"), "这个 测试")

    def test_domain_correction_is_not_enabled_globally(self):
        self.assertEqual(normalize_text("蓝色的册试"), "蓝色的册试")

    def test_explicit_domain_profile_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "owned_example.json"
            profile.write_text(json.dumps({"册试": "测试"}), encoding="utf-8")
            with patch.dict("os.environ", {"ASR_CORRECTIONS_PROFILE": str(profile)}):
                self.assertEqual(normalize_text("这是册试，蓝色"), "这是测试,蓝色")

    def test_exact_duplicate_is_not_appended(self):
        stabilizer = TextStabilizer()
        first = stabilizer.feed("今天我想介绍我们的研究成果", 0, 2)
        second = stabilizer.feed("今天我想介绍我们的研究成果", 2, 4)
        self.assertEqual(len(stabilizer.segments), 1)
        self.assertEqual(first[0]["segment_id"], second[0]["segment_id"])
        self.assertEqual(stabilizer.segments[0].end, 4)

    def test_overlap_is_merged(self):
        stabilizer = TextStabilizer()
        stabilizer.feed("我们今天讨论人工智能", 0, 2)
        stabilizer.feed("人工智能在教育中的应用", 2, 4)
        self.assertEqual(stabilizer.segments[0].text, "我们今天讨论人工智能在教育中的应用")


if __name__ == "__main__":
    unittest.main()
