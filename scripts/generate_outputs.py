from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_local import configure_environment

configure_environment(PROJECT_ROOT)

from src.file_pipeline import run_file_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="本地中文演讲 → 英文翻译文本 + 英文 TTS WAV"
    )
    parser.add_argument("audio", type=Path, help="输入 WAV 文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="输出目录（默认 outputs）",
    )
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=0.4,
        help="流式音频块大小（默认 0.4 秒，与网页演示一致）",
    )
    args = parser.parse_args()
    result = run_file_pipeline(
        args.audio,
        args.output_dir,
        block_seconds=args.block_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
