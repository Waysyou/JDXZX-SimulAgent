from __future__ import annotations

import os
import shutil
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE_MODEL = (
    PROJECT_ROOT
    / ".cache"
    / "models"
    / "piper-en_US-lessac-medium"
    / "en_US-lessac-medium.onnx"
)
DEFAULT_ESPEAK_DATA_DIR = PROJECT_ROOT / "runtime" / "piper_espeak_data"
DEFAULT_WINDOWS_ESPEAK_CACHE_DIR = (
    Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "JDXZX_PiperData"
)


def _native_safe_espeak_data_dir(source: Path) -> Path:
    """Return an ASCII eSpeak data path for Piper's Windows native library.

    Some Piper Windows builds cannot initialize eSpeak from a path containing
    Chinese characters. The project may be extracted into such a path, so a
    small (about 18 MB) copy is staged under the public Windows profile only
    when necessary. An explicit ASCII ``PIPER_ESPEAK_DATA_DIR`` needs no copy.
    """
    source = Path(source)
    if not (source / "phontab").exists():
        raise FileNotFoundError(
            "Piper eSpeak data is missing. Expected phontab under: " f"{source}"
        )
    if str(source).isascii():
        return source

    cache_dir = Path(
        os.environ.get("PIPER_ESPEAK_CACHE_DIR") or DEFAULT_WINDOWS_ESPEAK_CACHE_DIR
    )
    if not str(cache_dir).isascii():
        raise RuntimeError(
            "Piper requires an ASCII eSpeak runtime path on this Windows build. "
            "Set PIPER_ESPEAK_CACHE_DIR to an ASCII writable directory."
        )
    if not (cache_dir / "phontab").exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, cache_dir, dirs_exist_ok=True)
    return cache_dir


class PiperTTS:
    """Lazy wrapper for a local Piper voice model.

    The wrapper does not import or load an audio model until synthesize() is
    called, so it is safe to keep the web server open while another program is
    using the GPU.
    """

    def __init__(self, voice_model: str | os.PathLike[str] | None = None) -> None:
        self.voice_model = Path(
            voice_model
            or os.environ.get("PIPER_VOICE_MODEL", "")
            or DEFAULT_VOICE_MODEL
        )
        self.espeak_data_dir = Path(
            os.environ.get("PIPER_ESPEAK_DATA_DIR") or DEFAULT_ESPEAK_DATA_DIR
        )
        self.length_scale = float(os.environ.get("PIPER_LENGTH_SCALE", "0.75"))
        self._voice = None

    def load(self):
        if not self.voice_model.exists():
            raise FileNotFoundError(
                "Piper voice model is not configured. Set PIPER_VOICE_MODEL to an open voice model."
            )
        from piper import PiperVoice

        if self._voice is None:
            self.espeak_data_dir = _native_safe_espeak_data_dir(self.espeak_data_dir)
            load_kwargs = {"espeak_data_dir": str(self.espeak_data_dir)}
            self._voice = PiperVoice.load(str(self.voice_model), **load_kwargs)
        return self._voice

    def synthesize(self, text: str, output_path: str | os.PathLike[str]) -> Path:
        from piper.config import SynthesisConfig

        voice = self.load()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Piper 1.x exposes synthesize_wav; keeping this call isolated makes
        # it easy to adapt to another open voice backend later.
        with wave.open(str(output), "wb") as wav_file:
            voice.synthesize_wav(
                text,
                wav_file,
                syn_config=SynthesisConfig(length_scale=self.length_scale),
            )
        return output
