from __future__ import annotations

import json
import os
import re
import shutil
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / ".cache" / "models" / "marian-zh-en"
DEFAULT_NLLB_MODEL_DIR = PROJECT_ROOT / ".cache" / "models" / "nllb-200-distilled-600m"
DEFAULT_QWEN_MODEL_DIR = PROJECT_ROOT / ".cache" / "models" / "qwen2.5-1.5b-instruct"
TRANSLATION_GLOSSARY_PATH = PROJECT_ROOT / "config" / "translation_glossary.json"
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_PROMPT_LEAK_RE = re.compile(
    r"(?:<\/?(?:current_segment|context_for_disambiguation_only)>|"
    r"(?:translation|answer)\s*:)",
    re.IGNORECASE,
)
_ZH_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_ZH_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_ZH_BIG_UNITS = {"万": 10_000, "亿": 100_000_000}
_ZH_NUMBER_PATTERN = r"[零〇一二两三四五六七八九十百千万亿点\d,.]+"
_EN_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
}
_EN_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_NEGATION_ZH_RE = re.compile(r"(?:不|没|未|无|非|不得|不能|并非|并不|禁止|避免)")
_NEGATION_EN_RE = re.compile(
    r"\b(?:no|not|never|neither|nor|without|cannot|can't|won't|mustn't|"
    r"shouldn't|needn't|unable|lack|lacks|lacking|prohibit(?:ed)?|forbid(?:den)?)\b",
    re.IGNORECASE,
)


class TranslationQualityError(RuntimeError):
    """Raised when a model response cannot safely enter TXT or TTS output."""

    def __init__(self, source: str, candidate: str, issues: list[str]) -> None:
        self.source = source
        self.candidate = candidate
        self.issues = tuple(issues)
        super().__init__("、".join(issues))


def _decimal_text(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _parse_chinese_number(token: str) -> Decimal | None:
    """Parse common spoken Chinese integers/decimals without external data."""
    token = str(token).replace(",", "").strip()
    if not token:
        return None
    # Spoken decimals commonly retain a Chinese magnitude suffix, for example
    # "三点五万" or "3.5亿". Handle that form before parsing the decimal part.
    if "点" in token and token[-1:] in _ZH_BIG_UNITS:
        base = _parse_chinese_number(token[:-1])
        return base * _ZH_BIG_UNITS[token[-1]] if base is not None else None
    try:
        return Decimal(token)
    except InvalidOperation:
        pass
    if "点" in token:
        integer_text, decimal_text = token.split("点", 1)
        integer = _parse_chinese_number(integer_text or "零")
        if integer is None or not decimal_text:
            return None
        digits = []
        for character in decimal_text:
            if character.isdigit():
                digits.append(character)
            elif character in _ZH_DIGITS:
                digits.append(str(_ZH_DIGITS[character]))
            else:
                return None
        return Decimal(f"{int(integer)}.{''.join(digits)}")
    if all(character.isdigit() or character in _ZH_DIGITS for character in token):
        digits = [
            character if character.isdigit() else str(_ZH_DIGITS[character])
            for character in token
        ]
        return Decimal("".join(digits))
    total = 0
    section = 0
    number = 0
    saw_number = False
    for character in token:
        if character.isdigit():
            number = number * 10 + int(character)
            saw_number = True
        elif character in _ZH_DIGITS:
            number = _ZH_DIGITS[character]
            saw_number = True
        elif character in _ZH_SMALL_UNITS:
            unit = _ZH_SMALL_UNITS[character]
            section += (number or 1) * unit
            number = 0
            saw_number = True
        elif character in _ZH_BIG_UNITS:
            unit = _ZH_BIG_UNITS[character]
            section += number
            total += (section or 1) * unit
            section = 0
            number = 0
            saw_number = True
        else:
            return None
    return Decimal(total + section + number) if saw_number else None


def _extract_english_number_values(text: str) -> set[Decimal]:
    """Extract Arabic and ordinary English number phrases from a translation."""
    tokens = re.findall(
        r"\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th)?|[A-Za-z]+",
        text,
    )
    values: set[Decimal] = set()
    current = Decimal(0)
    total = Decimal(0)
    active = False
    previous_was_value = False

    def flush() -> None:
        nonlocal current, total, active, previous_was_value
        if active:
            values.add(total + current)
        current = Decimal(0)
        total = Decimal(0)
        active = False
        previous_was_value = False

    for raw_token in tokens:
        token = raw_token.lower()
        numeric = re.fullmatch(r"([\d,]+(?:\.\d+)?)(?:st|nd|rd|th)?", token)
        if numeric:
            if previous_was_value:
                flush()
            current += Decimal(numeric.group(1).replace(",", ""))
            active = True
            previous_was_value = True
        elif token in _EN_SMALL_NUMBERS or token in {"a", "an"}:
            # Chinese "一个" is often rendered idiomatically by the English
            # singular articles "a/an", not the literal word "one".  Treat
            # that article as the value 1 so valid speech translation is not
            # rejected while all explicit multi-value checks stay strict.
            value = Decimal(_EN_SMALL_NUMBERS.get(token, 1))
            if previous_was_value:
                can_extend_tens = current >= 20 and current % 10 == 0 and value < 10
                can_extend_hundreds = current >= 100 and value < 100
                if not (can_extend_tens or can_extend_hundreds):
                    flush()
            current += value
            active = True
            previous_was_value = True
        elif token == "hundred" and active:
            current = max(current, Decimal(1)) * 100
            previous_was_value = False
        elif token in {"thousand", "million", "billion"} and active:
            scale = {
                "thousand": Decimal(1_000),
                "million": Decimal(1_000_000),
                "billion": Decimal(1_000_000_000),
            }[token]
            total += max(current, Decimal(1)) * scale
            current = Decimal(0)
            previous_was_value = False
        elif token == "and" and active:
            previous_was_value = False
        else:
            flush()
    flush()
    return values


def _semantic_constraints(source: str) -> list[tuple[str, Decimal | str, str]]:
    """Extract high-confidence facts that an English translation must preserve."""
    constraints: list[tuple[str, Decimal | str, str]] = []

    def add(kind: str, value: Decimal | str | None, label: str) -> None:
        if value is None:
            return
        item = (kind, value, label)
        if item not in constraints:
            constraints.append(item)

    for match in re.finditer(rf"({_ZH_NUMBER_PATTERN})\s*年", source):
        add("year", _parse_chinese_number(match.group(1)), match.group(0))
    for match in re.finditer(rf"百分之\s*({_ZH_NUMBER_PATTERN})", source):
        add("percent", _parse_chinese_number(match.group(1)), match.group(0))
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", source):
        add("percent", Decimal(match.group(1)), match.group(0))
    for match in re.finditer(
        rf"({_ZH_NUMBER_PATTERN})\s*(?:元|块钱|人民币|美元|美金)", source
    ):
        add("currency", _parse_chinese_number(match.group(1)), match.group(0))
    for match in re.finditer(rf"({_ZH_NUMBER_PATTERN})\s*月", source):
        add("month", _parse_chinese_number(match.group(1)), match.group(0))
    for match in re.finditer(rf"({_ZH_NUMBER_PATTERN})\s*(?:日|号)", source):
        add("day", _parse_chinese_number(match.group(1)), match.group(0))
    for match in re.finditer(
        rf"({_ZH_NUMBER_PATTERN})\s*(?:点|时)(?![零〇一二两三四五六七八九\d])",
        source,
    ):
        add("hour", _parse_chinese_number(match.group(1)), match.group(0))
    count_units = (
        "毫秒|分钟|小时|阶段|模块|指标|个人|个|项|组|人|次|段|份|种|台|套"
    )
    for match in re.finditer(rf"({_ZH_NUMBER_PATTERN})\s*(?:{count_units})", source):
        add("quantity", _parse_chinese_number(match.group(1)), match.group(0))
    for match in re.finditer(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", source):
        add("label", match.group(1), match.group(0))
    return constraints


def _target_hours(text: str) -> set[Decimal]:
    hours: set[Decimal] = set()
    for match in re.finditer(
        r"\b(\d{1,2})(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)\b",
        text,
        re.IGNORECASE,
    ):
        hour = int(match.group(1)) % 12
        if match.group(2).lower().startswith("p"):
            hour += 12
        hours.add(Decimal(hour))
    return hours


def semantic_consistency_issues(source: str, candidate: str) -> list[str]:
    """Return deterministic hard-fact mismatches in a translation."""
    values = _extract_english_number_values(candidate)
    target_labels = {
        match.group(1).upper()
        for match in re.finditer(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", candidate)
    }
    lower_candidate = candidate.lower()
    target_months = {
        value for name, value in _EN_MONTHS.items() if re.search(rf"\b{name}\b", lower_candidate)
    }
    target_hours = _target_hours(candidate)
    issues: list[str] = []
    for kind, value, label in _semantic_constraints(source):
        preserved = False
        if kind == "label":
            preserved = str(value).upper() in target_labels
        elif kind == "month":
            preserved = value in values or int(value) in target_months
        elif kind == "hour":
            preserved = value in values or value in target_hours
        else:
            preserved = value in values
        if not preserved:
            display = str(value) if isinstance(value, str) else _decimal_text(value)
            kind_label = {
                "year": "年份",
                "percent": "百分比",
                "currency": "金额",
                "month": "月份",
                "day": "日期",
                "hour": "时间",
                "quantity": "数量",
                "label": "编号",
            }[kind]
            issues.append(f"{kind_label}不一致或缺失：{label} 必须保留为 {display}")
    return issues


def translation_semantic_warnings(source: str, candidate: str) -> list[str]:
    """Return soft semantic risks that should trigger one retry but not block."""
    if _NEGATION_ZH_RE.search(source) and not _NEGATION_EN_RE.search(candidate):
        return ["原文包含否定关系，但英文中未检测到明确否定表达"]
    return []


def _constraint_prompt(source: str) -> str:
    constraints = _semantic_constraints(source)
    if not constraints:
        return ""
    rendered = []
    for kind, value, source_label in constraints:
        display = str(value) if isinstance(value, str) else _decimal_text(value)
        if (
            kind == "currency"
            and isinstance(value, Decimal)
            and value == value.to_integral_value()
            and abs(value) >= 1000
        ):
            arabic = f"{int(value):,}"
        else:
            arabic = display
        if kind in {"currency", "percent", "quantity", "year"}:
            rendered.append(
                f'{kind} from "{source_label}" must preserve the exact value '
                f'{display}; use the unambiguous Arabic numeral "{arabic}"'
            )
        elif kind == "month":
            rendered.append(f"month from {source_label}={display} (a month name is allowed)")
        elif kind == "hour":
            rendered.append(f"hour from {source_label}={display} (12-hour AM/PM is allowed)")
        else:
            rendered.append(f"{kind} from {source_label}={display}")
    return (
        "Required factual constraints. Never calculate, round, omit, or alter them: "
        + "; ".join(rendered)
        + ".\n"
    )


def _load_translation_glossary() -> dict[str, str]:
    if not TRANSLATION_GLOSSARY_PATH.exists():
        return {}
    return json.loads(TRANSLATION_GLOSSARY_PATH.read_text(encoding="utf-8"))


_TRANSLATION_GLOSSARY = _load_translation_glossary()


def _replace_glossary_terms(text: str) -> str:
    for source, target in _TRANSLATION_GLOSSARY.items():
        text = text.replace(source, f" {target} ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _normalize_translation_candidate(text: str) -> str:
    """Remove harmless wrappers before running the strict quality gate."""
    text = str(text or "").strip()
    text = re.sub(r"^```(?:english)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:English\s+)?translation\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip()
    return _replace_glossary_terms(text)


def translation_quality_issues(source: str, candidate: str) -> list[str]:
    """Return deterministic reasons why a translation must be rejected."""
    candidate = _normalize_translation_candidate(candidate)
    issues: list[str] = []
    if not candidate:
        issues.append("翻译结果为空")
    if _CJK_RE.search(candidate):
        issues.append("翻译结果仍含中文字符")
    if _PROMPT_LEAK_RE.search(candidate):
        issues.append("翻译结果包含提示词或标签")
    if len(candidate) > max(120, len(source) * 7):
        issues.append("翻译结果长度异常")
    if candidate and not re.search(r"[A-Za-z]", candidate):
        issues.append("翻译结果不含英文单词")
    if candidate:
        issues.extend(semantic_consistency_issues(source, candidate))
    return issues


def validate_translation(source: str, candidate: str) -> str:
    """Normalize and validate one result before it can be saved or spoken."""
    normalized = _normalize_translation_candidate(candidate)
    issues = translation_quality_issues(source, normalized)
    if issues:
        raise TranslationQualityError(source, normalized, issues)
    return normalized


def _ascii_tokenizer_dir(
    model_dir: Path,
    name: str,
    required_files: tuple[str, ...],
) -> Path:
    """Mirror only tokenizer files when Windows cannot read Unicode paths."""
    if os.name != "nt" or str(model_dir).isascii():
        return model_dir
    runtime_dir = Path(model_dir.anchor) / ".simul_agent_runtime" / name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for filename in required_files:
        source = model_dir / filename
        target = runtime_dir / filename
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
    return runtime_dir


class MarianTranslator:
    """Lazy local MarianMT translator.

    The model is loaded only when translate() is called, so importing this
    module never consumes GPU memory or starts a model process.
    """

    def __init__(self, model_dir: str | os.PathLike[str] | None = None) -> None:
        self.model_dir = Path(model_dir or os.environ.get("TRANSLATION_MODEL_DIR", DEFAULT_MODEL_DIR))
        self.device = os.environ.get("TRANSLATION_DEVICE", "auto")
        self._tokenizer = None
        self._model = None
        self.load_seconds: float | None = None

    def _tokenizer_dir(self) -> Path:
        return _ascii_tokenizer_dir(
            self.model_dir,
            "marian-zh-en-tokenizer",
            (
            "config.json",
            "source.spm",
            "target.spm",
            "tokenizer_config.json",
            "vocab.json",
            ),
        )

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Translation model is not installed: {self.model_dir}. "
                "Download it before enabling real translation."
        )
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        started = time.perf_counter()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer_dir = self._tokenizer_dir()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir), local_files_only=True)
        self._model.to(self.device)
        self._model.eval()
        self.load_seconds = time.perf_counter() - started

    def translate(self, text: str) -> str:
        return self.translate_many([text])[0]

    def translate_many(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        self.load()
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        import torch

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_length=192,
                num_beams=int(os.environ.get("TRANSLATION_BEAMS", "1")),
            )
        decoded = self._tokenizer.batch_decode(output, skip_special_tokens=True)
        return [validate_translation(source, text) for source, text in zip(texts, decoded)]


class NLLBTranslator:
    """Local NLLB-200 distilled Chinese-to-English translator."""

    def __init__(self, model_dir: str | os.PathLike[str] | None = None) -> None:
        self.model_dir = Path(
            model_dir
            or os.environ.get("NLLB_MODEL_DIR", DEFAULT_NLLB_MODEL_DIR)
        )
        self.device = os.environ.get("TRANSLATION_DEVICE", "auto")
        self._tokenizer = None
        self._model = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"NLLB model is not installed: {self.model_dir}. "
                "Download it before enabling the NLLB backend."
            )
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        started = time.perf_counter()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer_dir = _ascii_tokenizer_dir(
            self.model_dir,
            "nllb-tokenizer",
            (
                "config.json",
                "generation_config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            src_lang="zho_Hans",
            local_files_only=True,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self._model.to(self.device)
        self._model.eval()
        self.load_seconds = time.perf_counter() - started

    def translate(self, text: str) -> str:
        return self.translate_many([text])[0]

    def translate_many(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        self.load()
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        import torch

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=160,
                num_beams=int(os.environ.get("TRANSLATION_BEAMS", "2")),
            )
        decoded = self._tokenizer.batch_decode(output, skip_special_tokens=True)
        return [validate_translation(source, text) for source, text in zip(texts, decoded)]


class QwenTranslator:
    """Local Qwen instruction model used for context-aware speech translation."""

    SYSTEM_PROMPT = (
        "Translate the Chinese sentence into natural, accurate, concise English "
        "for a formal speech. Preserve names, numbers, negation, and causality. "
        "Correct obvious speech-recognition homophones using the context. "
        "Output only the translation of the current segment in English. "
        "Never include Chinese characters, explanations, or the previous context."
    )

    def __init__(self, model_dir: str | os.PathLike[str] | None = None) -> None:
        self.model_dir = Path(
            model_dir
            or os.environ.get("QWEN_MODEL_DIR", DEFAULT_QWEN_MODEL_DIR)
        )
        self.device = os.environ.get("TRANSLATION_DEVICE", "auto")
        self._tokenizer = None
        self._model = None
        self.load_seconds: float | None = None
        self._recent_source: list[str] = []

    def reset_context(self) -> None:
        self._recent_source.clear()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Qwen model is not installed: {self.model_dir}. "
                "Download it before enabling the Qwen backend."
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self._tokenizer.padding_side = "left"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            dtype=dtype,
        )
        self._model.to(self.device)
        self._model.eval()
        self.load_seconds = time.perf_counter() - started

    def translate(self, text: str) -> str:
        return self.translate_many([text])[0]

    def translate_many(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        self.load()
        prompts = []
        running_context = list(self._recent_source)
        for text in texts:
            terms = [
                f"{source}={target}"
                for source, target in _TRANSLATION_GLOSSARY.items()
                if source in text
            ]
            terminology = (
                "Terminology: " + "; ".join(terms) + ".\n"
                if terms
                else ""
            )
            factual_constraints = _constraint_prompt(text)
            if running_context:
                user_content = (
                    terminology
                    + factual_constraints
                    + "<context_for_disambiguation_only>"
                    + " ".join(running_context[-2:])
                    + "</context_for_disambiguation_only>\n"
                    + "<current_segment>"
                    + text
                    + "</current_segment>"
                )
            else:
                user_content = (
                    terminology
                    + factual_constraints
                    + "<current_segment>"
                    + text
                    + "</current_segment>"
                )
            prompts.append(
                self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
                )
            )
            running_context.append(text)
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        import torch

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        translations = [
            text.strip()
            for text in self._tokenizer.batch_decode(
                output[:, prompt_length:],
                skip_special_tokens=True,
            )
        ]
        for index, (source, translated) in enumerate(zip(texts, translations)):
            hard_issues = translation_quality_issues(source, translated)
            soft_warnings = translation_semantic_warnings(source, translated)
            if hard_issues or soft_warnings:
                failure_reason = "; ".join(hard_issues + soft_warnings)
                retry_prompt = self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Translate only the current Chinese segment below. "
                                "Do not use or repeat any previous segment. Return "
                                "one concise English sentence with no Chinese "
                                "characters. Preserve every required factual value "
                                "exactly and preserve negation.\n"
                                + _constraint_prompt(source)
                                + "The previous candidate failed these checks: "
                                + failure_reason
                                + "\n<current_segment>"
                                + source
                                + "</current_segment>"
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                retry_inputs = self._tokenizer(
                    retry_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                retry_inputs = {
                    key: value.to(self.device)
                    for key, value in retry_inputs.items()
                }
                with torch.inference_mode():
                    retry_output = self._model.generate(
                        **retry_inputs,
                        max_new_tokens=64,
                        do_sample=False,
                    )
                retry_length = retry_inputs["input_ids"].shape[1]
                translations[index] = self._tokenizer.decode(
                    retry_output[0, retry_length:],
                    skip_special_tokens=True,
                ).strip()
        translations = [
            validate_translation(source, translated)
            for source, translated in zip(texts, translations)
        ]
        self._recent_source = running_context[-6:]
        return translations


def build_translator():
    """Select the strongest installed open model, with an explicit override."""
    backend = os.environ.get("TRANSLATION_BACKEND", "auto").lower()
    if backend == "marian":
        return MarianTranslator()
    if backend == "nllb":
        return NLLBTranslator()
    if backend == "qwen":
        return QwenTranslator()
    if DEFAULT_QWEN_MODEL_DIR.exists():
        return QwenTranslator()
    if DEFAULT_NLLB_MODEL_DIR.exists():
        return NLLBTranslator()
    return MarianTranslator()


class DemoTranslator:
    """Small deterministic translator for UI wiring tests only.

    This class must never be used for final competition outputs. It lets us
    test the page and segment mapping while model downloads are paused.
    """

    _PHRASES = {
        "今天我想介绍我们的研究成果": "Today, I would like to introduce our research findings.",
        "这个系统可以帮助演讲者进行实时翻译": "This system can help speakers perform real-time translation.",
    }

    def translate(self, text: str) -> str:
        return self._PHRASES.get(text, f"[Translation pending] {text}")
