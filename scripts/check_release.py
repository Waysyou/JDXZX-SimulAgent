"""Small source-only publication check; not a complete secret/license audit."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {".git", ".venv", "venv", ".cache", "__pycache__", ".pytest_cache", "output", "outputs", "logs", "backups", "runtime"}
FORBIDDEN_SUFFIXES = {".wav", ".mp3", ".mp4", ".onnx", ".safetensors", ".bin", ".pt", ".pth", ".zip", ".7z", ".whl", ".dmp", ".pdf", ".docx", ".pptx"}
SECRET_PATTERNS = {
    "private-user-path": re.compile(r"(?:[A-Z]:[\\/]+Users[\\/]+(?!Public\b)[^\\/\s]+|wxid_[A-Za-z0-9_]+)", re.I),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:ghp|gho|ghu|ghs)_[A-Za-z0-9]{24,}\b"),
    "api-token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "private-share-link": re.compile(r"pan\.baidu\.com/s/|pwd=[A-Za-z0-9]{4}\b"),
}


def publication_files(root: Path = ROOT) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in PRIVATE_DIRS for part in relative.parts):
            continue
        if path.name == "local_settings.json" or path.name.startswith(".env"):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def inspect(root: Path = ROOT) -> dict:
    findings = []
    files = publication_files(root)
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"file": relative, "issue": "forbidden-publication-asset"})
            continue
        if any(part.lower() in {"xfdata", "user_data", "prediction_result", "final_submission", "code_audit", "valid", "test"} for part in Path(relative).parts):
            findings.append({"file": relative, "issue": "contest-or-output-directory"})
        if path.name in {".gitignore", ".gitattributes"}:
            continue
        try:
            content = path.read_text(encoding="utf-8-sig")
        except (UnicodeError, OSError):
            findings.append({"file": relative, "issue": "unreviewed-binary"})
            continue
        for number, line in enumerate(content.splitlines(), 1):
            for kind, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({"file": relative, "line": number, "issue": kind})
        try:
            if path.suffix == ".py":
                ast.parse(content, filename=relative)
            if path.suffix == ".json":
                json.loads(content)
        except (SyntaxError, ValueError) as exc:
            findings.append({"file": relative, "issue": type(exc).__name__})
    return {"automated_checks_ok": not findings, "source_file_count": len(files),
            "findings": findings, "public_release_authorized": False,
            "manual_review_required": ["team consent and attribution", "contest/award obligations", "project and dependency licenses", "full secret/data review", "fresh install and real model inference"]}


if __name__ == "__main__":
    report = inspect()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["automated_checks_ok"] else 1)
