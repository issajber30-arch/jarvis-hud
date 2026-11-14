"""Secret redaction for everything bound for Claude, logs, the HUD or TTS."""
import re

_R = "[REDACTED]"
_PEM = re.compile(r"-----BEGIN [A-Z ]*-----.*?(?:-----END [A-Z ]*-----|\Z)", re.S)
_TOKENS = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[abprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|eyJ[\w\-]+\.[\w\-]+\.[\w\-]+"
    r"|AIza[\w-]{35}|sk_(?:live|test)_\w+|(?i:bearer)\s+[\w.~+/=-]{16,}")
_ASSIGN = re.compile(r"(?i)([\w-]*(?:password|passwd|pwd|token|secret|api[_-]?key)[\w-]*)(\"?\s*[=:]\s*)"
                     r"(\"[^\"]*\"|'[^']*'|[^\s,;]+)")


def redact(text: str) -> str:
    text = _PEM.sub(_R, text)
    text = _TOKENS.sub(_R, text)
    return _ASSIGN.sub(lambda m: f"{m.group(1)}{m.group(2)}{_R}", text)
