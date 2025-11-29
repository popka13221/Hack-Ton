import re


_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[\\n\\t]+")


def normalize_text(text: str) -> str:
    """Minimal normalization used for both training and inference."""
    lowered = text.lower().strip()
    cleaned = _punct_re.sub(" ", lowered)
    collapsed = _whitespace_re.sub(" ", cleaned)
    return collapsed.strip()
