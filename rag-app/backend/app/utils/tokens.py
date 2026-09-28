"""Token counting. Provider-reported usage is authoritative; these are estimates."""
from __future__ import annotations

_enc = None


def _encoder():
    global _enc
    if _enc is None:
        try:  # pragma: no cover - depends on optional dep
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc = False
    return _enc


def count_tokens(text: str) -> int:
    """Estimated token count (labelled as an estimate everywhere it is shown)."""
    if not text:
        return 0
    enc = _encoder()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)
