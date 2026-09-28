from __future__ import annotations

import pytest

from app.ingestion.pipeline import SUPPORTED_EXTENSIONS
from app.llm.groq_client import GroqClient
from app.config import Settings
from app.storage.files import resolve_stored, sanitize_filename, store_upload, validate_upload
from app.utils.errors import RagError

MAX = 1024 * 1024


def test_extension_rejected():
    with pytest.raises(RagError) as exc:
        validate_upload("payload.exe", b"MZ...", "application/octet-stream", MAX, SUPPORTED_EXTENSIONS)
    assert "Unsupported file type" in exc.value.message


def test_oversized_upload_rejected():
    with pytest.raises(RagError) as exc:
        validate_upload("notes.txt", b"x" * 200, "text/plain", 100, SUPPORTED_EXTENSIONS)
    assert "too large" in exc.value.message


def test_empty_upload_rejected():
    with pytest.raises(RagError):
        validate_upload("notes.txt", b"", "text/plain", MAX, SUPPORTED_EXTENSIONS)
    with pytest.raises(RagError):
        validate_upload("notes.txt", b"   \n", "text/plain", MAX, SUPPORTED_EXTENSIONS)


def test_extension_content_mismatch_rejected():
    with pytest.raises(RagError):
        validate_upload("fake.pdf", b"just text, no PDF header", "application/pdf", MAX, SUPPORTED_EXTENSIONS)
    with pytest.raises(RagError):
        validate_upload("notes.txt", b"hello", "application/x-msdownload", MAX, SUPPORTED_EXTENSIONS)


def test_path_traversal_is_sanitized(tmp_path):
    safe = validate_upload("../../etc/passwd.txt", b"root:x", "text/plain", MAX, SUPPORTED_EXTENSIONS)
    assert "/" not in safe and ".." not in safe
    assert sanitize_filename("..\\..\\windows\\system32\\evil.txt") == "windows_system32_evil.txt"

    _, stored = store_upload(tmp_path, safe, b"root:x")
    assert stored.parent == tmp_path.resolve()

    with pytest.raises(RagError):
        resolve_stored(tmp_path, "../../../etc/passwd")


def test_model_allowlist_enforced():
    client = GroqClient(Settings(model_20b="openai/gpt-oss-20b", model_120b="openai/gpt-oss-120b"))
    assert client.validate_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert client.validate_model(None) == "openai/gpt-oss-20b"
    with pytest.raises(RagError):
        client.validate_model("some/unapproved-model")


def test_error_messages_never_leak_details():
    err = RagError("Readable message.", 502, detail="Bearer sk-secret-token in stack trace")
    assert "secret" not in err.message
