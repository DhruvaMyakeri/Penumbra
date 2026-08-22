"""Credential and path resolution. Secrets are read here and never logged."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"

load_dotenv(REPO_ROOT / ".env")

# The .env in this repo uses lowercase / trailing-space keys, so accept several
# spellings rather than forcing the user to edit the file.
_REACTOR_KEYS = ("REACTOR_API_KEY", "Reactor_api", "reactor_api", "REACTOR_API")
_GEMINI_KEYS = ("GEMINI_API_KEY", "gemini_api", "GEMINI_API")


def _first(names: tuple[str, ...]) -> str | None:
    normalised = {k.strip().lower(): v for k, v in os.environ.items()}
    for name in names:
        value = normalised.get(name.strip().lower())
        if value and value.strip():
            return value.strip()
    return None


def reactor_api_key() -> str:
    key = _first(_REACTOR_KEYS)
    if not key:
        raise RuntimeError("No Reactor API key found in environment/.env")
    return key


def gemini_api_key() -> str | None:
    return _first(_GEMINI_KEYS)


def redacted(secret: str) -> str:
    """A safe fingerprint of a secret, for logs. Never reveals the secret."""
    if not secret:
        return "<empty>"
    return f"<{len(secret)} chars, sha1:{__import__('hashlib').sha1(secret.encode()).hexdigest()[:8]}>"
