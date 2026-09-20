"""Shared test helpers: an offline embedding function and scripted LLM fakes."""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

# The suite must never write into the agent's real memory. Found 2026-09-16:
# every run added rows to backend/data/agent.db — 163 fake test domains and 32
# fake skills — and bumped the dashboard's task and skill counts, because the
# tests used the same settings as the running agent. Environment variables win
# over the project's .env, and nothing from `app` has been imported yet, so the
# whole session uses a scratch database and memory store.
_TEST_DATA = tempfile.mkdtemp(prefix="agent-tests-")
os.environ["DB_URL"] = "sqlite+aiosqlite:///" + (Path(_TEST_DATA) / "agent.db").as_posix()
os.environ["CHROMA_PATH"] = str(Path(_TEST_DATA) / "chroma")
# Nor the user's mailbox: email tests use local fake servers (tests/mail_servers.py).
os.environ["EMAIL_ADDRESS"] = ""
os.environ["EMAIL_APP_PASSWORD"] = ""
atexit.register(shutil.rmtree, _TEST_DATA, ignore_errors=True)

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class HashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding: shared words give high cosine
    similarity. Offline and instant, so memory tests never download a model."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = [0.0] * 128
            for word in text.lower().replace(",", " ").replace(".", " ").split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors

    @staticmethod
    def name() -> str:
        return "test-hash-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()


@pytest.fixture
def semantic_memory(tmp_path):
    from app.state.semantic_memory import SemanticMemory

    return SemanticMemory(path=tmp_path / "chroma", embedding_function=HashEmbedding())


def llm_message(content: str | None = None, tool_calls: list | None = None):
    """A LiteLLM-shaped chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls or None), finish_reason="stop")]
    )


def tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class ScriptedCompletion:
    """Async completion function returning queued responses (or raising queued
    exceptions) and recording every call's keyword arguments."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0) if self.responses else llm_message("")
        if isinstance(item, Exception):
            raise item
        return item
