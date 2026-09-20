"""
Level 3 learning upgrades and the free Groq + Gemini provider chain:

  * lessons indexed on their own, so a task can be recalled through what it
    taught (tech audit 7: skills combining across differently worded tasks)
  * the task and its lesson embedded in one batched call (tech audit 2.3)
  * two free Gemini models tried in order, and an immediate switch to Gemini
    when every Groq key is cooling
  * step results in the live stream (tech audit 5.3)
  * wake-word tasks shown in the app window, clean shutdown for the desktop app
  * Vosk wake / stop word detection on real human recordings
"""

from __future__ import annotations

import asyncio
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.explain import build_explanation
from app.agent.toolkit import describe_result
from app.config import get_settings
from app.state.semantic_memory import SemanticMemory
from app.utils import litellm_patch, llm
from tests.conftest import HashEmbedding, llm_message

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "voice"


@pytest.fixture
def memory_settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "semantic_memory_enabled", True)
    monkeypatch.setattr(s, "semantic_min_similarity", 0.35)
    monkeypatch.setattr(s, "semantic_lesson_min_similarity", 0.2)
    monkeypatch.setattr(s, "semantic_recall_k", 5)
    return s


class CountingEmbedding(HashEmbedding):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def __call__(self, input):
        self.calls.append(len(input))
        return super().__call__(input)


CART_LESSON = "Worked: closed the login popup then opened the product and clicked add to cart"


# =============================================================================
# Lessons collection and batched embeddings
# =============================================================================

def test_task_is_recalled_through_its_lesson(tmp_path, memory_settings):
    memory = SemanticMemory(path=tmp_path / "chroma", embedding_function=HashEmbedding())
    memory.upsert_sync(task_id="f1", instruction="search headphones flipkart", domain="flipkart.com",
                       scope="browser", success=True, trajectory=[{"action_type": "click_element"}], lesson=CART_LESSON)

    found = memory.recall_sync("add laptop to cart on new site")

    assert [e.task_id for e in found] == ["f1"]
    assert found[0].matched_on == "lesson"
    assert found[0].lesson == CART_LESSON


def test_similar_wording_still_matches_on_the_task(tmp_path, memory_settings):
    memory = SemanticMemory(path=tmp_path / "chroma", embedding_function=HashEmbedding())
    memory.upsert_sync(task_id="w1", instruction="open whatsapp and message rakesh hello", domain="local",
                       scope="local", success=True, trajectory=[], lesson="use ctrl f")
    found = memory.recall_sync("open whatsapp and message priyal hello")
    assert found[0].task_id == "w1" and found[0].matched_on == "task"


def test_task_and_lesson_are_embedded_in_one_call(tmp_path, memory_settings):
    ef = CountingEmbedding()
    memory = SemanticMemory(path=tmp_path / "chroma", embedding_function=ef)
    memory.upsert_sync(task_id="b1", instruction="search shoes on myntra", domain="myntra.com", scope="browser",
                       success=True, trajectory=[], lesson=CART_LESSON)
    assert ef.calls == [2], "one batched call for the instruction and its lesson"

    ef.calls.clear()
    memory.recall_sync("search sandals on myntra")
    assert ef.calls == [1], "the query is embedded once and reused for both searches"


def test_old_experiences_get_their_lessons_indexed(tmp_path, memory_settings):
    memory = SemanticMemory(path=tmp_path / "chroma", embedding_function=HashEmbedding())
    collection = memory._get_collection()
    # Stored the way experiences were before the lessons collection existed.
    collection.upsert(
        ids=["old1"], documents=["search headphones flipkart"],
        metadatas=[{"task_id": "old1", "domain": "flipkart.com", "scope": "browser", "success": True,
                    "tools": "[]", "lesson": CART_LESSON, "error": "", "feedback": 0, "user_disputed": False,
                    "feedback_comment": "", "created_at": "2026-09-10T00:00:00+00:00"}],
    )
    assert memory.recall_sync("add laptop to cart on new site") == []
    assert memory.backfill_lessons() == 1
    assert memory.backfill_lessons() == 0
    assert memory.recall_sync("add laptop to cart on new site")[0].task_id == "old1"
    assert memory.warm() is True


def test_query_retries_while_the_index_is_being_saved(monkeypatch):
    import time as real_time

    from app.state import semantic_memory as sm

    monkeypatch.setattr(sm, "time", SimpleNamespace(sleep=lambda seconds: None, monotonic=real_time.monotonic))
    calls = []

    def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError("Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk")
        return {"ok": True}

    assert sm._query_with_retry(flaky, n_results=1) == {"ok": True}
    assert len(calls) == 3

    def broken(**kwargs):
        raise ValueError("invalid include")

    with pytest.raises(ValueError):
        sm._query_with_retry(broken)


def test_index_still_not_ready_falls_back_to_direct_search(monkeypatch):
    import time as real_time

    from app.state import semantic_memory as sm

    monkeypatch.setattr(sm, "time", SimpleNamespace(sleep=lambda seconds: None, monotonic=real_time.monotonic))

    def never_ready(**kwargs):
        raise RuntimeError("Error creating hnsw segment reader: Nothing found on disk")

    answer = sm._query_with_retry(never_ready, fallback=lambda **kwargs: {"direct": kwargs["n_results"]}, n_results=3)
    assert answer == {"direct": 3}


def test_direct_search_matches_the_indexed_search(tmp_path, memory_settings):
    from app.state import semantic_memory as sm

    memory = SemanticMemory(path=tmp_path / "chroma", embedding_function=HashEmbedding())
    memory.upsert_sync(task_id="w", instruction="open whatsapp and message rakesh hello", domain="local",
                       scope="local", success=True, trajectory=[], lesson="")
    memory.upsert_sync(task_id="a", instruction="search iphone price on amazon", domain="amazon.in",
                       scope="browser", success=True, trajectory=[], lesson="")
    query = memory._embed(["message priyal hello on whatsapp"])
    direct = sm._direct_query(memory._collection, memory._embed, query_embeddings=query, n_results=2,
                              include=["documents", "metadatas", "distances"])
    assert direct["ids"][0][0] == "w"
    assert direct["documents"][0][0] == "open whatsapp and message rakesh hello"
    assert direct["distances"][0][0] < direct["distances"][0][1]


def test_explanation_says_when_a_lesson_matched(tmp_path, memory_settings):
    memory = SemanticMemory(path=tmp_path / "chroma", embedding_function=HashEmbedding())
    memory.upsert_sync(task_id="f1", instruction="search headphones flipkart", domain="flipkart.com",
                       scope="browser", success=True, trajectory=[], lesson=CART_LESSON)
    experiences = memory.recall_sync("add laptop to cart on new site")
    text = build_explanation(trajectory=[], success=True, experiences=experiences, strategy="",
                             strategy_from_llm=False, retried=False, first_attempt_error="",
                             retry_blocked_reason="", error="")
    assert "its lesson is" in text and "relevant" in text


# =============================================================================
# Free provider chain: Groq -> Gemini 3.6 Flash -> Gemini 2.5 Flash-Lite
# =============================================================================

class _Pool:
    size = 2

    def __init__(self, cooling: float = 0.0):
        self.cooling = cooling

    def acquire(self, model=None):
        return "gsk_test"

    def mark_success(self, key, model=None):
        pass

    def mark_rate_limited(self, key, cooldown_seconds=None, model=None):
        pass

    def min_remaining_cooldown(self, model=None):
        return self.cooling


@pytest.fixture
def provider_chain(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", "gemini-test-key")
    monkeypatch.setattr(s, "llm_fallback_model", "gemini/gemini-3.6-flash, gemini/gemini-2.5-flash-lite")
    monkeypatch.setattr(s, "groq_max_cooldown_wait_seconds", 3.0)
    monkeypatch.setattr(litellm_patch, "_MIN_CALL_INTERVAL", 0.0)
    llm._reset_for_tests()  # no Gemini cooldowns left over from other tests
    calls: list[str] = []

    def install(pool: _Pool, failing: set[str]):
        monkeypatch.setattr("app.utils.key_pool.get_key_pool", lambda: pool)

        async def fake(*args, **kwargs):
            calls.append(kwargs["model"])
            if any(kwargs["model"].startswith(prefix) for prefix in failing):
                raise RuntimeError("503 Service Unavailable" if kwargs["model"].startswith("groq/") else "429 quota exhausted")
            return llm_message(f"answer from {kwargs['model']}")

        monkeypatch.setattr(litellm_patch, "_orig_acompletion", fake)
        return calls

    yield install
    llm._reset_for_tests()


async def test_second_gemini_model_is_tried_when_the_first_fails(provider_chain):
    calls = provider_chain(_Pool(), failing={"groq/", "gemini/gemini-3.6-flash"})
    result = await litellm_patch._sanitized_acompletion(
        model="groq/openai/gpt-oss-120b", messages=[{"role": "user", "content": "hi"}]
    )
    assert result.choices[0].message.content == "answer from gemini/gemini-2.5-flash-lite"
    assert calls[-2:] == ["gemini/gemini-3.6-flash", "gemini/gemini-2.5-flash-lite"]


async def test_all_groq_keys_cooling_switches_to_gemini_without_waiting(provider_chain):
    calls = provider_chain(_Pool(cooling=30.0), failing=set())
    started = asyncio.get_running_loop().time()
    result = await litellm_patch._sanitized_acompletion(
        model="groq/openai/gpt-oss-20b", messages=[{"role": "user", "content": "hi"}]
    )
    assert asyncio.get_running_loop().time() - started < 2
    assert calls == ["gemini/gemini-3.6-flash"]
    assert result.choices[0].message.content == "answer from gemini/gemini-3.6-flash"


async def test_every_provider_failing_raises_the_last_error(provider_chain):
    provider_chain(_Pool(), failing={"groq/", "gemini/"})
    with pytest.raises(RuntimeError, match="quota"):
        await litellm_patch._sanitized_acompletion(model="groq/openai/gpt-oss-120b", messages=[{"role": "user", "content": "x"}])


def test_key_pool_reads_keys_from_settings_when_not_in_environment(monkeypatch):
    from app.utils.key_pool import build_key_pool_from_env

    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(get_settings(), "groq_api_keys", "gsk_first1,gsk_second2")
    pool = build_key_pool_from_env()
    assert pool is not None and pool.size == 2


def test_llama_4_scout_is_preferred_for_vision_when_groq_offers_it(monkeypatch):
    monkeypatch.setattr(get_settings(), "groq_vision_model", "qwen/qwen3.8-27b")
    llm._set_available_for_tests({"meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3.8-27b"})
    assert llm.resolve_groq_vision_model() == "meta-llama/llama-4-scout-17b-16e-instruct"
    llm._set_available_for_tests({"qwen/qwen3.8-27b"})
    assert llm.resolve_groq_vision_model() == "qwen/qwen3.8-27b"
    llm._reset_for_tests()


async def test_vision_tries_each_gemini_model(monkeypatch):
    from app.utils import vision

    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", "k")
    monkeypatch.setattr(s, "gemini_vision_model", "gemini/a,gemini/b")
    llm._reset_for_tests()
    tried: list[str] = []

    async def fake(**kwargs):
        tried.append(kwargs["model"])
        if kwargs["model"] == "gemini/a":
            raise RuntimeError("quota")
        return llm_message("a login form")

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake)
    assert await vision._describe_with_gemini("aGk=", "what is shown?", "image/png", 100) == "a login form"
    assert tried == ["gemini/a", "gemini/b"]
    llm._reset_for_tests()


# =============================================================================
# Live stream results, wake-word tasks in the window, desktop shutdown
# =============================================================================

def test_describe_result_uses_the_tools_first_line():
    assert describe_result("list_recent_files", "Recent files in Downloads:\n1. a.pdf") == "✅ Recent files in Downloads:"
    assert describe_result("navigate_browser", '{"url": "x"}\nNavigated to https://x.com') == "✅ Navigated to https://x.com"
    assert describe_result("send_keys", "Sent keys") is None
    assert describe_result("find_files", "") is None
    assert describe_result("read_file", "x" * 200).endswith("…")


async def test_wake_word_tasks_are_broadcast_to_open_windows(monkeypatch):
    from app.wake_listener import _ui_broadcaster

    received = []
    fake_server = SimpleNamespace(broadcast=lambda message: _record(received, message))
    fake_main = types.ModuleType("app.main")
    fake_main.ws_server = fake_server
    monkeypatch.setitem(sys.modules, "app.main", fake_main)

    send = _ui_broadcaster()
    send("status-message")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert received == ["status-message"]


async def _record(bucket, message):
    bucket.append(message)


def test_wake_listener_status_without_a_listener():
    from app.wake_listener import get_wake_listener_status

    assert get_wake_listener_status()["running"] is False


async def test_shutdown_endpoint_is_local_only(monkeypatch):
    import signal

    from app import main

    raised = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))
    outside = await main.app_shutdown(SimpleNamespace(client=SimpleNamespace(host="192.168.1.20")))
    assert outside.status_code == 403
    local = await main.app_shutdown(SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")))
    assert local == {"status": "shutting_down"}
    await asyncio.sleep(0.5)
    assert raised == [signal.SIGINT]


def test_dont_alone_counts_as_done_but_not_inside_a_sentence():
    from app.wake_listener import WakeWordListener

    listener = WakeWordListener(wake_word="hello", stop_word="done")
    assert listener._matches_stop_word("don't")
    assert not listener._matches_stop_word("i don't")
    assert not listener._matches_stop_word("please don't open it")
    assert not listener._matches_wake_word("don't")
    assert WakeWordListener(wake_word="hello", stop_word="finish")._matches_stop_word("don't") is False


# =============================================================================
# Vosk on real human recordings (Lingua Libre, CC BY-SA 4.0 — see ATTRIBUTION.md)
# =============================================================================

def _listener_with_vosk(monkeypatch):
    pytest.importorskip("vosk")
    from app.wake_listener import WakeWordListener

    model_path = Path(__file__).resolve().parents[1] / "data" / "vosk-model-small-en-us-0.15"
    if not model_path.exists():
        pytest.skip("Vosk model not installed (scripts/install_vosk_model.py)")
    monkeypatch.setattr(get_settings(), "wake_word_engine", "vosk")
    monkeypatch.setattr(get_settings(), "vosk_model_path", str(model_path))
    listener = WakeWordListener(wake_word="hello", stop_word="done")
    listener._load_vosk_if_configured()
    assert listener.engine_in_use == "vosk"
    return listener


def _heard(listener, wav_path: Path) -> str:
    import speech_recognition as sr

    with wave.open(str(wav_path)) as w:
        audio = sr.AudioData(w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth())
    return listener._recognize_vosk(audio)


def _recordings(prefix: str) -> list[Path]:
    return sorted(FIXTURES.glob(f"{prefix}_*.wav"))


def test_vosk_hears_hello_from_real_speakers(monkeypatch):
    files = _recordings("hello")
    if not files:
        pytest.skip("no recordings")
    listener = _listener_with_vosk(monkeypatch)
    missed = [f.name for f in files if not listener._matches_wake_word(_heard(listener, f))]
    assert not missed, f"wake word not detected in {missed}"


def test_vosk_hears_done_from_real_speakers(monkeypatch):
    files = _recordings("done")
    if not files:
        pytest.skip("no recordings")
    listener = _listener_with_vosk(monkeypatch)
    heard = {f.name: _heard(listener, f) for f in files}
    missed = {name: text for name, text in heard.items() if not listener._matches_stop_word(text)}
    assert not missed, f"stop word not detected (file: heard): {missed}; grammar={listener._vosk_grammar}"


def test_vosk_ignores_similar_sounding_words(monkeypatch):
    files = _recordings("neg")
    if not files:
        pytest.skip("no recordings")
    listener = _listener_with_vosk(monkeypatch)
    triggered = {
        f.name: text for f in files
        if (text := _heard(listener, f)) and (listener._matches_wake_word(text) or listener._matches_stop_word(text))
    }
    assert not triggered, f"false triggers: {triggered}"
