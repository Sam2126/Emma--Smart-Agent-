"""
Semantic experience memory (Level 3 learning).

Every finished task becomes one "experience" in a local ChromaDB store, and
before a new task, experiences are recalled by meaning instead of exact domain
or skill-type match. That is the difference from the SQL skill tables in
brain.py: a lesson learned on "open WhatsApp and message Bhawesh" is found
again for "send Priyal a hello on WhatsApp", and a popup lesson learned on one
shopping site is found for a different shopping site.

Two collections:
  experiences  the instruction (embedded) plus outcome, tools, lesson and
               user feedback as metadata
  lessons      the lesson text of the same task, embedded on its own

A new task is matched against both. The lessons collection is what lets
skills combine across tasks that are worded differently. Measured with the
local MiniLM model: "Add laptop to cart on new-site" scored 0.03 against the
instruction "Search headphones on Amazon", but 0.36 against a stored lesson
"closed the login popup, searched the product ... clicked Add to cart".

Embeddings come from ChromaDB's default all-MiniLM-L6-v2 model, run locally on
CPU through onnxruntime: no API call and no cost. The task and its lesson are
embedded in ONE batched call when stored, and a new instruction is embedded
once and reused for both searches.

Everything here degrades gracefully. If ChromaDB cannot start, recall returns
an empty list and writes are skipped, so a task never fails because of memory.
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_NAME = "experiences"
LESSONS_COLLECTION_NAME = "lessons"

# Metadata strings are capped so one very chatty run cannot bloat the store.
_MAX_LESSON_CHARS = 1500
_MAX_ERROR_CHARS = 500
_MAX_TOOLS = 15
_MAX_STEP_DETAILS = 12
_FEEDBACK_CLAMP = 5
_BACKFILL_LIMIT = 1000


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_BACKEND_ROOT / p).resolve()


def tool_sequence(trajectory: list[dict[str, Any]]) -> list[str]:
    """Ordered tool names from a trajectory, with consecutive repeats collapsed.

    "send_keys, send_keys, send_keys" becomes one "send_keys" so two runs that
    followed the same approach produce the same strategy signature even if
    one typed in more keystroke batches than the other.
    """
    seq: list[str] = []
    for step in trajectory or []:
        if not isinstance(step, dict):
            continue
        name = str(step.get("action_type") or "").strip()
        if name and (not seq or seq[-1] != name):
            seq.append(name)
    return seq[:_MAX_TOOLS]


def step_details(trajectory: list[dict[str, Any]]) -> list[str]:
    """The run's steps as short readable lines: what was called, and with what.

    Stored with the experience so a run the user confirmed can be shown to the
    planner as a flow to learn from, not just a list of tool names.
    """
    details: list[str] = []
    for step in trajectory or []:
        if not isinstance(step, dict):
            continue
        target = str(step.get("input_value") or "").strip()
        mark = "" if step.get("success") else " [FAILED]"
        details.append(f"{step.get('action_type')}({target[:70]}){mark}")
        if len(details) >= _MAX_STEP_DETAILS:
            break
    return details


def _lesson_document(lesson: str) -> str:
    return (lesson or "").strip()[:_MAX_LESSON_CHARS]


_TRANSIENT_INDEX_ERRORS = ("nothing found on disk", "hnsw segment reader")


def _query_with_retry(query_fn, attempts: int = 5, fallback=None, **kwargs: Any):
    """Run a ChromaDB query, retrying while a just-written index is still being saved.

    Found 2026-09-15: a query moments after a collection's first writes can
    fail with "Error creating hnsw segment reader: Nothing found on disk",
    because ChromaDB 1.x writes its vector index in the background. The same
    query succeeds a moment later, so only that error is retried. Under load
    the index can take longer than the retries; then `fallback` (a direct
    search over the stored vectors) answers instead of returning nothing.
    """
    for attempt in range(attempts):
        try:
            return query_fn(**kwargs)
        except Exception as e:
            transient = any(m in str(e).lower() for m in _TRANSIENT_INDEX_ERRORS)
            if not transient:
                raise
            if attempt == attempts - 1:
                if fallback is None:
                    raise
                logger.info("vector_index_not_ready_using_direct_search")
                return fallback(**kwargs)
            time.sleep(0.2 * (attempt + 1))


_DIRECT_SEARCH_LIMIT = 2000


def _direct_query(collection, embed, query_embeddings, n_results: int, include=None, **_: Any) -> dict[str, Any]:
    """The same answer as collection.query, computed without the vector index.

    ChromaDB keeps the stored vectors in the very index that is still being
    written, so reading them fails the same way. The documents and metadata live
    in its SQLite store, so they are read from there and embedded again in one
    batched call (local model, a personal memory of hundreds to a few thousand
    tasks), then ranked by cosine distance.
    """
    got = collection.get(include=["documents", "metadatas"], limit=_DIRECT_SEARCH_LIMIT)
    documents = got.get("documents") or []
    if not documents:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    query = [float(x) for x in query_embeddings[0]]
    query_norm = math.sqrt(sum(x * x for x in query)) or 1.0
    scored: list[tuple[float, int]] = []
    for i, vector in enumerate(embed([document or "" for document in documents])):
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(x * x for x in values)) or 1.0
        cosine = sum(a * b for a, b in zip(query, values)) / (query_norm * norm)
        scored.append((1.0 - cosine, i))
    scored.sort()
    top = scored[:n_results]
    documents = got.get("documents") or [None] * len(got["ids"])
    return {
        "ids": [[got["ids"][i] for _, i in top]],
        "documents": [[documents[i] for _, i in top]],
        "metadatas": [[got["metadatas"][i] for _, i in top]],
        "distances": [[distance for distance, _ in top]],
    }


@dataclass
class Experience:
    """One remembered task, as returned by recall."""

    task_id: str
    instruction: str
    domain: str
    scope: str
    success: bool
    tools: list[str]
    lesson: str
    error: str
    feedback: int
    user_disputed: bool
    created_at: str
    steps_detail: list[str] = field(default_factory=list)
    feedback_comment: str = ""
    similarity: float = 0.0
    score: float = 0.0
    matched_on: str = "task"  # "task": similar instruction; "lesson": relevant lesson
    user_confirmed: bool = False  # 👍 on a run the agent had marked as failed

    @property
    def strategy_signature(self) -> str:
        return " -> ".join(self.tools) if self.tools else "(no tools)"

    def age_text(self, now: datetime | None = None) -> str:
        """Human-friendly age such as "3 days ago", for explanations."""
        try:
            created = datetime.fromisoformat(self.created_at)
        except (TypeError, ValueError):
            return "earlier"
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        seconds = max(0.0, (now - created).total_seconds())
        if seconds < 90:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds // 60)} min ago"
        if seconds < 86400:
            hours = int(seconds // 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = int(seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"

    @property
    def effective_success(self) -> bool:
        """Success as the user sees it: a 👎 turns a "success" into a failure, a 👍 turns a "failure" into a success.

        Found 2026-09-15: a task that really worked was marked failed by an
        over-strict completion check; the user's 👍 had no way to correct it.
        """
        return (self.success and not self.user_disputed) or self.user_confirmed

    @property
    def verified(self) -> bool:
        """The user gave this run a 👍: its flow is worth learning from."""
        return self.feedback > 0 and self.effective_success


def summarize_strategies(experiences: list[Experience]) -> list[tuple[str, int, int]]:
    """Group recalled experiences by approach and count how often each worked.

    Returns (signature, successes, attempts), best success rate first. This is
    the "A/B" view: when two different approaches were tried on similar tasks,
    the planner sees which one actually worked more often.
    """
    buckets: dict[str, list[int]] = {}
    for exp in experiences:
        wins_total = buckets.setdefault(exp.strategy_signature, [0, 0])
        wins_total[1] += 1
        if exp.effective_success:
            wins_total[0] += 1
    ranked = [(sig, wins, total) for sig, (wins, total) in buckets.items()]
    ranked.sort(key=lambda item: (item[1] / item[2], item[2]), reverse=True)
    return ranked


def anti_skills(experiences: list[Experience], min_attempts: int = 2) -> list[tuple[str, int, int]]:
    """Approaches that consistently fail on similar tasks ("anti-skills").

    An approach qualifies when it was tried at least `min_attempts` times and
    worked in fewer than a third of them (a 👎 on a "success" counts as a
    failure). The strategy note tells the planner never to use these, and the
    explanation tells the user they were avoided.
    """
    return [
        (signature, wins, total)
        for signature, wins, total in summarize_strategies(experiences)
        if total >= min_attempts and wins * 3 < total and signature != "(no tools)"
    ]


def _is_incomplete_index(error: BaseException) -> bool:
    """True for the error ChromaDB gives when its index files are half written."""
    text = str(error).lower()
    return "nothing found on disk" in text or "hnsw segment reader" in text


class SemanticMemory:
    """Local ChromaDB-backed store of task experiences.

    All public async methods run the synchronous ChromaDB calls in a worker
    thread, so the event loop (WebSocket server, other tasks) never blocks on
    embedding or disk I/O.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        settings = get_settings()
        self._path = _resolve_path(path or settings.chroma_path)
        self._embedding_function = embedding_function
        self._ef = None
        self._lock = threading.RLock()
        self._collection = None
        self._lessons = None
        self._last_init_failure: float = 0.0
        # Feedback that arrived before the task's experience was written
        # (the write happens in the background right after the task ends, so
        # a fast 👍 click can beat it). Applied as soon as the write lands.
        self._pending_feedback: dict[str, tuple[int, str]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_collection(self):
        with self._lock:
            if self._collection is not None:
                return self._collection
            # Don't hammer a broken install on every call; retry once a minute.
            if self._last_init_failure and time.monotonic() - self._last_init_failure < 60:
                return None
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(self._path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                ef = self._embedding_function
                if ef is None:
                    from chromadb.utils import embedding_functions

                    ef = embedding_functions.DefaultEmbeddingFunction()
                lessons = client.get_or_create_collection(
                    name=LESSONS_COLLECTION_NAME,
                    embedding_function=ef,
                    metadata={"hnsw:space": "cosine"},
                )
                experiences = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=ef,
                    metadata={"hnsw:space": "cosine"},
                )
                self._ef, self._lessons, self._collection = ef, lessons, experiences
                logger.info(
                    "semantic_memory_ready",
                    path=str(self._path),
                    count=experiences.count(),
                    lessons=lessons.count(),
                )
                return self._collection
            except Exception as e:
                self._last_init_failure = time.monotonic()
                logger.warning("semantic_memory_unavailable", error=str(e)[:300])
                return None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """One batched embedding call for all `texts`."""
        return [[float(x) for x in vector] for vector in self._ef(texts)]

    @property
    def available(self) -> bool:
        return self._get_collection() is not None

    def warm(self) -> bool:  # noqa: D401 - see the docstring below
        """Load the embedding model and the index ahead of the first task,
        and index lessons of experiences stored before the lessons collection."""
        collection = self._get_collection()
        if collection is None:
            return False
        try:
            vector = self._embed(["warm up"])
            if collection.count() > 0:
                _query_with_retry(collection.query, query_embeddings=vector, n_results=1)
            self.backfill_lessons()
            return True
        except Exception as e:
            # The backend exits with os._exit(0), so ChromaDB is not always
            # given time to finish writing its index: measured here, the
            # experiences index came back missing link_lists.bin and reported
            # "Error creating hnsw segment reader: Nothing found on disk".
            # Asking again rebuilds the missing part from the stored vectors,
            # which is far better than starting the day with no memory.
            if _is_incomplete_index(e):
                logger.warning("semantic_memory_index_incomplete_rebuilding", error=str(e)[:160])
                try:
                    self._collection = None
                    collection = self._get_collection()
                    if collection is not None and collection.count() > 0:
                        _query_with_retry(
                            collection.query, query_embeddings=self._embed(["warm up"]), n_results=1
                        )
                    self.backfill_lessons()
                    logger.info("semantic_memory_index_rebuilt")
                    return True
                except Exception as retry_error:
                    logger.warning("semantic_memory_rebuild_failed", error=str(retry_error)[:200])
                    return False
            logger.warning("semantic_memory_warm_failed", error=str(e)[:200])
            return False

    def backfill_lessons(self) -> int:
        """Embed lessons that exist on experiences but not in the lessons collection."""
        collection = self._get_collection()
        if collection is None or self._lessons is None:
            return 0
        try:
            got = collection.get(include=["metadatas"], limit=_BACKFILL_LIMIT)
            with_lesson = {
                task_id: _lesson_document((meta or {}).get("lesson", ""))
                for task_id, meta in zip(got["ids"], got["metadatas"])
                if _lesson_document((meta or {}).get("lesson", ""))
            }
            if not with_lesson:
                return 0
            indexed = set(self._lessons.get(ids=list(with_lesson), include=["metadatas"])["ids"])
            todo = [task_id for task_id in with_lesson if task_id not in indexed]
            if not todo:
                return 0
            documents = [with_lesson[task_id] for task_id in todo]
            self._lessons.upsert(
                ids=todo,
                documents=documents,
                metadatas=[{"task_id": task_id} for task_id in todo],
                embeddings=self._embed(documents),
            )
            logger.info("lessons_backfilled", count=len(todo))
            return len(todo)
        except Exception as e:
            logger.warning("lessons_backfill_failed", error=str(e)[:200])
            return 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_sync(
        self,
        *,
        task_id: str,
        instruction: str,
        domain: str,
        scope: str,
        success: bool,
        trajectory: list[dict[str, Any]],
        lesson: str,
        error: str | None = None,
    ) -> bool:
        collection = self._get_collection()
        if collection is None or not instruction.strip():
            return False
        with self._lock:
            feedback, comment = self._pending_feedback.pop(task_id, (0, ""))
        metadata = {
            "task_id": task_id,
            "domain": domain or "general",
            "scope": scope or "browser",
            "success": bool(success),
            "tools": json.dumps(tool_sequence(trajectory)),
            "steps_detail": json.dumps(step_details(trajectory)),
            "lesson": (lesson or "")[:_MAX_LESSON_CHARS],
            "error": (error or "")[:_MAX_ERROR_CHARS],
            "feedback": max(-_FEEDBACK_CLAMP, min(_FEEDBACK_CLAMP, feedback)),
            "user_disputed": bool(feedback < 0 and success),
            "user_confirmed": bool(feedback > 0 and not success),
            "feedback_comment": comment[:300],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        lesson_doc = _lesson_document(lesson)
        try:
            # The task and its lesson are embedded together in one call.
            vectors = self._embed([instruction.strip()] + ([lesson_doc] if lesson_doc else []))
            collection.upsert(
                ids=[task_id], documents=[instruction.strip()], metadatas=[metadata], embeddings=[vectors[0]]
            )
            if lesson_doc and self._lessons is not None:
                self._lessons.upsert(
                    ids=[task_id], documents=[lesson_doc], metadatas=[{"task_id": task_id}], embeddings=[vectors[1]]
                )
            logger.info("experience_stored", task_id=task_id, success=success, tools=metadata["tools"], lesson=bool(lesson_doc))
            return True
        except Exception as e:
            logger.warning("experience_store_failed", task_id=task_id, error=str(e)[:200])
            return False

    async def upsert(self, **kwargs: Any) -> bool:
        return await asyncio.to_thread(self.upsert_sync, **kwargs)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall_sync(
        self,
        instruction: str,
        k: int | None = None,
        min_similarity: float | None = None,
        exclude_task_id: str | None = None,
    ) -> list[Experience]:
        collection = self._get_collection()
        if collection is None or not instruction.strip():
            return []
        settings = get_settings()
        k = k or settings.semantic_recall_k
        floor = settings.semantic_min_similarity if min_similarity is None else min_similarity
        lesson_floor = settings.semantic_lesson_min_similarity if min_similarity is None else min_similarity

        best: dict[str, Experience] = {}
        try:
            total = collection.count()
            if total == 0:
                return []
            query = self._embed([instruction.strip()])  # embedded once, used for both searches
            n_results = min(total, max(k * 3, k + 1))
            result = _query_with_retry(
                collection.query,
                fallback=functools.partial(_direct_query, collection, self._embed),
                query_embeddings=query,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            ):
                meta = meta or {}
                task_id = str(meta.get("task_id", ""))
                if exclude_task_id and task_id == exclude_task_id:
                    continue
                similarity = 1.0 - float(distance)
                if similarity >= floor:
                    best[task_id] = _experience_from(doc, meta, similarity)

            lesson_hits = self._lesson_hits(query, n_results, lesson_floor, exclude_task_id, best)
            if lesson_hits:
                got = collection.get(ids=list(lesson_hits), include=["documents", "metadatas"])
                for task_id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                    exp = _experience_from(doc, meta or {}, lesson_hits[task_id])
                    exp.matched_on = "lesson"
                    best[task_id] = exp
        except Exception as e:
            logger.warning("experience_recall_failed", error=str(e)[:200])
            return []

        experiences = list(best.values())
        for exp in experiences:
            # Rank by closeness first, then nudge by what actually happened:
            # user feedback matters most, a verified success a little.
            exp.score = (
                exp.similarity
                + 0.08 * max(-3, min(3, exp.feedback))
                + (0.05 if exp.effective_success else 0.0)
                # A run the user confirmed is the best evidence there is.
                + (0.12 if exp.verified else 0.0)
            )
        experiences.sort(key=lambda e: e.score, reverse=True)
        return experiences[:k]

    def _lesson_hits(
        self,
        query: list[list[float]],
        n_results: int,
        floor: float,
        exclude_task_id: str | None,
        already: dict[str, Experience],
    ) -> dict[str, float]:
        """task_id -> lesson similarity, for lessons closer than the task match."""
        if self._lessons is None:
            return {}
        try:
            count = self._lessons.count()
            if count == 0:
                return {}
            result = _query_with_retry(
                self._lessons.query,
                fallback=functools.partial(_direct_query, self._lessons, self._embed),
                query_embeddings=query, n_results=min(count, n_results), include=["metadatas", "distances"],
            )
        except Exception as e:
            # Lessons add matches; losing them must not lose the task matches.
            logger.warning("lesson_recall_failed", error=str(e)[:200])
            return {}
        hits: dict[str, float] = {}
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            task_id = str((meta or {}).get("task_id", ""))
            similarity = 1.0 - float(distance)
            if not task_id or task_id == exclude_task_id or similarity < floor:
                continue
            if task_id in already and already[task_id].similarity >= similarity:
                continue
            hits[task_id] = similarity
        return hits

    async def recall(self, instruction: str, **kwargs: Any) -> list[Experience]:
        return await asyncio.to_thread(self.recall_sync, instruction, **kwargs)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def record_feedback_sync(self, task_id: str, rating: int, comment: str = "") -> Experience | None:
        """Apply a 👍 (+1) or 👎 (-1) to a stored experience.

        Returns the updated experience, or None when the task isn't stored
        yet, in which case the rating is queued and applied on write.
        """
        rating = 1 if rating > 0 else -1
        collection = self._get_collection()
        if collection is None:
            return None
        try:
            got = collection.get(ids=[task_id], include=["documents", "metadatas"])
        except Exception as e:
            logger.warning("feedback_lookup_failed", task_id=task_id, error=str(e)[:200])
            return None

        if not got["ids"]:
            with self._lock:
                prev, prev_comment = self._pending_feedback.get(task_id, (0, ""))
                self._pending_feedback[task_id] = (prev + rating, comment or prev_comment)
            logger.info("feedback_queued_until_experience_stored", task_id=task_id, rating=rating)
            return None

        meta = dict(got["metadatas"][0] or {})
        meta["feedback"] = max(-_FEEDBACK_CLAMP, min(_FEEDBACK_CLAMP, int(meta.get("feedback", 0)) + rating))
        meta["user_disputed"] = bool(meta.get("success")) and meta["feedback"] < 0
        meta["user_confirmed"] = (not bool(meta.get("success"))) and meta["feedback"] > 0
        if comment:
            meta["feedback_comment"] = comment[:300]
        try:
            collection.update(ids=[task_id], metadatas=[meta])
        except Exception as e:
            logger.warning("feedback_update_failed", task_id=task_id, error=str(e)[:200])
            return None
        logger.info("feedback_recorded", task_id=task_id, rating=rating, total=meta["feedback"])
        return _experience_from(got["documents"][0], meta, similarity=1.0)

    async def record_feedback(self, task_id: str, rating: int, comment: str = "") -> Experience | None:
        return await asyncio.to_thread(self.record_feedback_sync, task_id, rating, comment)

    def count(self) -> int:
        collection = self._get_collection()
        try:
            return collection.count() if collection is not None else 0
        except Exception:
            return 0


def _experience_from(document: str, meta: dict[str, Any], similarity: float) -> Experience:
    try:
        tools = json.loads(meta.get("tools") or "[]")
    except (TypeError, ValueError):
        tools = []
    try:
        details = json.loads(meta.get("steps_detail") or "[]")
    except (TypeError, ValueError):
        details = []
    return Experience(
        task_id=str(meta.get("task_id", "")),
        instruction=document or "",
        domain=str(meta.get("domain", "")),
        scope=str(meta.get("scope", "")),
        success=bool(meta.get("success", False)),
        tools=[str(t) for t in tools],
        lesson=str(meta.get("lesson", "")),
        error=str(meta.get("error", "")),
        feedback=int(meta.get("feedback", 0) or 0),
        user_disputed=bool(meta.get("user_disputed", False)),
        user_confirmed=bool(meta.get("user_confirmed", False)),
        created_at=str(meta.get("created_at", "")),
        steps_detail=[str(step) for step in details],
        feedback_comment=str(meta.get("feedback_comment", "")),
        similarity=similarity,
    )


_singleton: SemanticMemory | None = None
_singleton_lock = threading.Lock()


def get_semantic_memory() -> SemanticMemory:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SemanticMemory()
        return _singleton
