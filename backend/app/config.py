"""
Application configuration loaded from environment variables.

Uses Pydantic Settings to validate and type-check all config values.
Copy .env.example to .env and fill in your values before running.
"""

from pathlib import Path
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All application settings, loaded from .env file and environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM API (Groq / OpenAI) ---
    groq_api_key: str = ""
    # Multi-key rotation pool: comma-separated list of Groq API keys.
    # Keys are cycled in round-robin; on 429, the rate-limited key is put in
    # cooldown and the next available key is used immediately (no wasted sleep).
    groq_api_keys: str = ""  # e.g. "gsk_key1,gsk_key2,gsk_key3,gsk_key4"
    openai_api_key: str = ""

    # High-quality reasoning model for planning & verification
    groq_planning_model: str = "openai/gpt-oss-120b"

    # Fast action model for execution steps
    groq_action_model: str = "openai/gpt-oss-120b"

    # --- Browser / Playwright ---
    cdp_endpoint: str = "http://127.0.0.1:9222"
    # The agent's Chrome always runs on its own profile
    # (~/.self_improving_agent/chrome_profile), never on a copy of yours; see
    # app/browser/controller.py. Older .env files may still set
    # CHROME_PROFILE_DIR or CHROME_USER_DATA_DIR: both are ignored.

    # Vision model (Groq) used by see_window for desktop app UI understanding.
    # NOTE: the previous default, meta-llama/llama-4-scout-17b-16e-instruct, has
    # been retired by Groq and now returns HTTP 404 "model_not_found" for every
    # request. That made see_window fail silently on EVERY call — it always
    # returned its "Vision unavailable, fall back to keyboard shortcuts" message,
    # so the agent was permanently blind inside desktop apps and could only
    # guess blindly at shortcuts like Ctrl+F. qwen/qwen3.8-27b is vision-capable
    # and verified working against this account's key pool.
    groq_vision_model: str = "qwen/qwen3.8-27b"

    # --- Local computer access (files, apps, shell) ---
    local_tools_enabled: bool = True
    shell_commands_enabled: bool = True

    # --- Email (search_emails, read_email, create_email_draft, send_email) ---
    # The user's own mailbox over IMAP/SMTP (app/tools/mail.py). Gmail needs an
    # App Password, not the Google password; setup_email.bat writes both lines.
    # The servers are chosen from the address (Gmail, Yahoo, iCloud, Zoho,
    # Outlook; a college or company domain by its mail servers); set the hosts
    # below only for another provider.
    email_address: str = ""
    email_app_password: str = ""
    email_display_name: str = ""
    email_imap_host: str = ""
    email_imap_port: int = 993
    email_smtp_host: str = ""
    email_smtp_port: int = 0                    # 0 -> the provider's port (465, or 587 with STARTTLS)

    # --- WebSocket (Extension <-> Backend) ---
    ws_host: str = "localhost"
    ws_port: int = 8765

    # --- FastAPI ---
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    # --- Database ---
    db_url: str = "sqlite+aiosqlite:///./data/agent.db"

    # --- Agent Behavior ---
    max_step_retries: int = 3
    action_delay_min: float = 1.0
    action_delay_max: float = 3.0
    max_task_steps: int = 30
    reflexion_enabled: bool = True

    # --- Level 3 learning: semantic memory + LLM reflection ---
    # Every finished task is stored in a local ChromaDB collection, embedded
    # with all-MiniLM-L6-v2 (runs on CPU through onnxruntime, no API calls).
    # Before a task, similar past tasks are recalled by meaning rather than
    # exact domain/skill-type match, so a lesson learned on one site or app
    # can be reused on another.
    semantic_memory_enabled: bool = True
    chroma_path: str = "./data/chroma"          # relative to the backend folder
    semantic_recall_k: int = 5
    # Cosine similarity floor. Verified on this machine with MiniLM: two
    # WhatsApp-message tasks scored 0.67, a WhatsApp task vs an Amazon search
    # scored 0.01, so 0.35 keeps related tasks and drops unrelated ones.
    semantic_min_similarity: float = 0.35
    # Floor for matches found through a past task's LESSON rather than its
    # wording. Measured: "Add laptop to cart on new-site" scored 0.36 against a
    # stored add-to-cart lesson but 0.03 against the instruction it came from.
    semantic_lesson_min_similarity: float = 0.3
    llm_reflection_enabled: bool = True
    reflection_model: str = ""                  # empty -> groq_planning_model
    reflection_timeout_seconds: float = 12.0
    reflection_cache_ttl_seconds: int = 3600
    # Retry a failed task once with a changed strategy, but only when the
    # failed attempt performed no step that would be harmful to repeat
    # (typing, clicking, sending, writing files, running commands).
    auto_retry_on_failure: bool = True

    # --- Free fallback LLM provider (Google AI Studio) ---
    # Text calls fall back to Gemini when every Groq key is rate-limited or
    # Groq is down, and Gemini is the first vision provider (vision_provider).
    # Leave the key empty for Groq-only behaviour. Both settings below take a
    # comma-separated list tried in order; each Gemini model has its own free
    # daily quota, so the second one extends the free budget. Verified
    # 2026-09-15 with this project's key: gemini-2.0-flash (named in the tech
    # audit) was removed by Google ("no longer available"); gemini-3.6-flash
    # and gemini-2.5-flash-lite both answered through LiteLLM.
    gemini_api_key: str = ""
    llm_fallback_model: str = "gemini/gemini-3.6-flash,gemini/gemini-2.5-flash-lite"
    gemini_vision_model: str = "gemini/gemini-3.6-flash,gemini/gemini-2.5-flash-lite"
    # How long a Groq call may wait for a rate-limited key to cool down before
    # the request goes to Gemini instead.
    groq_max_cooldown_wait_seconds: float = 3.0
    # Groq's free tier counts a request's input AND its max_tokens against the
    # model's tokens-per-minute limit (8,000 for gpt-oss-120b), and refuses a
    # single request above it. The actor asks for fewer output tokens when a
    # large prompt would otherwise pass it. 0 turns this off.
    groq_request_token_limit: int = 8000
    # The most text one request may carry, counted in characters across every
    # message. Groq's free tier refuses a request above its per-minute token
    # limit, and five page readings in a row reached about thirty thousand
    # characters. Roughly three and a half characters to a token, leaving room
    # for the tool definitions and the reply.
    llm_request_char_budget: int = 14000
    # "auto" -> Gemini first when GEMINI_API_KEY is set, else Groq
    # "gemini" / "groq" -> that provider first, the other as fallback
    vision_provider: str = "auto"
    browser_vision_enabled: bool = True        # see_page tool + perceive_page augmentation
    vision_verification_enabled: bool = True   # screenshot judge in task verification

    # --- Emma's voice (text to speech, app/tts.py) ---
    # Three voices tried in this order; one that fails or is rate limited sits
    # out for a short while and the next answers, so Emma is never silent.
    # The Groq model needs its terms accepted once at console.groq.com.
    tts_enabled: bool = True
    tts_order: str = "groq,gemini,windows"
    tts_groq_model: str = "canopylabs/orpheus-v1-english"
    tts_groq_voice: str = "tara"
    tts_gemini_model: str = "gemini-3.1-flash-tts-preview"
    tts_gemini_voice: str = "Kore"
    # The offline Windows voice: also the ONLY voice used for private text
    # (passwords, one-time codes, the contents of mail and documents).
    tts_windows_voice: str = "Microsoft Zira Desktop"
    tts_windows_rate: int = 1                  # -10 (slow) .. 10 (fast)
    tts_timeout_seconds: float = 20.0
    tts_max_chars: int = 600                   # one spoken reply, not a whole report
    tts_cache_enabled: bool = True
    # Step-by-step chatter is dropped when this many lines are already waiting:
    # an update that arrives after the step it describes is noise.
    tts_progress_queue_limit: int = 3

    # Names Emma should expect to hear: contacts, shops, anything she gets
    # wrong. Comma separated, added to the vocabulary Whisper is given and to
    # the repair pass that turns "Mentra" back into "Myntra".
    speech_extra_names: str = ""

    # --- Agent engine (LangGraph) ---
    # Two-tier actor: a fast model handles routine steps while they keep
    # succeeding; the action model handles the first step, any step after a
    # failure, and every 5th turn.
    two_tier_actor: bool = True
    groq_fast_action_model: str = "llama-3.1-8b-instant"
    # Prefer Groq's Llama models when the account offers them; otherwise the
    # configured models are used. Validated against the account's real model
    # list at startup (utils/llm.refresh_model_availability).
    prefer_llama_models: bool = True
    # Optional paid provider: LLM_PROVIDER=openai with OPENAI_API_KEY routes the
    # planner, actor, reflection and judge to OPENAI_MODEL.
    llm_provider: str = "groq"
    openai_model: str = "gpt-4o-mini"
    actor_max_iterations_local: int = 30
    actor_max_iterations_browser: int = 20
    task_timeout_seconds: int = 180
    human_confirmation_timeout_seconds: int = 120

    # --- Wake Word Listener (Hands-Free Voice Activation) ---
    wake_word_enabled: bool = True          # Start listener on backend startup
    wake_word: str = "emma"                 # Word that activates recording ("hello" also works)
    wake_word_stop: str = "done"            # Word that ends recording
    wake_word_listen_timeout: int = 30      # Max seconds to record after wake word
    # Keyword spotting engine for the wake/stop words:
    #   "vosk"   -> offline Vosk recognizer restricted to the wake/stop words
    #               (default; no internet, no rate limits). Needs the model
    #               folder below (scripts/install_vosk_model.py). Falls back to
    #               Google automatically if the package or model is missing.
    #   "google" -> Google's free online recognizer
    # The spoken instruction itself is always transcribed by Groq Whisper.
    wake_word_engine: str = "vosk"
    vosk_model_path: str = "./data/vosk-model-small-en-us-0.15"

    # --- Logging ---
    log_level: str = "INFO"

    @field_validator("db_url")
    @classmethod
    def _anchor_relative_sqlite_path(cls, value: str) -> str:
        """A relative SQLite path means the backend folder, wherever the process started.

        chroma_path and vosk_model_path were already read that way; the database
        was not. Found 2026-09-16: a stale data/agent.db sat in the project root,
        left by a run started from there, which had used that empty, separate
        database — none of the agent's memory, and nothing said so.
        """
        for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
            if value.startswith(prefix):
                path = value[len(prefix):]
                if path and not path.startswith((":memory:", "/")) and not Path(path).is_absolute():
                    return prefix + (_BACKEND_DIR / path).resolve().as_posix()
                break
        return value


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings (singleton pattern)."""
    return Settings()
