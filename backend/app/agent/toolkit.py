"""
Tool registry and execution for the agent engine.

  build_toolset(scope)     which tools a task may use
  execute_tool_call(...)   validate arguments, run the tool in a worker thread,
                           and return a structured trajectory step
  describe_action(...)     human-readable progress line for streaming
  retry_block_reason(...)  whether an automatic retry is safe
  local_completion_shortfall(...)  catch "opened the app and stopped" runs

Scope separation is strict: extension tasks (scope "browser") get browser tools
only. The Local Agent UI and the wake word (scope "local") get local, desktop
and browser tools, because mixed tasks like "download X then open it" must work
in one run.

Tools run through asyncio.to_thread. That keeps the event loop free while a
tool waits on PowerShell or on a browser action scheduled back onto the loop
(see tools/runtime.py), and it carries context variables such as the human
confirmation channel into the tool's thread.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import structlog

from app.config import get_settings
from app.tools.base import BaseTool
from app.tools.runtime import DEFAULT_TOOL_TIMEOUT_SECONDS

logger = structlog.get_logger(__name__)

LOCAL_SCOPE = "local"
BROWSER_SCOPE = "browser"

# Text fragments that mean a tool result describes a failure even though nothing
# raised: local tools report "not found" / "failed" as plain sentences. Only the
# start of a result is scanned, because failures are reported up front, while
# long successful results (a page text sample, a file's contents) often contain
# words like "error" further down and were misread as failures before.
SOFT_FAILURE_MARKERS = (
    "not found", "no files found", "no files matching", "failed", "error",
    "could not", "unable", "permission denied", "no such", "missing",
    "timed out", "does not exist",
    # Found 2026-09-15: a report saying "the required tool ... is not available
    # ... I cannot attach the file ... the task cannot be completed" was shown
    # as COMPLETED, because none of the words above appeared in it.
    "cannot", "can't", "can not", "not available", "not possible", "not supported",
    "wasn't able", "was not able", "were not able", "did not work", "didn't work",
    # Found 2026-09-17: "The email system is not configured on this computer ...
    # please run setup_email.bat" was shown as COMPLETED.
    "not set up", "isn't set up", "not been set up", "not configured", "isn't configured", "not been configured",
    # run_command's own refusals counted as successful steps (found 2026-09-17).
    "not run:", "blocked for safety", "commands are disabled", "tools are disabled",
)
_FAILURE_SCAN_CHARS = 240

# Tools whose effects are harmless to repeat. Anything else (typing, clicking,
# sending, writing, running commands) blocks an automatic retry once it has
# succeeded, because a retry could send a message twice or run a command twice.
SAFE_TO_REPEAT_TOOLS = {
    "list_recent_files", "find_files", "search_in_files", "read_file", "list_folder", "list_windows",
    "see_window", "focus_window", "open_app", "navigate_browser", "perceive_page",
    "read_page_as_markdown", "read_window_as_markdown",
    "see_page", "extract_page_data", "recall_domain_memory", "scroll_page", "wait_for_login",
    "create_folder",  # making a folder that already exists changes nothing
    "search_emails", "read_email", "find_email_address",  # reading mail changes nothing (read-only)
}

# Words that mean the user wanted something done INSIDE an app, not merely that
# the app be launched.
_IN_APP_INTENT_WORDS = (
    "search", "find", "send", "message", "msg", "text ", "type", "write",
    "reply", "call", "play", "click", "open the chat", "say ",
)
# Words that mean the user asked for something to be DONE on the computer, so a
# run that called no tool at all cannot be a success. Found 2026-09-16: a stray
# microphone recording became a task that ran nothing, reported success, and was
# stored as a good experience.
_ACTION_WORDS = (
    "open", "create", "make", "write", "save", "delete", "remove", "move", "copy", "rename",
    "search", "find", "send", "play", "download", "upload", "attach", "install", "run",
    "set ", "change", "add", "print", "start", "close", "type",
)
# Tools that actually do something inside a running app or web page. Looking at
# a window or page (see_window, see_page) or focusing it does not count. Found
# 2026-09-15: "open gemini and search ..." typed and submitted the question in
# the browser, yet was marked incomplete because only send_keys / click_window
# counted as in-app work.
_IN_APP_TOOLS = {
    "send_keys", "click_window",
    "type_into_element", "click_element", "press_key", "select_dropdown_option", "upload_file",
    "download_file", "add_to_cart",
    # Email work is done by these tools themselves, not inside an app window.
    "search_emails", "read_email", "find_email_address", "create_email_draft", "send_email",
}

# Tools whose successful output DESCRIBES something — a page, a window, a file
# or folder. That text naturally contains words like "error": a see_page answer
# "There is no popup, modal, cookie banner, login or sign-in wall, CAPTCHA, or
# error" was recorded as a failed step (2026-09-15). For these tools only their
# own failure messages, at the start of the output, count.
_DESCRIPTIVE_TOOLS = {
    "see_page", "perceive_page", "extract_page_data", "see_window", "list_windows",
    "read_file", "list_folder", "list_recent_files", "search_in_files", "clipboard",
}
_DESCRIPTIVE_FAILURE_PREFIXES = (
    "error", "failed", "could not", "unable", "vision is unavailable", "page vision is disabled",
    "screenshot failed", "file not found", "folder not found", "no files found", "no visible window",
    "local computer tools are disabled", "invalid arguments", "unknown tool",
    "refused",  # a Chrome window named to a desktop tool (tools/desktop.py)
)

# Tools whose one success message is known, and which repeat what the user gave
# them — a path, a file name, the text typed into an app. For these, the output
# is a success exactly when it starts with that message. Found 2026-09-16:
#   * send_keys reports "Sent to 'WhatsApp' (...): <the message>", so a message
#     such as "sorry, I can't come" read as a failure ("can't"). The send was not
#     recorded by the duplicate guard, and the automatic retry was allowed — the
#     same message could go to a real person twice.
#   * the opposite: "Found a window ... but Windows refused ..., so NOTHING was
#     sent", "No visible window ... nothing was clicked" and "Refused: ... will
#     not be deleted" contain none of the failure words, so they counted as
#     successful steps, and a send that never happened was recorded as sent.
_SUCCESS_PREFIXES = {
    "send_keys": ("sent to ",),
    "click_window": ("clicked (",),
    "focus_window": ("focused ",),
    "open_app": ("launched: ", "started '"),
    "open_file_or_folder": ("opened: ",),
    "write_file": ("wrote ", "appended to "),
    "create_folder": ("folder ready: ",),
    "move_path": ("moved ",),
    "copy_path": ("copied ",),
    "delete_path": ("moved to the recycle bin: ",),
    "find_files": ("found ",),
    "install_app": ("installed: ", "already installed: "),
    "download_file": ("downloaded: ",),
    "add_to_cart": ("added to cart: ",),
    # Email text can say anything ("I can't make it"), so only these openings count.
    "search_emails": ("found ",),
    "read_email": ("opened email ",),
    "find_email_address": ("found ",),
    "create_email_draft": ("draft saved: ",),
    "send_email": ("email sent: ",),
}

# "No errors", "without any problems", "no popup ... or error": mentions of a
# problem that say it did NOT happen.
_NEGATED_PROBLEM = re.compile(
    r"\b(?:no|not any|without|zero|never|nor|nothing)\b[^.;:\n]{0,80}?"
    r"\b(?:errors?|failures?|failed|problems?|issues?|missing)\b"
)

_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def strip_think(text: str | None) -> str:
    return _THINK.sub("", text or "").strip()


# =============================================================================
# Registry
# =============================================================================

def browser_tools() -> list[BaseTool]:
    from app.tools.browser import (
        AddToCartTool,
        ClickElementTool,
        ClickPositionTool,
        DownloadFileTool,
        ExtractPageDataTool,
        NavigateBrowserTool,
        PageMarkdownTool,
        PerceivePageTool,
        PressKeyTool,
        RecallDomainMemoryTool,
        ScrollPageTool,
        SeePageTool,
        SelectOptionTool,
        TypeElementTool,
        UploadFileTool,
        WaitForLoginTool,
    )

    return [
        NavigateBrowserTool(),
        PageMarkdownTool(),
        PerceivePageTool(),
        SeePageTool(),
        TypeElementTool(),
        ClickElementTool(),
        ClickPositionTool(),
        UploadFileTool(),
        DownloadFileTool(),
        AddToCartTool(),
        SelectOptionTool(),
        PressKeyTool(),
        ScrollPageTool(),
        ExtractPageDataTool(),
        RecallDomainMemoryTool(),
        WaitForLoginTool(),
    ]


def local_tools() -> list[BaseTool]:
    from app.tools.local import ALL_LOCAL_TOOLS

    return [tool_cls() for tool_cls in ALL_LOCAL_TOOLS]


def build_toolset(scope: str) -> dict[str, BaseTool]:
    tools = local_tools() + browser_tools() if scope == LOCAL_SCOPE else browser_tools()
    return {tool.name: tool for tool in tools}


def tool_schemas(toolset: dict[str, BaseTool]) -> list[dict[str, Any]]:
    return [tool.to_openai_tool() for tool in toolset.values()]


# Local tasks send the browser tools only when the request looks like it needs
# the web. Their schemas are ~6,000 characters (~1,500 tokens) resent on every
# model call; on Groq's free tier that pushed a WhatsApp task into
# tokens-per-minute limits on all four keys (2026-09-15). The model can call
# `use_browser` to get them for any web task this check misses, and a browser
# tool called by name still runs regardless.
USE_BROWSER_TOOL = "use_browser"
BROWSER_TOOL_NAMES = frozenset({
    "navigate_browser", "perceive_page", "read_page_as_markdown", "see_page", "type_into_element",
    "click_element", "click_at_position", "select_dropdown_option", "press_key", "scroll_page", "extract_page_data",
    "recall_domain_memory", "wait_for_login", "upload_file", "download_file", "add_to_cart",
})
_WEB_HINTS = re.compile(
    r"https?://|www\.|\.(?:com|in|org|net|io|dev|ai|co)\b|"
    r"\b(?:web|website|site|browser|chrome|online|internet|google|youtube|amazon|flipkart|myntra|cart|shop|shopping|"
    r"url|link|webpage|login|log in|sign in|"
    r"gemini|chatgpt|claude|perplexity|instagram|facebook|linkedin|github|reddit|wikipedia|twitter)\b",
    re.IGNORECASE,
)
# An email address ("priya@company.in") is not a website, and Gmail is handled
# by the email tools. Found 2026-09-17: an email task with such an address
# loaded all 14 browser tools as well, and the request (8,573 tokens) was over
# Groq's free limit of 8,000 tokens a minute, so it could never be served.
_EMAIL_ADDRESS = re.compile(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+")

# The email tools and their rules (~2,000 tokens) are sent only for email tasks;
# any other task can switch them on with `use_email`.
USE_EMAIL_TOOL = "use_email"
EMAIL_TOOL_NAMES = frozenset({"search_emails", "read_email", "find_email_address", "create_email_draft", "send_email"})
_EMAIL_HINTS = re.compile(
    r"@[\w-]+\.|\b(?:e-?mails?|mail(?:s|ed|ing|box)?|gmail|outlook|inbox|drafts?|cc|bcc|compose|"
    r"repl(?:y|ies)|forward|unread|attachments?)\b",
    re.IGNORECASE,
)


def instruction_needs_web(instruction: str) -> bool:
    return bool(_WEB_HINTS.search(_EMAIL_ADDRESS.sub(" ", instruction or "")))


def instruction_needs_email(instruction: str) -> bool:
    return bool(_EMAIL_HINTS.search(instruction or ""))


def use_browser_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": USE_BROWSER_TOOL,
            "description": (
                "Enable the browser tools (navigate_browser, perceive_page, see_page, type_into_element, "
                "click_element, click_at_position, press_key, upload_file, download_file, add_to_cart, ...) "
                "for this task. Call it "
                "only when the task needs a website, including anything in Chrome."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    }


def use_email_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": USE_EMAIL_TOOL,
            "description": (
                "Enable the email tools (search_emails, read_email, find_email_address, create_email_draft, "
                "send_email) and their rules. Call it when the task involves reading, writing or sending email."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    }


def initial_tool_schemas(
    toolset: dict[str, BaseTool], scope: str, instruction: str
) -> tuple[list[dict[str, Any]], bool]:
    """Tool schemas for the first model call, and whether browser tools were held back."""
    if scope != LOCAL_SCOPE:
        return tool_schemas(toolset), False
    held: set[str] = set()
    browser_deferred = not instruction_needs_web(instruction) and bool(BROWSER_TOOL_NAMES & set(toolset))
    if browser_deferred:
        held |= BROWSER_TOOL_NAMES
    email_deferred = not instruction_needs_email(instruction) and bool(EMAIL_TOOL_NAMES & set(toolset))
    if email_deferred:
        held |= EMAIL_TOOL_NAMES
    schemas = tool_schemas({name: tool for name, tool in toolset.items() if name not in held})
    if browser_deferred:
        schemas.append(use_browser_schema())
    if email_deferred:
        schemas.append(use_email_schema())
    return schemas, browser_deferred


def enable_tool_group(current: list[dict[str, Any]], toolset: dict[str, BaseTool], names: frozenset[str],
                      switch: str) -> tuple[list[dict[str, Any]], list[str]]:
    """`current` with the tools in `names` added and the `switch` tool removed; also the names added."""
    present = {schema["function"]["name"] for schema in current}
    added = {name: tool for name, tool in toolset.items() if name in names and name not in present}
    kept = [schema for schema in current if schema["function"]["name"] != switch]
    return kept + tool_schemas(added), sorted(added)


# =============================================================================
# Execution
# =============================================================================

def summarize_tool_args(tool_args: Any) -> str:
    """Best-effort human-readable summary of a tool call's arguments.

    Used as the learned "target" of a step: a CSS selector for browser tools,
    or a file path / app name / command / query / typed text for local tools.
    """
    if isinstance(tool_args, dict):
        for key in ("selector", "path", "name", "window_hint", "window", "query", "pattern",
                    "command", "url", "text", "keys", "value", "key",
                    "to", "draft_id", "email_id", "from_address", "subject", "folder"):
            value = tool_args.get(key)
            if value:
                return str(value)[:200]
        try:
            return json.dumps(tool_args)[:200]
        except (TypeError, ValueError):
            return str(tool_args)[:200]
    return str(tool_args)[:200]


def output_indicates_failure(output: str, tool_name: str | None = None) -> bool:
    head = (output or "")[:_FAILURE_SCAN_CHARS].lower()
    if head.startswith(("error executing tool", "invalid arguments", "unknown tool")):
        return True
    if tool_name in _SUCCESS_PREFIXES:
        return not head.lstrip().startswith(_SUCCESS_PREFIXES[tool_name])
    if tool_name in _DESCRIPTIVE_TOOLS:
        return head.lstrip().startswith(_DESCRIPTIVE_FAILURE_PREFIXES)
    return any(marker in head for marker in SOFT_FAILURE_MARKERS)


# Text the agent itself sent or typed, by tool: a report quoting it ("sent
# 'Sorry, I can't come'") describes the message, not a failure.
_WRITTEN_TEXT_ARGS = {
    "send_keys": ("text",),
    "type_into_element": ("text",),
    "write_file": ("content",),
    "send_email": ("subject", "body"),
    "create_email_draft": ("subject", "body"),
}


def _plain(text: str) -> str:
    """Lower case, straight quotes, single spaces; line breaks kept (they end a negation)."""
    text = (text or "").lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return "\n".join(" ".join(line.split()) for line in text.splitlines())


def _written_fragments(trajectory: list[dict[str, Any]] | None) -> list[str]:
    """Sentences and lines the run successfully typed or sent, longest first."""
    fragments: set[str] = set()
    for step in trajectory or []:
        if not isinstance(step, dict) or not step.get("success"):
            continue
        args = step.get("args") or {}
        for key in _WRITTEN_TEXT_ARGS.get(str(step.get("action_type") or ""), ()):
            value = args.get(key) if isinstance(args, dict) else None
            if not isinstance(value, str):
                continue
            for piece in re.split(r"(?<=[.!?])\s+|\n+", value.replace("\\n", "\n")):
                piece = _plain(piece)
                if len(piece) >= 6:
                    fragments.add(piece)
    return sorted(fragments, key=len, reverse=True)


def report_indicates_failure(report: str, trajectory: list[dict[str, Any]] | None = None) -> bool:
    """Whether the actor's final report describes a failure (negated mentions ignored).

    Found 2026-09-17: "Sent the email to Rakesh saying 'Sorry, I can't come
    tomorrow'" counted as a failed task because of the message's own "can't".
    Text the run itself sent or typed (from `trajectory`) is left out of the
    scan; anything else in the report, quoted tool errors included, still counts.
    """
    text = _plain(report)
    for fragment in _written_fragments(trajectory):
        text = text.replace(fragment, " ")
    text = _NEGATED_PROBLEM.sub(" ", text)
    return any(marker in text for marker in SOFT_FAILURE_MARKERS)


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Parse model-supplied tool arguments. Returns (args, error)."""
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    text = str(raw).strip()
    if not text:
        return {}, None
    try:
        data = json.loads(text)
        if isinstance(data, str):  # double-encoded JSON
            data = json.loads(data)
    except ValueError as e:
        return {}, f"arguments were not valid JSON ({e})"
    if not isinstance(data, dict):
        return {}, "arguments must be a JSON object"
    return data, None


TOOL_TIMEOUT_SECONDS = DEFAULT_TOOL_TIMEOUT_SECONDS


@dataclass
class ToolCallResult:
    record: dict[str, Any]
    output: str


async def execute_tool_call(toolset: dict[str, BaseTool], name: str, raw_arguments: Any) -> ToolCallResult:
    args, parse_error = parse_arguments(raw_arguments)
    tool = toolset.get(name)
    started = time.monotonic()

    if tool is None:
        output = f"Unknown tool '{name}'. Available tools: {', '.join(sorted(toolset))}."
    elif parse_error:
        output = f"Invalid arguments for {name}: {parse_error}. Send a JSON object."
    else:
        try:
            # The cap gives the task back if a tool hangs (a PowerShell call
            # that never returns); the stuck thread is left to finish on its own.
            output = await asyncio.wait_for(asyncio.to_thread(tool.run, **args), TOOL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("tool_timed_out", tool=name, seconds=TOOL_TIMEOUT_SECONDS)
            output = f"Error executing tool: {name} did not finish within {int(TOOL_TIMEOUT_SECONDS)}s."
        except Exception as e:
            logger.warning("tool_raised", tool=name, error=str(e)[:300])
            output = f"Error executing tool: {e}"

    output = output if isinstance(output, str) else str(output)
    failed = output_indicates_failure(output, name)
    record = {
        "action_type": name,
        "selector_used": summarize_tool_args(args),
        "input_value": summarize_tool_args(args),
        "args": args,
        "success": not failed,
        "output": output[:300],
        "error": output[:300] if failed else None,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    logger.info("tool_step", tool=name, success=not failed, ms=record["duration_ms"], target=record["input_value"][:80])
    return ToolCallResult(record=record, output=output)


# =============================================================================
# Streaming descriptions
# =============================================================================

_SPECIAL_KEYS = {
    "{ENTER}": "Enter", "~": "Enter", "{TAB}": "Tab", "{ESC}": "Escape", "{DOWN}": "Down",
    "{UP}": "Up", "{BACKSPACE}": "Backspace", "^f": "Ctrl+F", "^a": "Ctrl+A",
}


def _keys_label(keys: str) -> str:
    label = keys
    for token, name in _SPECIAL_KEYS.items():
        label = label.replace(token, f" {name} ")
    return " ".join(label.split()) or keys


def describe_action(name: str, args: dict[str, Any] | None) -> str:
    """One short progress line for the UI, e.g. "⌨️ Typing 'Rakesh' in WhatsApp"."""
    a = args or {}

    def quoted(key: str, limit: int = 50) -> str:
        value = str(a.get(key) or "").strip()
        return f"'{value[:limit]}'" if value else ""

    if name == "navigate_browser":
        return f"🔍 Navigating to {a.get('url', 'the site')}"
    if name == "perceive_page":
        return "👀 Reading the page"
    if name == "read_page_as_markdown":
        return "📖 Reading the page as text"
    if name == "see_page":
        return "👀 Looking at the page"
    if name == "type_into_element":
        suffix = " and pressing Enter" if a.get("press_enter") else ""
        return f"⌨️ Typing {quoted('text')}{suffix}"
    if name == "click_element":
        return f"👆 Clicking {quoted('selector', 60)}"
    if name == "click_at_position":
        where = quoted("description", 40) or f"({a.get('x')}, {a.get('y')})"
        return f"👆 Clicking {where} on the page"
    if name == "press_key":
        return f"⏎ Pressing {a.get('key', 'Enter')}"
    if name == "scroll_page":
        return f"↕️ Scrolling {a.get('direction', 'down')}"
    if name == "select_dropdown_option":
        return f"🔽 Choosing {quoted('option_value_or_label')}"
    if name == "extract_page_data":
        return "📋 Reading page data"
    if name == "wait_for_login":
        return "🔐 Waiting for you to log in"
    if name == "upload_file":
        file_name = str(a.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1] or "a file"
        return f"📎 Attaching '{file_name[:60]}' to the page"
    if name == "recall_domain_memory":
        return f"🧠 Recalling what I know about {a.get('domain', 'this site')}"
    if name == "open_app":
        return f"🚀 Opening {a.get('name', 'the app')}"
    if name == "send_keys":
        window = a.get("window_hint") or "the app"
        if a.get("text"):
            return f"⌨️ Typing {quoted('text')} in {window}"
        keys = str(a.get("keys") or "")
        if keys and not any(ch in keys for ch in "{}^~%+"):
            return f"⌨️ Typing '{keys[:50]}' in {window}"
        return f"⌨️ Pressing {_keys_label(keys)} in {window}"
    if name == "read_window_as_markdown":
        return f"📖 Reading {quoted('window', 40) or 'the window'} as text"
    if name == "see_window":
        return f"👀 Looking at {a.get('window') or 'the screen'}"
    if name == "click_window":
        return f"👆 Clicking at ({a.get('x')}, {a.get('y')}) in {a.get('window') or 'the screen'}"
    if name in ("list_windows", "focus_window"):
        return "🪟 " + ("Listing open windows" if name == "list_windows" else f"Focusing {a.get('window', 'the window')}")
    if name in ("find_files", "list_recent_files", "list_folder"):
        target = a.get("pattern") or a.get("query") or a.get("location") or a.get("path") or ""
        return f"📂 Looking for files {('matching ' + str(target)) if target else ''}".strip()
    if name == "open_file_or_folder":
        return f"📂 Opening {quoted('path', 70)}"
    if name == "read_file":
        return f"📄 Reading {quoted('path', 70)}"
    if name == "write_file":
        return f"✏️ Writing {quoted('path', 70)}"
    if name == "run_command":
        return f"💻 Running {quoted('command', 60)}"
    if name == "create_folder":
        return f"📁 Creating folder {quoted('path', 70)}"
    if name in ("move_path", "copy_path"):
        verb = "📦 Moving" if name == "move_path" else "📑 Copying"
        return f"{verb} {quoted('source', 50)} to {quoted('destination', 50)}"
    if name == "delete_path":
        return f"🗑️ Sending {quoted('path', 60)} to the Recycle Bin"
    if name == "search_in_files":
        return f"🔎 Searching files for {quoted('text', 50)}"
    if name == "clipboard":
        return "📋 Copying to the clipboard" if a.get("text") else "📋 Reading the clipboard"
    if name == "search_emails":
        wanted = a.get("query") or a.get("from_address") or a.get("subject") or ""
        where = str(a.get("folder") or "inbox")
        return f"📬 Searching {where} for '{str(wanted)[:50]}'" if wanted else f"📬 Checking {where}"
    if name == "read_email":
        return f"📧 Reading email {a.get('email_id', '')}".rstrip()
    if name == "find_email_address":
        return f"📇 Looking up the email address of {quoted('name')}"
    if name == "create_email_draft":
        return f"📝 Saving a draft to {quoted('to', 60) or 'the recipient'}"
    if name == "send_email":
        if a.get("draft_id"):
            return f"📤 Sending draft {a.get('draft_id')}"
        return f"📤 Sending the email to {quoted('to', 60) or 'the recipient'}"
    return f"🔧 {name}"


# Tools whose result is worth a line of its own in the live stream
# ("✅ Found 234 results" in the tech audit's streaming example).
_RESULT_TOOLS = {
    "navigate_browser", "see_page", "extract_page_data", "wait_for_login", "upload_file",
    "find_files", "list_recent_files", "list_folder", "open_app", "open_file_or_folder",
    "see_window", "list_windows", "read_file", "write_file", "run_command",
    "search_in_files", "create_folder", "move_path", "copy_path", "delete_path", "clipboard",
    "search_emails", "read_email", "find_email_address", "create_email_draft", "send_email",
}


def describe_result(name: str, output: str) -> str | None:
    """A short "✅ ..." line from a successful tool's own first line of output."""
    if name not in _RESULT_TOOLS:
        return None
    first = next(
        (line.strip() for line in (output or "").splitlines() if line.strip() and not line.strip().startswith(("{", "["))),
        "",
    )
    if not first:
        return None
    return "✅ " + (first[:90].rstrip() + "…" if len(first) > 90 else first)


# =============================================================================
# Safety and completion checks
# =============================================================================

def retry_block_reason(trajectory: list[dict[str, Any]], failure_kind: str) -> str | None:
    """Why an automatic retry would be unsafe, or None when it is safe."""
    if failure_kind == "timeout":
        return "the first attempt used the whole time budget"
    done = [
        step.get("action_type") for step in trajectory
        if isinstance(step, dict) and step.get("success") and step.get("action_type") not in SAFE_TO_REPEAT_TOOLS
    ]
    if done:
        unique = ", ".join(dict.fromkeys(str(d) for d in done))
        return f"the first attempt already performed {unique}, and repeating that could duplicate its effect"
    return None


def local_completion_shortfall(instruction: str, trajectory: list[dict[str, Any]]) -> str | None:
    """Why a local run is incomplete, or None if it looks genuinely done.

    Guards the failure mode this agent kept repeating: the request asks for work
    inside an app ("open WhatsApp and search X and say hello"), the run launches
    the app and nothing else, and the final report says the task is complete.
    """
    text = (instruction or "").lower()
    used = {str(step.get("action_type") or "").lower() for step in trajectory if isinstance(step, dict)}
    if not used:
        if any(word in text for word in _ACTION_WORDS):
            return "no tool ran at all, so nothing was actually done on the computer"
        return None
    if not any(word in text for word in _IN_APP_INTENT_WORDS):
        return None
    if used & _IN_APP_TOOLS:
        return None
    if "open_app" in used:
        return (
            "the request asked for actions inside the app, but nothing was typed or "
            "clicked in it — no typing, clicking or key press ran in the app or the browser"
        )
    return None


_REPEAT_REQUEST = re.compile(r"\b(twice|again|repeat\w*|\d+\s*times|thrice)\b", re.IGNORECASE)


class SendTracker:
    """Remembers messages already sent in desktop apps so none is sent twice.

    Found in production: "open WhatsApp and search Rakesh and say hello to him"
    typed 'Hello', pressed Enter (sent), then looked at the window, misread the
    screenshot, clicked the message box and typed 'Hello' again — twice — and
    sent a duplicate. Nothing stopped a repeat of an already-completed send.

    A message counts as sent when send_keys typed `text` into a window and a
    later send_keys pressed Enter there. Text typed after Ctrl+F is a search
    (its Enter opens a chat), not a message. Repeats stay allowed when the
    request itself asks for them ("say hello twice", "send it again").
    """

    def __init__(self, instruction: str) -> None:
        self.enabled = not _REPEAT_REQUEST.search(instruction or "")
        self._pending: dict[str, str] = {}
        self._searching: set[str] = set()
        self.sent: dict[tuple[str, str], str] = {}

    @staticmethod
    def _window(args: dict[str, Any]) -> str:
        return str(args.get("window_hint") or "").strip().lower()

    @staticmethod
    def _email_key(args: dict[str, Any]) -> tuple[str, str]:
        """The same email, however the model spells the call: recipients, subject and text."""
        if str(args.get("draft_id") or "").strip():
            return ("email draft", str(args.get("draft_id")).strip())
        people = " ".join(str(args.get(k) or "") for k in ("to", "cc", "bcc")).lower()
        addresses = ",".join(sorted(set(re.findall(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+", people))))
        subject = " ".join(str(args.get("subject") or "").lower().split())
        body = " ".join(str(args.get("body") or "").lower().split())[:400]
        reply = str(args.get("reply_to_id") or "").strip()
        return ("email", f"{addresses}|{subject}|{body}|{reply}")

    def duplicate_of(self, name: str, args: dict[str, Any]) -> str | None:
        """The already-sent text this call would type again, or None."""
        if not self.enabled:
            return None
        if name == "send_email":
            return self.sent.get(self._email_key(args))
        if name != "send_keys":
            return None
        text = str(args.get("text") or "").strip()
        window = self._window(args)
        if not text or window in self._searching:
            return None
        return self.sent.get((window, text.lower()))

    def record(self, name: str, args: dict[str, Any]) -> str:
        """Update after a successful call. Returns a note for the model, or ""."""
        if name == "send_email":
            if not self.enabled:
                return ""
            label = f"the email '{args.get('subject') or ''}' to {args.get('to') or 'its recipients'}"
            if str(args.get("draft_id") or "").strip():
                label = f"draft {str(args.get('draft_id')).strip()}"
            self.sent[self._email_key(args)] = label
            return (
                f"\n\n[SENT] {label} was sent. That step is DONE: do not send it again. If every part of the "
                "request is complete, write the final report now."
            )
        if name != "send_keys":
            return ""
        window = self._window(args)
        keys = str(args.get("keys") or "")
        text = str(args.get("text") or "").strip()
        note = ""
        # In one send_keys call `keys` are sent before `text`.
        if keys:
            upper = keys.upper()
            if "^F" in upper:
                self._searching.add(window)
                self._pending.pop(window, None)
            if "{ENTER}" in upper or "~" in keys:
                if window in self._searching:
                    self._searching.discard(window)
                elif window in self._pending:
                    message = self._pending.pop(window)
                    self.sent[(window, message.lower())] = message
                    app = args.get("window_hint") or "the app"
                    note = (
                        f"\n\n[SENT] '{message}' was typed and sent with Enter in {app}. That step is DONE: "
                        "do not click the message box, type or send it again. If every part of the request "
                        "is complete, write the final report now (one see_window check at most)."
                    )
            elif "^F" not in upper:
                self._pending.pop(window, None)
        if text and window not in self._searching:
            self._pending[window] = text
        return note


def _shorten(content: str, limit: int) -> str:
    """Keep the beginning and the end of a result, which is where its news is."""
    if len(content) <= limit:
        return content
    return content[: limit // 2] + "\n...[older result shortened]...\n" + content[-limit // 3:]


def compact_messages(
    messages: list[dict[str, Any]],
    keep_recent: int = 8,
    limit: int = 700,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """Shorten old tool results so a long run still fits in one request.

    Two passes, because one was not enough. The first keeps the system prompt,
    the task and the most recent messages whole and shortens older tool
    outputs. The second exists because of a real failure: five page readings in
    a row are all "recent", so nothing was shortened, the request reached about
    thirty thousand characters and Groq refused it with "Request too large for
    model openai/gpt-oss-120b". The whole conversation is then squeezed from
    the oldest result forward until it fits, and only the newest result - the
    one the agent is about to act on - is protected from being cut small.

    Messages are never removed, only shortened: each tool result has to stay
    paired with the tool call that asked for it.
    """
    budget = budget or get_settings().llm_request_char_budget
    compacted = []
    cutoff = len(messages) - keep_recent
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if 2 <= i < cutoff and msg.get("role") == "tool" and isinstance(content, str) and len(content) > limit:
            msg = {**msg, "content": _shorten(content, limit)}
        compacted.append(msg)

    def total() -> int:
        return sum(len(m.get("content") or "") for m in compacted)

    if total() <= budget:
        return compacted

    tool_positions = [i for i, m in enumerate(compacted) if m.get("role") == "tool"]
    newest = tool_positions[-1] if tool_positions else None
    for floor in (limit, 400, 200):
        for i in tool_positions:
            if total() <= budget:
                break
            if i == newest:
                continue           # the agent is acting on this one
            content = compacted[i].get("content") or ""
            if len(content) > floor:
                compacted[i] = {**compacted[i], "content": _shorten(content, floor)}
        if total() <= budget:
            break

    # Still over: the newest result is itself enormous (a whole page of text).
    if total() > budget and newest is not None:
        content = compacted[newest].get("content") or ""
        room = max(1500, budget - (total() - len(content)))
        compacted[newest] = {**compacted[newest], "content": _shorten(content, room)}

    logger.info("messages_compacted", chars=total(), budget=budget, messages=len(compacted))
    return compacted
