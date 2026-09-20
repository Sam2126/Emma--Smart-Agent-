"""Tests for voice input, task scoping, and the local/browser tool split."""

from __future__ import annotations

import asyncio
import base64

import pytest

from app.agent.prompts import actor_system_prompt, planner_user_prompt
from app.agent.state import create_initial_state
from app.agent.toolkit import build_toolset
from app.websocket.protocol import TaskSubmitMessage, VoiceTaskMessage, parse_incoming_message


def test_voice_task_message_roundtrip():
    data = {
        "type": "voice_task",
        "audio_base64": base64.b64encode(b"fakeaudio").decode(),
        "mime": "audio/webm",
        "scope": "local",
    }
    msg = parse_incoming_message(data)
    assert isinstance(msg, VoiceTaskMessage)
    assert msg.scope == "local"
    assert base64.b64decode(msg.audio_base64) == b"fakeaudio"


def test_task_submit_defaults_to_browser_scope():
    msg = parse_incoming_message({"type": "task_submit", "instruction": "hi"})
    assert isinstance(msg, TaskSubmitMessage)
    assert msg.scope == "browser"


def test_initial_state_carries_scope():
    assert create_initial_state("do a local thing", scope="local")["scope"] == "local"
    assert create_initial_state("browse a site")["scope"] == "browser"


def test_a_task_naming_a_local_file_keeps_the_local_tools():
    """Found 2026-09-16: "open chrome search gemini and add the last ppt from
    downloads and ask him to create a summary in that" matched "open chrome"
    first, ran in browser scope, and the agent said it had no tool that could
    list the newest PowerPoint in Downloads — true of the 12 browser tools."""
    from app.websocket.server import _resolve_voice_scope

    instruction = "open chrome search gemini and add the last ppt from downloads and ask him to create a summary in that"
    assert _resolve_voice_scope(instruction, "browser") == "local"

    names = set(build_toolset(_resolve_voice_scope(instruction, "browser")))
    assert {"list_recent_files", "find_files", "upload_file"} <= names
    assert {"navigate_browser", "type_into_element", "click_element"} <= names


@pytest.mark.parametrize("instruction", [
    "open chrome and attach my resume to the form",
    "in chrome upload the latest pdf from downloads",
    "use chrome to search gemini and add the last pptx from my computer",
    "open the browser and send the screenshot from desktop",
])
def test_mixed_web_and_local_tasks_route_to_local(instruction):
    from app.websocket.server import _resolve_voice_scope

    assert _resolve_voice_scope(instruction, "browser") == "local"


@pytest.mark.parametrize("instruction", [
    "open chrome and search for pizza near me",
    "go to youtube.com and play lofi",
    "navigate to amazon.in and find headphones",
    "use the browser to check the news",
])
def test_purely_web_tasks_still_route_to_browser(instruction):
    """The extension must keep its browser-only scope for web-only work."""
    from app.websocket.server import _resolve_voice_scope

    assert _resolve_voice_scope(instruction, "browser") == "browser"


def test_explicit_device_phrases_and_caller_scope_still_win():
    from app.websocket.server import _resolve_voice_scope

    assert _resolve_voice_scope("open chrome on this device", "browser") == "local"
    assert _resolve_voice_scope("open notepad and write a note", "local") == "local"
    # A web-only task keeps the leaner browser toolset even from the Local Agent:
    # local scope resends every local tool schema on each model call, which is what
    # hit Groq's free-tier token limit on 2026-09-15. Nothing loses file access by
    # this, because a task naming a local file is caught by the rule above first.
    assert _resolve_voice_scope("go to youtube.com", "local") == "browser"
    assert _resolve_voice_scope("go to youtube.com and attach the last pdf", "local") == "local"


def test_local_scope_has_full_access():
    """The Local Agent window and the wake word have BOTH local and browser tools."""
    names = set(build_toolset("local"))
    assert {"list_recent_files", "find_files", "run_command", "open_file_or_folder", "open_app", "send_keys"} <= names
    assert {"see_window", "click_window"} <= names
    assert {"navigate_browser", "perceive_page", "see_page", "click_element"} <= names


def test_browser_scope_has_no_local_tools():
    """Strict scope separation: extension tasks must not touch the laptop."""
    names = set(build_toolset("browser"))
    assert "navigate_browser" in names and "see_page" in names
    assert not names & {"list_recent_files", "run_command", "open_app", "send_keys", "write_file"}


def test_local_prompts_skip_browser_protocol():
    plan_prompt = planner_user_prompt("open my recent download", "local")
    assert "BOTH local and browser access" in plan_prompt
    assert "navigate_browser" not in plan_prompt
    assert "BOTH local tools" in actor_system_prompt("local")
    assert "LOGIN-GATE PROTOCOL" not in actor_system_prompt("local")
    assert "LOGIN-GATE PROTOCOL" in actor_system_prompt("browser")


def test_transcribe_rejects_empty_audio():
    from app.voice import transcribe_audio

    with pytest.raises(ValueError):
        asyncio.run(transcribe_audio(b""))
