"""
Tests for the shared LiteLLM patch's tool_choice fix.

Verified empirically against this project's actual Groq account (see session
notes) that openai/gpt-oss-* models reject tool_choice="none" with a 400
("Tool choice is none, but model called a tool") as soon as a prior tool
result is in the conversation — and that removing `tools` from the request
does NOT avoid this; Groq 400s either way. The only request shape that
works is tool_choice="auto" with tools kept in the request. This test locks
in that behavior so a future "helpful-looking" revert back to stripping
tools doesn't quietly reintroduce the crash.
"""

from __future__ import annotations

from app.utils.litellm_patch import _fix_tool_choice


def test_tool_choice_none_with_tools_becomes_auto():
    """The crash-causing case: CrewAI asks for a forced final answer via
    tool_choice='none' while tools are still attached. Stripping tools does
    not help (verified against the real API) — convert to 'auto' instead,
    keeping tools available."""
    kwargs = {
        "tool_choice": "none",
        "tools": [{"type": "function", "function": {"name": "find_files"}}],
    }
    _fix_tool_choice(kwargs)
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"], "tools must be kept, not stripped"


def test_tool_choice_none_dict_form_with_tools_becomes_auto():
    kwargs = {
        "tool_choice": {"type": "none"},
        "tools": [{"type": "function", "function": {"name": "find_files"}}],
    }
    _fix_tool_choice(kwargs)
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"]


def test_tool_choice_none_without_tools_is_stripped():
    """No tools available at all — nothing to set to 'auto' against, so the
    old behavior (drop both keys) is still correct here."""
    kwargs = {"tool_choice": "none", "tools": []}
    _fix_tool_choice(kwargs)
    assert "tool_choice" not in kwargs
    assert "tools" not in kwargs


def test_tool_choice_auto_is_left_alone():
    kwargs = {
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "find_files"}}],
    }
    _fix_tool_choice(kwargs)
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"]


def test_no_tool_choice_and_no_tools_is_a_noop():
    kwargs = {"messages": [{"role": "user", "content": "hi"}]}
    _fix_tool_choice(kwargs)
    assert "tool_choice" not in kwargs
    assert "tools" not in kwargs


# =============================================================================
# Tool schema relaxation. CrewAI's strict-mode conversion marks EVERY argument
# required, so Groq rejected send_keys(window_hint='WhatsApp', keys='^f') with
# "missing properties: 'text', 'delay_ms'" and failed the whole task. Verified
# against the real Groq API: the same call is accepted once only arguments
# without a default are required and "strict" is removed.
# =============================================================================

from app.utils.litellm_patch import _relax_tool_schemas  # noqa: E402


def test_relax_tool_schemas_makes_defaulted_args_optional_and_drops_strict():
    kwargs = {"tools": [{"type": "function", "function": {
        "name": "send_keys",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "window_hint": {"type": "string"},
                "text": {"type": "string", "default": ""},
                "keys": {"type": "string", "default": ""},
                "delay_ms": {"type": "integer", "default": 600},
            },
            "required": ["window_hint", "text", "keys", "delay_ms"],
        },
    }}]}
    _relax_tool_schemas(kwargs)
    fn = kwargs["tools"][0]["function"]
    assert fn["parameters"]["required"] == ["window_hint"]
    assert "strict" not in fn


def test_relax_tool_schemas_leaves_requests_without_tools_untouched():
    kwargs = {"messages": [{"role": "user", "content": "hi"}]}
    _relax_tool_schemas(kwargs)
    assert kwargs == {"messages": [{"role": "user", "content": "hi"}]}
