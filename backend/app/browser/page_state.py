"""
Page state extraction — accessibility tree and DOM snapshot.

Captures a compressed, LLM-friendly representation of the current page state.
Uses Playwright's accessibility tree as the primary perception method,
supplemented by targeted DOM queries for known key elements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from playwright.async_api import Page

logger = structlog.get_logger(__name__)


@dataclass
class PageState:
    """
    Compressed representation of a web page's current state.

    Designed to be token-efficient when serialized for LLM consumption
    (target: ~2000-4000 tokens for a typical product/cart page).
    """

    url: str = ""
    title: str = ""
    # Simplified accessibility tree (list of interactive/meaningful elements)
    accessibility_tree: list[dict[str, Any]] = field(default_factory=list)
    # Key-value pairs extracted from known selectors (e.g., cart_count, price)
    extracted_values: dict[str, str] = field(default_factory=dict)
    # Raw text content of the page (truncated)
    visible_text: str = ""
    # Error if state extraction failed
    error: str | None = None

    def to_llm_context(self, max_tokens: int = 4000) -> str:
        """
        Serialize this page state into a compact string for LLM prompts.

        Prioritizes: URL + title → extracted values → interactive elements → text.
        Truncates to stay within the token budget (~4 chars/token estimate).
        """
        max_chars = max_tokens * 4
        parts: list[str] = []

        # Always include URL and title
        parts.append(f"URL: {self.url}")
        parts.append(f"Title: {self.title}")

        # Include extracted values (high-signal, low-token-cost)
        if self.extracted_values:
            parts.append("\n--- Key Values ---")
            for key, value in self.extracted_values.items():
                parts.append(f"  {key}: {value}")

        # Include interactive elements from the a11y tree
        if self.accessibility_tree:
            parts.append("\n--- Interactive Elements ---")
            for elem in self.accessibility_tree:
                elem_str = self._format_a11y_node(elem)
                if elem_str:
                    parts.append(f"  {elem_str}")

        # Include visible text (lowest priority, truncated)
        if self.visible_text:
            parts.append("\n--- Page Text (truncated) ---")
            parts.append(self.visible_text)

        result = "\n".join(parts)

        # Truncate to budget
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... [truncated]"

        return result

    @staticmethod
    def _format_a11y_node(node: dict[str, Any]) -> str:
        """Format a single accessibility tree node for LLM consumption."""
        role = node.get("role", "")
        name = node.get("name", "")
        value = node.get("value", "")
        description = node.get("description", "")

        # Skip decorative / non-interactive nodes
        if role in ("none", "generic", "presentation", "StaticText") and not name:
            return ""

        parts = [f"[{role}]"]
        if name:
            parts.append(f'"{name}"')
        if value:
            parts.append(f"value={value}")
        if description:
            parts.append(f"({description})")

        return " ".join(parts)


async def extract_page_state(
    page: Page,
    known_selectors: dict[str, str] | None = None,
) -> PageState:
    """
    Extract a compressed page state from the given Playwright page.

    Args:
        page: The Playwright Page to extract state from.
        known_selectors: Optional dict of {name: css_selector} to extract
                        specific values from (e.g., {"cart_count": "#nav-cart-count"}).

    Returns:
        A PageState object with the current page representation.
    """
    state = PageState()

    try:
        # --- Basic info ---
        state.url = page.url
        state.title = await page.title()

        # --- Accessibility tree ---
        # Playwright removed page.accessibility (it is gone in 1.62, so this
        # tree was always empty); the ARIA snapshot is its replacement.
        try:
            snapshot = await page.locator("body").aria_snapshot(timeout=3000)
            state.accessibility_tree = _parse_aria_snapshot(snapshot)
        except Exception as e:
            logger.debug("a11y_snapshot_skipped", error=str(e)[:200])

        # --- Extract known selector values ---
        if known_selectors:
            for name, selector in known_selectors.items():
                try:
                    element = page.locator(selector).first
                    if await element.is_visible(timeout=2000):
                        text = await element.inner_text(timeout=2000)
                        state.extracted_values[name] = text.strip()
                except Exception:
                    # Element not found or not visible — skip silently
                    pass

        # --- Visible text (truncated) ---
        try:
            body_text = await page.locator("body").inner_text(timeout=5000)
            # Take first ~2000 chars of visible text
            state.visible_text = body_text[:2000].strip() if body_text else ""
        except Exception as e:
            logger.warning("body_text_extraction_failed", error=str(e))

    except Exception as e:
        logger.error("page_state_extraction_failed", error=str(e))
        state.error = str(e)

    logger.info(
        "page_state_extracted",
        url=state.url,
        a11y_nodes=len(state.accessibility_tree),
        extracted_values=len(state.extracted_values),
    )

    return state


_ARIA_INTERACTIVE_ROLES = {
    "button", "link", "textbox", "combobox", "checkbox", "radio",
    "menuitem", "tab", "searchbox", "slider", "spinbutton",
    "switch", "option", "listbox", "menu", "menubar",
    "heading", "img", "alert", "dialog", "navigation",
}
# One ARIA snapshot line, e.g.:  - link "ChatGPT" [level=2]:   or   - textbox "Search"
_ARIA_LINE = re.compile(r'^\s*-\s+([a-z]+)(?:\s+"((?:[^"\\]|\\.)*)")?(?:\s+\[([^\]]*)\])?')


def _parse_aria_snapshot(snapshot: str, max_nodes: int = 150) -> list[dict[str, Any]]:
    """Flatten Playwright's YAML ARIA snapshot into the same node list as before."""
    nodes: list[dict[str, Any]] = []
    for line in (snapshot or "").splitlines():
        match = _ARIA_LINE.match(line)
        if not match:
            continue
        role, name, attrs = match.group(1), (match.group(2) or "").replace('\\"', '"'), match.group(3) or ""
        if role not in _ARIA_INTERACTIVE_ROLES and not name:
            continue
        node: dict[str, Any] = {"role": role, "name": name}
        for attr in attrs.split():
            key, _, value = attr.partition("=")
            if key == "level" and value.isdigit():
                node["level"] = int(value)
            elif key == "checked":
                node["checked"] = value != "false"
            elif key == "disabled":
                node["disabled"] = True
        nodes.append(node)
        if len(nodes) >= max_nodes:
            break
    return nodes


def _flatten_a11y_tree(
    node: dict[str, Any],
    max_depth: int = 6,
    max_nodes: int = 150,
    _depth: int = 0,
    _count: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Flatten the hierarchical accessibility tree into a flat list of nodes.

    Filters out decorative/non-interactive nodes to keep the representation
    compact. Caps at max_nodes to stay within LLM token budgets.

    Args:
        node: Root accessibility tree node from Playwright.
        max_depth: Maximum tree depth to traverse.
        max_nodes: Maximum number of nodes to include.
        _depth: Current recursion depth (internal).
        _count: Mutable counter for node limiting (internal).

    Returns:
        Flat list of meaningful accessibility nodes.
    """
    if _count is None:
        _count = [0]

    if _depth > max_depth or _count[0] >= max_nodes:
        return []

    result: list[dict[str, Any]] = []
    role = node.get("role", "")
    name = node.get("name", "")

    # Include interactive and meaningful roles
    _INTERACTIVE_ROLES = {
        "button", "link", "textbox", "combobox", "checkbox", "radio",
        "menuitem", "tab", "searchbox", "slider", "spinbutton",
        "switch", "option", "listbox", "menu", "menubar",
        "heading", "img", "alert", "dialog", "navigation",
    }

    if role in _INTERACTIVE_ROLES or (role and name):
        flat_node = {
            "role": role,
            "name": name,
        }
        if node.get("value"):
            flat_node["value"] = node["value"]
        if node.get("description"):
            flat_node["description"] = node["description"]
        if node.get("checked") is not None:
            flat_node["checked"] = node["checked"]
        if node.get("disabled"):
            flat_node["disabled"] = True
        if node.get("level"):
            flat_node["level"] = node["level"]

        result.append(flat_node)
        _count[0] += 1

    # Recurse into children
    children = node.get("children", [])
    for child in children:
        if _count[0] >= max_nodes:
            break
        result.extend(
            _flatten_a11y_tree(child, max_depth, max_nodes, _depth + 1, _count)
        )

    return result
