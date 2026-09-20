"""
Tests for reading a screen as text instead of guessing at it.

Two tools are covered, one per surface:

  read_window_as_markdown  a desktop window, through Windows' own accessibility
                           service (UI Automation) - exact, free, no vision model
  read_page_as_markdown    a web page, through one pass of JavaScript inside it

The desktop tests stub the PowerShell call, so they check the parsing, the
coordinate arithmetic and the wording of every refusal without needing a window
open on the machine running them. The page tests drive a real headless browser,
because the whole point of that tool is what real DOM and shadow DOM contain.
"""

from __future__ import annotations

import asyncio

import pytest

from app.tools.desktop import ReadWindowTool
from app.utils.vision import percent_to_pixels


# =============================================================================
# Positions reported by a vision model
# =============================================================================

def test_percentages_become_pixels_of_the_real_capture():
    assert percent_to_pixels("`Send` -> (25%, 80%)", 1280, 720) == "`Send` -> (320, 576)"


def test_whole_screen_positions_carry_the_captured_origin():
    # A monitor placed left of the main one starts at a negative x: the centre
    # of that virtual desktop is screen x=0, not x=1920.
    assert percent_to_pixels("`A` -> (50%, 50%)", 3840, 1080, -1920, 0) == "`A` -> (0, 540)"


def test_values_that_are_already_pixels_are_left_alone():
    assert percent_to_pixels("`A` -> (300, 400)", 800, 600) == "`A` -> (300, 400)"


# =============================================================================
# read_window_as_markdown
# =============================================================================

def _ps_returning(*lines: str):
    """Stand in for the PowerShell call with a fixed transcript."""
    return lambda script, timeout=None: "\n".join(lines)


def _window(left=100, top=50, width=800, height=600):
    return f"WIN|notepad|Untitled - Notepad|{left}|{top}|{width}|{height}"


def _node(depth, ctype, name, x, y, w, h, state="", value=""):
    return f"N|{depth}|{ctype}|{name}|{x}|{y}|{w}|{h}|{state}|{value}"


def test_a_window_is_read_as_an_outline_with_window_relative_positions(monkeypatch):
    monkeypatch.setattr(
        "app.tools.desktop._ps",
        _ps_returning(
            _window(left=100, top=50),
            "CLOAKED|0",
            # centre of this button is at screen (250, 150) -> (150, 100) in the window
            _node(1, "Button", "Compose", 200, 130, 100, 40),
            _node(2, "Edit", "Message", 200, 200, 400, 30, "", "Hello Shrey"),
            _node(2, "CheckBox", "Send a copy", 200, 300, 20, 20, "on"),
            "END|3|0",
        ),
    )
    out = ReadWindowTool()._run(window="Notepad")

    assert "`Compose` — Button at (150, 100)" in out
    assert 'holds "Hello Shrey"' in out
    assert "`Send a copy` — CheckBox [on]" in out
    assert "measured, not estimated" in out


def test_a_control_with_no_rectangle_is_listed_without_a_position(monkeypatch):
    monkeypatch.setattr(
        "app.tools.desktop._ps",
        _ps_returning(_window(), "CLOAKED|0",
                      _node(1, "MenuItem", "File", -1, -1, 0, 0, "off screen"), "END|1|0"),
    )
    out = ReadWindowTool()._run(window="Notepad")
    assert "`File` — MenuItem [off screen]" in out
    assert " at (" not in out


def test_a_suspended_app_says_so_instead_of_blaming_the_app(monkeypatch):
    # Measured on Settings: Windows suspends a Store app and cloaks its window,
    # leaving no interface to read and nothing drawn on screen.
    monkeypatch.setattr(
        "app.tools.desktop._ps",
        _ps_returning(_window(), "CLOAKED|1", "END|0|0"),
    )
    out = ReadWindowTool()._run(window="Settings")
    assert "SUSPENDED" in out and "focus_window" in out
    # It must NOT send the agent to a screenshot: a cloaked window is not drawn,
    # so the screenshot would show whatever app is in front of it.
    assert "see_window would photograph" in out


def test_a_window_showing_only_title_bar_buttons_is_called_out(monkeypatch):
    # WhatsApp and Electron apps expose their window furniture and nothing else.
    monkeypatch.setattr(
        "app.tools.desktop._ps",
        _ps_returning(
            _window(), "CLOAKED|0",
            _node(1, "Button", "Minimize", 1700, 10, 40, 30),
            _node(1, "Button", "Restore", 1750, 10, 40, 30),
            _node(1, "Button", "Close", 1800, 10, 40, 30),
            "END|3|0",
        ),
    )
    out = ReadWindowTool()._run(window="WhatsApp")
    assert "Only the window's own title-bar buttons" in out
    assert "Use see_window" in out


def test_a_missing_window_is_reported_without_reading_anything(monkeypatch):
    monkeypatch.setattr("app.tools.desktop._ps", _ps_returning("NOWINDOW"))
    out = ReadWindowTool()._run(window="Spotify")
    assert "No visible window matching 'Spotify'" in out
    assert "list_windows" in out


def test_an_app_with_no_accessibility_tree_is_sent_to_see_window(monkeypatch):
    monkeypatch.setattr("app.tools.desktop._ps", _ps_returning("NOTREE"))
    out = ReadWindowTool()._run(window="SomeGame")
    assert "Use see_window" in out


def test_chrome_is_never_read_with_the_desktop_tools():
    out = ReadWindowTool()._run(window="Chrome")
    assert out.startswith("Refused:")
    assert "read_page_as_markdown" not in out.lower() or "browser tools" in out


def test_reading_needs_a_named_window():
    out = ReadWindowTool()._run(window="")
    assert "needs a window" in out and "see_window" in out


# =============================================================================
# read_page_as_markdown (real browser)
# =============================================================================

PAGE = """
<html><head><title>Inbox</title></head><body>
<main>
  <h1>Inbox</h1>
  <p>You have <strong>3</strong> new messages. See <a href="/help">the help page</a>.</p>
  <ul><li>Shrey - Project update</li></ul>
  <table><tr><th>From</th><th>Subject</th></tr><tr><td>Shrey</td><td>Update</td></tr></table>
  <button aria-label="Compose new message">+</button>
  <input id="q" placeholder="Search mail" value="invoice">
  <input type="checkbox" id="c1" checked><label for="c1">Select all</label>
  <button disabled aria-label="Send">Send</button>
  <canvas id="board" width="150" height="80"></canvas>
  <my-widget></my-widget>
  <nav><a href="/sale">T-shirts Under 299</a></nav>
  <div class="vertical-filters">
    <label class="common-customCheckbox"><input type="checkbox" style="visibility:hidden">Roadster(2826)</label>
    <label class="common-customCheckbox"><input type="checkbox" style="visibility:hidden">Leotude(2243)</label>
  </div>
  <a href="/p/123" class="card">Jockey
Sizes: S
Rs. 799</a>
  <div style="height:2000px"></div>
  <button id="far" aria-label="Far below">bottom</button>
</main>
<script>
  class MyWidget extends HTMLElement {
    connectedCallback() {
      this.attachShadow({mode:'open'}).innerHTML =
        '<button id="deep" aria-label="Inside a web component">go</button>';
    }
  }
  customElements.define('my-widget', MyWidget);
</script>
</body></html>
"""


@pytest.fixture(scope="module")
def page_reading():
    """Run the real reader against a real page once; return (data, locator counts)."""
    from playwright.async_api import async_playwright
    from app.tools.browser import _PAGE_MARKDOWN_JS

    async def _read():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 1000, "height": 700})
                await page.set_content(PAGE)
                await page.wait_for_timeout(150)
                data = await page.evaluate(_PAGE_MARKDOWN_JS, {"max_text": 8000, "max_controls": 60})
                # Every selector the tool offers must actually resolve.
                counts = {}
                for control in data["controls"]:
                    if control["selector"]:
                        counts[control["name"]] = await page.locator(control["selector"]).count()
                return data, counts
            finally:
                await browser.close()

    return asyncio.run(_read())


@pytest.fixture(scope="module")
def controls(page_reading):
    data, _ = page_reading
    return {c["name"]: c for c in data["controls"]}


def test_an_icon_only_button_is_named_the_way_a_screen_reader_names_it(controls):
    assert "Compose new message" in controls
    assert controls["Compose new message"]["role"] == "button"


def test_a_control_inside_a_web_component_is_found(controls):
    # Shadow DOM is invisible to an ordinary querySelectorAll, which is why
    # custom widgets used to look empty.
    assert "Inside a web component" in controls


def test_a_field_reports_what_it_already_holds(controls):
    assert controls["Search mail"]["state"] == 'contains "invoice"'


def test_a_ticked_box_is_not_described_by_its_value(controls):
    # A checkbox's value is the string "on" and means nothing.
    assert controls["Select all"]["state"] == "checked"


def test_a_disabled_button_says_so(controls):
    assert controls["Send"]["state"] == "disabled"


def test_every_control_carries_a_measured_rectangle(controls):
    box = controls["Compose new message"]["box"]
    # Each value is rounded on its own, so the centre is within a pixel of the
    # middle of the rectangle rather than exactly equal to it.
    assert abs(box["cx"] - (box["x"] + box["w"] / 2)) <= 1
    assert abs(box["cy"] - (box["y"] + box["h"] / 2)) <= 1
    assert box["x"] < box["cx"] < box["x"] + box["w"]
    assert box["y"] < box["cy"] < box["y"] + box["h"]
    assert box["on_screen"] is True


def test_something_below_the_fold_is_marked_off_screen(controls):
    assert controls["Far below"]["box"]["on_screen"] is False


def test_every_offered_selector_resolves_on_the_page(page_reading):
    _, counts = page_reading
    unresolved = [name for name, count in counts.items() if count < 1]
    assert not unresolved, f"selectors that match nothing: {unresolved}"


def test_the_page_is_returned_as_markdown(page_reading):
    data, _ = page_reading
    markdown = data["markdown"]
    assert "# Inbox" in markdown
    assert "[the help page](/help)" in markdown       # links survive
    assert "**3**" in markdown                        # emphasis survives
    assert "| From | Subject |" in markdown           # tables survive
    assert "<div" not in markdown and "<button" not in markdown   # no raw HTML


# =============================================================================
# What the page tools hand back to the agent
# =============================================================================

class _FakeMouse:
    def __init__(self):
        self.clicks = []

    async def click(self, x, y):
        self.clicks.append((x, y))


class _FakePage:
    """Just enough of a Playwright page for the tool bodies."""

    def __init__(self, data=None, width=1000, height=700):
        self._data = data or {}
        self._size = [width, height]
        self.url = "https://example.com/inbox"
        self.mouse = _FakeMouse()

    async def evaluate(self, script, arg=None):
        # The reader's own JavaScript mentions innerWidth too, so match the
        # viewport query exactly rather than by keyword.
        if "=> [window.innerWidth" in script:
            return self._size
        return self._data

    async def title(self):
        return "Inbox"

    async def wait_for_load_state(self, *a, **k):
        return None


class _FakeController:
    def __init__(self, page):
        self._page = page
        self.download_seq = 0
        self.downloads = []

    async def get_active_page(self):
        return self._page

    async def get_all_pages(self):
        return [self._page]

    async def follow_new_tab(self, _before):
        return None


@pytest.fixture
def fake_browser(monkeypatch):
    def _install(data=None, width=1000, height=700):
        page = _FakePage(data, width, height)
        monkeypatch.setattr("app.tools.browser.get_browser_controller", lambda: _FakeController(page))
        return page
    return _install


def _reading(**overrides):
    reading = {
        "title": "Inbox", "url": "https://example.com/inbox",
        "viewport": [1000, 700], "scroll": [0, 240], "page_height": 2600,
        "markdown": "# Inbox\n\nYou have **3** new messages.", "truncated": False,
        "controls": [
            {"name": "Compose", "role": "button", "selector": "#compose", "state": "",
             "box": {"x": 20, "y": 120, "w": 120, "h": 40, "cx": 80, "cy": 140, "on_screen": True}},
            {"name": "Archive", "role": "button", "selector": "#archive", "state": "disabled",
             "box": {"x": 20, "y": 2400, "w": 120, "h": 40, "cx": 80, "cy": 2420, "on_screen": False}},
        ],
    }
    reading.update(overrides)
    return reading


def test_the_reading_names_every_control_with_its_position(fake_browser):
    from app.tools.browser import PageMarkdownTool
    fake_browser(_reading())
    out = PageMarkdownTool()._run()

    assert "# Inbox" in out and "https://example.com/inbox" in out
    assert "Visible area 1000x700 px" in out and "2600 px tall" in out
    assert "## Controls (2)" in out
    assert "`Compose` - button selector: #compose at (80, 140)" in out
    assert "[disabled]" in out
    assert "## Page content" in out


def test_a_control_below_the_fold_tells_the_agent_to_scroll(fake_browser):
    from app.tools.browser import PageMarkdownTool
    fake_browser(_reading())
    out = PageMarkdownTool()._run()
    archive = [line for line in out.splitlines() if "Archive" in line][0]
    assert "OFF SCREEN, scroll_page first" in archive
    assert "OFF SCREEN" not in [line for line in out.splitlines() if "Compose" in line][0]


def test_a_page_with_no_controls_points_at_the_screenshot(fake_browser):
    from app.tools.browser import PageMarkdownTool
    fake_browser(_reading(controls=[]))
    out = PageMarkdownTool()._run()
    assert "## Controls (0)" in out and "see_page" in out


def test_cut_off_text_says_so(fake_browser):
    from app.tools.browser import PageMarkdownTool
    fake_browser(_reading(truncated=True))
    assert "text cut off at the limit" in PageMarkdownTool()._run()


def test_clicking_a_position_inside_the_page_reaches_the_mouse(fake_browser):
    from app.tools.browser import ClickPositionTool
    page = fake_browser()
    out = ClickPositionTool()._run(x=80, y=140, description="the Compose button")
    assert page.mouse.clicks == [(80, 140)]
    assert "Clicked the Compose button at (80, 140)" in out
    # A position click gives no confirmation of its own.
    assert "see_page or perceive_page" in out


def test_clicking_outside_the_visible_area_is_refused(fake_browser):
    from app.tools.browser import ClickPositionTool
    page = fake_browser()
    out = ClickPositionTool()._run(x=80, y=2420, description="the Archive button")
    assert page.mouse.clicks == [], "nothing may be clicked outside the visible area"
    assert "outside the visible page area" in out and "scroll_page" in out


# =============================================================================
# Controls a shop hides behind its own styling
# =============================================================================
# Measured on Myntra: every brand and price facet is an <input type=checkbox>
# with visibility:hidden inside a visible <label>. Before this was handled the
# reader found none of them, and offered promotional links from the menu
# ("T-shirts Under 299") as if they were price filters.

def test_a_checkbox_a_shop_hides_behind_its_label_is_still_found(controls):
    assert "Roadster(2826)" in controls
    assert controls["Roadster(2826)"]["role"] == "checkbox"


def test_such_a_control_is_clicked_on_its_visible_label(controls):
    brand = controls["Roadster(2826)"]
    # The hidden input has no usable position; the label around it does.
    assert brand["box"]["w"] > 1 and brand["box"]["h"] > 1
    assert brand["selector"].startswith("label")


def test_a_filter_is_told_apart_from_a_promotional_link(controls):
    assert controls["Roadster(2826)"]["region"] == "filters"
    assert controls["T-shirts Under 299"]["region"] == "menu"


def test_a_link_is_addressed_by_its_href(controls):
    # `:has-text()` matches raw text, so a product card whose text runs over
    # several lines was unreachable through a normalised text selector.
    assert controls["Jockey Sizes: S Rs. 799"]["selector"] == 'a[href="/p/123"]'


def test_the_reading_says_which_controls_are_filters(fake_browser):
    from app.tools.browser import PageMarkdownTool
    fake_browser(_reading(controls=[
        {"name": "Roadster", "role": "checkbox", "selector": "label:has-text(\"Roadster\")",
         "state": "", "region": "filters",
         "box": {"x": 20, "y": 400, "w": 200, "h": 17, "cx": 120, "cy": 408, "on_screen": True}},
        {"name": "T-shirts Under 299", "role": "link", "selector": "a[href=\"/sale\"]",
         "state": "", "region": "menu",
         "box": {"x": 20, "y": 10, "w": 120, "h": 20, "cx": 80, "cy": 20, "on_screen": True}},
    ]))
    out = PageMarkdownTool()._run()
    assert "`Roadster` - checkbox (in the page's filters)" in out
    assert "`T-shirts Under 299` - link (in the menu, not a filter)" in out
