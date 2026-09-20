"""
Universal site-agnostic browser automation tools for the agent engine.

These tools allow the autonomous agent to perceive and operate ANY web page
(Amazon, Flipkart, eBay, Walmart, Google, Apple, Nike, etc.) without hardcoding.

Perception output is returned as newline-separated JSON objects so the LLM
can reliably parse element selectors and categories rather than guessing from
Python repr() strings.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Type, Callable
from urllib.parse import urlparse

import structlog
from pydantic import BaseModel, Field
from app.tools.base import BaseTool

from app.browser.registry import get_browser_controller
from app.config import get_settings
from app.tools.runtime import run_sync as _run_sync
from app.browser.actions import ActionType, _count, execute_action
from app.state.domain_memory import DomainMemoryStore, extract_domain

logger = structlog.get_logger(__name__)
domain_store = DomainMemoryStore()




# =============================================================================
# Shared helpers: human confirmation gate and page vision
# =============================================================================

async def _human_gate(confirmer, action_description: str, url: str) -> str | None:
    """Ask the user before an irreversible action. Returns None to go ahead.

    With no confirmation channel (a task started from REST or the wake word),
    the action is refused, which is what the safety gate always did. With the
    extension connected, the user sees a confirmation dialog and decides.
    """
    if confirmer is None:
        logger.warning("safety_gate_blocked_irreversible_action", action=action_description, url=url)
        return (
            f"SAFETY GATE TRIGGERED: '{action_description}' is an IRREVERSIBLE TRANSACTION "
            "(place order / payment / checkout) and no user is connected to confirm it, so it "
            "was NOT performed. Stop here and report that the user must confirm this step."
        )
    approved = await confirmer.request(action_description, {"url": url})
    if approved:
        logger.info("safety_gate_user_approved", action=action_description, url=url)
        return None
    logger.info("safety_gate_user_declined", action=action_description, url=url)
    return (
        f"The user DECLINED '{action_description}', so it was NOT performed. "
        "Do not retry it. Report that the user declined this step."
    )


async def _page_vision(page, question: str, with_positions: bool = False) -> str | None:
    """Screenshot the visible page and describe it with the vision provider chain.

    With `with_positions`, the model is asked for each element's centre as a
    percentage of the image and the answer comes back in CSS pixels, ready for
    click_at_position. Percentages are asked for because a vision model judges
    relative placement well and absolute pixel counts badly; the conversion uses
    the page's own innerWidth/innerHeight, which is the coordinate space the
    mouse works in whatever the screen's scaling is.
    """
    from app.utils.vision import describe_image, percent_to_pixels

    try:
        shot = await page.screenshot(type="jpeg", quality=60, full_page=False, timeout=10000)
    except Exception as e:
        logger.warning("page_screenshot_failed", error=str(e)[:150])
        return None
    prompt = (
        "You are looking at a screenshot of a web page in a browser.\n"
        f"Question: {question}\n\n"
        "Answer concisely. Always mention any popup, modal, cookie banner, login or sign-in "
        "wall, CAPTCHA or error message covering the page. Then list the most important "
        "visible buttons, links and inputs by their exact visible text, top to bottom. "
        "Do not invent anything that is not visible."
    )
    if with_positions:
        prompt += (
            "\n\nWrite EVERY clickable element on its own line, exactly as:\n"
            "`label` -> (X%, Y%)\n"
            "where X% and Y% are the CENTRE of that element as a PERCENTAGE (0-100) of the "
            "image width and height. Example: `Compose` -> (7%, 18%). Never give pixel values."
        )
    text = await describe_image(base64.b64encode(shot).decode(), prompt, mime="image/jpeg", max_tokens=700)
    if text and with_positions:
        width, height = await _viewport_size(page)
        text = percent_to_pixels(text, width, height)
    return text


async def _viewport_size(page) -> tuple[int, int]:
    """The visible page area in CSS pixels - the space mouse coordinates live in."""
    try:
        size = await page.evaluate("() => [window.innerWidth, window.innerHeight]")
        return int(size[0]), int(size[1])
    except Exception:
        box = page.viewport_size or {}
        return int(box.get("width") or 1280), int(box.get("height") or 720)


_BOT_CHECK_URL_MARKERS = ("__cf_chl", "/cdn-cgi/challenge-platform")
_BOT_CHECK_TITLES = ("just a moment", "attention required", "verify you are human")


async def _bot_check_remains(page, wait_seconds: float = 15.0) -> bool:
    """True when the page still shows a "verify you are human" check after waiting.

    Found 2026-09-15: chatgpt.com answered the agent's browser with Cloudflare's
    check page (URL containing __cf_chl). In a normal Chrome window the check
    often clears by itself within seconds, so it is waited out briefly; if it
    stays, the agent is told plainly instead of re-opening the site.
    """
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        try:
            title = (await asyncio.wait_for(page.title(), 2)).lower()
        except Exception:
            title = ""
        on_check = any(m in (page.url or "") for m in _BOT_CHECK_URL_MARKERS) or any(t in title for t in _BOT_CHECK_TITLES)
        if not on_check:
            return False
        if asyncio.get_running_loop().time() >= deadline:
            return True
        await asyncio.sleep(1)


# =============================================================================
# 1. Universal Navigate Tool
# =============================================================================

class NavigateInput(BaseModel):
    url: str = Field(..., description="The complete website URL (e.g. 'https://www.flipkart.com', 'https://www.amazon.in', 'https://www.apple.com')")


class NavigateBrowserTool(BaseTool):
    name: str = "navigate_browser"
    description: str = "Navigates the browser to any target website URL and returns the loaded URL and title."
    args_schema: Type[BaseModel] = NavigateInput

    def _run(self, url: str) -> str:
        async def _async_run():
            target_url = url.strip()
            if not target_url.startswith("http://") and not target_url.startswith("https://"):
                target_url = f"https://{target_url}"

            controller = get_browser_controller()
            page = await controller.get_active_page()
            result = await execute_action(page, ActionType.NAVIGATE, input_value=target_url)
            if result.success and await _bot_check_remains(page):
                return (
                    f"Reached {page.url[:120]}, but the site is showing a 'verify you are human' check "
                    "(Cloudflare) that the agent could not get past. Stop here and tell the user to complete "
                    "the check in the agent's Chrome window, then run the task again. Do not re-open the site."
                )
            if result.success:
                try:
                    title = await asyncio.wait_for(page.title(), 3)
                except Exception:
                    title = ""
                if result.metadata.get("still_loading"):
                    return (
                        f"Reached {result.url_after}, but the page is still loading (title so far: '{title}'). "
                        "Do NOT navigate to it again: call perceive_page or see_page to work with what is there."
                    )
                return f"Successfully navigated to {result.url_after}. Title: '{title}'."
            return (
                f"Navigation failed: {(result.error or '')[:300]} "
                "Do not open the same URL again more than once; try another way (for example a search)."
            )

        return _run_sync(_async_run)


# =============================================================================
# 2. Universal Page Perception Tool
# =============================================================================

# Module-level constant so tests can import and reuse the exact same JS
# the agent uses — no duplication, guaranteed consistency.
_PERCEIVE_JS = """
() => {
    const results = [];

    // Helper: clean text extractor
    const getText = (el) => {
        let t = (el.innerText || el.textContent || el.value || el.getAttribute('aria-label') || el.title || el.placeholder || el.querySelector('img')?.getAttribute('alt') || '').trim();
        // Fallback for prominent IDs that represent game overlays or start banners
        if (!t && el.id) {
            const idLow = el.id.toLowerCase();
            if (idLow.includes('instruction') || idLow.includes('start') || idLow.includes('play') || idLow.includes('overlay') || idLow.includes('launch')) {
                t = 'Click to Play / Start Overlay';
            }
        }
        return t;
    };

    // Helper: the element's exact rectangle, in CSS pixels measured from the
    // top-left of the VISIBLE page area. Added 2026-09-20: the scan named
    // elements but never said where they were, so anything the model could see
    // but had no reliable selector for (Gmail's custom buttons, canvases,
    // shadow-DOM widgets) was unreachable. cx/cy go straight to
    // click_at_position. Elements scrolled out of view are reported too, with
    // on_screen false, so the agent scrolls instead of clicking blind.
    const boxOf = (el) => {
        try {
            const r = el.getBoundingClientRect();
            if (!r || r.width < 1 || r.height < 1) return null;
            return {
                x: Math.round(r.left), y: Math.round(r.top),
                w: Math.round(r.width), h: Math.round(r.height),
                cx: Math.round(r.left + r.width / 2),
                cy: Math.round(r.top + r.height / 2),
                on_screen: r.top >= 0 && r.left >= 0 &&
                           r.bottom <= window.innerHeight && r.right <= window.innerWidth
            };
        } catch (e) { return null; }
    };


    // 1. Text Inputs, Search Boxes & Code/Rich Editors (max 6)
    const inputSelectors = [
        'input[type="text"]', 'input[type="search"]', 'input:not([type])', 'textarea', '[contenteditable="true"]', '[role="textbox"]',
        '.CodeMirror', '.monaco-editor', '.ace_editor'
    ];
    document.querySelectorAll(inputSelectors.join(', ')).forEach((el) => {
        if (el.offsetParent !== null && results.filter(r => r.category === 'input' || r.category === 'code_or_rich_editor').length < 6) {
            const isEditor = el.classList.contains('CodeMirror') || el.classList.contains('monaco-editor') || el.classList.contains('ace_editor') || el.getAttribute('contenteditable') === 'true';
            const sel = el.id ? `#${el.id}` : (el.name ? `${el.tagName.toLowerCase()}[name="${el.name}"]` : (el.placeholder ? `${el.tagName.toLowerCase()}[placeholder*="${el.placeholder.slice(0,15)}"]` : el.tagName.toLowerCase()));
            results.push({
                category: isEditor ? 'code_or_rich_editor' : 'input',
                box: boxOf(el),
                selector: sel,
                placeholder: (el.placeholder || el.getAttribute('aria-label') || '').slice(0, 30),
                id: el.id || ''
            });
        }
    });

    // 2. Clickable Action Buttons, Call-to-Actions & Prompts (max 14)
    const buttonSelectors = [
        'button', 'input[type="submit"]', 'input[type="button"]', 'a.btn', '[role="button"]', '[role="tab"]',
        '[class*="btn" i]', '[class*="button" i]', '[id*="play" i]', '[id*="start" i]', '[id*="instructions" i]',
        'div[onclick]', 'span[onclick]'
    ];
    const seenButtons = new Set();
    document.querySelectorAll(buttonSelectors.join(', ')).forEach((el) => {
        const txt = getText(el);
        if (el.offsetParent !== null && (txt || el.id) && results.filter(r => r.category === 'button').length < 12) {
            const key = el.id || txt.slice(0, 25);
            if (!seenButtons.has(key)) {
                seenButtons.add(key);
                let sel = el.id ? `#${el.id}` : (txt ? `${el.tagName.toLowerCase()}:has-text("${txt.slice(0, 20)}")` : 'button');
                results.push({
                    category: 'button',
                    box: boxOf(el),
                    text: txt.slice(0, 30),
                    selector: sel,
                    id: el.id || ''
                });
            }
        }
    });

    // Capture prominent CTA text (e.g. "CLICK TO PLAY", "Play Now", "Start Game", "Quick Match")
    const actionKeywords = ['click to play', 'play now', 'start game', 'quick match', 'play free', 'tap to play', 'press to start'];
    document.querySelectorAll('div, span, h1, h2, h3, a').forEach((el) => {
        if (el.offsetParent !== null && el.children.length <= 1 && results.filter(r => r.category === 'button').length < 15) {
            const txt = (el.innerText || '').trim();
            const lower = txt.toLowerCase();
            if (actionKeywords.some(kw => lower === kw || (lower.startsWith(kw) && lower.length < 30))) {
                const key = el.id || txt;
                if (!seenButtons.has(key)) {
                    seenButtons.add(key);
                    results.push({
                        category: 'button',
                        box: boxOf(el),
                        text: txt.slice(0, 30),
                        selector: el.id ? `#${el.id}` : `text="${txt}"`,
                        id: el.id || ''
                    });
                }
            }
        }
    });

    // 3. Dropdowns & Selection Controls (max 4)
    document.querySelectorAll('select, [role="combobox"]').forEach((el) => {
        if (el.offsetParent !== null && results.filter(r => r.category === 'dropdown_or_select').length < 4) {
            let optionsList = [];
            if (el.tagName === 'SELECT') {
                optionsList = Array.from(el.options).slice(0, 6).map(o => o.text.trim());
            }
            results.push({
                category: 'dropdown_or_select',
                box: boxOf(el),
                selector: el.id ? `#${el.id}` : (el.name ? `select[name="${el.name}"]` : 'select'),
                id: el.id || '',
                options_sample: optionsList
            });
        }
    });

    // 4. Navigation Filters, Categories, Facets & Checkboxes (max 8)
    const filterSelectors = [
        '#s-refinements a', '[class*="filter" i] a', '[class*="refinement" i] a',
        '[id*="filter" i] a', 'nav a', 'li a.a-link-normal',
        'input#low-price', 'input#high-price', '[role="checkbox"]', 'input[type="checkbox"]'
    ];
    document.querySelectorAll(filterSelectors.join(', ')).forEach((el) => {
        const txt = getText(el);
        if (el.offsetParent !== null && (txt || el.id) && results.filter(r => r.category === 'filter_or_facet').length < 8) {
            let sel = el.id ? `#${el.id}` : (el.tagName === 'A' ? `a:has-text("${txt.slice(0, 25)}")` : (el.tagName === 'INPUT' ? `input[name="${el.name}"]` : `text="${txt.slice(0, 25)}"`));
            results.push({
                category: 'filter_or_facet',
                box: boxOf(el),
                text: txt.slice(0, 35),
                selector: sel,
                id: el.id || ''
            });
        }
    });

    // 5. Interactive Cards, Search Results & Entity Links (max 12)
    const seenLinks = new Set();
    const candidateLinks = Array.from(document.querySelectorAll('a[href]')).filter(el => {
        if (el.closest('header, nav, footer, [role="navigation"]')) return false;
        const href = el.getAttribute('href') || '';
        if (!href || href.startsWith('#') || href.startsWith('javascript:') || href.startsWith('mailto:')) return false;
        if (/(facebook|twitter|instagram|tiktok|discord|linkedin|privacy|terms|contact|fandom|wiki)/i.test(href)) return false;
        return true;
    });

    candidateLinks.forEach(el => {
        const href = el.getAttribute('href') || '';
        if (seenLinks.has(href)) return;
        seenLinks.add(href);

        let txt = getText(el).replace(/\\s+/g, ' ').trim();
        if (!txt || txt.length < 2) return;

        if (el.offsetParent !== null && results.filter(r => r.category === 'product_link').length < 12) {
            let sel = el.id ? `#${el.id}` : (txt ? `a:has-text("${txt.slice(0, 25)}")` : (href ? `a[href="${href}"]` : 'a'));
            results.push({
                category: 'product_link',
                box: boxOf(el),
                title: txt.slice(0, 45),
                selector: sel,
                href: href.slice(0, 60)
            });
        }
    });


    // 6. Interactive Canvases, Game Viewports & Media Frames (max 2)
    document.querySelectorAll('canvas, iframe, video, #game, #canvas, [id*="game" i]').forEach((el) => {
        if (el.offsetParent !== null && results.filter(r => r.category === 'canvas_or_game_viewport').length < 2) {
            const sel = el.id ? `#${el.id}` : el.tagName.toLowerCase();
            results.push({
                category: 'canvas_or_game_viewport',
                box: boxOf(el),
                selector: sel,
                id: el.id || ''
            });
        }
    });

    // 7. Output / Status areas (max 2)
    document.querySelectorAll('#output, .output, pre, [role="log"], [role="status"]').forEach((el) => {
        if (el.offsetParent !== null && results.filter(r => r.category === 'output_area').length < 2) {
            results.push({
                category: 'output_area',
                box: boxOf(el),
                selector: el.id ? `#${el.id}` : (el.className ? `.${el.className.trim().split(/\\s+/)[0]}` : 'pre'),
                preview_text: (el.innerText || '').slice(0, 100)
            });
        }
    });

    return results;
}
"""





# =============================================================================
# 2b. The page as Markdown, with every control named
# =============================================================================
# Added 2026-09-20. Cloudflare sells this idea as "Markdown for Agents": a page
# handed to a model as clean text instead of raw HTML. Their version converts
# HTML at the edge, costs money, only covers sites behind Cloudflare and is
# read-only. This does the same conversion inside the page the agent already
# controls, for nothing, on any site - and goes further, because reading is
# only half the problem: every control is listed with the name a screen reader
# would announce, so the agent acts on "Compose new message" instead of
# guessing which div to click.
#
# Three things it finds that the older DOM scan could not:
#   - icon-only buttons, named through aria-label / aria-labelledby / <label>
#   - controls inside a web component's shadow DOM (invisible to an ordinary
#     querySelectorAll, which is why custom widgets looked empty)
#   - each control's state: checked, expanded, disabled, and what a field holds

_PAGE_MARKDOWN_JS = """
(limits) => {
    const MAX_TEXT = limits.max_text || 8000;
    const MAX_CONTROLS = limits.max_controls || 60;

    const visible = (el) => {
        const s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
        const r = el.getBoundingClientRect();
        return r.width > 1 && r.height > 1;
    };

    const boxOf = (el) => {
        const r = el.getBoundingClientRect();
        return {
            x: Math.round(r.left), y: Math.round(r.top),
            w: Math.round(r.width), h: Math.round(r.height),
            cx: Math.round(r.left + r.width / 2),
            cy: Math.round(r.top + r.height / 2),
            on_screen: r.bottom > 0 && r.right > 0 &&
                       r.top < window.innerHeight && r.left < window.innerWidth
        };
    };

    const clean = (t) => (t || '').replace(/\\s+/g, ' ').trim();

    // What a person actually clicks. A site that styles its own checkboxes and
    // radios hides the real input (visibility:hidden) and puts the click on the
    // label around it - Myntra's brand and price facets, and most shopping
    // filters, are built exactly this way. Reporting the hidden input's
    // position would send the click nowhere, and skipping it hid every filter
    // on the page, so the visible wrapper is reported instead.
    const CLICKABLE_WRAPPER = /checkbox|radio|label|filter|facet|option|control|toggle|switch/i;
    const clickTargetOf = (el) => {
        if (visible(el)) return el;
        let parent = el.parentElement;
        for (let hops = 0; parent && hops < 4; hops++, parent = parent.parentElement) {
            if (!visible(parent)) continue;
            const cls = typeof parent.className === 'string' ? parent.className : '';
            if (parent.tagName === 'LABEL' || parent.tagName === 'BUTTON' ||
                parent.getAttribute('role') || parent.hasAttribute('onclick') ||
                CLICKABLE_WRAPPER.test(cls)) {
                return parent;
            }
        }
        return null;
    };

    // Where the control sits. A promotional "T-shirts Under Rs.299" link in the
    // menu is not the price filter, and mistaking one for the other produces a
    // wrong result that looks like a right one.
    const regionOf = (el) => {
        try {
            if (el.closest('[class*="filter" i], [id*="filter" i], [class*="facet" i], [class*="refinement" i], aside')) return 'filters';
            if (el.closest('nav, header, [role="navigation"], [class*="nav" i], [class*="menu" i]')) return 'menu';
            if (el.closest('footer')) return 'footer';
        } catch (e) {}
        return '';
    };

    // The accessible name: what a person would call this control. Same order a
    // screen reader uses, so an icon button with no text is still named.
    const nameOf = (el) => {
        let n = clean(el.getAttribute('aria-label'));
        if (!n) {
            const by = el.getAttribute('aria-labelledby');
            if (by) {
                n = clean(by.split(/\\s+/).map(id => {
                    const t = document.getElementById(id);
                    return t ? t.textContent : '';
                }).join(' '));
            }
        }
        if (!n && el.id) {
            const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
            if (lab) n = clean(lab.textContent);
        }
        if (!n && el.closest) {
            const wrap = el.closest('label');
            if (wrap) n = clean(wrap.textContent);
        }
        if (!n) n = clean(el.getAttribute('placeholder'));
        if (!n) n = clean(el.getAttribute('title'));
        if (!n && el.tagName !== 'INPUT') n = clean(el.innerText || el.textContent);
        if (!n) {
            const img = el.querySelector && el.querySelector('img[alt]');
            if (img) n = clean(img.getAttribute('alt'));
        }
        if (!n) n = clean(el.getAttribute('name'));
        return n.slice(0, 80);
    };

    const ROLE_BY_TAG = {
        A: 'link', BUTTON: 'button', SELECT: 'dropdown', TEXTAREA: 'textbox',
        SUMMARY: 'expander', CANVAS: 'canvas', IFRAME: 'frame', VIDEO: 'video', AUDIO: 'audio'
    };
    const roleOf = (el) => {
        const explicit = clean(el.getAttribute('role'));
        if (explicit) return explicit;
        if (el.tagName === 'INPUT') {
            const t = (el.getAttribute('type') || 'text').toLowerCase();
            if (t === 'checkbox' || t === 'radio') return t;
            if (t === 'submit' || t === 'button' || t === 'reset') return 'button';
            if (t === 'file') return 'file upload';
            return 'textbox';
        }
        if (el.isContentEditable) return 'textbox';
        return ROLE_BY_TAG[el.tagName] || 'clickable';
    };

    const selectorOf = (el) => {
        if (el.id) return '#' + CSS.escape(el.id);
        const test = el.getAttribute('data-testid');
        if (test) return '[data-testid="' + test + '"]';
        const aria = el.getAttribute('aria-label');
        if (aria) return el.tagName.toLowerCase() + '[aria-label="' + aria.replace(/"/g, '').slice(0, 40) + '"]';
        const nm = el.getAttribute('name');
        if (nm) return el.tagName.toLowerCase() + '[name="' + nm + '"]';
        if (el.tagName === 'A') {
            const href = el.getAttribute('href') || '';
            if (href && !href.startsWith('#') && !href.includes('"')) return 'a[href="' + href + '"]';
        }
        // `:has-text()` matches the element's RAW text, so a normalised string
        // never matches an element whose text runs over several lines. The
        // first line does.
        const firstLine = (el.innerText || '').split('\\n').map(t => t.trim()).filter(Boolean)[0] || '';
        if (firstLine) {
            return el.tagName.toLowerCase() + ':has-text("' + firstLine.replace(/"/g, '').slice(0, 25) + '")';
        }
        return '';
    };

    const stateOf = (el) => {
        const bits = [];
        const role = roleOf(el);
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') bits.push('disabled');
        if (el.checked || el.getAttribute('aria-checked') === 'true') bits.push('checked');
        const exp = el.getAttribute('aria-expanded');
        if (exp) bits.push(exp === 'true' ? 'expanded' : 'collapsed');
        if (el.getAttribute('aria-selected') === 'true') bits.push('selected');
        if (el.required) bits.push('required');
        // A checkbox's value is the string "on" and means nothing; only a
        // typed-in field's contents are worth reporting.
        if (role === 'textbox' && el.value) bits.push('contains "' + clean(el.value).slice(0, 30) + '"');
        if (el.tagName === 'SELECT' && el.value) bits.push('set to "' + clean(el.value).slice(0, 30) + '"');
        return bits.join(', ');
    };

    const CONTROL_SEL = [
        'a[href]', 'button', 'input', 'select', 'textarea', 'summary', 'canvas',
        '[role="button"]', '[role="link"]', '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
        '[role="menuitem"]', '[role="option"]', '[role="switch"]', '[role="combobox"]',
        '[role="textbox"]', '[role="searchbox"]', '[contenteditable="true"]', '[onclick]'
    ].join(', ');

    const controls = [];
    const seen = new Set();
    const collect = (root) => {
        let found;
        try { found = root.querySelectorAll(CONTROL_SEL); } catch (e) { return; }
        found.forEach((el) => {
            if (controls.length >= MAX_CONTROLS || seen.has(el)) return;
            const target = clickTargetOf(el);
            if (!target || seen.has(target)) return;
            seen.add(el);
            seen.add(target);
            const name = nameOf(el) || nameOf(target);
            const role = roleOf(el);
            if (!name && role !== 'canvas' && role !== 'frame') return;
            controls.push({
                name: name || '(unnamed ' + role + ')',
                role: role,
                selector: selectorOf(target),
                state: stateOf(el),
                region: regionOf(el),
                box: boxOf(target)
            });
        });
        try {
            root.querySelectorAll('*').forEach((el) => {
                if (el.shadowRoot) collect(el.shadowRoot);
            });
        } catch (e) {}
    };
    collect(document);

    // ---- the readable page, as Markdown ------------------------------------
    const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'SVG', 'TEMPLATE', 'IFRAME', 'HEAD']);
    const lines = [];
    let used = 0;
    const push = (s) => {
        if (used >= MAX_TEXT) return;
        lines.push(s);
        used += s.length;
    };

    const inline = (el) => {
        let out = '';
        el.childNodes.forEach((n) => {
            if (n.nodeType === 3) { out += n.textContent; return; }
            if (n.nodeType !== 1 || SKIP.has(n.tagName)) return;
            const t = clean(n.innerText || n.textContent);
            if (n.tagName === 'A' && t) {
                const href = n.getAttribute('href') || '';
                out += href ? '[' + t + '](' + href.slice(0, 120) + ')' : t;
            } else if ((n.tagName === 'STRONG' || n.tagName === 'B') && t) {
                out += '**' + t + '**';
            } else if ((n.tagName === 'EM' || n.tagName === 'I') && t) {
                out += '*' + t + '*';
            } else if (n.tagName === 'CODE' && t) {
                out += '`' + t + '`';
            } else if (n.tagName === 'IMG') {
                const alt = clean(n.getAttribute('alt'));
                if (alt) out += '![' + alt + ']';
            } else {
                out += n.textContent;
            }
        });
        return clean(out);
    };

    const walk = (el, depth) => {
        if (used >= MAX_TEXT || depth > 25 || SKIP.has(el.tagName)) return;
        if (el.nodeType !== 1 || !visible(el)) return;
        const tag = el.tagName;

        if (/^H[1-6]$/.test(tag)) {
            const t = inline(el);
            if (t) push('\\n' + '#'.repeat(parseInt(tag[1], 10)) + ' ' + t + '\\n');
            return;
        }
        if (tag === 'P' || tag === 'BLOCKQUOTE') {
            const t = inline(el);
            if (t) push((tag === 'BLOCKQUOTE' ? '> ' : '') + t + '\\n');
            return;
        }
        if (tag === 'LI') {
            const t = inline(el);
            if (t) push('- ' + t);
            return;
        }
        if (tag === 'PRE') {
            const t = (el.innerText || '').trim();
            if (t) push('```\\n' + t.slice(0, 600) + '\\n```\\n');
            return;
        }
        if (tag === 'HR') { push('\\n---\\n'); return; }
        if (tag === 'TABLE') {
            const rows = Array.from(el.querySelectorAll('tr')).slice(0, 15);
            rows.forEach((tr, i) => {
                const cells = Array.from(tr.querySelectorAll('th, td')).map(c => clean(c.innerText).slice(0, 40));
                if (!cells.length) return;
                push('| ' + cells.join(' | ') + ' |');
                if (i === 0) push('|' + cells.map(() => '---').join('|') + '|');
            });
            push('');
            return;
        }
        if (tag === 'IMG') {
            const alt = clean(el.getAttribute('alt'));
            if (alt) push('![' + alt + ']');
            return;
        }
        if (!el.children.length) {
            const t = clean(el.innerText || el.textContent);
            if (t) push(t);
            return;
        }
        Array.from(el.children).forEach((child) => walk(child, depth + 1));
    };

    const main = document.querySelector('main, [role="main"], article') || document.body;
    if (main) walk(main, 0);

    const out = [];
    let prev = null;
    lines.forEach((l) => {
        const key = l.trim();
        if (key && key === prev) return;
        prev = key;
        out.push(l);
    });

    return {
        title: document.title || '',
        url: location.href,
        viewport: [window.innerWidth, window.innerHeight],
        scroll: [Math.round(window.scrollX), Math.round(window.scrollY)],
        page_height: Math.round(document.documentElement.scrollHeight),
        markdown: out.join('\\n').replace(/\\n{3,}/g, '\\n\\n').trim().slice(0, MAX_TEXT),
        truncated: used >= MAX_TEXT,
        controls: controls
    };
}
"""


class PageMarkdownInput(BaseModel):
    max_text: int = Field(default=8000, description="Largest amount of page text to return, in characters.")
    max_controls: int = Field(default=60, description="Most controls to list.")


class PageMarkdownTool(BaseTool):
    name: str = "read_page_as_markdown"
    description: str = (
        "Reads the whole page as clean Markdown AND lists every control by the name a screen "
        "reader would announce - icon-only buttons, controls inside web components, each with "
        "its state (checked, expanded, disabled, what a field already holds), a selector and a "
        "position. Use it FIRST on any unfamiliar page: it shows what is really there instead "
        "of leaving the agent to guess a selector, and it needs no screenshot and no vision "
        "model. Follow it with click_element on a listed selector, or click_at_position."
    )
    args_schema: Type[BaseModel] = PageMarkdownInput

    def _run(self, max_text: int = 8000, max_controls: int = 60) -> str:
        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()
            try:
                data = await page.evaluate(
                    _PAGE_MARKDOWN_JS,
                    {"max_text": max(500, int(max_text)), "max_controls": max(5, int(max_controls))},
                )
            except Exception as e:
                return f"Could not read the page: {str(e)[:200]}"

            width, height = data.get("viewport") or [0, 0]
            scroll_y = (data.get("scroll") or [0, 0])[1]
            head = [
                f"# {data.get('title') or '(no title)'}",
                f"URL: {data.get('url')}",
                f"Visible area {width}x{height} px; the page is {data.get('page_height')} px tall, "
                f"scrolled to {scroll_y} px. Positions below are inside the visible area.",
            ]

            controls = data.get("controls") or []
            lines = [f"\n## Controls ({len(controls)})"]
            if not controls:
                lines.append("(none found - the page may still be loading, or draws itself on a canvas: use see_page)")
            for c in controls:
                box = c.get("box") or {}
                bits = [f"- `{c.get('name')}` - {c.get('role')}"]
                if c.get("region") == "filters":
                    bits.append("(in the page's filters)")
                elif c.get("region") in ("menu", "footer"):
                    bits.append(f"(in the {c['region']}, not a filter)")
                if c.get("state"):
                    bits.append(f"[{c['state']}]")
                if c.get("selector"):
                    bits.append(f"selector: {c['selector']}")
                bits.append(f"at ({box.get('cx')}, {box.get('cy')})")
                if not box.get("on_screen"):
                    bits.append("- OFF SCREEN, scroll_page first")
                lines.append(" ".join(bits))

            body = ["\n## Page content"]
            body.append(data.get("markdown") or "(no readable text)")
            if data.get("truncated"):
                body.append("\n(text cut off at the limit - scroll_page or raise max_text for more)")
            return "\n".join(head + lines + body)

        return _run_sync(_async_run)


class PerceivePageInput(BaseModel):
    target_description: str = Field(
        default="all",
        description="What you are looking for (e.g. 'search input', 'product listings', 'add to cart button', 'code editor', 'all interactive elements')"
    )


class PerceivePageTool(BaseTool):
    name: str = "perceive_page"
    description: str = (
        "Inspects the live page DOM on ANY website and returns interactive elements: "
        "code editors (CodeMirror, Monaco, Ace, contenteditable), search inputs, buttons, "
        "dropdowns (select), tabs, and output consoles with their CSS selectors and state."
    )
    args_schema: Type[BaseModel] = PerceivePageInput

    def _run(self, target_description: str = "all") -> str:
        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()

            try:
                elements = await page.evaluate(_PERCEIVE_JS)
                title = await page.title()
                url = page.url

                output = [f"Page URL: {url}", f"Page Title: {title}", "--- Detected Page Elements (JSON) ---"]
                for item in elements:
                    # json.dumps produces parseable JSON; str() would give Python repr (single quotes)
                    output.append(json.dumps(item, ensure_ascii=False))
                # A DOM scan that finds almost nothing usually means a canvas or
                # game, a single-page app still rendering, or an overlay the
                # selectors cannot see. A screenshot read by the vision model
                # fills that gap instead of leaving the agent blind.
                if len(elements) < 3 and get_settings().browser_vision_enabled:
                    visual = await _page_vision(
                        page,
                        "The DOM scan found almost no interactive elements. What is actually "
                        "visible, and what can be clicked or typed into?",
                    )
                    if visual:
                        output.append("--- Visual view (screenshot; the DOM scan found few elements) ---")
                        output.append(visual)
                return "\n".join(output)
            except Exception as e:
                return f"Error perceiving page: {str(e)}"

        return _run_sync(_async_run)


class SeePageInput(BaseModel):
    question: str = Field(
        default=(
            "Describe what is visible: popups, overlays, login walls, banners, the main "
            "content and the most important buttons with their visible text."
        ),
        description="What to look for on the page, e.g. 'is a cookie banner covering the results?'",
    )


class SeePageTool(BaseTool):
    name: str = "see_page"
    description: str = (
        "Takes a screenshot of the current browser page and describes what is VISUALLY on "
        "screen: popups, overlays, login walls, cookie banners, canvas or game content, and "
        "buttons that perceive_page's DOM scan can miss, each with the pixel position of its "
        "centre for click_at_position. Use it when perceive_page shows few or confusing "
        "elements, when a click seems to do nothing, or to confirm the visible result of an action."
    )
    args_schema: Type[BaseModel] = SeePageInput

    def _run(self, question: str = "") -> str:
        async def _async_run():
            if not get_settings().browser_vision_enabled:
                return "Page vision is disabled (browser_vision_enabled=false). Use perceive_page."
            controller = get_browser_controller()
            page = await controller.get_active_page()
            text = await _page_vision(
                page, question or SeePageInput.model_fields["question"].default, with_positions=True
            )
            if not text:
                return (
                    "Vision is unavailable right now, so I cannot see the page. Use perceive_page's "
                    "element list instead and do not guess at elements."
                )
            width, height = await _viewport_size(page)
            return (
                f"Page URL: {page.url}\n"
                f"Visible page area: {width}x{height} px. Any (x, y) below is a position inside it, "
                "ready for click_at_position - prefer click_element when the element has a clear "
                "selector, and use the position when it does not.\n"
                f"Visual description:\n{text}"
            )

        return _run_sync(_async_run)


# =============================================================================
# 3. Universal Type Element Tool
# =============================================================================


class TypeElementInput(BaseModel):
    selector: str = Field(..., description="CSS selector or element description (e.g. '.CodeMirror', '#twotabsearchtextbox', 'input[name=\"q\"]', '#cmd_line_args')")
    text: str = Field(..., description="The text, code snippet, or search keyword to type")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing (default False; set True for search submission)")


class TypeElementTool(BaseTool):
    name: str = "type_into_element"
    description: str = (
        "Types or pastes text/code into any input field, search box, textarea, contenteditable container, "
        "or code editor (CodeMirror, Monaco, Ace, Online IDEs) on any website."
    )
    args_schema: Type[BaseModel] = TypeElementInput

    def _run(self, selector: str, text: str, press_enter: bool = False) -> str:
        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()

            active_selector = selector
            # Normalize hidden child elements (e.g. .ace_text-input) to the parent editor container
            if ".ace_text-input" in active_selector or ".ace_content" in active_selector:
                active_selector = active_selector.replace(" .ace_text-input", "").replace(" .ace_content", "") or ".ace_editor"
            elif ".CodeMirror textarea" in active_selector:
                active_selector = active_selector.replace(" textarea", "") or ".CodeMirror"

            # Execute typing via execute_action
            res = await execute_action(page, ActionType.TYPE, selector=active_selector, input_value=text, timeout=2000)

            if not res.success:
                return f"Failed to type into '{selector}': {res.error}"


            if press_enter:
                await execute_action(page, ActionType.PRESS_KEY, input_value="Enter")
                await asyncio.sleep(1.5)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass  # a slow results page is still usable; perceive_page will show it

            return f"Successfully typed text into '{active_selector}' (press_enter={press_enter})."

        return _run_sync(_async_run)


# =============================================================================
# 4. Universal Click Element Tool
# =============================================================================

class ClickElementInput(BaseModel):
    selector: str = Field(..., description="CSS selector, button text, or element identifier to click (e.g. '#control-btn-run', 'button:has-text(\"Run\")', 'button:has-text(\"Add to Cart\")', '#nav-search-submit-button')")


class ClickElementTool(BaseTool):
    name: str = "click_element"
    description: str = "Clicks any button, tab, link, run button, product result, or interactive element on any website."
    args_schema: Type[BaseModel] = ClickElementInput

    def _run(self, selector: str) -> str:
        # Read the confirmation channel here, in the tool's worker thread: the
        # engine sets it as a context variable and asyncio.to_thread carries it
        # into this thread, but not into the coroutine scheduled on the loop.
        from app.utils.confirmation import get_confirmer
        confirmer = get_confirmer()

        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()

            # Safety gate for irreversible actions: place order, pay, checkout.
            # The page URL is checked too, so any click on a payment page asks.
            from app.verifier.engine import requires_human_confirmation
            if requires_human_confirmation(action_type="click", target_description=selector, current_url=page.url):
                gate = await _human_gate(confirmer, f"Click '{selector}'", page.url)
                if gate is not None:
                    return gate

            pages_before = len(await controller.get_all_pages())
            download_marker = controller.download_seq
            res = await execute_action(page, ActionType.CLICK, selector=selector, timeout=5000)

            if not res.success:
                return (
                    f"Failed to click '{selector}': {res.error} "
                    "Call perceive_page and click a selector from its element list instead of guessing."
                )

            await asyncio.sleep(1.5)
            try:
                # Default timeout was 30 s; a slow page after a click stalled the task.
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass

            # A link that opened a new tab: keep working in that tab.
            page = await controller.follow_new_tab(pages_before) or page

            try:
                title = await asyncio.wait_for(page.title(), 3)
            except Exception:
                title = ""
            message = f"Clicked '{selector}'. Current Page: '{title}' (URL: {page.url})"
            started = [d for d in controller.downloads if d["seq"] > download_marker]
            if started:
                message += (
                    f" The click started a download: '{started[0]['name']}' is being saved to "
                    f"{started[0]['path']} (use download_file to wait for it)."
                )
            return message

        return _run_sync(_async_run)


# =============================================================================
# 4a. Click a position inside the page
# =============================================================================
# Added 2026-09-20. click_element needs a selector, so anything the agent could
# SEE but not name was unreachable: canvases and games, shadow-DOM widgets, and
# the custom controls Gmail and other web apps build out of plain divs. Both
# eyes now report exact positions - perceive_page gives every element's box
# from the DOM, see_page gives the centre of what the vision model recognises -
# and this tool clicks them. Positions are CSS pixels from the top-left of the
# visible page area, the same space window.innerWidth/innerHeight describe, so
# they are unaffected by the screen's scaling.


class ClickPositionInput(BaseModel):
    x: int = Field(..., description="X position in pixels from the left edge of the visible page area.")
    y: int = Field(..., description="Y position in pixels from the top edge of the visible page area.")
    description: str = Field(
        default="",
        description="What is being clicked there, e.g. 'the Compose button' - used for the confirmation prompt and the log.",
    )


class ClickPositionTool(BaseTool):
    name: str = "click_at_position"
    description: str = (
        "Clicks a position inside the web page, for something that has no usable selector: "
        "a canvas or game, a custom widget, or a control only the screenshot revealed. "
        "Use the coordinates perceive_page reports in an element's `box` (cx, cy) or the "
        "(x, y) see_page gives - never guessed ones, and take a fresh reading after anything "
        "that changes the page. Prefer click_element whenever the element has a clear selector."
    )
    args_schema: Type[BaseModel] = ClickPositionInput

    def _run(self, x: int, y: int, description: str = "") -> str:
        from app.utils.confirmation import get_confirmer
        confirmer = get_confirmer()

        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()
            label = description.strip() or f"position ({x}, {y})"

            width, height = await _viewport_size(page)
            if not (0 <= x < width and 0 <= y < height):
                return (
                    f"({x}, {y}) is outside the visible page area ({width}x{height} px), so nothing was "
                    "clicked. Scroll the element into view with scroll_page, then take a fresh "
                    "perceive_page or see_page reading - positions are measured from the visible area, "
                    "not from the top of the whole page."
                )

            # Same gate as click_element: an irreversible action (pay, order,
            # checkout) asks the user first, and so does any click on a payment page.
            from app.verifier.engine import requires_human_confirmation
            if requires_human_confirmation(action_type="click", target_description=label, current_url=page.url):
                gate = await _human_gate(confirmer, f"Click {label}", page.url)
                if gate is not None:
                    return gate

            pages_before = len(await controller.get_all_pages())
            download_marker = controller.download_seq
            try:
                await page.mouse.click(x, y)
            except Exception as e:
                return f"Failed to click ({x}, {y}): {str(e)[:200]}"

            await asyncio.sleep(1.5)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass

            page = await controller.follow_new_tab(pages_before) or page
            try:
                title = await asyncio.wait_for(page.title(), 3)
            except Exception:
                title = ""
            logger.info("browser_click_position", x=x, y=y, label=label[:60])
            message = (
                f"Clicked {label} at ({x}, {y}). Current Page: '{title}' (URL: {page.url}). "
                "A position click gives no confirmation of its own - call see_page or perceive_page "
                "to check what actually changed before the next step."
            )
            started = [d for d in controller.downloads if d["seq"] > download_marker]
            if started:
                message += (
                    f" The click started a download: '{started[0]['name']}' is being saved to "
                    f"{started[0]['path']} (use download_file to wait for it)."
                )
            return message

        return _run_sync(_async_run)


# =============================================================================
# 4b. Upload a local file to the page (chat attachments, upload forms)
# =============================================================================
# Added 2026-09-15 after "open gemini and send him the last downloads ppt and
# ask him to explain it": no tool could attach a file, so the agent clicked the
# attach icon, could not drive the Windows file dialog and gave up. Playwright
# fills the file picker directly, so the OS dialog is never involved.

_ATTACH_OPENERS = (
    'button[aria-label*="upload" i]',
    'button[aria-label*="attach" i]',
    'button[aria-label*="add file" i]',
    'button[aria-label*="add photos" i]',
    '[data-testid*="upload" i]',
    '[data-testid*="attach" i]',
    'button:has-text("Upload")',
    'button:has-text("Attach")',
)
_UPLOAD_MENU_ITEMS = (
    '[role="menuitem"]:has-text("Upload")',
    '[role="menuitem"]:has-text("file")',
    'button:has-text("Upload files")',
    'button:has-text("Upload from computer")',
    'text=Upload files',
)
_PRIVATE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".kdbx", ".ovpn", ".ppk")
_PRIVATE_FOLDERS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker"}
# Browser and Windows credential stores
_PRIVATE_NAMES = {"login data", "cookies", "web data", "local state", "key4.db", "logins.json"}


def _looks_private(path: Path) -> bool:
    """Credential and key files are never uploaded to a website.

    Deliberately narrow: ordinary files in AppData or Temp (chat exports,
    downloaded attachments) must still upload.
    """
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    return (
        name.startswith(".env")
        or "id_rsa" in name
        or "id_ed25519" in name
        or "password" in name
        or "credential" in name
        or name in _PRIVATE_NAMES
        or path.suffix.lower() in _PRIVATE_SUFFIXES
        or bool(parts & _PRIVATE_FOLDERS)
    )


async def _click_expecting_chooser(page, locator, file_path: str, wait_ms: int = 4000) -> bool:
    """Click something that should open a file picker, and fill the picker."""
    try:
        async with page.expect_file_chooser(timeout=wait_ms) as chooser_info:
            await locator.click(timeout=3000)
        chooser = await chooser_info.value
        await chooser.set_files(file_path)
        return True
    except Exception:
        return False


async def _set_on_file_input(page, file_path: str) -> bool:
    inputs = page.locator('input[type="file"]')
    if await _count(inputs) == 0:
        return False
    try:
        await inputs.first.set_input_files(file_path, timeout=10000)
        return True
    except Exception:
        return False


async def _attach_file(page, file_path: str, selector: str = "") -> str | None:
    """Attach `file_path` to the page. Returns how it was attached, or None."""
    if not selector and await _set_on_file_input(page, file_path):
        return "the page's file input"
    openers = [selector] if selector else list(_ATTACH_OPENERS)
    for opener in openers:
        try:
            locator = page.locator(opener).first
        except Exception:
            continue
        if await _count(locator) == 0:
            continue
        if await _click_expecting_chooser(page, locator, file_path):
            return f"'{opener}'"
        # The button may have opened a menu ("Upload files") or revealed a file input.
        for item in _UPLOAD_MENU_ITEMS:
            menu_item = page.locator(item).first
            if await _count(menu_item) > 0 and await _click_expecting_chooser(page, menu_item, file_path):
                return f"'{opener}' then '{item}'"
        if await _set_on_file_input(page, file_path):
            return f"'{opener}' then the page's file input"
    return None


class UploadFileInput(BaseModel):
    path: str = Field(..., description="Full path of the file on this computer (get it from list_recent_files or find_files first).")
    selector: str = Field(default="", description="Optional: the page's upload / attach button from perceive_page. Leave empty to find it automatically.")


class UploadFileTool(BaseTool):
    name: str = "upload_file"
    description: str = (
        "Attaches a file from this computer to the current web page: chat attachments (Gemini, ChatGPT, Claude), "
        "upload forms, email attachments. It clicks the page's attach button and fills the file picker itself — "
        "never try to operate the Windows file dialog with see_window or click_window. After it succeeds, check "
        "with see_page that the upload finished, then type and send the message."
    )
    args_schema: Type[BaseModel] = UploadFileInput

    def _run(self, path: str, selector: str = "") -> str:
        from app.utils.confirmation import get_confirmer
        confirmer = get_confirmer()

        from app.tools.local import _resolve_user_path

        # The same resolution as the file tools, so "Downloads\\deck.pptx" and a
        # hand-built %USERPROFILE%\\Desktop path reach the user's real folders.
        file_path = _resolve_user_path(path)
        if not file_path.is_file():
            return f"File not found: {file_path}. Get the exact path with list_recent_files or find_files first."
        if _looks_private(file_path):
            logger.warning("upload_blocked_private_file", path=str(file_path))
            return (
                f"Upload blocked: '{file_path.name}' looks like a private key, password or credential file, so it "
                "could not be uploaded to a website. Tell the user."
            )
        size_mb = file_path.stat().st_size / (1024 * 1024)

        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()
            site = urlparse(page.url).netloc or page.url
            # Sending a file off this computer is always the user's call when
            # someone is there to ask (Local Agent window, extension).
            if confirmer is not None:
                approved = await confirmer.request(
                    f"Upload '{file_path.name}' ({size_mb:.1f} MB) from your computer to {site}",
                    {"file": str(file_path), "url": page.url},
                )
                if not approved:
                    return f"The user declined, so '{file_path.name}' could not be uploaded. Do not retry; report it."
            how = await _attach_file(page, str(file_path), selector.strip())
            if how is None:
                return (
                    "Could not find a way to attach a file on this page. Call perceive_page or see_page to find the "
                    "upload or attach button, then call upload_file again with selector set to that button."
                )
            await asyncio.sleep(2)
            logger.info("file_uploaded_to_page", file=file_path.name, site=site, via=how)
            return (
                f"Attached '{file_path.name}' ({size_mb:.1f} MB) to {site} using {how}. "
                "Check with see_page that the attachment finished uploading, then type the message and send it."
            )

        return _run_sync(_async_run)



# =============================================================================
# 4c. Download a file into the user's Downloads folder
# =============================================================================
# Added 2026-09-17: asked to download something, the agent clicked the link and
# reported success, but with Playwright attached Chrome hands downloads to a
# temporary folder that is deleted on disconnect. The controller now saves every
# download to Downloads (BrowserController._watch_downloads); this tool starts
# one and waits until the file is there.

# A click or link that has not started a download within this time never will.
_DOWNLOAD_START_SECONDS = 10.0


async def _wait_for_download(controller, marker: int, start_timeout: float) -> dict[str, Any] | None:
    """The first download that started after `marker`, or None if none starts in time."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + start_timeout
    while True:
        for record in controller.downloads:
            if record["seq"] > marker:
                return record
        if loop.time() >= deadline:
            return None
        await asyncio.sleep(0.25)


class DownloadFileInput(BaseModel):
    selector: str = Field(default="", description="The page's download link or button (from perceive_page), e.g. 'a:has-text(\"Download\")'. Leave empty when giving url.")
    url: str = Field(default="", description="A direct link to the file, when there is one.")
    timeout_seconds: int = Field(default=120, description="How long to wait for the file to finish downloading (at most 240).")


class DownloadFileTool(BaseTool):
    name: str = "download_file"
    description: str = (
        "Downloads a file from a website into the user's Downloads folder and returns its full path: it clicks "
        "the page's download link or button (selector) or opens a direct file link (url), then waits until the "
        "file is saved. Use it for PDFs, images, documents, music, datasets and program installers. To install "
        "an app (Microsoft Store or a well-known program), use install_app instead."
    )
    args_schema: Type[BaseModel] = DownloadFileInput

    def _run(self, selector: str = "", url: str = "", timeout_seconds: int = 120) -> str:
        from app.utils.confirmation import get_confirmer
        confirmer = get_confirmer()

        async def _async_run():
            if not selector.strip() and not url.strip():
                return "Nothing to download: give the download button's selector or the file's url."
            controller = get_browser_controller()
            page = await controller.get_active_page()
            marker = controller.download_seq
            if selector.strip():
                from app.verifier.engine import requires_human_confirmation
                if requires_human_confirmation(action_type="click", target_description=selector, current_url=page.url):
                    gate = await _human_gate(confirmer, f"Click '{selector}'", page.url)
                    if gate is not None:
                        return gate
                res = await execute_action(page, ActionType.CLICK, selector=selector, timeout=5000)
                if not res.success:
                    return (
                        f"Could not click '{selector}' to start the download: {res.error} "
                        "Call perceive_page and use a selector from its list."
                    )
            else:
                target_url = url.strip()
                if not target_url.startswith(("http://", "https://")):
                    target_url = f"https://{target_url}"
                try:
                    await page.goto(target_url, wait_until="commit", timeout=20000)
                except Exception as e:
                    # Opening a file link starts a download and aborts the navigation.
                    if "download" not in str(e).lower() and "err_aborted" not in str(e).lower():
                        return f"Could not open {target_url}: {str(e)[:200]}"
            record = await _wait_for_download(controller, marker, _DOWNLOAD_START_SECONDS)
            if record is None:
                return (
                    f"No download started within {_DOWNLOAD_START_SECONDS:.0f} s. The link may lead to a page rather than the file: look at "
                    "it with see_page and use that page's real download button, or sign in first if the site asks."
                )
            limit = max(10, min(int(timeout_seconds or 120), 240))
            deadline = asyncio.get_running_loop().time() + limit
            while record["status"] == "downloading" and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.5)
            if record["status"] == "saved":
                size_mb = record["size"] / (1024 * 1024)
                return f"Downloaded: {record['path']} ({size_mb:.1f} MB) from {record['url'][:150]}"
            if record["status"] == "failed":
                return f"The download of '{record['name']}' failed: {record['error']}"
            return (
                f"The download of '{record['name']}' is still running after {limit} s; it will be saved as "
                f"{record['path']} when it finishes."
            )

        return _run_sync(_async_run)


# =============================================================================
# 4d. Add the product on the page to the cart
# =============================================================================
# Added 2026-09-17: "add ... to my cart" kept failing. The button is labelled
# differently on every shop (Add to Cart, ADD TO BAG, Add to Basket), Myntra
# needs a size first, and the result is easy to misread. One tool finds the
# button, picks a size or colour when asked, and checks the cart really changed.

_CART_BADGES = (
    "#nav-cart-count", "[data-cart-count]", ".cart-count", "#cart-count",
    'a[href*="cart"] span', 'button[aria-label*="cart" i]',
)
_CART_BUTTONS = (
    "#add-to-cart-button",                       # Amazon
    "input[name='submit.add-to-cart']",          # Amazon
    "[data-testid*='add-to-cart' i]",
    "[aria-label*='add to cart' i]",
    "[aria-label*='add to bag' i]",
    "input[type='submit'][value*='add to cart' i]",
    "input[type='button'][value*='add to cart' i]",
    "[id*='add-to-cart' i]",
    "[id*='addtocart' i]",
    "[class*='add-to-cart' i]",
    "[class*='addtocart' i]",
    "[class*='add-to-bag' i]",
)
_CART_TEXT = re.compile(r"\badd\s+(?:item\s+)?to\s+(?:cart|bag|basket|trolley)\b", re.IGNORECASE)
_NOT_A_CART_BUTTON = re.compile(r"buy\s*now|check\s*out|place\s+order|\bpay\b", re.IGNORECASE)
_ADDED_SIGNALS = (
    "added to cart", "added to your cart", "added to bag", "added to your bag", "added to basket",
    "item added", "go to cart", "go to bag", "view cart", "view bag", "proceed to checkout", "proceed to buy",
)
_NEEDS_CHOICE = re.compile(
    r"(?:please\s+)?(?:select|choose)\s+(?:a\s+|your\s+)?(size|colou?r|variant|style)", re.IGNORECASE
)
_OPTIONS_JS = """
() => {
  const out = [];
  const boxes = document.querySelectorAll('[class*="size" i], [id*="size" i], [aria-label*="size" i], [data-testid*="size" i]');
  for (const box of boxes) {
    for (const el of box.querySelectorAll('button, li, label, a, span')) {
      if (el.children.length > 1 || el.offsetParent === null) continue;
      const t = (el.innerText || '').trim();
      if (t && t.length <= 8 && !out.includes(t)) out.push(t);
      if (out.length >= 20) return out;
    }
  }
  return out;
}
"""


async def _cart_badge_text(page) -> str | None:
    for sel in _CART_BADGES:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=500):
                return (await el.inner_text()).strip()
        except Exception:
            pass
    return None


def _as_count(text: str | None) -> int | None:
    match = re.search(r"\d+", text or "")
    return int(match.group()) if match else None


async def _usable_cart_button(locator) -> bool:
    try:
        if not await locator.is_visible():
            return False
        label = await locator.evaluate("e => (e.innerText || e.value || e.getAttribute('aria-label') || '')")
    except Exception:
        return False
    if _NOT_A_CART_BUTTON.search(label or ""):
        return False
    try:
        return await locator.is_enabled()
    except Exception:
        return True


async def _find_cart_button(page, learned: list[str]):
    """The visible add-to-cart control and how it was found, or (None, "")."""
    for sel in [*learned, *_CART_BUTTONS]:
        try:
            loc = page.locator(sel)
            n = min(await _count(loc), 5)
        except Exception:
            continue
        for i in range(n):
            if await _usable_cart_button(loc.nth(i)):
                return loc.nth(i), sel
    for loc, how in (
        (page.get_by_role("button", name=_CART_TEXT), "button labelled add to cart"),
        (page.get_by_text(_CART_TEXT), "text add to cart"),
    ):
        n = min(await _count(loc), 8)
        for i in range(n):
            if await _usable_cart_button(loc.nth(i)):
                return loc.nth(i), how
    return None, ""


async def _press(locator) -> None:
    try:
        await locator.scroll_into_view_if_needed(timeout=3000)
    except Exception:
        pass
    try:
        await locator.click(timeout=4000)
        return
    except Exception:
        pass
    try:
        await locator.click(force=True, timeout=2000)
        return
    except Exception:
        pass
    await locator.evaluate("e => e.click()")


async def _choose_option(page, option: str) -> bool:
    """Click the size / colour / variant shown exactly as `option`."""
    exact = re.compile(rf"^\s*{re.escape(option.strip())}\s*$", re.IGNORECASE)
    for loc in (
        page.get_by_role("button", name=exact),
        page.get_by_role("radio", name=exact),
        page.get_by_role("option", name=exact),
        page.get_by_text(exact),
    ):
        n = min(await _count(loc), 10)
        for i in range(n):
            candidate = loc.nth(i)
            try:
                if await candidate.is_visible():
                    await _press(candidate)
                    await asyncio.sleep(0.8)
                    return True
            except Exception:
                continue
    selects = page.locator("select")
    for i in range(min(await _count(selects), 5)):
        try:
            await selects.nth(i).select_option(label=option.strip(), timeout=1500)
            return True
        except Exception:
            continue
    return False


class AddToCartInput(BaseModel):
    product_selector: str = Field(default="", description="Optional: a product link or card to open first (from perceive_page), when the page is a search or listing page.")
    option: str = Field(default="", description="Optional: the size, colour or variant to pick first, exactly as the page shows it (e.g. 'M', '9', 'Black').")


class AddToCartTool(BaseTool):
    name: str = "add_to_cart"
    description: str = (
        "Adds the product on the current page to the shop's cart or bag (Amazon, Flipkart, Myntra and other shops): "
        "finds the 'Add to Cart' / 'Add to Bag' button, clicks it and checks that the item was really added. Give "
        "product_selector to open a product from a search page first, and option to pick the size or colour the "
        "site requires. It never buys, pays or checks out."
    )
    args_schema: Type[BaseModel] = AddToCartInput

    def _run(self, product_selector: str = "", option: str = "") -> str:
        from app.utils.confirmation import get_confirmer
        confirmer = get_confirmer()

        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()

            if product_selector.strip():
                pages_before = len(await controller.get_all_pages())
                res = await execute_action(page, ActionType.CLICK, selector=product_selector, timeout=5000)
                if not res.success:
                    return (
                        f"Could not open the product '{product_selector}': {res.error} "
                        "Call perceive_page and pick a product link from its list."
                    )
                await asyncio.sleep(1.5)
                page = await controller.follow_new_tab(pages_before) or page
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=8000)
                except Exception:
                    pass

            from app.verifier.engine import requires_human_confirmation
            if requires_human_confirmation(action_type="click", target_description="add to cart", current_url=page.url):
                gate = await _human_gate(confirmer, f"Add to cart on {page.url}", page.url)
                if gate is not None:
                    return gate

            if option.strip() and not await _choose_option(page, option):
                return (
                    f"Could not find the option '{option}' on this page, so nothing was added. "
                    "Look at the available sizes or colours with see_page."
                )

            try:
                learned = list((await domain_store.get_domain_memory(page.url)).get("add_to_cart_selectors") or [])
            except Exception:
                learned = []
            button, how = await _find_cart_button(page, learned)
            if button is None:
                return (
                    "No 'Add to Cart' or 'Add to Bag' button is visible on this page, so nothing was added. If this is "
                    "a search or listing page, call add_to_cart with product_selector set to a product from "
                    "perceive_page; otherwise look with see_page (the item may be out of stock)."
                )

            before = await _cart_badge_text(page)
            await _press(button)
            await asyncio.sleep(2.5)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass
            after = await _cart_badge_text(page)
            try:
                body = (await page.locator("body").inner_text(timeout=3000))[:6000].lower()
            except Exception:
                body = ""
            try:
                title = await asyncio.wait_for(page.title(), 3)
            except Exception:
                title = ""
            site = urlparse(page.url).netloc or page.url

            before_n, after_n = _as_count(before), _as_count(after)
            count_went_up = after_n is not None and (before_n is None or after_n > before_n)
            if count_went_up or any(signal in body for signal in _ADDED_SIGNALS):
                if how.startswith(("#", "[", "input")):
                    try:
                        await domain_store.record_domain_learning(page.url, add_to_cart_selectors=[how], count_run=False)
                    except Exception as e:
                        logger.warning("cart_selector_learning_failed", error=str(e)[:120])
                detail = f"cart count {before or 0} → {after}" if count_went_up else "the site confirmed it"
                logger.info("added_to_cart", site=site, via=how)
                return f"Added to cart: '{title[:100]}' on {site} ({detail}). Nothing was bought."

            choice = _NEEDS_CHOICE.search(body)
            if choice:
                try:
                    options = await page.evaluate(_OPTIONS_JS)
                except Exception:
                    options = []
                listed = f" Options on the page: {', '.join(options)}." if options else ""
                return (
                    f"Not added yet: the site asks to choose a {choice.group(1).lower()} first.{listed} Call add_to_cart "
                    "again with option set to the one the user wants; if they did not say, ask in your report."
                )
            return (
                f"Clicked the add-to-cart button ({how}) but could not confirm the item was added "
                f"(cart count {before or 'not shown'} → {after or 'not shown'}). Check with see_page before trying "
                "again, so the item is not added twice."
            )

        return _run_sync(_async_run)

# =============================================================================
# 5. Universal Select Option Tool (Dropdowns / Language Pickers)
# =============================================================================

class SelectOptionInput(BaseModel):
    selector: str = Field(..., description="CSS selector for the select/dropdown element (e.g. 'select', '#select-lang', 'select[name=\"lang\"]')")
    option_value_or_label: str = Field(..., description="The value or text label of the option to choose (e.g. 'C++', 'Python3', 'Java', 'Electronics')")


class SelectOptionTool(BaseTool):
    name: str = "select_dropdown_option"
    description: str = "Selects an option from any standard dropdown menu (<select>) or language selector on any website."
    args_schema: Type[BaseModel] = SelectOptionInput

    def _run(self, selector: str, option_value_or_label: str) -> str:
        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()
            res = await execute_action(page, ActionType.SELECT_OPTION, selector=selector, input_value=option_value_or_label)
            if res.success:
                return f"Successfully selected option '{option_value_or_label}' in '{selector}'."
            return f"Failed to select option '{option_value_or_label}': {res.error}"

        return _run_sync(_async_run)



# =============================================================================
# 5. Universal Press Key Tool
# =============================================================================

class PressKeyInput(BaseModel):
    key: str = Field(default="Enter", description="Key name to press (e.g. 'Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd', 'Enter', 'Escape', 'Tab')")


class PressKeyTool(BaseTool):
    name: str = "press_key"
    description: str = (
        "Presses a keyboard key on the active page (e.g. 'Space', 'ArrowUp', 'ArrowDown', "
        "'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd', 'Enter', 'Escape'). "
        "Essential for playing games, interacting with web apps, submitting searches, or dismissing popups."
    )
    args_schema: Type[BaseModel] = PressKeyInput

    def _run(self, key: str = "Enter") -> str:
        from app.utils.confirmation import get_confirmer
        confirmer = get_confirmer()

        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()
            # Enter on a checkout or payment page can submit an order.
            from app.verifier.engine import requires_human_confirmation
            if key.strip().lower() in ("enter", "return", "numpadenter") and requires_human_confirmation(
                action_type="press_key", target_description=key, current_url=page.url
            ):
                gate = await _human_gate(confirmer, f"Press {key} on {page.url}", page.url)
                if gate is not None:
                    return gate
            res = await execute_action(page, ActionType.PRESS_KEY, input_value=key)
            if res.success:
                return f"Successfully pressed key '{key}'."
            return f"Failed to press key '{key}': {res.error}"

        return _run_sync(_async_run)


# =============================================================================
# 6. Universal Scroll Tool
# =============================================================================

class ScrollInput(BaseModel):
    direction: str = Field(default="down", description="Scroll direction: 'down' or 'up'")
    amount: int = Field(default=600, description="Pixels to scroll (default 600)")


class ScrollPageTool(BaseTool):
    name: str = "scroll_page"
    description: str = "Scrolls the page up or down to reveal lazy-loaded products, buttons, or reviews."
    args_schema: Type[BaseModel] = ScrollInput

    def _run(self, direction: str = "down", amount: int = 600) -> str:
        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()
            delta = amount if direction == "down" else -amount
            res = await execute_action(page, ActionType.SCROLL, input_value=str(delta))
            if res.success:
                return f"Scrolled page {direction} by {amount}px."
            return f"Failed to scroll: {res.error}"

        return _run_sync(_async_run)


# =============================================================================
# 7. Universal Extract Page Data Tool
# =============================================================================

class ExtractDataInput(BaseModel):
    query_description: str = Field(default="all", description="Specific data to inspect: 'cart_badge', 'title', 'price', 'product_names', 'all'")


class ExtractPageDataTool(BaseTool):
    name: str = "extract_page_data"
    description: str = "Extracts text content, prices, cart badge count, and page title from any website for verification."
    args_schema: Type[BaseModel] = ExtractDataInput

    def _run(self, query_description: str = "all") -> str:
        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()

            title = await page.title()
            url = page.url
            
            cart_text = await _cart_badge_text(page) or "N/A"

            # Body text sample
            body_sample = ""
            try:
                body_sample = (await page.locator("body").inner_text(timeout=3000))[:1000]
            except Exception:
                pass

            return (
                f"URL: {url}\n"
                f"Title: {title}\n"
                f"Cart Badge Count: {cart_text}\n"
                f"Page Text Sample:\n{body_sample}"
            )

        return _run_sync(_async_run)


# =============================================================================
# 8. Recall Domain Memory Tool (Self-Improvement)
# =============================================================================

class RecallMemoryInput(BaseModel):
    domain: str = Field(..., description="Domain name or URL to query memory for (e.g. 'amazon.in', 'flipkart.com', 'ebay.com')")


class RecallDomainMemoryTool(BaseTool):
    name: str = "recall_domain_memory"
    description: str = "Recalls learned strategies, working selectors, overlay dismissal rules, and past lessons for any specific website."
    args_schema: Type[BaseModel] = RecallMemoryInput

    def _run(self, domain: str) -> str:
        async def _async_run():
            mem = await domain_store.get_domain_memory(domain)
            return (
                f"Domain: {mem.get('domain')}\n"
                f"Known Search Selectors: {mem.get('search_selectors')}\n"
                f"Known Product Selectors: {mem.get('product_link_selectors')}\n"
                f"Known Add to Cart Selectors: {mem.get('add_to_cart_selectors')}\n"
                f"Known Overlay Dismiss Rules: {mem.get('popup_dismiss_selectors')}\n"
                f"Past Lessons & Tips: {mem.get('general_tips')}\n"
                f"Experience: {mem.get('successful_runs')} successes, {mem.get('failed_runs')} failures."
            )

        return _run_sync(_async_run)


# =============================================================================
# 9. Human Login Gate — pause the task until the user finishes signing in
# =============================================================================

# Text/url signals that mean "this page wants a login"
_LOGIN_URL_PATTERNS = (
    "login", "signin", "sign-in", "sign_up", "signup", "accounts.google",
    "auth", "oauth", "session/new",
)
# Elements that only appear while signed OUT
_LOGIN_EL_JS = """
() => {
    const sels = [
        'button', 'a', 'input[type="submit"]', 'input[type="button"]'
    ];
    const out = [];
    document.querySelectorAll(sels.join(', ')).forEach((el) => {
        if (el.offsetParent === null) return;
        const t = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().toLowerCase();
        if (!t) return;
        if ((t === 'log in' || t === 'login' || t === 'sign in' || t.includes('sign up')
             || t.includes('log in to')) && out.length < 5) {
            out.push(t);
        }
    });
    const body = (document.body.innerText || '').toLowerCase();
    const pageHint = /log in to (continue|create|edit|get)|sign in to (continue|view)|you need to (log in|sign in)|authentication required/.test(body);
    return { loginButtons: out, pageHint };
}
"""


class WaitForLoginInput(BaseModel):
    timeout_seconds: int = Field(
        default=180,
        description="Max seconds to wait for the user to complete the login (default 180)."
    )


class WaitForLoginTool(BaseTool):
    name: str = "wait_for_login"
    description: str = (
        "Pauses the task and waits until the user completes the login on the current page "
        "(a login form, 'Log in' overlay, or 'log in to continue' gate). "
        "Bring the browser window to the front, sign in (2FA included), and the tool returns "
        "AUTOMATICALLY once the page is signed in — the task then continues where it left off. "
        "Use it whenever a site blocks the requested action behind a login."
    )
    args_schema: Type[BaseModel] = WaitForLoginInput

    def _run(self, timeout_seconds: int = 180) -> str:
        async def _async_run():
            controller = get_browser_controller()
            page = await controller.get_active_page()
            loop = asyncio.get_event_loop()
            start = loop.time()

            try:
                await page.bring_to_front()
            except Exception:
                pass

            while loop.time() - start < timeout_seconds:
                try:
                    url = page.url or ""
                    if not any(p in url.lower() for p in _LOGIN_URL_PATTERNS):
                        probe = await page.evaluate(_LOGIN_EL_JS)
                        if not probe["loginButtons"] and not probe.get("pageHint"):
                            elapsed = int(loop.time() - start)
                            # Learning: remember that this domain needed a manual
                            # login and that the session is now persisted in the
                            # agent-linked Chrome profile — future runs skip it.
                            try:
                                await domain_store.record_domain_learning(
                                    url or (page.url or ""),
                                    general_tips=(
                                        "LOGIN SESSION SAVED: user completed sign-in manually in the "
                                        "agent Chrome window. The session cookie persists in the linked "
                                        "profile — future tasks on this domain should NOT need a new login. "
                                        "If a login gate still appears, call wait_for_login; the saved "
                                        "session usually restores it automatically."
                                    ),
                                    success=True,
                                    count_run=False,  # a tip, not a finished run
                                )
                            except Exception as e:
                                logger.warning("login_learning_record_failed", error=str(e)[:120])
                            return (
                                f"LOGIN COMPLETE (after {elapsed}s) — the page is signed in. "
                                "Continue the task from where it was blocked. "
                                "The session is saved in the agent profile; you will NOT be asked again for this site."
                            )
                except Exception:
                    pass  # page navigating mid-login — keep waiting
                await asyncio.sleep(2.0)

            return (
                f"TIMEOUT after {timeout_seconds}s — still not signed in. "
                "Report clearly that the task is blocked on login and what was completed so far."
            )

        return _run_sync(_async_run)
