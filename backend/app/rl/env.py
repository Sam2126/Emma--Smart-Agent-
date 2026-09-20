"""
Gymnasium-compliant Browser Gym Environment for RL (GRPO / PPO).

Implements the standard 5-tuple step API:
    obs, reward, terminated, truncated, info = env.step(action)

Features:
- Dual backends:
    - Real browser controller (live Playwright smoke testing)
    - SandboxBrowserBackend (hermetic, deterministic, millisecond-fast fixtures for training)
- Strict `sandbox_only=True` enforcement guard (raises SandboxViolationError if attempted on live domains)
- Step truncation at max_steps (default 25)
- Integrated ProcessRewardModel (PRM) for dense step rewards and anti-gaming penalties
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
import structlog

from app.rl.fixtures.site_fixtures import FIXTURE_DOM_STATES
from app.rl.reward import ProcessRewardModel, RewardBreakdown
from app.verifier.rule_checks import CheckResult

logger = structlog.get_logger(__name__)


class SandboxViolationError(PermissionError):
    """Raised when an action or environment reset violates the sandbox-only safety invariant."""
    pass


@dataclass
class AgentAction:
    """Standardized action representation for RL agent."""
    type: str  # "click", "type", "select", "press", "scroll", "navigate"
    selector: str = ""
    text: str = ""
    url: str = ""
    confirmed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentAction:
        return cls(
            type=data.get("type", data.get("action_type", "")),
            selector=data.get("selector", data.get("selector_used", "")),
            text=data.get("text", data.get("input_value", "")),
            url=data.get("url", ""),
            confirmed=data.get("confirmed", False),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "selector": self.selector,
            "text": self.text,
            "url": self.url,
            "confirmed": self.confirmed,
            "metadata": self.metadata,
        }


@dataclass
class Observation:
    """Gymnasium observation of browser page state under token budget."""
    url: str
    title: str
    visible_text: str
    elements: list[dict[str, Any]]
    dom_hash: str
    extracted_values: dict[str, Any] = field(default_factory=dict)
    milestones: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "visible_text": self.visible_text[:1000],
            "elements": self.elements[:50],  # Token-budget friendly
            "dom_hash": self.dom_hash,
            "extracted_values": self.extracted_values,
            "milestones": self.milestones,
        }


class SandboxBrowserBackend:
    """
    Hermetic in-memory browser backend driven by deterministic fixture transitions.
    Executes in <1ms per step without internet connectivity or live website risks.
    """

    def __init__(self) -> None:
        self.current_page: str = "home"
        self.state_data: dict[str, Any] = copy.deepcopy(FIXTURE_DOM_STATES["home"])

    def navigate(self, url: str) -> Observation:
        if "s?k=" in url or "/search" in url:
            if "filter=" in url or "rh=" in url:
                self.current_page = "filtered_results"
            else:
                self.current_page = "results"
        elif "/dp/" in url or "product" in url:
            self.current_page = "product"
        elif "/cart" in url:
            self.current_page = "cart"
        else:
            self.current_page = "home"

        self.state_data = copy.deepcopy(FIXTURE_DOM_STATES[self.current_page])
        return self.get_observation()

    def step_action(self, action: AgentAction) -> Observation:
        act_type = action.type.lower()
        sel = action.selector.lower()

        if act_type in ("navigate", "goto"):
            return self.navigate(action.url or action.text)

        elif act_type in ("type", "type_search_query"):
            # Typing query into search box
            if "#twotabsearchtextbox" in sel or "search" in sel or "q" in sel:
                self.current_page = "results"
                self.state_data = copy.deepcopy(FIXTURE_DOM_STATES["results"])
                if action.text:
                    self.state_data["extracted_values"]["query"] = action.text
                    self.state_data["title"] = f"Sandbox Store: Search Results for '{action.text}'"

        elif act_type == "click":
            if "add-to-cart" in sel or "cart" in sel:
                # Adding product to cart
                self.current_page = "cart"
                self.state_data = copy.deepcopy(FIXTURE_DOM_STATES["cart"])
            elif "product" in sel or "wh-1000xm5" in sel or "b09xyz" in sel or "product-link" in sel:
                # Clicking product link
                self.current_page = "product"
                self.state_data = copy.deepcopy(FIXTURE_DOM_STATES["product"])
            elif "facet" in sel or "filter" in sel or ("brand" in sel and "button" in sel) or "button:has-text('sony')" in sel:
                # Clicking facet filter
                self.current_page = "filtered_results"
                self.state_data = copy.deepcopy(FIXTURE_DOM_STATES["filtered_results"])
            elif "nav-search-submit" in sel:
                self.current_page = "results"
                self.state_data = copy.deepcopy(FIXTURE_DOM_STATES["results"])

        return self.get_observation()

    def get_observation(self) -> Observation:
        raw_dom = f"{self.state_data['url']}|{self.state_data['title']}|{len(self.state_data['elements'])}"
        dom_hash = hashlib.md5(raw_dom.encode("utf-8")).hexdigest()[:16]
        return Observation(
            url=self.state_data["url"],
            title=self.state_data["title"],
            visible_text=self.state_data["visible_text"],
            elements=self.state_data["elements"],
            dom_hash=dom_hash,
            extracted_values=dict(self.state_data.get("extracted_values", {})),
        )


class LiveBrowserBackend:
    """
    Real-browser backend adapter (spec §7.2): wraps BrowserController + the
    shared action executor / page-state flattener behind the same sync
    navigate()/step_action()/get_observation() interface as SandboxBrowserBackend.

    Use for smoke tests and human demos ONLY — never for GRPO/PPO training
    (the trainers enforce sandbox_only=True at construction). Playwright calls
    are async; this adapter drives its own event loop per call to keep the Gym
    step API synchronous.
    """

    def __init__(self, cdp_endpoint: str = "http://127.0.0.1:9222") -> None:
        self.cdp_endpoint = cdp_endpoint
        self._controller: Any = None
        self._current_state: Any = None

    def _get_controller(self) -> Any:
        if self._controller is None:
            from app.browser.controller import BrowserController

            self._controller = BrowserController()
            # connect() is async; run it on a private loop synchronously
            asyncio.run(self._controller.connect(self.cdp_endpoint))
        return self._controller

    def _run_async(self, coro: Any) -> Any:
        return asyncio.run(coro)

    def _extract_observation(self) -> Observation:
        controller = self._get_controller()

        async def _extract() -> Observation:
            from app.browser.page_state import extract_page_state

            page = await controller.get_active_page()
            state = await extract_page_state(page)
            dom_hash = hashlib.md5(f"{state.url}|{state.title}".encode("utf-8")).hexdigest()[:16]
            return Observation(
                url=state.url,
                title=state.title,
                visible_text=state.to_llm_context(max_tokens=1000),
                elements=list(getattr(state, "elements", []) or []),
                dom_hash=dom_hash,
                extracted_values=dict(getattr(state, "extracted_values", {}) or {}),
            )

        self._current_state = self._run_async(_extract())
        return self._current_state

    def navigate(self, url: str) -> Observation:
        controller = self._get_controller()

        async def _nav() -> None:
            from app.browser.actions import ActionType, execute_action

            page = await controller.get_active_page()
            await execute_action(page, ActionType.NAVIGATE, input_value=url)

        self._run_async(_nav())
        return self._extract_observation()

    def step_action(self, action: AgentAction) -> Observation:
        controller = self._get_controller()

        async def _step() -> None:
            from app.browser.actions import ActionType, execute_action

            page = await controller.get_active_page()
            mapping = {
                "click": ActionType.CLICK,
                "type": ActionType.TYPE,
                "type_search_query": ActionType.TYPE,
                "select": ActionType.SELECT_OPTION,
                "press": ActionType.PRESS_KEY,
                "scroll": ActionType.SCROLL,
                "navigate": ActionType.NAVIGATE,
                "goto": ActionType.NAVIGATE,
            }
            act_type = mapping.get(action.type.lower())
            if act_type is None:
                return
            input_value = action.text if act_type != ActionType.NAVIGATE else (action.url or action.text)
            await execute_action(page, act_type, selector=action.selector, input_value=input_value)

        self._run_async(_step())
        return self._extract_observation()

    def get_observation(self) -> Observation:
        if self._current_state is None:
            return self._extract_observation()
        return self._current_state


class BrowserGymEnv:
    """
    Gymnasium-compatible browser automation environment.

    Usage:
        env = BrowserGymEnv(sandbox_only=True)
        obs, info = env.reset("Search for Sony headphones and add to cart")
        action = {"type": "type", "selector": "#twotabsearchtextbox", "text": "Sony headphones"}
        obs, reward, terminated, truncated, info = env.step(action)

    Backends (spec §7.2):
        - "sandbox" (default): SandboxBrowserBackend — hermetic fixture pages, no
          network. Required for all GRPO/PPO training.
        - "browser": LiveBrowserBackend — real Chrome via BrowserController (CDP).
          Smoke tests / human demos only; never used by the training rail.
        - Any object exposing navigate()/step_action()/get_observation() with the
          same signatures as SandboxBrowserBackend.
    """

    def __init__(
        self,
        sandbox_only: bool = True,
        max_steps: int = 25,
        deprecated_selectors: set[str] | list[str] | None = None,
        backend: Any | None = None,
    ) -> None:
        self.sandbox_only = sandbox_only
        self.max_steps = max_steps
        if backend is None:
            backend = SandboxBrowserBackend()
        self.backend = backend
        self.prm = ProcessRewardModel(deprecated_selectors=deprecated_selectors)
        self.step_count: int = 0
        self.current_task: str = ""
        self.target_domain: str = "sandbox://ecommerce"
        self.current_obs: Observation | None = None
        self.terminated: bool = False
        self.truncated: bool = False
        self.task_success: bool = False

    def reset(self, task: str | dict[str, Any]) -> tuple[Observation, dict[str, Any]]:
        """
        Reset environment for a new task.

        Raises SandboxViolationError if sandbox_only is True and a live domain is requested.
        """
        if isinstance(task, dict):
            task_desc = task.get("instruction", "")
            domain = task.get("domain", "sandbox://ecommerce")
        else:
            task_desc = str(task)
            domain = "sandbox://ecommerce"

        # Hard safety rail: check sandbox domain
        if self.sandbox_only:
            self._assert_sandbox_domain(domain)

        self.current_task = task_desc
        self.target_domain = domain
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.task_success = False
        self.prm.reset()

        # Reset backend to home state
        self.current_obs = self.backend.navigate("sandbox://ecommerce/")

        info = {
            "task": self.current_task,
            "target_domain": self.target_domain,
            "sandbox_mode": self.sandbox_only,
            "step": 0,
        }
        return self.current_obs, info

    @staticmethod
    def _assert_sandbox_domain(domain: str) -> None:
        """Raise SandboxViolationError unless `domain` is a hermetic sandbox target."""
        d = (domain or "").lower()
        live_domains = ("amazon.in", "amazon.com", "flipkart.com", "ebay.com", "walmart.com")
        # Live-domain check first: a "sandbox://" prefix must not smuggle a real site.
        if any(ld in d for ld in live_domains):
            raise SandboxViolationError(
                f"Cannot reset to live domain '{domain}' when sandbox_only=True. "
                "All GRPO and PPO training rollouts must use sandbox fixtures."
            )
        is_sandbox = (
            d.startswith("sandbox://")
            or "localhost" in d
            or "127.0.0.1" in d
            or d.rstrip("/") in ("", "sandbox")
        )
        if not is_sandbox:
            raise SandboxViolationError(
                f"Cannot reset to live domain '{domain}' when sandbox_only=True. "
                "All GRPO and PPO training rollouts must use sandbox fixtures."
            )

    def step(
        self, action: AgentAction | dict[str, Any]
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """
        Execute one action in the environment.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step in an environment that has terminated or truncated. Call reset() first.")

        if isinstance(action, dict):
            agent_act = AgentAction.from_dict(action)
        else:
            agent_act = action

        self.step_count += 1
        state_before = self.current_obs

        # Execute in sandbox backend
        state_after = self.backend.step_action(agent_act)
        self.current_obs = state_after

        # Check goal conditions
        task_lower = self.current_task.lower()
        terminal_pass = False
        terminal_fail = False

        # Evaluate progress rules
        rule_results: list[CheckResult] = []

        # Check URL change
        if state_after.url != state_before.url:
            rule_results.append(CheckResult("url_changed", True, details="URL updated"))

        # Check search goal
        if ("search" in task_lower) and ("/s?k=" in state_after.url or "results" in state_after.url):
            rule_results.append(CheckResult("url_contains:search", True, details="Navigated to search results"))
            if "filter" not in task_lower and "cart" not in task_lower:
                terminal_pass = True

        # Check filter goal
        if ("filter" in task_lower or "facet" in task_lower or "sony" in task_lower) and "filter=applied" in state_after.url:
            rule_results.append(CheckResult("facet_applied", True, details="Filter facet successfully applied"))
            if "cart" not in task_lower:
                terminal_pass = True

        # Check add-to-cart goal
        if ("cart" in task_lower or "add" in task_lower) and state_after.url.endswith("/cart"):
            rule_results.append(CheckResult("numeric_increment:cart_count", True, details="Cart count incremented"))
            terminal_pass = True

        # Truncation check
        if self.step_count >= self.max_steps:
            self.truncated = True
            if not terminal_pass:
                terminal_fail = True

        # Compute PRM dense reward
        rb: RewardBreakdown = self.prm.step_reward(
            state_before=state_before,
            state_after=state_after,
            action=agent_act.to_dict(),
            rules=rule_results,
            is_terminal=(terminal_pass or terminal_fail),
            terminal_pass=terminal_pass,
            terminal_fail=terminal_fail,
        )

        # Check if anti-gaming requested abort
        if rb.abort_requested:
            self.terminated = True
            terminal_fail = True
            rb.total -= 2.0  # Apply terminal fail consequence
            rb.terminal -= 2.0
            rb.details.append("-2.0 Terminal FAIL from anti-gaming abort")
            # Keep the PRM's accumulated total consistent with the returned reward
            self.prm.total_reward -= 2.0

        if terminal_pass:
            self.terminated = True
            self.task_success = True
        elif terminal_fail and not self.truncated:
            self.terminated = True

        info = {
            "step": self.step_count,
            "reward_breakdown": rb.to_dict(),
            "gaming_flags": rb.gaming_flags,
            "task_success": self.task_success,
            "total_accumulated_reward": self.prm.total_reward,
        }

        return self.current_obs, rb.total, self.terminated, self.truncated, info

    def close(self) -> None:
        """Close the environment."""
        pass
