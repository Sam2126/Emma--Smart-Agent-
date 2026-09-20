"""
LLM-as-judge — second-tier verification, as text and as vision.

Text judge: one routed LiteLLM call (role "judge", see app.utils.llm) that reads
the page state and decides PASS/FAIL. It used to go through a separate
langchain-groq ChatGroq client — the only LLM call in verification that
bypassed the shared LiteLLM key pool, rate-limit handling and Gemini fallback.

Vision judge: the same decision made from a screenshot of the final page,
through app.utils.vision (Gemini when configured, Groq otherwise). It sees what
DOM text misses — overlays, rendered results, canvas content — which is what
lets task verification work on sites that have no hand-written rules.

Neither is the sole arbiter; VerificationEngine decides how verdicts combine.
Never let a judge alone gate an irreversible action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import structlog

from app.browser.page_state import PageState

logger = structlog.get_logger(__name__)


@dataclass
class LLMJudgment:
    """Result of an LLM verification judgment."""
    passed: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    raw_response: str = ""
    source: str = "text"  # "text" | "vision"


_STEP_VERIFICATION_PROMPT = """You are a verification judge for a browser automation agent.

Your job: Determine whether a browser action succeeded based on the page state before and after the action.

You must respond in EXACTLY this format (no markdown, no extra text):
VERDICT: PASS or FAIL
CONFIDENCE: 0.0 to 1.0
REASONING: one paragraph explaining your judgment

Rules:
- Be conservative — if unsure, say FAIL with low confidence
- Look for concrete evidence in the page state (URL changes, element presence, text content)
- Do NOT assume success just because no error is reported
- A page showing a CAPTCHA, login form, or error message means FAIL"""

_TASK_VERIFICATION_PROMPT = """You are a verification judge for a browser automation agent.

Your job: Determine whether an entire task was completed successfully based on the original instruction and the final page state.

You must respond in EXACTLY this format (no markdown, no extra text):
VERDICT: PASS or FAIL
CONFIDENCE: 0.0 to 1.0
REASONING: one paragraph explaining your judgment

Rules:
- The task must be FULLY completed, not just partially
- Look for concrete evidence: correct product in cart, right page loaded, expected text visible
- If the page shows an error, CAPTCHA, or unexpected content, that's a FAIL
- Be conservative — partial completion is still FAIL"""

_VISUAL_TASK_PROMPT = """You are a verification judge for a browser automation agent. The image is a screenshot of the browser page AFTER the agent finished.

Decide whether the original instruction was FULLY completed, judging only from what is visible in the screenshot.

Respond in EXACTLY this format (no markdown, no extra text):
VERDICT: PASS or FAIL
CONFIDENCE: 0.0 to 1.0
REASONING: one paragraph naming the concrete visual evidence

Rules:
- A login wall, CAPTCHA, error page, empty results or an unrelated page is FAIL
- A popup or modal still covering the result is FAIL unless the instruction was about that popup
- Partial completion is FAIL; if the screenshot cannot show the result, say FAIL with low confidence"""


class LLMJudge:
    """Verification judge using routed LiteLLM calls and the vision chain."""

    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = completion_fn

    async def verify_step(
        self,
        action_description: str,
        before: PageState | None,
        after: PageState,
    ) -> LLMJudgment:
        """Judge whether a single action step succeeded."""
        before_context = before.to_llm_context(max_tokens=1500) if before else "N/A (first action)"
        after_context = after.to_llm_context(max_tokens=1500)
        user_msg = (
            f"ACTION ATTEMPTED: {action_description}\n\n"
            f"PAGE STATE BEFORE:\n{before_context}\n\n"
            f"PAGE STATE AFTER:\n{after_context}"
        )
        return await self._judge(_STEP_VERIFICATION_PROMPT, user_msg)

    async def verify_task(
        self,
        task_instruction: str,
        final_state: PageState,
        action_summary: str,
    ) -> LLMJudgment:
        """Judge whether an entire task was completed, from the final page state."""
        state_context = final_state.to_llm_context(max_tokens=2000)
        user_msg = (
            f"ORIGINAL INSTRUCTION: {task_instruction}\n\n"
            f"ACTIONS TAKEN:\n{action_summary}\n\n"
            f"FINAL PAGE STATE:\n{state_context}"
        )
        return await self._judge(_TASK_VERIFICATION_PROMPT, user_msg)

    async def verify_task_visual(
        self,
        task_instruction: str,
        action_summary: str,
        screenshot_b64: str,
        mime: str = "image/jpeg",
    ) -> LLMJudgment | None:
        """Judge task completion from a screenshot. None when vision is unavailable."""
        from app.utils.vision import describe_image

        prompt = (
            f"{_VISUAL_TASK_PROMPT}\n\nORIGINAL INSTRUCTION: {task_instruction}\n\n"
            f"ACTIONS TAKEN:\n{(action_summary or '(none recorded)')[:1500]}"
        )
        raw = await describe_image(screenshot_b64, prompt, mime=mime, max_tokens=500)
        if not raw:
            return None
        judgment = self._parse_judgment(raw)
        judgment.source = "vision"
        return judgment

    async def _judge(self, system_prompt: str, user_message: str) -> LLMJudgment:
        """Run the text judgment and parse the response."""
        from app.utils.llm import chat

        try:
            response = await chat(
                "judge",
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=600,
                temperature=0.0,
                timeout=45,
                completion_fn=self._completion_fn,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_judgment(raw)
        except Exception as e:
            logger.error("llm_judge_failed", error=str(e)[:300])
            # On LLM failure, return a conservative FAIL with low confidence
            return LLMJudgment(
                passed=False,
                confidence=0.0,
                reasoning=f"LLM judge failed with error: {str(e)}",
                raw_response="",
            )

    @staticmethod
    def _parse_judgment(raw: str) -> LLMJudgment:
        """Parse the structured response into a LLMJudgment.

        Tolerates light markdown (e.g. "**VERDICT:** PASS"), which models add
        despite being told not to.
        """
        verdict = False
        confidence = 0.0
        reasoning = ""

        for line in raw.strip().split("\n"):
            line = line.strip().lstrip("*-# `").replace("**", "").strip()
            upper = line.upper()
            if upper.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper().startswith("PASS")
            elif upper.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip().rstrip("%"))
                    if confidence > 1.0:
                        confidence /= 100.0
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            elif upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return LLMJudgment(
            passed=verdict,
            confidence=confidence,
            reasoning=reasoning or "No reasoning provided",
            raw_response=raw,
        )
