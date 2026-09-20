"""
Amazon India — verification rules for Phase 0 task types.

Maps (site, action_type) → list of RuleChecks to run after that action.
These are the hard-coded, high-trust checks that form the first verification tier.
"""

from __future__ import annotations

from app.verifier.rule_checks import (
    RuleCheck,
    URLContainsCheck,
    URLChangedCheck,
    ElementExistsCheck,
    ElementTextContainsCheck,
    NumericIncrementCheck,
    TitleContainsCheck,
    VisibleTextContainsCheck,
)


# =============================================================================
# Step-level verification rules
# (run after each individual action)
# =============================================================================

STEP_RULES: dict[str, list[RuleCheck]] = {
    # After navigating to amazon.in
    "navigate_to_amazon": [
        URLContainsCheck("amazon.in"),
        TitleContainsCheck("Amazon"),
    ],

    # After typing in the search box
    "type_search_query": [
        # No strict check — just ensure we're still on Amazon
        URLContainsCheck("amazon"),
    ],

    # After clicking the search button / pressing Enter
    "submit_search": [
        URLContainsCheck("/s?k="),
        ElementExistsCheck("cart_count"),
    ],

    # After clicking a product from search results
    "click_product": [
        URLChangedCheck(),
        # Product page should have a title
        ElementExistsCheck("product_title"),
    ],

    # After clicking "Add to Cart"
    "add_to_cart": [
        NumericIncrementCheck("cart_count", min_increment=1),
    ],

    # After navigating to the cart page
    "view_cart": [
        URLContainsCheck("/cart"),
    ],
}


# =============================================================================
# Task-level verification rules
# (run after the entire task is believed complete)
# =============================================================================

TASK_RULES: dict[str, list[RuleCheck]] = {
    # "Search for X and add to cart" — final verification
    "search_and_add_to_cart": [
        # Cart count should have incremented from the start of the task
        NumericIncrementCheck("cart_count", min_increment=1),
    ],

    # "View cart contents" — just verify we're on the cart page
    "view_cart": [
        URLContainsCheck("/cart"),
        ElementExistsCheck("cart_count"),
    ],
}


def get_step_rules(step_type: str) -> list[RuleCheck]:
    """
    Get the rule checks for a specific step type.

    Args:
        step_type: The type of step (e.g., "add_to_cart", "submit_search").

    Returns:
        List of RuleCheck instances. Empty list if no rules defined for this step.
    """
    return STEP_RULES.get(step_type, [])


def get_task_rules(task_type: str) -> list[RuleCheck]:
    """
    Get the rule checks for a specific task type.

    Args:
        task_type: The type of task (e.g., "search_and_add_to_cart").

    Returns:
        List of RuleCheck instances. Empty list if no rules defined for this task.
    """
    return TASK_RULES.get(task_type, [])
