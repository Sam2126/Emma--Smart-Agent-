"""
Unit tests for the rule-based verifier checks and verification engine.
"""

import pytest
from app.browser.page_state import PageState
from app.verifier.rule_checks import (
    URLContainsCheck,
    URLChangedCheck,
    ElementExistsCheck,
    ElementTextContainsCheck,
    ValueChangedCheck,
    NumericIncrementCheck,
    TitleContainsCheck,
    VisibleTextContainsCheck,
)
from app.verifier.engine import VerificationEngine, VerificationTier


@pytest.mark.asyncio
async def test_url_contains_check():
    check = URLContainsCheck("/s?k=")
    state_pass = PageState(url="https://www.amazon.in/s?k=headphones")
    state_fail = PageState(url="https://www.amazon.in/cart")

    res_pass = await check.check(None, state_pass)
    assert res_pass.passed is True

    res_fail = await check.check(None, state_fail)
    assert res_fail.passed is False


@pytest.mark.asyncio
async def test_url_changed_check():
    check = URLChangedCheck()
    before = PageState(url="https://www.amazon.in/")
    after = PageState(url="https://www.amazon.in/s?k=laptop")
    same = PageState(url="https://www.amazon.in/")

    res_changed = await check.check(before, after)
    assert res_changed.passed is True

    res_same = await check.check(before, same)
    assert res_same.passed is False


@pytest.mark.asyncio
async def test_element_exists_check():
    check = ElementExistsCheck("cart_count")
    state_has = PageState(extracted_values={"cart_count": "2"})
    state_empty = PageState(extracted_values={"cart_count": ""})
    state_missing = PageState(extracted_values={})

    assert (await check.check(None, state_has)).passed is True
    assert (await check.check(None, state_empty)).passed is False
    assert (await check.check(None, state_missing)).passed is False


@pytest.mark.asyncio
async def test_numeric_increment_check():
    check = NumericIncrementCheck("cart_count", min_increment=1)
    before = PageState(extracted_values={"cart_count": "1"})
    after_inc = PageState(extracted_values={"cart_count": "2"})
    after_same = PageState(extracted_values={"cart_count": "1"})
    after_dec = PageState(extracted_values={"cart_count": "0"})

    assert (await check.check(before, after_inc)).passed is True
    assert (await check.check(before, after_same)).passed is False
    assert (await check.check(before, after_dec)).passed is False


@pytest.mark.asyncio
async def test_element_text_contains_check():
    check = ElementTextContainsCheck("product_title", "Sony", case_sensitive=False)
    state_pass = PageState(extracted_values={"product_title": "Sony WH-1000XM5 Wireless Headphones"})
    state_fail = PageState(extracted_values={"product_title": "Bose QuietComfort 45"})

    assert (await check.check(None, state_pass)).passed is True
    assert (await check.check(None, state_fail)).passed is False


@pytest.mark.asyncio
async def test_title_contains_check():
    check = TitleContainsCheck("Amazon")
    state_pass = PageState(title="Online Shopping site in India: Shop Online for Mobiles, Books, Watches - Amazon.in")
    state_fail = PageState(title="Flipkart")

    assert (await check.check(None, state_pass)).passed is True
    assert (await check.check(None, state_fail)).passed is False


@pytest.mark.asyncio
async def test_visible_text_contains_check():
    check = VisibleTextContainsCheck("Added to Cart")
    state_pass = PageState(visible_text="Success! Item has been Added to Cart. View Cart (1)")
    state_fail = PageState(visible_text="Out of stock. Currently unavailable.")

    assert (await check.check(None, state_pass)).passed is True
    assert (await check.check(None, state_fail)).passed is False


@pytest.mark.asyncio
async def test_engine_verify_step_rules_conclusive():
    engine = VerificationEngine()
    
    # Step: navigate_to_amazon
    after_state = PageState(
        url="https://www.amazon.in/",
        title="Amazon.in Shopping",
    )
    result = await engine.verify_step("navigate_to_amazon", before=None, after=after_state)
    assert result.passed is True
    assert result.tier == VerificationTier.RULE
    assert result.confidence == 1.0


def test_requires_human_confirmation():
    assert VerificationEngine.requires_human_confirmation("place_order") is True
    assert VerificationEngine.requires_human_confirmation("confirm_payment") is True
    assert VerificationEngine.requires_human_confirmation("checkout") is True
    assert VerificationEngine.requires_human_confirmation("type_search_query") is False
    assert VerificationEngine.requires_human_confirmation("click_product") is False
