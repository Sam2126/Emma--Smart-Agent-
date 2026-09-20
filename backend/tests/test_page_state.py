"""
Unit tests for PageState data structures, formatting, and tree flattening.
"""

from app.browser.page_state import PageState, _flatten_a11y_tree


def test_page_state_to_llm_context():
    state = PageState(
        url="https://www.amazon.in/dp/B09XS7JWHH",
        title="Sony WH-1000XM5 Wireless Headphones",
        extracted_values={"cart_count": "1", "price": "29,990"},
        accessibility_tree=[
            {"role": "button", "name": "Add to Cart"},
            {"role": "button", "name": "Buy Now"},
            {"role": "heading", "name": "Product Details"},
        ],
        visible_text="Sony WH-1000XM5 Noise Cancelling Headphones. Rating 4.5 out of 5 stars.",
    )

    context = state.to_llm_context(max_tokens=500)
    assert "URL: https://www.amazon.in/dp/B09XS7JWHH" in context
    assert "Title: Sony WH-1000XM5 Wireless Headphones" in context
    assert "cart_count: 1" in context
    assert "price: 29,990" in context
    assert '[button] "Add to Cart"' in context
    assert "Sony WH-1000XM5 Noise Cancelling" in context


def test_flatten_a11y_tree():
    sample_tree = {
        "role": "RootWebArea",
        "name": "Amazon",
        "children": [
            {
                "role": "navigation",
                "name": "Main Menu",
                "children": [
                    {
                        "role": "searchbox",
                        "name": "Search Amazon.in",
                        "value": "",
                    },
                    {
                        "role": "button",
                        "name": "Go",
                    },
                ],
            },
            {
                "role": "none",
                "name": "",
                "children": [
                    {
                        "role": "link",
                        "name": "Electronics",
                    }
                ],
            },
        ],
    }

    flat = _flatten_a11y_tree(sample_tree, max_depth=5, max_nodes=50)
    roles = [node["role"] for node in flat]
    names = [node["name"] for node in flat]

    assert "navigation" in roles
    assert "searchbox" in roles
    assert "button" in roles
    assert "link" in roles
    assert "Search Amazon.in" in names
    assert "Go" in names
    assert "Electronics" in names
    # Empty decorative role 'none' should be filtered out
    assert "none" not in roles
