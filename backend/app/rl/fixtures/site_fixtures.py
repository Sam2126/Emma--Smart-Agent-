"""
Deterministic sandbox fixtures for hermetic RL training.

Provides structured mock DOM states and fixture representations for:
- Home / Search page
- Search results with facet links
- Product detail page
- Cart page with numeric badge
"""

from __future__ import annotations
from typing import Any


FIXTURE_DOM_STATES: dict[str, dict[str, Any]] = {
    "home": {
        "url": "sandbox://ecommerce/",
        "title": "Sandbox Online Store: Shop Electronics, Books & More",
        "visible_text": "Sandbox Store Search Electronics Deals Cart (0 items)",
        "elements": [
            {"id": "search_input", "selector": "#twotabsearchtextbox", "tag": "input", "text": "", "role": "searchbox", "attributes": {"placeholder": "Search products...", "name": "q"}},
            {"id": "search_submit", "selector": "#nav-search-submit-button", "tag": "button", "text": "Search", "role": "button"},
            {"id": "nav_cart", "selector": "#nav-cart", "tag": "a", "text": "Cart 0", "role": "link"},
        ],
        "extracted_values": {
            "cart_count": "0",
            "page_type": "home",
        },
        "dom_hash": "dom_home_hash_001",
    },
    "results": {
        "url": "sandbox://ecommerce/s?k=headphones",
        "title": "Sandbox Store: Search Results for 'headphones'",
        "visible_text": "Search Results: 1-16 of over 1,000 results for 'headphones'. Filters: Brand Sony, Rating 4 Stars & Up.",
        "elements": [
            {"id": "search_input", "selector": "#twotabsearchtextbox", "tag": "input", "text": "headphones", "role": "searchbox"},
            {"id": "facet_sony", "selector": "button:has-text('Sony')", "tag": "button", "text": "Sony", "role": "checkbox", "attributes": {"data-facet": "brand", "data-value": "sony"}},
            {"id": "facet_rating", "selector": "a:has-text('4 Stars & Up')", "tag": "a", "text": "4 Stars & Up", "role": "link", "attributes": {"data-facet": "rating"}},
            {"id": "product_1", "selector": "a.product-link:has-text('Sony WH-1000XM5')", "tag": "a", "text": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "role": "link", "attributes": {"href": "/dp/B09XYZ"}},
            {"id": "product_2", "selector": "a.product-link:has-text('Bose 700')", "tag": "a", "text": "Bose Noise Cancelling Headphones 700", "role": "link"},
            {"id": "nav_cart", "selector": "#nav-cart", "tag": "a", "text": "Cart 0", "role": "link"},
        ],
        "extracted_values": {
            "cart_count": "0",
            "query": "headphones",
            "page_type": "search_results",
        },
        "dom_hash": "dom_results_hash_002",
    },
    "filtered_results": {
        "url": "sandbox://ecommerce/s?k=headphones&rh=p_n_brand:sony&filter=applied",
        "title": "Sandbox Store: Sony Headphones Filtered Results",
        "visible_text": "Filtered Results: Sony Headphones (Applied Facet: Brand Sony). 12 items found.",
        "elements": [
            {"id": "search_input", "selector": "#twotabsearchtextbox", "tag": "input", "text": "headphones", "role": "searchbox"},
            {"id": "active_facet", "selector": ".applied-filter:has-text('Sony')", "tag": "span", "text": "Sony [x]", "role": "status"},
            {"id": "product_1", "selector": "a.product-link:has-text('Sony WH-1000XM5')", "tag": "a", "text": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "role": "link", "attributes": {"href": "/dp/B09XYZ"}},
            {"id": "nav_cart", "selector": "#nav-cart", "tag": "a", "text": "Cart 0", "role": "link"},
        ],
        "extracted_values": {
            "cart_count": "0",
            "active_filters": "brand:sony",
            "page_type": "filtered_search_results",
        },
        "dom_hash": "dom_filtered_hash_003",
    },
    "product": {
        "url": "sandbox://ecommerce/dp/B09XYZ",
        "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones - Black",
        "visible_text": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones. Price: ₹24,990. In Stock. Add to Cart.",
        "elements": [
            {"id": "product_title", "selector": "#productTitle", "tag": "h1", "text": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "role": "heading"},
            {"id": "price", "selector": ".a-price-whole", "tag": "span", "text": "24,990", "role": "text"},
            {"id": "add_to_cart", "selector": "#add-to-cart-button", "tag": "button", "text": "Add to Cart", "role": "button"},
            {"id": "buy_now", "selector": "#buy-now-button", "tag": "button", "text": "Buy Now", "role": "button"},
            {"id": "nav_cart", "selector": "#nav-cart", "tag": "a", "text": "Cart 0", "role": "link"},
        ],
        "extracted_values": {
            "product_title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            "price": "₹24,990",
            "cart_count": "0",
            "page_type": "product_detail",
        },
        "dom_hash": "dom_product_hash_004",
    },
    "cart": {
        "url": "sandbox://ecommerce/cart",
        "title": "Shopping Cart (1 item) - Sandbox Store",
        "visible_text": "Added to Cart. 1 Item in Cart. Subtotal: ₹24,990. Proceed to checkout.",
        "elements": [
            {"id": "cart_badge", "selector": "#nav-cart-count", "tag": "span", "text": "1", "role": "status"},
            {"id": "cart_item", "selector": ".cart-item-title", "tag": "span", "text": "Sony WH-1000XM5", "role": "text"},
            {"id": "checkout", "selector": "#proceed-to-checkout", "tag": "button", "text": "Proceed to checkout", "role": "button"},
        ],
        "extracted_values": {
            "cart_count": "1",
            "subtotal": "₹24,990",
            "page_type": "cart",
        },
        "dom_hash": "dom_cart_hash_005",
    },
}
