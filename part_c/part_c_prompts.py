CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT_PROMPT = """
You are a customer support ticket classifier.

CLASSIFY the customer ticket into exactly one category:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category":"...", "reason":"..."}
"""

FEW_SHOT_PROMPT = """
You are a customer support ticket classifier.

CLASSIFY the customer ticket into exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:

Ticket: I was charged twice for my order.
{"category":"billing","reason":"Duplicate payment"}

Ticket: Someone used my account without permission.
{"category":"fraud","reason":"Unauthorized account activity"}

Ticket: My order has not arrived.
{"category":"delivery","reason":"Order not delivered"}

Ticket: The app crashes when I open it.
{"category":"technical","reason":"Application error"}

Return JSON only in this format:
{"category":"...", "reason":"..."}
"""

# Validation
assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "EXAMPLES:" not in ZERO_SHOT_PROMPT

assert "CLASSIFY" in FEW_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT

for category in CATEGORIES:
    assert category in ZERO_SHOT_PROMPT
    assert category in FEW_SHOT_PROMPT

print("Zero-shot prompt validation passed!")
print("Few-shot prompt validation passed!")
