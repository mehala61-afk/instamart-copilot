import pandas as pd
import re

# Load tickets
tickets = pd.read_csv("data/raw/tickets.csv")

# PII patterns
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"


def redact_pii(text):

    if pd.isna(text):
        return ""

    text = str(text)

    email_count = len(re.findall(EMAIL_PATTERN, text))
    phone_count = len(re.findall(PHONE_PATTERN, text))
    order_count = len(re.findall(ORDER_PATTERN, text))

    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(PHONE_PATTERN, "[PHONE]", text)
    text = re.sub(ORDER_PATTERN, "[ORDER_ID]", text)

    return text, email_count, phone_count, order_count


# Process all tickets
email_total = 0
phone_total = 0
order_total = 0

redacted_texts = []

for text in tickets["text"]:

    result = redact_pii(text)

    if isinstance(result, tuple):
        redacted, emails, phones, orders = result
    else:
        redacted = result
        emails = phones = orders = 0

    redacted_texts.append(redacted)

    email_total += emails
    phone_total += phones
    order_total += orders


tickets["redacted_text"] = redacted_texts

print("Email redactions:", email_total)
print("Phone redactions:", phone_total)
print("Order ID redactions:", order_total)

print("\nTotal PII redactions:",
      email_total + phone_total + order_total)


# Verify no PII remains
remaining = 0

for text in tickets["redacted_text"]:

    remaining += len(re.findall(EMAIL_PATTERN, text))
    remaining += len(re.findall(PHONE_PATTERN, text))
    remaining += len(re.findall(ORDER_PATTERN, text))

print("Remaining PII:", remaining)

assert email_total == 55
assert phone_total == 91
assert order_total == 132
assert remaining == 0

print("\nPII redaction validation passed!")
