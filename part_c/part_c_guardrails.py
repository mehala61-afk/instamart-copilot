import pandas as pd
import math

# Load tickets
tickets = pd.read_csv("data/raw/tickets.csv")

MAX_INPUT_TOKENS = 400

# Token estimation
def est_tokens(text):
    return math.ceil(len(text.split()) * 1.3)


# Check each ticket
def check_ticket(row):

    text = str(row["text"]) if pd.notna(row["text"]) else ""

    # Empty ticket check
    if not text.strip():
        return "skipped_empty"

    # Long ticket check
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"

    return "eligible"


# Apply guardrails
tickets["status"] = tickets.apply(check_ticket, axis=1)


# Display results
print("Total tickets:", len(tickets))

print("\nStatus Counts:")
print(tickets["status"].value_counts())

print("\nEligible tickets:",
      (tickets["status"] == "eligible").sum())

print("\nEmpty tickets:",
      (tickets["status"] == "skipped_empty").sum())

print("\nLong tickets:",
      (tickets["status"] == "flagged_long").sum())


# Validation
assert len(tickets) == 164
assert (tickets["status"] == "skipped_empty").sum() == 4
assert (tickets["status"] == "flagged_long").sum() == 4
assert (tickets["status"] == "eligible").sum() == 156

print("\nInput guardrails validation passed!")
