import json
import re
import time
import random
import pandas as pd
import mock_llm

from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

from part_c_prompts import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT
from part_c_retry import call_with_retry


# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent

TICKETS_PATH = BASE_DIR / "data" / "raw" / "tickets.csv"
SEGMENTS_PATH = BASE_DIR / "outputs" / "customer_segments.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# CONFIGURATION
ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

MAX_INPUT_TOKENS = 400

# LOAD DATA
tickets = pd.read_csv(TICKETS_PATH)
segments = pd.read_csv(SEGMENTS_PATH)

premium_lookup = (
    segments.set_index("customer_id")["is_premium"].to_dict()
)

# PII REDACTION
def redact_text(text):

    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text
    )

    text = re.sub(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text
    )

    text = re.sub(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        text
    )

    return text

# RESPONSE VALIDATION
def validate_response(response_text):

    try:
        result = json.loads(response_text)

        if (
            isinstance(result, dict)
            and result.get("category") in ALLOWED_CATEGORIES
        ):
            return result["category"], "classified_json"

        return None, "flagged_invalid_output"

    except json.JSONDecodeError:

        words = re.findall(
            r"[a-z]+",
            response_text.lower()
        )

        matches = [
            word for word in words
            if word in ALLOWED_CATEGORIES
        ]

        if len(matches) == 1:
            return matches[0], "classified_keyword"

        return None, "flagged_invalid_output"


# ROUTING
def get_routing(status, category, customer_id):

    if status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if premium_lookup.get(customer_id, False):
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"

# CLASSIFICATION PIPELINE
def run_classification(prompt, run_name):

    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, ticket in tickets.iterrows():

        raw_text = ticket["text"]

        # Empty ticket
        if pd.isna(raw_text) or str(raw_text).strip() == "":

            status = "skipped_empty"
            category = None
            tokens_used = 0

        else:

            text = str(raw_text)

            # Long ticket
            if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:

                status = "flagged_long"
                category = None
                tokens_used = 0

            else:

                # Redact PII
                redacted_text = redact_text(text)

                # PII assertion
                assert not re.search(
                    r"[\w.+-]+@[\w-]+\.[\w.]+",
                    redacted_text
                )

                assert not re.search(
                    r"\b[6-9]\d{9}\b",
                    redacted_text
                )

                assert not re.search(
                    r"ORD-\d{6}",
                    redacted_text
                )

                # Call model with retry
                response = call_with_retry(
                    prompt,
                    redacted_text,
                    ticket["ticket_id"]
                )

                tokens_used = response.usage.total_tokens
                total_tokens += tokens_used

                response_text = (
                    response.choices[0].message.content
                )

                category, status = validate_response(
                    response_text
                )

        customer_id = ticket["customer_id"]

        segment_row = segments[
            segments["customer_id"] == customer_id
        ]

        if not segment_row.empty:
            segment_name = segment_row.iloc[0]["segment_name"]
        else:
            segment_name = None

        route, priority = get_routing(
            status,
            category,
            customer_id
        )

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used
        })

    result_df = pd.DataFrame(results)

    print(f"\n--- {run_name} RESULTS ---")
    print("Total tickets:", len(result_df))
    print("\nStatus counts:")
    print(result_df["status"].value_counts())
    print("\nTotal tokens:", total_tokens)

    return result_df


# RUN ZERO-SHOT
zero_results = run_classification(
    ZERO_SHOT_PROMPT,
    "ZERO-SHOT"
)


# RUN FEW-SHOT
few_results = run_classification(
    FEW_SHOT_PROMPT,
    "FEW-SHOT"
)

# ACCURACY
zero_valid = zero_results[
    zero_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

few_valid = few_results[
    few_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]


zero_actual = tickets.set_index("ticket_id").loc[
    zero_valid["ticket_id"],
    "true_category"
]

few_actual = tickets.set_index("ticket_id").loc[
    few_valid["ticket_id"],
    "true_category"
]


zero_accuracy = accuracy_score(
    zero_actual,
    zero_valid["category"]
)

few_accuracy = accuracy_score(
    few_actual,
    few_valid["category"]
)


print("\n========== PROMPT COMPARISON ==========")

print("Zero-shot classified:", len(zero_valid))
print("Zero-shot accuracy:", round(zero_accuracy * 100, 2), "%")

print("Few-shot classified:", len(few_valid))
print("Few-shot accuracy:", round(few_accuracy * 100, 2), "%")


# CONFUSION MATRIX
labels = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]

cm = confusion_matrix(
    zero_actual,
    zero_valid["category"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n========== ZERO-SHOT CONFUSION MATRIX ==========")
print(cm_df)

# TOKEN COMPARISON
zero_tokens = zero_results["tokens_used"].sum()
few_tokens = few_results["tokens_used"].sum()

print("\n========== TOKEN USAGE ==========")

print("Zero-shot total tokens:", zero_tokens)
print("Few-shot total tokens:", few_tokens)

print(
    "Zero-shot tokens per classified ticket:",
    round(zero_tokens / max(len(zero_valid), 1), 2)
)

print(
    "Few-shot tokens per classified ticket:",
    round(few_tokens / max(len(few_valid), 1), 2)
)

# SAVE ZERO/FEW-SHOT COMPARISON
comparison = pd.DataFrame([
    {
        "prompt": "zero_shot",
        "classified": len(zero_valid),
        "accuracy": zero_accuracy,
        "total_tokens": zero_tokens,
        "tokens_per_classified": zero_tokens / max(len(zero_valid), 1)
    },
    {
        "prompt": "few_shot",
        "classified": len(few_valid),
        "accuracy": few_accuracy,
        "total_tokens": few_tokens,
        "tokens_per_classified": few_tokens / max(len(few_valid), 1)
    }
])

comparison.to_csv(
    OUTPUT_DIR / "prompt_comparison.csv",
    index=False
)

print("\nPrompt comparison saved successfully!")

# BUDGET-AWARE CLASSIFICATION
BASE_DELAY = 0.05
MAX_RETRIES = 3


def run_with_budget(prompt, user_content, ticket_id, total_tokens, budget):

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]

    for retry_number in range(MAX_RETRIES + 1):

        # Check budget before EVERY model call, including retries
        if total_tokens >= budget:
            return None, total_tokens, "budget_exhausted"

        try:
            response = mock_llm.chat(messages)

            used_tokens = response.usage.total_tokens
            total_tokens += used_tokens

            return response, total_tokens, None

        except mock_llm.TransientLLMError:

            if retry_number == MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"{ticket_id} | Retry {retry_number} "
                f"| Wait: {wait:.4f}s"
            )

            time.sleep(wait)

    return None, total_tokens, "budget_exhausted"


def run_budgeted_classification(prompt, run_name, budget):

    # Reset immediately before each full run
    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, ticket in tickets.iterrows():

        raw_text = ticket["text"]
        customer_id = ticket["customer_id"]

        category = None
        tokens_used = 0

        # Empty ticket
        if pd.isna(raw_text) or str(raw_text).strip() == "":

            status = "skipped_empty"

        else:

            text = str(raw_text)

            # Long ticket
            if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:

                status = "flagged_long"

            else:

                # Redact PII
                redacted_text = redact_text(text)

                # Ensure PII is removed
                assert not re.search(
                    r"[\w.+-]+@[\w-]+\.[\w.]+",
                    redacted_text
                )

                assert not re.search(
                    r"\b[6-9]\d{9}\b",
                    redacted_text
                )

                assert not re.search(
                    r"ORD-\d{6}",
                    redacted_text
                )

                response, total_tokens, budget_status = (
                    run_with_budget(
                        prompt,
                        redacted_text,
                        ticket["ticket_id"],
                        total_tokens,
                        budget
                    )
                )

                if budget_status == "budget_exhausted":

                    status = "budget_exhausted"

                else:

                    tokens_used = response.usage.total_tokens

                    response_text = (
                        response.choices[0].message.content
                    )

                    category, status = validate_response(
                        response_text
                    )

        # Customer segment
        segment_row = segments[
            segments["customer_id"] == customer_id
        ]

        if not segment_row.empty:
            segment_name = segment_row.iloc[0]["segment_name"]
        else:
            segment_name = None

        # Routing and priority
        route, priority = get_routing(
            status,
            category,
            customer_id
        )

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used
        })

    result_df = pd.DataFrame(results)

    print(f"\n========== {run_name} ==========")
    print("Total tickets:", len(result_df))
    print("Token budget:", budget)
    print("Tokens used:", total_tokens)
    print("\nStatus counts:")
    print(result_df["status"].value_counts())

    return result_df

# ZERO-SHOT BUDGET DEMO
budget_demo = run_budgeted_classification(
    ZERO_SHOT_PROMPT,
    "ZERO-SHOT BUDGET DEMO",
    budget=6000
)

budget_demo.to_csv(
    OUTPUT_DIR / "triage_budget_demo.csv",
    index=False
)

print("\ntriage_budget_demo.csv saved successfully!")


# FINAL FEW-SHOT RUN
final_results = run_budgeted_classification(
    FEW_SHOT_PROMPT,
    "FINAL FEW-SHOT RUN",
    budget=100000
)

final_results.to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False
)

print("\ntriage_results.csv saved successfully!")


# FINAL VALIDATION
print("\n========== FINAL VALIDATION ==========")

print("Total rows:", len(final_results))

print("\nStatus counts:")
print(final_results["status"].value_counts())

print("\nRoute counts:")
print(final_results["route"].value_counts())

print("\nPriority counts:")
print(final_results["priority"].value_counts())

assert len(budget_demo) == 164
assert len(final_results) == 164

assert list(final_results.columns) == [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

assert (budget_demo["status"] == "budget_exhausted").sum() > 0

print("\nAll final validation checks passed!")