# Overview

Instamart Customer Intelligence & Support Copilot is a reproducible capstone project for customer intelligence and support workflows. Part A creates and validates the cleaned data and SQLite database, Part B builds RFM segments and a lapse model, Part C safely triages support tickets, and Part D provides a retrieval-grounded policy assistant and a tool-calling refund agent. All language-model behavior is supplied by the offline `mock_llm.py`; no API key, account, network call, or paid service is required.


# Reproduce

Run these commands from the repository root in the listed order. The supplied `generate_data.py` and `mock_llm.py` are used unmodified.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\generate_data.py
.\.venv\Scripts\python.exe .\rebuild_database.py
.\.venv\Scripts\python.exe .\part_a\part_a_eda.py
.\.venv\Scripts\python.exe .\part_a\part_a_validation.py
.\.venv\Scripts\python.exe .\part_b\part_b_rfm.py
.\.venv\Scripts\python.exe .\part_b\part_b_segmentation.py
.\.venv\Scripts\python.exe .\part_b\part_b_lapse_prediction.py
.\.venv\Scripts\python.exe .\part_c\part_c_pipeline.py
.\.venv\Scripts\python.exe .\part_c\part_c_validation.py
.\.venv\Scripts\python.exe .\part_d\part_d_policy_rag.py
.\.venv\Scripts\python.exe .\part_d\part_d_refund_agent.py
```

The SQL submitted for Part A is in `sql/queries.sql`. The generated outputs are committed under `outputs/`, and Part D checkpoints are retained under `state/`.


# Part A
## Data Foundation: Clean, Store, Query

# Scope and validation
Part A covers data cleaning, database setup, required SQL analysis, and Pandas EDA. Its validation completed successfully: all 10 checkpoints passed and the script printed `ALL CHECKPOINTS PASSED`.

Validated database contents:

| Table      | Rows |
|------------|-----:|
| `customers`| 1,200|
| `orders`   |12,309|
| `products` |    40|

Validated data-quality and order results:

- Duplicate order IDs: 0.
- Missing order amounts: 0.
- Completed orders: 11,110.
- Cancelled orders: 1,199.
- Pending orders: 0.
- Completed-order revenue: INR 2,998,668.80.

## Verified SQL and customer-analysis outputs

The seven required SQL queries were reported as completed in the Part A workflow. The following verified outputs are available for the README:

- 25 customers have no completed order.
- 550 customers are above average spend.
- Daily revenue analysis returned 183 rows.

The exact SQL text and result exports are not currently in this workspace, so they should be restored before this section is treated as fully reproducible.

## EDA results

All EDA figures below use completed orders.

### City summary

| City     | Total revenue (INR) | Average order (INR) | Order count |
|----------|--------------------:|--------------------:|---------:|
| Bengaluru| 523,643.40 | 276.47 | 1,894 |
| Chennai  | 463,629.15 | 272.72 | 1,700 |
| Delhi    | 502,829.80 | 243.62 | 2,064 |
| Hyderabad| 528,589.90 | 284.95 | 1,855 |
| Mumbai   | 479,058.50 | 297.00 | 1,613 |
| Pune     | 500,918.05 | 252.48 | 1,984 |

Hyderabad had the highest completed-order revenue (INR 528,589.90). Mumbai had the highest average order value (INR 297.00).

### Distribution and outliers

| Metric              |   Result   |
|---------------------|-----------:|
| Mean order amount   | INR 269.91 |
| Median order amount | INR 135.00 |
| Skewness            | 4.61       |
| Q1                  | INR 64.80 |
| Q3                  | INR 297.00 |
| IQR | INR 232.20 |
| Lower fence | INR -283.50 |
| Upper fence | INR 645.30 |
| Below-fence outliers | 0 |
| Above-fence outliers | 897 |

The large gap between mean and median, together with a skewness of 4.61, indicates a strongly right-skewed order-amount distribution. The IQR rule identified 897 high-value orders above INR 645.30 and none below the lower fence.

### Correlation

| Relationship with `order_amount` | Correlation |
|---|---:|
| `discount_pct` | -0.095106 |
| `quantity` | 0.087294 |
| `delivery_minutes` | -0.003135 |

The strongest absolute relationship is `discount_pct` versus `order_amount` (-0.095), but it is very weak. These correlations do not establish causation.


# Part B
## Customer Segmentation and Lapse Prediction

### 1. RFM Analysis

RFM analysis was performed using completed orders.

- Snapshot date: `2025-10-01`
- Total customers: `1,175`
- Average Recency: `33.56 days`
- Average Frequency: `9.46 orders`
- Average Monetary: `₹2,552.06`

Output: `data/clean/customer_rfm.csv`

### 2. K-Means Customer Segmentation

K-Means clustering was used to segment customers based on RFM features.

The selected number of clusters was **K = 4**, based on silhouette score and cluster interpretability.

| Segment           | Customers | Avg Recency | Avg Frequency | Avg Monetary |
|-------------------|----------:|------------:|--------------:|-------------:|
| Regular Low-Spend |       436 |       28.86 |          3.77 |    ₹1,057.62 |
| Premium           |       138 |       24.12 |          6.77 |    ₹7,129.70 |
| Frequent Active   |       419 |        8.77 |         19.40 |    ₹3,410.02 |
| At-Risk Inactive  |       182 |      109.04 |          2.20 |      ₹685.99 |

### 3. PCA

PCA was used for dimensionality reduction and visualization.

- PC1 explained variance: `63.43%`
- PC2 explained variance: `21.80%`
- Total explained variance: `85.23%`

Output: `outputs/customer_segments.csv`

### 4. Lapse Prediction

A customer was labelled as lapsed if they had no completed order during September 2025.

- Cohort cutoff: `2025-09-01`
- Label window: `2025-09-01` to `2025-09-30`
- Total customers: `1,156`
- Lapsed customers: `443`
- Lapse percentage: `38.32%`

Features were calculated using only pre-cutoff order data to avoid target leakage.

### 5. Model Training and Evaluation

Logistic Regression and Random Forest models were evaluated.

Random Forest results:
- Accuracy: `0.7112`
- Precision: `0.6146`
- Recall: `0.6629`
- F1 Score: `0.6378`

The selected Random Forest model achieved the required F1 threshold of `0.55`.

### 6. Balanced Logistic Regression

Oversampling was applied only to the training data to improve the identification of lapsed customers.

| Metric   | Score |
|----------|------:|
| Accuracy |0.7241 |
| Precision|0.5954 |
| Recall   |0.8764 |
| F1 Score |0.7091 |

### 7. Feature Importance

- Top Random Forest feature: `num__frequency` — `0.337822`
- Top positive Logistic Regression coefficient: `num__monetary` — `0.199426`
- Top negative Logistic Regression coefficient: `num__frequency` — `-1.588850`

### 8. Model Persistence

The selected Random Forest pipeline was saved as:

`outputs/lapse_model.joblib`

Reload validation:

`Reload Prediction Match: True`

### 9. Retention Shortlist

The top 10 Premium customers were ranked using predicted lapse probabilities.

- Actually lapsed customers in the Top 10: `1`

This shortlist is illustrative and requires further validation before use in real retention campaigns.

### 10. Limitations

- Lapse prediction is based on the September 2025 label window.
- The dataset is limited to the available historical period.
- Oversampling was applied only to training data.
- The retention shortlist is illustrative and requires further validation.
- Cluster labels describe the current clustering run and are not permanent customer identities.

# Part C
##Customer Support Ticket Triage with Guardrail

### 1. Input Guardrails
Processed 164 customer support tickets in their original file order.

- Total tickets: 164
- Eligible tickets: 156
- Empty tickets: 4 (`skipped_empty`)
- Long tickets: 4 (`flagged_long`)
- Maximum input limit: 400 estimated tokens.
- Empty and long tickets were not sent to the LLM.

### 2. PII Redaction
Implemented regular-expression-based PII redaction before sending ticket content to the LLM.

| PII Type | Redacted Count |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

- Rechecked redacted text using the same patterns.
- No matching PII remained after redaction.
- The user message contained only the redacted ticket text.

### 3. Prompt Engineering
Implemented two classification prompts:

- **Zero-shot prompting:** Classification instructions without examples.
- **Few-shot prompting:** Classification instructions with labeled examples, including a fraud example.

Both prompts classify tickets into four categories:
- `billing`
- `delivery`
- `technical`
- `fraud`

Both prompts require JSON-only output containing `category` and `reason`.

### 4. Retry and Resilience
Implemented retry handling for transient LLM errors.
- Maximum retries: 3 per ticket.
- Base delay: 0.05 seconds.
- Exponential backoff with jitter.
- Retry delay formula:

  `BASE_DELAY * 2**n * random.uniform(0.9, 1.15)`

- The mock LLM was reset immediately before each full classification run.
- Retry logs record ticket ID, retry number, and wait duration.

### 5. Output Validation
Implemented response validation to prevent unsupported category predictions.

| Status | Description |
|---|---|
| `classified_json` | Valid JSON containing an allowed category |
| `classified_keyword` | Non-JSON response containing exactly one allowed category keyword |
| `flagged_invalid_output` | Invalid JSON or disallowed category; sent for human review |

Final Few-shot classification results:

| Status | Tickets |
|---|---:|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |
| **Total** | **164** |

Invalid outputs were not automatically assigned a guessed category.

### 6. Zero-shot vs Few-shot Evaluation
Evaluated both prompts using the 156 eligible tickets.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 10,024 | 18,136 |
| Tokens per classified ticket | 68.19 | 123.37 |

Few-shot prompting improved classification accuracy by 14.29 percentage points compared with zero-shot prompting.

However, Few-shot used approximately 81% more tokens. In this experiment, the additional token cost was justified by the improvement in classification accuracy.

A zero-shot confusion matrix was also generated to examine classification errors.

### 7. Token Budget and Output Files
Implemented token-budget checks before every model call, including retry attempts.

- Zero-shot budget demo: 6,000 tokens.
- Final Few-shot budget: 100,000 tokens.
- Tickets that could not be processed because the budget was exhausted were assigned `budget_exhausted` and routed to human review.

Generated output files:

- `outputs/prompt_comparison.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

The final `triage_results.csv` contains exactly 164 rows with these columns:

`ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`

### 8. Ticket Routing and Priority
Implemented rule-based ticket routing using classification results and customer segments from Part B.

| Condition | Route | Priority |
|---|---|---|
| Empty, long, invalid, or budget-exhausted ticket | `human_review` | NORMAL |
| Fraud category | `security_team` | HIGH |
| Premium customer | `premium_desk` | HIGH |
| Other valid tickets | `support_queue` | NORMAL |

Final routing results:

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |
| **Total** | **164** |

Final priority results:

| Priority | Tickets |
|---|---:|
| NORMAL | 119 |
| HIGH | 45 |
| **Total** | **164** |

### 9. Validation and Limitations
Final validation checks passed.

- All 164 tickets were included in the final output.
- Empty and long tickets were excluded from model calls.
- PII was redacted before model calls.
- Invalid outputs were routed to human review without guessing.
- Ticket routing and priority followed the specified rules.
- The final output contained exactly 164 rows and the required columns.

**Limitations:**
- The project uses the provided `mock_llm.py`, not a production LLM.
- The reported accuracy represents the mock LLM experiment and does not guarantee real-world performance.
- Invalid outputs and tickets that cannot be processed safely require human review.

# Part D
## Part D – Policy RAG and Refund Agent

### Policy retrieval and grounding

The five policy documents are split into 40-word chunks with a 10-word overlap. This produces 15 chunks: `refund_policy.txt` 4, `delivery_policy.txt` 3, `escalation_policy.txt` 3, `payment_policy.txt` 3 and `privacy_policy.txt` 2. The query `refund policy late delivery` retrieves three chunks, all from `refund_policy.txt`.

TF (term frequency) measures how often a term occurs in one chunk, while IDF (inverse document frequency) gives less weight to terms that occur in many chunks. A limitation is that TF-IDF mainly matches words, so paraphrases with the same meaning but different vocabulary can rank less well.

| Evaluation | Expected document in top 3 | Expected fact in answer | Unsupported-question fallback |
| --- | ---: | ---: | ---: |
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the refund-eligibility question misses `more than 45 minutes`, because the supporting sentence is split at a 40-word chunk boundary. The failed-payment-refund question similarly misses `48 hours`; overlap preserves this boundary context.

### Refund-agent evaluation

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
| --- | --- | ---: | ---: | ---: |
| S1 | REFUND | 128.25 | 5 | 892 |
| S2 | REFUND | 135.00 | 5 | 892 |
| S3 | ESCALATE | 1999.00 | 5 | 879 |
| S4 | DENIED | 0.00 | 4 | 601 |
| S5 | DENIED | 0.00 | 3 | 334 |
| S6 | NOT_FOUND | 0.00 | 2 | 70 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2994 |

The default 6,000-token limit allows S7 to complete all six model calls before returning `FALLBACK_MAX_ITERATIONS`. A run of S1 with `token_limit=200` returns `FALLBACK_TOKEN_LIMIT` after three calls and 351 tokens, without executing the next requested tool. A run with `time_limit_seconds=0` returns `FALLBACK_TIMEOUT` before its first model call.

The crash-and-resume demonstration saves a checkpoint after its second tool observation. Across the interrupted and resumed S1 run, the tools are executed exactly once in this order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`. The resumed run completes with `REFUND` for INR 128.25.

### ReAct versus plan-and-execute

The ReAct agent asks the model once per observed step: S1 used five calls, S5 used three and S6 used two. This consumes more calls and prompt tokens than a plan-and-execute design that first asks for a complete plan and replans only after an error. In return, ReAct can immediately adapt when a tool says `NOT_FOUND` or when an order is cancelled; it does not continue blindly with an invalid precomputed plan. For this refund workflow, I would choose ReAct because each action changes the information available to the next decision, and incorrect automatic refunds are more costly than a few extra local model calls.

The Part A numerical results are verified, but the corresponding source data, scripts, SQL files, dependency manifest, and result exports are absent from the current workspace. Reproduction cannot be verified until those assets are restored.

Next, restore or locate the Part A project files and run the documented workflow from a clean environment. Once reproducibility is confirmed, begin Part B with customer-level RFM feature creation, segmentation, and lapse-prediction evaluation; update this README only with measured results.

# Limitations and Next Steps

- This project uses the supplied deterministic `mock_llm.py`; results do not establish performance with a production language model.
- TF-IDF is lexical and can miss semantic paraphrases; the zero-overlap evaluation demonstrates sensitivity to chunk boundaries.
- Refund decisions are evaluated on the supplied simulated orders and policy documents. A production service would need authenticated customer identity, audit controls and human-review operations.
- The lapse model and ticket-routing results are based on the supplied historical dataset and should be monitored before operational use.

Next steps are to test semantic retrieval, validate models on newer data, and add secure production integrations without changing the reproducible offline baseline.


