import mock_llm
from part_c_retry import call_with_retry

# Reset mock LLM before testing
mock_llm.reset()

# Test retry function
response = call_with_retry(
    "CLASSIFY\nCategories: billing, delivery, technical, fraud",
    "My order is delayed",
    "TEST001"
)

# Display output
print("\n--- RETRY TEST RESULT ---")

print("Response:")
print(response.choices[0].message.content)

print("\nTotal tokens:")
print(response.usage.total_tokens)
