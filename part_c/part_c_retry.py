import time
import random
import mock_llm

BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_with_retry(system_prompt, user_content, ticket_id):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    for retry_number in range(MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)
            return response

        except mock_llm.TransientLLMError:

            if retry_number == MAX_RETRIES:
                print(f"{ticket_id}: retries exhausted")
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
