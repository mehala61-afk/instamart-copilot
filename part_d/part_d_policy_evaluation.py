import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_d_policy_rag import retrieve_policy


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "part_d"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        question = row.get("question") or row.get("Question")

        if not question:
            question = next(
                (value for key, value in row.items()
                 if key and "question" in key.lower()),
                ""
            )

        expected = row.get("expected_doc") or row.get("expected") or row.get("Expected") or ""

        result = retrieve_policy(question)

        predicted = "NONE" if result["answer"] == "NONE" else "POLICY"

        results.append({
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "answer": result["answer"],
            "source": result["source"],
            "score": result["score"]
        })

output_file = OUTPUT_DIR / "policy_evaluation_results.csv"

with open(output_file, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("POLICY EVALUATION")
print("-----------------")
print("Total questions:", len(results))
print("Results saved:", output_file)
print("NONE predictions:", sum(r["predicted"] == "NONE" for r in results))