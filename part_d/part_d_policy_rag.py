"""Part D policy RAG: required 40-word chunks, TF-IDF retrieval and evaluation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402

POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"


def chunk_document(text: str, source: str, chunk_size: int = 40, overlap: int = 10) -> list[dict]:
    """Return word chunks, retaining each chunk's originating policy filename."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append({"source": source, "text": " ".join(piece)})
        if start + chunk_size >= len(words):
            break
    return chunks


def build_index(overlap: int = 10) -> tuple[list[dict], TfidfVectorizer, object]:
    chunks: list[dict] = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), path.name, overlap=overlap))
    if not chunks:
        raise FileNotFoundError(f"No policy documents found in {POLICY_DIR}")
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return chunks, vectorizer, matrix


_CACHE: dict[int, tuple[list[dict], TfidfVectorizer, object]] = {}


def retrieve(query: str, k: int = 3, overlap: int = 10) -> list[dict]:
    """Rank policy chunks by cosine similarity and return source, score and text."""
    if overlap not in _CACHE:
        _CACHE[overlap] = build_index(overlap)
    chunks, vectorizer, matrix = _CACHE[overlap]
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:k]
    return [
        {"source": chunks[i]["source"], "score": float(scores[i]), "text": chunks[i]["text"]}
        for i in ranked
    ]


def answer(question: str, overlap: int = 10) -> tuple[str, list[str]]:
    """Answer exclusively from the three retrieved chunks via the supplied mock LLM."""
    hits = retrieve(question, k=3, overlap=overlap)
    context = "\n".join(f"[{hit['source']}]\n{hit['text']}" for hit in hits)
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer only from the supplied context. "
                "If the information is absent, say you could not find it."
            ),
        },
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
    ]
    response = mock_llm.chat(messages)
    return response.choices[0].message.content, [hit["source"] for hit in hits]


def _read_eval_rows() -> list[dict]:
    with (RAW_DIR / "policy_eval.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(overlap: int) -> tuple[list[dict], dict]:
    """Evaluate grounding/retrieval, supporting the supplied policy_eval.csv columns."""
    rows = []
    for item in _read_eval_rows():
        question = item["question"]
        expected_doc = (item.get("expected_doc") or "").strip()
        expected_fact = item.get("expected_fact") or ""
        text, sources = answer(question, overlap=overlap)
        unanswerable = expected_doc.upper() in {"", "NONE", "N/A"}
        rows.append(
            {
                "question": question,
                "expected_doc": "" if unanswerable else expected_doc,
                "expected_fact": expected_fact,
                "answer": text,
                "sources": "; ".join(sources),
                "expected_doc_in_top_3": (expected_doc in sources) if not unanswerable else "",
                "expected_fact_in_answer": (expected_fact in text) if not unanswerable else "",
                "fallback": text.startswith("I could not find"),
                "overlap": overlap,
            }
        )
    answerable = [row for row in rows if row["expected_doc"]]
    unanswerable = [row for row in rows if not row["expected_doc"]]
    summary = {
        "overlap": overlap,
        "answerable": len(answerable),
        "expected_doc_in_top_3": sum(row["expected_doc_in_top_3"] is True for row in answerable),
        "expected_fact_in_answer": sum(row["expected_fact_in_answer"] is True for row in answerable),
        "unanswerable": len(unanswerable),
        "fallback": sum(row["fallback"] is True for row in unanswerable),
    }
    return rows, summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    chunks, _, _ = build_index(overlap=10)
    by_source = {}
    for chunk in chunks:
        by_source[chunk["source"]] = by_source.get(chunk["source"], 0) + 1
    if len(chunks) != 15:
        raise AssertionError(f"Expected 15 chunks, got {len(chunks)}: {by_source}")

    overlap_rows, overlap_summary = evaluate(overlap=10)
    zero_rows, zero_summary = evaluate(overlap=0)
    all_rows = overlap_rows + zero_rows
    fieldnames = list(all_rows[0])
    with (OUTPUT_DIR / "policy_evaluation_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    report = {"chunk_count": len(chunks), "chunks_by_source": by_source,
              "overlap_10": overlap_summary, "overlap_0": zero_summary}
    (OUTPUT_DIR / "policy_evaluation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


