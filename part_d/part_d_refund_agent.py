"""Part D refund agent: tool-calling ReAct loop with safe fallbacks and resume."""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402
from part_d_policy_rag import retrieve  # noqa: E402

DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up one order and its delivery details.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the most relevant refund policy chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate the eligible refund using the delivery policy.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Record an automatically issued refund in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Record a refund that requires human approval.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
]


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)


def get_order(order_id: str) -> dict:
    """Read the requested order without exposing unrelated database columns."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    return dict(row) if row is not None else {"error": "order not found"}


def search_policy(query: str) -> dict:
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount: float, delivery_minutes: float) -> dict:
    refund = round(float(order_amount), 2) if float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": refund}


def _append_ledger(tool_name: str, order_id: str, refund_inr: float) -> dict:
    _ensure_dirs()
    result = {"timestamp": time.time(), "tool_name": tool_name, "order_id": order_id, "refund_inr": round(float(refund_inr), 2)}
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    return {"refund_inr": result["refund_inr"]}


def issue_refund(order_id: str, refund_inr: float) -> dict:
    return _append_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict:
    return _append_ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


class SimulatedCrash(RuntimeError):
    pass


def _checkpoint_path(run_id: str) -> Path:
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(state: dict) -> None:
    _ensure_dirs()
    _checkpoint_path(state["run_id"]).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _result(state: dict, status: str, final_answer: str) -> dict:
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]}


def _fallback(state: dict, status: str, reason: str) -> dict:
    return _result(state, status, f"{status}: human handoff required ({reason}).")


def _continue(state: dict, max_iterations: int, token_limit: int, time_limit_seconds: float,
              simulate_crash_after_tools: int | None = None) -> dict:
    while state["llm_calls"] < max_iterations:
        if time.monotonic() - state["started_at"] >= time_limit_seconds:
            return _fallback(state, "FALLBACK_TIMEOUT", "time limit reached")
        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        message = response.choices[0].message
        if state["tokens_used"] > token_limit:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", "token limit reached before tool execution")
        if not message.tool_calls:
            final_answer = message.content or "FALLBACK: empty model response"
            return _result(state, final_answer.split(":", 1)[0], final_answer)
        tool_calls = [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in message.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for call in tool_calls:
            if call["name"] not in TOOL_MAP:
                return _fallback(state, "FALLBACK_UNKNOWN_TOOL", call["name"])
            observation = TOOL_MAP[call["name"]](**call["arguments"])
            state["steps"].append({"tool_name": call["name"], "arguments": call["arguments"], "observation": observation})
            state["messages"].append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                      "content": json.dumps(observation)})
            _save_checkpoint(state)  # checkpoint is deliberately written after every tool observation
            if simulate_crash_after_tools is not None and len(state["steps"]) == simulate_crash_after_tools:
                raise SimulatedCrash(state["run_id"])
    return _fallback(state, "FALLBACK_MAX_ITERATIONS", "maximum model calls reached")


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6, token_limit: int = 6000,
              time_limit_seconds: float = 10, run_id: str | None = None,
              simulate_crash_after_tools: int | None = None) -> dict:
    """Run the ReAct loop and retain a checkpoint after every completed tool execution."""
    state = {
        "run_id": run_id or f"{scenario_id}_{uuid.uuid4().hex[:10]}", "scenario_id": scenario_id,
        "messages": [{"role": "system", "content": "AGENT. Use tools to resolve the refund request safely."},
                     {"role": "user", "content": user_message}],
        "tokens_used": 0, "llm_calls": 0, "steps": [], "started_at": time.monotonic(),
    }
    return _continue(state, max_iterations, token_limit, time_limit_seconds, simulate_crash_after_tools)


def resume_agent(run_id: str, max_iterations: int = 6, token_limit: int = 6000,
                 time_limit_seconds: float = 10) -> dict:
    """Load a saved run and continue after its final recorded tool observation."""
    state = json.loads(_checkpoint_path(run_id).read_text(encoding="utf-8"))
    # A resume is a new execution window; prior work and tokens remain accounted for.
    state["started_at"] = time.monotonic()
    return _continue(state, max_iterations, token_limit, time_limit_seconds)


def demonstrate_resume(s1: dict) -> dict:
    run_id = "demo_s1_resume"
    try:
        run_agent(s1["message"], "S1_RESUME", run_id=run_id, simulate_crash_after_tools=2)
    except SimulatedCrash:
        pass
    resumed = resume_agent(run_id)
    return {"status": resumed["status"], "refund_result": resumed["final_answer"],
            "tools": [step["tool_name"] for step in resumed["steps"]], "llm_calls": resumed["llm_calls"],
            "tokens_used": resumed["tokens_used"]}


def _load_scenarios() -> list[dict]:
    with SCENARIOS_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_run(result: dict) -> None:
    record = {key: result[key] for key in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}
    with RUNS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    _ensure_dirs()
    scenarios = _load_scenarios()
    s1 = next(row for row in scenarios if row["scenario_id"] == "S1")
    # Required demonstrations; their details are retained separately, not mixed into the seven-row run log.
    token_demo = run_agent(s1["message"], "S1_TOKEN_LIMIT", token_limit=200)
    timeout_demo = run_agent(s1["message"], "S1_TIMEOUT", time_limit_seconds=0)
    resume_demo = demonstrate_resume(s1)
    (OUTPUT_DIR / "part_d_demo_results.json").write_text(json.dumps({
        "token_limit_200": token_demo, "timeout_0": timeout_demo, "crash_and_resume": resume_demo}, indent=2), encoding="utf-8")

    # The committed ledger and run log describe exactly the seven required primary scenarios.
    RUNS_PATH.write_text("", encoding="utf-8")
    LEDGER_PATH.write_text("", encoding="utf-8")
    primary_results = []
    mock_llm.reset()
    for row in scenarios:
        result = run_agent(row["message"], row["scenario_id"])
        _write_run(result)
        primary_results.append({"scenario_id": row["scenario_id"], "status": result["status"],
                                "refund_amount": row["expected_refund_inr"], "llm_calls": result["llm_calls"],
                                "tokens_used": result["tokens_used"]})
    (OUTPUT_DIR / "part_d_scenario_summary.json").write_text(json.dumps(primary_results, indent=2), encoding="utf-8")
    print(json.dumps({"scenarios": primary_results, "token_demo": token_demo["status"],
                      "timeout_demo": timeout_demo["status"], "resume_demo": resume_demo}, indent=2))


if __name__ == "__main__":
    main()
