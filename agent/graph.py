"""The self-correcting loop as a LangGraph state machine.

    START -> generate -> execute --success--------------------------> END
                ^           |
                |           +--failure, budget left--> (back to generate, debug model)
                |           |
                |           +--failure, budget spent--> circuit_breaker -> END
                +--LLM error-----------------------------------------> END
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import AgentConfig
from .llm import LLMClient, LLMError
from .prompts import DEBUG_PROMPT, build_task, earlier_errors, extract_code, system_prompt, with_tests
from .sandbox import run_code

Status = Literal["running", "success", "failed"]


class Attempt(TypedDict):
    attempt: int
    model: str
    route_reason: str
    temperature: float
    code: str
    repeat: bool          # identical to an earlier attempt, so it wasn't re-run
    success: bool
    stdout: str
    error: str
    tokens: int
    llm_seconds: float
    exec_seconds: float


class AgentState(TypedDict, total=False):
    task: str
    tests: str | None
    code: str
    attempt: int                                   # attempts made so far
    tokens_used: int
    last_error: str
    last_stdout: str
    status: Status
    stop_reason: str
    history: Annotated[list[Attempt], operator.add]  # append-only audit log
    _pending: dict                                  # generate -> execute handoff


def select_model(attempt: int, config: AgentConfig) -> tuple[str, str]:
    """Route the next call. First draft is cheap; any retry escalates to the debug model."""
    if attempt == 0:
        return config.fast_model, "first attempt -> fast model"
    return config.debug_model, f"retry #{attempt} after failure -> debug model, temp {select_temperature(attempt, config):.1f}"


def select_temperature(attempt: int, config: AgentConfig) -> float:
    """Low temperature for the first draft; hotter on each retry so the model stops repeating a bad fix."""
    return min(1.0, config.temperature + attempt * config.retry_temperature_step)


def build_messages(state: AgentState, config: AgentConfig) -> list[dict]:
    messages = [
        {"role": "system", "content": system_prompt(config.allow_site_packages)},
        {"role": "user", "content": build_task(state["task"], state.get("tests"))},
    ]
    if state.get("attempt", 0) > 0:
        # The latest attempt is sent in full; earlier ones only as one-line error summaries.
        # That keeps the prompt small (it matters for CPU inference and small context
        # windows) while still stopping the model from reintroducing an old bug.
        messages.append({
            "role": "user",
            "content": DEBUG_PROMPT.format(
                attempt=state["attempt"], code=state.get("code", ""), error=state.get("last_error", ""),
                earlier=earlier_errors(state.get("history", [])[:-1]),
            ) + (REPEAT_NOTE if state.get("history") and state["history"][-1]["repeat"] else ""),
        })
    return messages


REPEAT_NOTE = ("\n\nNOTE: that is the same code you already submitted, and it fails the same way. "
               "Repeating it will not work. Take a different approach.")


def build_graph(llm: LLMClient, config: AgentConfig | None = None):
    config = config or AgentConfig()

    def generate(state: AgentState) -> dict:
        attempt = state.get("attempt", 0)
        model, reason = select_model(attempt, config)
        temperature = select_temperature(attempt, config)
        start = time.perf_counter()
        try:
            resp = llm.chat(model, build_messages(state, config), temperature=temperature)
        except LLMError as exc:
            return {"status": "failed", "stop_reason": f"llm_error: {exc}"}
        return {
            "code": extract_code(resp.text),
            "attempt": attempt + 1,
            "tokens_used": state.get("tokens_used", 0) + resp.total_tokens,
            "_pending": {
                "model": model,
                "route_reason": reason,
                "temperature": temperature,
                "tokens": resp.total_tokens,
                "llm_seconds": time.perf_counter() - start,
            },
        }

    def execute(state: AgentState) -> dict:
        code = state["code"]
        pending = state["_pending"]
        history = state.get("history", [])
        earlier = next((r for r in history if code.strip() and r["code"] == code), None)
        if not code.strip():
            ok, stdout, error, exec_s = False, "", "No Python code found in the model's reply.", 0.0
        elif earlier is not None:
            # Same code as before: the outcome is already known, so don't run it again.
            ok, stdout, error, exec_s = earlier["success"], earlier["stdout"], earlier["error"], 0.0
        else:
            result = run_code(
                with_tests(code, state.get("tests")),
                timeout_s=config.exec_timeout_s,
                max_output_chars=config.max_output_chars,
                max_memory_mb=config.max_memory_mb,
                mode=config.sandbox,
                allow_network=config.allow_network,
                allow_site_packages=config.allow_site_packages,
                docker_image=config.docker_image,
            )
            ok, stdout, exec_s = result.success, result.stdout, result.duration_s
            error = "" if ok else result.error

        record: Attempt = {
            "attempt": state["attempt"],
            "model": pending["model"],
            "route_reason": pending["route_reason"],
            "temperature": pending["temperature"],
            "code": code,
            "repeat": earlier is not None,
            "success": ok,
            "stdout": stdout,
            "error": error,
            "tokens": pending["tokens"],
            "llm_seconds": pending["llm_seconds"],
            "exec_seconds": exec_s,
        }
        update: dict = {"history": [record], "last_error": error, "last_stdout": stdout}
        if ok:
            update.update(status="success", stop_reason="tests_passed" if state.get("tests") else "ran_cleanly")
        elif earlier is not None and history and history[-1]["repeat"]:
            # Two repeats in a row, even with rising temperature: the model is stuck, and
            # more retries would only burn tokens. (A single repeat gets one more chance,
            # at a higher temperature and with a note saying it was a repeat.)
            update.update(status="failed", stop_reason="no_progress: model repeated earlier code twice in a row")
        return update

    def circuit_breaker(state: AgentState) -> dict:
        if state["attempt"] >= config.max_attempts:
            reason = f"max_attempts: gave up after {state['attempt']} attempts"
        else:
            reason = f"token_budget: {state['tokens_used']} >= {config.max_total_tokens} tokens"
        return {"status": "failed", "stop_reason": reason}

    def after_generate(state: AgentState) -> str:
        return END if state.get("status") == "failed" else "execute"

    def after_execute(state: AgentState) -> str:
        if state.get("status") in ("success", "failed"):
            return END
        if state["attempt"] >= config.max_attempts or state["tokens_used"] >= config.max_total_tokens:
            return "circuit_breaker"
        return "generate"

    g = StateGraph(AgentState)
    g.add_node("generate", generate)
    g.add_node("execute", execute)
    g.add_node("circuit_breaker", circuit_breaker)
    g.add_edge(START, "generate")
    g.add_conditional_edges("generate", after_generate, ["execute", END])
    g.add_conditional_edges("execute", after_execute, ["generate", "circuit_breaker", END])
    g.add_edge("circuit_breaker", END)
    return g.compile()


def run_agent(task: str, llm: LLMClient, config: AgentConfig | None = None,
              tests: str | None = None) -> AgentState:
    config = config or AgentConfig()
    graph = build_graph(llm, config)
    initial: AgentState = {
        "task": task, "tests": tests, "code": "", "attempt": 0, "tokens_used": 0,
        "last_error": "", "last_stdout": "", "status": "running", "stop_reason": "", "history": [],
    }
    # Each attempt is 2 graph steps; the recursion limit is a second, framework-level breaker.
    return graph.invoke(initial, {"recursion_limit": config.max_attempts * 2 + 5})
