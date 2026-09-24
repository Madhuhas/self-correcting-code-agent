import pytest

from agent import AgentConfig, ScriptedLLM, run_agent, select_model
from agent.llm import LLMError, LLMResponse
from agent.prompts import extract_code

CFG = AgentConfig(fast_model="fast", debug_model="smart", max_attempts=3, exec_timeout_s=5)

GOOD = "```python\nprint('ok')\n```"
BAD = "```python\nraise ValueError('bug')\n```"


def fence(code: str) -> str:
    return f"```python\n{code}\n```"


def test_succeeds_first_try_without_retry():
    llm = ScriptedLLM([GOOD])
    out = run_agent("say ok", llm, CFG)
    assert out["status"] == "success"
    assert out["attempt"] == 1
    assert [m for m, _ in llm.calls] == ["fast"]
    assert out["last_stdout"].strip() == "ok"


def test_self_corrects_and_escalates_model():
    llm = ScriptedLLM([BAD, GOOD])
    out = run_agent("say ok", llm, CFG)
    assert out["status"] == "success"
    assert out["attempt"] == 2
    assert [m for m, _ in llm.calls] == ["fast", "smart"]


def test_error_traceback_is_fed_back_to_model():
    llm = ScriptedLLM([BAD, GOOD])
    run_agent("say ok", llm, CFG)
    _, retry_messages = llm.calls[1]
    feedback = retry_messages[-1]["content"]
    assert "ValueError: bug" in feedback
    assert "raise ValueError('bug')" in feedback


def test_circuit_breaker_stops_at_max_attempts():
    llm = ScriptedLLM([fence(f"raise ValueError({i})") for i in range(10)])
    out = run_agent("never works", llm, CFG)
    assert out["status"] == "failed"
    assert out["stop_reason"].startswith("max_attempts")
    assert out["attempt"] == 3
    assert len(llm.calls) == 3
    assert len(out["history"]) == 3


def test_token_budget_trips_breaker():
    cfg = AgentConfig(fast_model="f", debug_model="s", max_attempts=10, max_total_tokens=250)
    llm = ScriptedLLM([fence(f"raise ValueError({i})") for i in range(10)], tokens_per_call=100)
    out = run_agent("never works", llm, cfg)
    assert out["stop_reason"].startswith("token_budget")
    assert len(llm.calls) == 3  # 100, 200, 300 -> stop


def test_one_repeat_gets_another_chance_without_rerunning():
    llm = ScriptedLLM([BAD, BAD, GOOD])
    out = run_agent("x", llm, CFG)
    assert out["status"] == "success"
    repeat = out["history"][1]
    assert repeat["repeat"] and repeat["exec_seconds"] == 0.0  # known result, not re-executed
    assert "ValueError: bug" in repeat["error"]
    assert "same code you already submitted" in llm.calls[2][1][-1]["content"]


def test_two_repeats_in_a_row_stop_early():
    llm = ScriptedLLM([BAD, BAD, BAD, GOOD])
    cfg = AgentConfig(fast_model="f", debug_model="s", max_attempts=6)
    out = run_agent("x", llm, cfg)
    assert out["status"] == "failed"
    assert out["stop_reason"].startswith("no_progress")
    assert len(llm.calls) == 3


def test_repeat_of_older_attempt_is_detected():
    other = fence("raise KeyError('other')")
    llm = ScriptedLLM([BAD, other, BAD, GOOD])
    cfg = AgentConfig(fast_model="f", debug_model="s", max_attempts=6)
    out = run_agent("x", llm, cfg)
    assert [r["repeat"] for r in out["history"]] == [False, False, True, False]
    assert out["status"] == "success"


def test_tests_are_appended_and_enforced():
    tests = "assert add(2, 2) == 4"
    llm = ScriptedLLM([fence("def add(a, b):\n    return a - b"), fence("def add(a, b):\n    return a + b")])
    out = run_agent("add two numbers", llm, CFG, tests=tests)
    assert out["status"] == "success"
    assert out["stop_reason"] == "tests_passed"
    assert "AssertionError" in out["history"][0]["error"]
    # Model was told about the tests up front.
    assert "assert add(2, 2) == 4" in llm.calls[0][1][1]["content"]


def test_reply_without_code_counts_as_failure():
    llm = ScriptedLLM(["I cannot help with that.", GOOD])
    llm_empty = ScriptedLLM(["", GOOD])
    assert run_agent("x", llm, CFG)["attempt"] == 2
    out = run_agent("x", llm_empty, CFG)
    assert "No Python code" in out["history"][0]["error"]


def test_timeout_is_reported_to_model():
    llm = ScriptedLLM([fence("while True: pass"), GOOD])
    cfg = AgentConfig(fast_model="f", debug_model="s", exec_timeout_s=1)
    out = run_agent("x", llm, cfg)
    assert out["status"] == "success"
    assert "TimeoutError" in llm.calls[1][1][-1]["content"]


def test_llm_outage_fails_cleanly():
    class Down:
        def chat(self, model, messages, temperature=None) -> LLMResponse:
            raise LLMError("connection refused")

    out = run_agent("x", Down(), CFG)
    assert out["status"] == "failed"
    assert "llm_error" in out["stop_reason"]
    assert out["history"] == []


@pytest.mark.parametrize("attempt,expected", [(0, "fast"), (1, "smart"), (2, "smart")])
def test_router(attempt, expected):
    assert select_model(attempt, CFG)[0] == expected


def test_temperature_rises_on_retries_and_is_capped():
    llm = ScriptedLLM([fence(f"raise ValueError({i})") for i in range(5)])
    cfg = AgentConfig(fast_model="f", debug_model="s", max_attempts=5)
    run_agent("x", llm, cfg)
    assert llm.temperatures == pytest.approx([0.2, 0.5, 0.8, 1.0, 1.0])


@pytest.mark.parametrize("reply,expected", [
    ("```python\nx = 1\n```", "x = 1"),
    ("Here:\n```py\nx = 1\n```\nDone.", "x = 1"),
    ("```\nx = 1\n```", "x = 1"),
    ("```python\nshort\n```\n```python\nlonger_block = 1\n```", "longer_block = 1"),
    ("```python\nx = 1\n", "x = 1"),  # unterminated fence
    ("x = 1", "x = 1"),
])
def test_extract_code(reply, expected):
    assert extract_code(reply) == expected


def test_retry_prompt_lists_earlier_failures():
    llm = ScriptedLLM([fence("raise KeyError('first')"), fence("raise IndexError('second')"), GOOD])
    run_agent("x", llm, CFG)
    third_prompt = llm.calls[2][1][-1]["content"]
    assert "IndexError: second" in third_prompt      # latest error, in full
    assert "attempt 1: KeyError: 'first'" in third_prompt  # earlier one, summarised


def test_first_retry_has_no_earlier_section():
    llm = ScriptedLLM([BAD, GOOD])
    run_agent("x", llm, CFG)
    assert "EARLIER FAILURES" not in llm.calls[1][1][-1]["content"]


def test_system_prompt_follows_package_setting():
    llm = ScriptedLLM([GOOD, GOOD])
    run_agent("x", llm, CFG)
    run_agent("x", llm, AgentConfig(fast_model="f", debug_model="s", allow_site_packages=True))
    assert "only the Python standard library" in llm.calls[0][1][0]["content"]
    assert "third-party packages" in llm.calls[1][1][0]["content"]
