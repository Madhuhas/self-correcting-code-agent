"""Prompt construction and code extraction."""

from __future__ import annotations

import re

SYSTEM_PROMPT = """You are a senior Python engineer. Write a single, complete, self-contained \
Python 3 script that solves the user's task.

Rules:
- {packages_rule}
- The script runs in a sandbox: no network, no subprocesses, and it may only write files \
in the current directory. Do not read from stdin.
- Include a short demonstration or assertions under `if __name__ == "__main__":` so running \
the script proves it works.
- Reply with exactly one ```python code block and nothing else."""

STDLIB_RULE = "Use only the Python standard library."
PACKAGES_RULE = "You may use third-party packages that are already installed; prefer the standard library."

DEBUG_PROMPT = """Your previous script failed when executed.

--- CODE (attempt {attempt}) ---
```python
{code}
```

--- ERROR ---
{error}
{earlier}
First, in one sentence, state which line is wrong and why. Then reply with the fully \
corrected script in one ```python code block. The code MUST differ from the code above. \
Do not remove tests or assertions to make the error go away; fix the logic instead."""

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def system_prompt(allow_packages: bool) -> str:
    return SYSTEM_PROMPT.format(packages_rule=PACKAGES_RULE if allow_packages else STDLIB_RULE)


def earlier_errors(history: list[dict]) -> str:
    """One line per earlier failure, so a fix doesn't bring back a bug that was already fixed."""
    if not history:
        return ""
    lines = []
    for rec in history:
        last = (rec["error"].strip().splitlines() or ["(no error text)"])[-1]
        lines.append(f"- attempt {rec['attempt']}: {last[:200]}")
    return "\n--- EARLIER FAILURES (fixed before; do not bring them back) ---\n" + "\n".join(lines) + "\n"


def extract_code(text: str) -> str:
    """Pull Python source out of a model reply.

    Prefers the longest fenced block (small models sometimes emit a usage snippet
    alongside the real solution); falls back to the raw text if there is no fence.
    """
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # Unterminated fence: model ran out of tokens or forgot to close it.
    m = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def build_task(task: str, tests: str | None) -> str:
    if not tests:
        return task
    return (
        f"{task}\n\nYour script will be executed with the following tests appended to it, "
        f"so the names they use must exist:\n```python\n{tests}\n```"
    )


def with_tests(code: str, tests: str | None) -> str:
    return f"{code}\n\n# --- harness tests ---\n{tests}\n" if tests else code
