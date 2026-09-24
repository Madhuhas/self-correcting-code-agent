"""CLI: python -m agent "task description" [--tests FILE] [--demo]"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_MODELS, AgentConfig
from .graph import run_agent
from .llm import LlamaCppClient, OllamaClient, ScriptedLLM

# Demo mode: a scripted model that makes a classic bug, then fixes it. Lets anyone see
# the loop working without installing Ollama or downloading a model.
DEMO_TASK = "Write a function mean(xs) that returns the average of a list, returning 0.0 for an empty list."
DEMO_TESTS = "assert mean([1, 2, 3]) == 2.0\nassert mean([]) == 0.0\nprint('all tests passed')"
DEMO_REPLIES = [
    "```python\ndef mean(xs):\n    return sum(xs) / len(xs)\n```",
    "The empty list divides by zero. Fixed:\n```python\ndef mean(xs):\n"
    "    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)\n```",
]


def _hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * max(0, 60 - len(title))}")


def main(argv: list[str] | None = None) -> int:
    # Model output can contain any Unicode; don't let a cp1252 Windows console crash the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="agent", description="Self-correcting code generation agent")
    p.add_argument("task", nargs="?", help="What the code should do")
    p.add_argument("--tests", type=Path, help="Python file of assertions appended to the generated code")
    p.add_argument("--backend", choices=sorted(DEFAULT_MODELS), help="llamacpp (in-process, default) or ollama")
    p.add_argument("--max-attempts", type=int, help="Circuit breaker: max generate/execute cycles")
    p.add_argument("--fast-model", help="Model for the first attempt")
    p.add_argument("--debug-model", help="Model for retries")
    p.add_argument("--sandbox", choices=["process", "docker"], help="Where generated code runs")
    p.add_argument("--allow-packages", action="store_true",
                   help="Let generated code import installed third-party packages")
    p.add_argument("--demo", action="store_true", help="Run with a scripted model (no Ollama needed)")
    p.add_argument("--json", type=Path, help="Write the full run trace to this JSON file")
    args = p.parse_args(argv)

    if args.backend:
        os.environ["AGENT_BACKEND"] = args.backend
    config = AgentConfig.from_env()
    overrides = {k: v for k, v in {
        "max_attempts": args.max_attempts, "fast_model": args.fast_model, "debug_model": args.debug_model,
        "sandbox": args.sandbox, "allow_site_packages": args.allow_packages or None,
    }.items() if v is not None}
    config = replace(config, **overrides)

    if args.demo:
        task, tests, llm = DEMO_TASK, DEMO_TESTS, ScriptedLLM(DEMO_REPLIES)
    else:
        if not args.task:
            p.error("a task is required (or use --demo)")
        task = args.task
        tests = args.tests.read_text(encoding="utf-8") if args.tests else None
        if config.backend == "ollama":
            llm = OllamaClient(config.ollama_url, config.llm_timeout_s, config.temperature)
        else:
            llm = LlamaCppClient(temperature=config.temperature)

    print(f"Task: {task}")
    print(f"Backend: {config.backend} | fast={Path(config.fast_model).name} "
          f"debug={Path(config.debug_model).name} | max_attempts={config.max_attempts} | "
          f"sandbox={config.sandbox}")
    if args.demo:
        print("[demo] model replies are scripted; code execution and the loop are real")

    final = run_agent(task, llm, config, tests=tests)

    for rec in final["history"]:
        mark = "PASS" if rec["success"] else "FAIL"
        _hr(f"Attempt {rec['attempt']} [{mark}] {Path(rec['model']).name}")
        print(f"route: {rec['route_reason']} | tokens: {rec['tokens']} | "
              f"llm: {rec['llm_seconds']:.1f}s | exec: {rec['exec_seconds']:.2f}s")
        print(rec["code"])
        if rec["error"]:
            print("--- error ---\n" + rec["error"].strip())

    _hr(f"RESULT: {final['status'].upper()}")
    print(f"stop_reason: {final['stop_reason']}")
    print(f"attempts: {final['attempt']} | total tokens: {final['tokens_used']}")
    if final["status"] == "success" and final.get("last_stdout"):
        print("--- program output ---\n" + final["last_stdout"].rstrip())

    if args.json:
        args.json.write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
        print(f"trace written to {args.json}")

    return 0 if final["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
