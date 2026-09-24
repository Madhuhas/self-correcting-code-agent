"""LLM clients. The graph depends only on the ``LLMClient`` protocol, so tests can
swap in a scripted fake and production can use llama.cpp, Ollama, or anything else."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    def chat(self, model: str, messages: Sequence[dict],
             temperature: float | None = None) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


class OllamaClient:
    """Minimal client for Ollama's /api/chat endpoint (stdlib only)."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: float = 300.0,
                 temperature: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature

    def chat(self, model: str, messages: Sequence[dict],
             temperature: float | None = None) -> LLMResponse:
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise LLMError(f"Ollama returned HTTP {exc.code} for model '{model}': {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.base_url} ({exc}). Is `ollama serve` running?"
            ) from exc

        return LLMResponse(
            text=body.get("message", {}).get("content", ""),
            prompt_tokens=body.get("prompt_eval_count", 0) or 0,
            completion_tokens=body.get("eval_count", 0) or 0,
        )


class LlamaCppClient:
    """Runs GGUF models in-process via llama-cpp-python: no server, no Ollama.

    ``model`` is a path to a .gguf file. Loaded models are cached, so the fast and
    debug models are each loaded once (~1 GB + ~2 GB of RAM for the 1.5B/3B Q4 models).
    """

    def __init__(self, n_ctx: int = 4096, temperature: float = 0.2, max_tokens: int = 1024,
                 n_threads: int | None = None) -> None:
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_threads = n_threads
        self._models: dict = {}

    def _load(self, model: str):
        if model not in self._models:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise LLMError("llama-cpp-python is not installed; see README") from exc
            if not Path(model).is_file():
                raise LLMError(f"Model file not found: {model}. Run `python -m agent.download`.")
            self._models[model] = Llama(model_path=model, n_ctx=self.n_ctx,
                                        n_threads=self.n_threads, verbose=False)
        return self._models[model]

    def chat(self, model: str, messages: Sequence[dict],
             temperature: float | None = None) -> LLMResponse:
        llm = self._load(model)
        try:
            temp = self.temperature if temperature is None else temperature
            out = llm.create_chat_completion(messages=list(messages), temperature=temp,
                                             max_tokens=self.max_tokens)
        except ValueError as exc:  # e.g. prompt exceeds context window
            raise LLMError(f"llama.cpp error: {exc}") from exc
        usage = out.get("usage") or {}
        return LLMResponse(
            text=out["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


class ScriptedLLM:
    """Deterministic fake that returns canned replies in order. Used by tests and demos."""

    def __init__(self, replies: Sequence[str], tokens_per_call: int = 100) -> None:
        self._replies = list(replies)
        self.tokens_per_call = tokens_per_call
        self.calls: list[tuple[str, list[dict]]] = []
        self.temperatures: list[float | None] = []

    def chat(self, model: str, messages: Sequence[dict],
             temperature: float | None = None) -> LLMResponse:
        self.calls.append((model, list(messages)))
        self.temperatures.append(temperature)
        if not self._replies:
            raise LLMError("ScriptedLLM ran out of replies")
        half = self.tokens_per_call // 2
        return LLMResponse(self._replies.pop(0), prompt_tokens=half, completion_tokens=half)
