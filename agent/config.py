"""Runtime configuration, overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Default models per backend. llama.cpp takes GGUF file paths; Ollama takes model tags.
DEFAULT_MODELS = {
    "llamacpp": (str(MODEL_DIR / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
                 str(MODEL_DIR / "qwen2.5-coder-3b-instruct-q4_k_m.gguf")),
    "ollama": ("qwen2.5-coder:1.5b", "qwen2.5-coder:3b"),
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return raw.strip().lower() in ("1", "true", "yes", "on") if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class AgentConfig:
    # "llamacpp" runs GGUF models in-process (no server); "ollama" talks to a local Ollama.
    backend: str = "llamacpp"
    # Model routing: cheap/fast model for the first draft, stronger model for debugging.
    fast_model: str = DEFAULT_MODELS["llamacpp"][0]
    debug_model: str = DEFAULT_MODELS["llamacpp"][1]
    ollama_url: str = "http://localhost:11434"
    llm_timeout_s: float = 300.0  # CPU inference is slow; be generous.
    temperature: float = 0.2
    # Added per retry so a stuck model explores instead of repeating itself; capped at 1.0.
    retry_temperature_step: float = 0.3

    # Circuit breaker: hard caps on retries and on total tokens spent.
    max_attempts: int = 3
    max_total_tokens: int = 20_000

    # Sandbox. "process" = local interpreter with OS limits + policy guard;
    # "docker" = container with no network and a read-only filesystem.
    sandbox: str = "process"
    docker_image: str = "python:3.11-slim"
    allow_network: bool = False
    allow_site_packages: bool = False  # let generated code import installed third-party packages
    exec_timeout_s: float = 10.0
    max_output_chars: int = 4_000
    max_memory_mb: int = 512  # enforced on POSIX only

    @classmethod
    def from_env(cls) -> "AgentConfig":
        d = cls()
        backend = os.getenv("AGENT_BACKEND", d.backend)
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"AGENT_BACKEND must be one of {sorted(DEFAULT_MODELS)}, got {backend!r}")
        fast, debug = DEFAULT_MODELS[backend]
        return cls(
            backend=backend,
            fast_model=os.getenv("AGENT_FAST_MODEL", fast),
            debug_model=os.getenv("AGENT_DEBUG_MODEL", debug),
            ollama_url=os.getenv("OLLAMA_URL", d.ollama_url),
            llm_timeout_s=_env_float("AGENT_LLM_TIMEOUT", d.llm_timeout_s),
            temperature=_env_float("AGENT_TEMPERATURE", d.temperature),
            max_attempts=_env_int("AGENT_MAX_ATTEMPTS", d.max_attempts),
            max_total_tokens=_env_int("AGENT_MAX_TOKENS", d.max_total_tokens),
            sandbox=os.getenv("AGENT_SANDBOX", d.sandbox),
            docker_image=os.getenv("AGENT_DOCKER_IMAGE", d.docker_image),
            allow_network=_env_bool("AGENT_ALLOW_NETWORK", d.allow_network),
            allow_site_packages=_env_bool("AGENT_ALLOW_PACKAGES", d.allow_site_packages),
            exec_timeout_s=_env_float("AGENT_EXEC_TIMEOUT", d.exec_timeout_s),
            max_output_chars=_env_int("AGENT_MAX_OUTPUT", d.max_output_chars),
            max_memory_mb=_env_int("AGENT_MAX_MEMORY_MB", d.max_memory_mb),
        )
