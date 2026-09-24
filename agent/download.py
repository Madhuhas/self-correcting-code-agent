"""Download the default GGUF models: python -m agent.download"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from .config import DEFAULT_MODELS

_REPOS = {"1.5b": "Qwen2.5-Coder-1.5B-Instruct-GGUF", "3b": "Qwen2.5-Coder-3B-Instruct-GGUF"}


def main() -> int:
    for path in map(Path, DEFAULT_MODELS["llamacpp"]):
        if path.is_file():
            print(f"already present: {path.name}")
            continue
        size = path.name.split("-")[2]  # qwen2.5-coder-<size>-instruct-...
        url = f"https://huggingface.co/Qwen/{_REPOS[size]}/resolve/main/{path.name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"downloading {path.name} ...")

        def progress(blocks: int, block_size: int, total: int) -> None:
            if total > 0:
                pct = min(100, blocks * block_size * 100 // total)
                print(f"\r  {pct:3d}% of {total / 1e9:.2f} GB", end="", flush=True)

        urllib.request.urlretrieve(url, tmp, progress)
        tmp.replace(path)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
