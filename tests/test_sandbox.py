import sys

import pytest

from agent.sandbox import _docker_command, docker_available, run_code


def test_success_captures_stdout():
    r = run_code("print('hello')")
    assert r.success and r.stdout.strip() == "hello" and r.exit_code == 0


def test_exception_returns_traceback():
    r = run_code("x = 1 / 0")
    assert not r.success
    assert "ZeroDivisionError" in r.error
    assert "Traceback" in r.error


def test_sandbox_path_is_redacted():
    r = run_code("raise ValueError('boom')")
    assert "<sandbox>" in r.error


def test_infinite_loop_is_killed():
    r = run_code("while True: pass", timeout_s=1.0)
    assert r.timed_out and not r.success
    assert "TimeoutError" in r.error


def test_timeout_excludes_interpreter_startup():
    # Code that runs just under the limit must pass, however long the interpreter took to start.
    r = run_code("import time; time.sleep(0.5); print('done')", timeout_s=1.0)
    assert r.success and r.stdout.strip() == "done"
    assert r.duration_s < 1.0


def test_syntax_error_is_reported():
    r = run_code("def f(:\n    pass")
    assert not r.success and "SyntaxError" in r.error


def test_bootstrap_frame_hidden_and_main_guard_runs():
    r = run_code("def f():\n    raise KeyError('k')\nif __name__ == '__main__':\n    f()")
    assert "KeyError" in r.error
    assert "_bootstrap" not in r.error
    assert "main.py" in r.error


def test_output_flood_is_killed():
    r = run_code("while True: print('x' * 1000)", timeout_s=30, max_output_bytes=1_000_000)
    assert r.timed_out and "output exceeded" in r.error


def test_output_is_truncated_keeping_the_tail():
    r = run_code("print('a' * 10000)\nraise RuntimeError('the real error')", max_output_chars=200)
    assert len(r.stdout) < 300
    assert "the real error" in r.error


def test_secrets_do_not_leak_into_child(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    r = run_code("import os; print(os.environ.get('OPENAI_API_KEY'))")
    assert r.stdout.strip() == "None"


def test_sys_exit_nonzero_is_failure():
    r = run_code("import sys; sys.exit(3)")
    assert not r.success and r.exit_code == 3


# --- policy guard -------------------------------------------------------------

def test_unicode_output_survives():
    r = run_code("print('✓ café 你好')")
    assert r.success and r.stdout.strip() == "✓ café 你好"


def test_files_can_be_written_inside_workdir():
    r = run_code("open('out.txt', 'w').write('hi')\nprint(open('out.txt').read())")
    assert r.success and r.stdout.strip() == "hi"


def test_tempfile_works():
    r = run_code("import tempfile\nwith tempfile.TemporaryFile('w+') as f:\n    f.write('x')\nprint('ok')")
    assert r.success


def test_write_outside_workdir_is_blocked(tmp_path):
    target = tmp_path / "escaped.txt"
    r = run_code(f"open({str(target)!r}, 'w').write('x')")
    assert not r.success and "sandbox policy" in r.error
    assert not target.exists()


def test_delete_outside_workdir_is_blocked(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("important")
    r = run_code(f"import os; os.remove({str(victim)!r})")
    assert "sandbox policy" in r.error and victim.exists()


@pytest.mark.parametrize("code", [
    "import urllib.request; urllib.request.urlopen('http://example.com', timeout=2)",
    "import socket; socket.create_connection(('1.1.1.1', 53), timeout=2)",
])
def test_network_is_blocked(code):
    r = run_code(code)
    assert "network access is not allowed" in r.error


@pytest.mark.parametrize("code", [
    "import subprocess, sys; subprocess.run([sys.executable, '-c', 'print(1)'])",
    "import os; os.system('echo hi')",
    "import ctypes; ctypes.CDLL(None)",
])
def test_process_spawning_and_ctypes_are_blocked(code):
    r = run_code(code)
    assert "sandbox policy" in r.error


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not enforce RLIMIT_AS")
def test_memory_limit_is_enforced():
    r = run_code("b = bytearray(1024 ** 3)", max_memory_mb=256)
    assert not r.success and "MemoryError" in r.error


def test_stdlib_imports_still_work():
    r = run_code("import json, re, sqlite3, csv, decimal, statistics, dataclasses, unittest\nprint('ok')")
    assert r.success, r.error


# --- docker -----------------------------------------------------------------

def test_docker_command_is_locked_down():
    cmd = _docker_command("n", "/tmp/x", "python:3.11-slim", 256, allow_network=False)
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--memory 256m", "--pids-limit"):
        assert flag in joined
    assert "--network bridge" in " ".join(_docker_command("n", "/tmp/x", "img", 256, allow_network=True))


@pytest.mark.skipif(not docker_available(), reason="Docker not installed or not running")
def test_docker_sandbox_runs_code():
    r = run_code("print('from container')", mode="docker", startup_timeout_s=300)
    assert r.success and r.stdout.strip() == "from container"
