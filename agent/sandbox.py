"""Run model-generated Python in a separate, locked-down process.

Two modes:

* ``process`` (default): a fresh interpreter in a throwaway directory. Limits come
  from three layers: OS limits (setrlimit on POSIX, a Job Object on Windows), a
  wall-clock timer, and an in-process policy guard (a PEP 578 audit hook) that blocks
  network access, process spawning, ctypes, and file writes outside the sandbox.
  This stops the mistakes a model actually makes, but deliberately hostile code
  could still find a way around an in-process guard.
* ``docker``: the same bootstrap runs inside a container with no network, a read-only
  root filesystem, dropped capabilities, and memory/PID limits. Use this for code you
  don't trust.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Minimal environment for the child: no API keys or tokens leak into generated code.
_SAFE_ENV_KEYS = ("PATH", "SYSTEMROOT", "LANG")


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    duration_s: float

    @property
    def error(self) -> str:
        """The feedback the model sees when a run fails."""
        return self.stderr or f"Process exited with code {self.exit_code} and no stderr."


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Keep the tail: the actual exception line lives at the end of a traceback.
    return f"...[{len(text) - limit} chars truncated]...\n" + text[-limit:]


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8", "replace") if path.exists() else ""


# --------------------------------------------------------------------------- child side

# Runs inside the sandboxed interpreter. Handshake with the parent:
#   1. touch the ready file  -> parent starts the execution timer (startup isn't billed)
#   2. wait for the go file  -> parent has applied OS limits (Windows Job Object)
#   3. install the policy guard, then run the model's code
# The bootstrap frame is dropped from tracebacks so the model only sees its own code.
_BOOTSTRAP = r'''
import os, sys, time, traceback

script, ready, go, root, allow_net = sys.argv[1:6]
open(ready, "w").close()
deadline = time.monotonic() + 30
while not os.path.exists(go):
    if time.monotonic() > deadline:
        sys.exit("sandbox: parent never released the child")
    time.sleep(0.005)


def install_guard(root, allow_net):
    root = os.path.normcase(os.path.realpath(root))
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    fs_events = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.symlink",
                 "os.link", "os.truncate", "os.utime", "shutil.rmtree", "shutil.move",
                 "shutil.copyfile", "shutil.copytree"}
    net_events = {"socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo",
                  "socket.gethostbyname", "socket.gethostbyaddr", "urllib.Request"}
    blocked = ("subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
               "os.startfile", "os.kill", "ctypes.", "winreg.", "_winapi.")

    def inside(path):
        if isinstance(path, int) or path is None:
            return True
        try:
            full = os.path.normcase(os.path.realpath(os.fsdecode(path)))
        except (TypeError, ValueError):
            return True
        return full == root or full.startswith(root + os.sep)

    def deny(what):
        raise PermissionError(f"sandbox policy: {what} is not allowed")

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            writing = any(c in mode for c in "wax+") if mode else bool(flags & write_flags)
            if writing and not inside(path):
                deny(f"writing to {os.fsdecode(path)!r} (only the working directory is writable)")
        elif event in fs_events:
            for arg in args:
                if isinstance(arg, (str, bytes, os.PathLike)) and not inside(arg):
                    deny(f"{event} on {os.fsdecode(arg)!r}")
        elif event in net_events:
            if not allow_net:
                deny("network access")
        elif event.startswith(blocked):
            deny(f"{event}")

    sys.addaudithook(hook)  # audit hooks cannot be removed once added


with open(script, encoding="utf-8") as f:
    src = f.read()
try:
    code = compile(src, script, "exec")
except SyntaxError as e:
    traceback.print_exception(type(e), e, None)
    sys.exit(1)

install_guard(root, allow_net == "1")
del install_guard
try:
    exec(code, {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__})
except SystemExit:
    raise
except BaseException as e:
    traceback.print_exception(type(e), e, e.__traceback__.tb_next)
    sys.exit(1)
'''


# ------------------------------------------------------------------------ OS limits

def _posix_limits(max_memory_mb: int, cpu_seconds: int):
    def apply() -> None:
        import resource

        mem = max_memory_mb * 1024 * 1024
        limits = [(resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu_seconds),
                  (resource.RLIMIT_FSIZE, 10 * 1024 * 1024)]
        # No forking. RLIMIT_NPROC is per-user and ignored for root, so skip it there.
        if hasattr(resource, "RLIMIT_NPROC") and os.getuid() != 0:
            limits.append((resource.RLIMIT_NPROC, 0))
        for kind, value in limits:
            try:
                resource.setrlimit(kind, (value, value))
            except (ValueError, OSError):
                pass  # e.g. macOS rejects RLIMIT_AS; the other layers still apply

    return apply


class _WindowsJob:
    """Windows Job Object: per-process memory cap, one active process (no children),
    and everything in the job is killed when the handle closes."""

    _LIMIT_ACTIVE_PROCESS = 0x8
    _LIMIT_PROCESS_MEMORY = 0x100
    _LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self, max_memory_mb: int) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("Basic", BasicLimits), ("Io", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32, self._ctypes = k32, ctypes

        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = ExtendedLimits()
        info.Basic.LimitFlags = (self._LIMIT_ACTIVE_PROCESS | self._LIMIT_PROCESS_MEMORY
                                 | self._LIMIT_KILL_ON_JOB_CLOSE)
        info.Basic.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = max_memory_mb * 1024 * 1024
        if not k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign(self, proc: subprocess.Popen) -> None:
        if not self._k32.AssignProcessToJobObject(self.handle, int(proc._handle)):  # type: ignore[attr-defined]
            raise OSError(self._ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _docker_command(name: str, workdir: str, image: str, max_memory_mb: int, allow_network: bool) -> list[str]:
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "bridge" if allow_network else "none",
        "--memory", f"{max_memory_mb}m", "--memory-swap", f"{max_memory_mb}m",
        "--cpus", "1", "--pids-limit", "32",
        "--read-only", "--tmpfs", "/tmp:rw,size=16m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-v", f"{workdir}:/sandbox", "-w", "/sandbox", "-e", "TMPDIR=/sandbox",
    ]
    if hasattr(os, "getuid"):
        # On Linux hosts the bind mount keeps host ownership; run as the host user so the
        # container can write the handshake files.
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    return cmd + [image, "python", "-I", "-B", "-X", "utf8", "/sandbox/_bootstrap.py",
                  "/sandbox/main.py", "/sandbox/_ready", "/sandbox/_go", "/sandbox",
                  "1" if allow_network else "0"]


# ----------------------------------------------------------------------- parent side

def run_code(
    code: str,
    *,
    timeout_s: float = 10.0,
    max_output_chars: int = 4_000,
    max_memory_mb: int = 512,
    startup_timeout_s: float = 60.0,
    max_output_bytes: int = 10 * 1024 * 1024,
    mode: str = "process",
    allow_network: bool = False,
    allow_site_packages: bool = False,
    docker_image: str = "python:3.11-slim",
) -> ExecutionResult:
    """Execute ``code`` in a fresh interpreter inside a throwaway directory.

    ``timeout_s`` limits the code's own run time. Interpreter (or container) startup has
    its own, separate ``startup_timeout_s`` budget.
    """
    if mode not in ("process", "docker"):
        raise ValueError(f"unknown sandbox mode {mode!r}")

    with tempfile.TemporaryDirectory(prefix="agent_sandbox_") as workdir:
        wd = Path(workdir)
        script, bootstrap = wd / "main.py", wd / "_bootstrap.py"
        ready, go = wd / "_ready", wd / "_go"
        out_path, err_path = wd / "_stdout", wd / "_stderr"
        script.write_text(code, encoding="utf-8")
        bootstrap.write_text(_BOOTSTRAP, encoding="utf-8")

        container = f"agent-sbx-{uuid.uuid4().hex[:12]}"
        job: _WindowsJob | None = None
        kwargs: dict = {}
        if mode == "docker":
            go.touch()  # the container itself is the resource boundary
            cmd = _docker_command(container, workdir, docker_image, max_memory_mb, allow_network)
            env = None
        else:
            env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
            env.update(TEMP=workdir, TMP=workdir, TMPDIR=workdir, HOME=workdir)
            # -I: ignore PYTHON* env vars, user site and cwd on sys.path. -B: no .pyc writes
            # (they would trip the write guard). -X utf8: UTF-8 stdio on every platform.
            # With allow_site_packages, -E keeps env isolation but lets installed packages load.
            flags = ["-E"] if allow_site_packages else ["-I"]
            cmd = [sys.executable, *flags, "-B", "-X", "utf8", str(bootstrap), str(script),
                   str(ready), str(go), workdir, "1" if allow_network else "0"]
            if os.name == "posix":
                kwargs["preexec_fn"] = _posix_limits(max_memory_mb, int(timeout_s) + 1)
            elif os.name == "nt":
                job = _WindowsJob(max_memory_mb)

        timed_out, kill_reason = False, ""
        launched = time.perf_counter()
        ready_at: float | None = None
        try:
            # Output goes to files rather than pipes: no deadlock on large output, and the
            # size can be checked while the process runs.
            with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
                try:
                    proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                                            stdout=out_f, stderr=err_f, **kwargs)
                except FileNotFoundError as exc:
                    if mode == "docker":
                        raise RuntimeError("sandbox mode 'docker' needs Docker installed and running") from exc
                    raise
                if mode == "process":
                    if job is not None:
                        job.assign(proc)
                    go.touch()  # limits are in place; let the child run the code

                while proc.poll() is None:
                    now = time.perf_counter()
                    if ready_at is None and ready.exists():
                        ready_at = now
                    if ready_at is None and now - launched > startup_timeout_s:
                        kill_reason = f"sandbox failed to start within {startup_timeout_s:.0f}s"
                    elif ready_at is not None and now - ready_at > timeout_s:
                        kill_reason = f"execution exceeded the {timeout_s:g}s time limit"
                    elif out_path.stat().st_size + err_path.stat().st_size > max_output_bytes:
                        kill_reason = f"output exceeded {max_output_bytes // (1024 * 1024)} MB"
                    if kill_reason:
                        if mode == "docker":
                            subprocess.run(["docker", "kill", container], capture_output=True, timeout=30)
                        proc.kill()
                        proc.wait()
                        timed_out = True
                        break
                    time.sleep(0.02)
        finally:
            if job is not None:
                job.close()  # kills anything still in the job
        duration = time.perf_counter() - (ready_at or launched)

        stdout = _read(out_path)
        stderr = _read(err_path).replace(workdir, "<sandbox>").replace("/sandbox/", "<sandbox>/")
        if timed_out:
            stderr = f"TimeoutError: {kill_reason}.\n{stderr}".strip()
        return ExecutionResult(
            success=not timed_out and proc.returncode == 0,
            stdout=_truncate(stdout, max_output_chars),
            stderr=_truncate(stderr, max_output_chars),
            exit_code=None if timed_out else proc.returncode,
            timed_out=timed_out,
            duration_s=duration,
        )
