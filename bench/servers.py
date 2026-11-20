"""Start and stop the inference servers under benchmark, one at a time.

Each engine runs from its own virtualenv with its defaults for everything that is a tuning
choice (batching limits, attention backend, CUDA graphs, prefix caching — on by default in
all three), and with the same model, dtype and context length. The exact command line of
every run is recorded in the results.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ("minfer", "vllm", "sglang")

# Per-engine environment overrides, recorded with the results.
ENVIRONMENT: dict[str, dict[str, str]] = {
    # DeepGEMM is SGLang's FP8 GEMM library; a bf16 model never calls it, but its JIT is
    # initialized at start-up and aborts unless nvcc >= 12.9 (this machine has 12.8).
    "sglang": {"SGLANG_ENABLE_JIT_DEEPGEMM": "0"},
}


def command(engine: str, model: str, port: int, extra: Sequence[str] = ()) -> list[str]:
    """The server command line for ``engine``."""
    if engine == "minfer":
        base = [".venv/bin/python", "-m", "minfer.cli", "serve", "--model", model]
        base += ["--port", str(port), "--log-level", "warning"]
    elif engine == "vllm":
        base = [".venv-vllm/bin/vllm", "serve", model, "--port", str(port), "--dtype", "bfloat16"]
        # Without this vLLM adopts the checkpoint's generation_config.json as request
        # defaults (sampling with a repetition penalty); the benchmark sends explicit
        # greedy parameters either way.
        base += ["--generation-config", "vllm", "--enable-prompt-tokens-details"]
    elif engine == "sglang":
        base = [".venv-sglang/bin/python", "-m", "sglang.launch_server", "--model-path", model]
        base += ["--port", str(port), "--dtype", "bfloat16", "--log-level", "warning"]
        # Observation only: expose /metrics and report cached tokens in `usage`.
        base += ["--enable-metrics", "--enable-cache-report"]
    else:
        raise ValueError(f"unknown engine {engine!r}")
    return [*base, *extra]


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _gpu_memory_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(int(line) for line in out.stdout.split())


@contextmanager
def launch(
    engine: str,
    model: str,
    port: int,
    log_path: Path,
    extra: Sequence[str] = (),
    startup_timeout: float = 900.0,
) -> Iterator[tuple[str, list[str]]]:
    """Run a server for the duration of the ``with`` block.

    Yields ``(base_url, command_line)`` once ``/health`` answers. On exit the whole process
    group is terminated and the function waits until the GPU's memory is released, so the
    next engine starts on an idle device.
    """
    argv = command(engine, model, port, extra)
    env = os.environ.copy()
    # Each engine's JIT (FlashInfer needs ninja) must find its own virtualenv's tools.
    venv_bin = ROOT / Path(argv[0]).parent
    env["PATH"] = f"{venv_bin}{os.pathsep}{env['PATH']}"
    env.update(ENVIRONMENT.get(engine, {}))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + startup_timeout
            while not _healthy(port):
                if process.poll() is not None:
                    raise RuntimeError(f"{engine} exited during start-up; see {log_path}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{engine} not healthy after {startup_timeout:.0f}s")
                time.sleep(1.0)
            yield f"http://127.0.0.1:{port}", argv
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            deadline = time.monotonic() + 120
            while _gpu_memory_used_mib() > 1024 and time.monotonic() < deadline:
                time.sleep(1.0)
