"""Serve minfer as an OpenAI-compatible endpoint on Modal (scale to zero).

    modal deploy deploy/modal_app.py

The deployment exposes `minfer serve` behind a public URL (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/health`, `/metrics`). A container with one H100 starts on
the first request and is stopped after five idle minutes.

    from openai import OpenAI
    client = OpenAI(base_url="https://<workspace>--minfer-serve.modal.run/v1", api_key="-")

Not deployed from the machine this repository was developed on (it has no Modal
credentials); see PROGRESS.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/minfer"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PORT = 8000
IGNORE = [".git", ".venv*", "**/__pycache__", "bench", "tests", ".ruff_cache"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential", "ninja-build")
    .pip_install("uv==0.13.4")
    .env({"HF_HOME": "/cache/hf", "UV_LINK_MODE": "copy", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(REPO, REMOTE, copy=True, ignore=IGNORE)
    .workdir(REMOTE)
    .run_commands("uv sync --frozen --no-dev --python 3.12")
)

app = modal.App("minfer", image=image)
weights = modal.Volume.from_name("minfer-hf-cache", create_if_missing=True)


@app.function(
    gpu="H100",
    volumes={"/cache": weights},
    scaledown_window=300,
    max_containers=1,
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=PORT, startup_timeout=600, label="minfer-serve")
def serve() -> None:
    """Start `minfer serve`; Modal routes HTTP traffic to it once the port answers."""
    command = [".venv/bin/python", "-m", "minfer.cli", "serve", "--model", MODEL]
    command += ["--host", "0.0.0.0", "--port", str(PORT)]
    subprocess.Popen(command, cwd=REMOTE)
