"""Reproduce the benchmark suite on a fresh H100 with Modal.

    modal run bench/modal_bench.py                    # everything `make bench` runs
    modal run bench/modal_bench.py --target bench-agent

The image starts from NVIDIA's CUDA development image (FlashInfer JIT-compiles its kernels
with nvcc), installs uv, copies this repository and builds the three pinned virtualenvs
with the same script used locally (`scripts/setup_envs.sh`). The function then runs the
requested Makefile target on one H100 and returns the regenerated `bench/results/` and
`bench/plots/`, which are written back into the local checkout.

Model weights are cached in a Modal volume so that reruns do not download them again.

Not run from the machine this repository was developed on (it has no Modal credentials);
see PROGRESS.md.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/minfer"
IGNORE = [".git", ".venv*", "**/__pycache__", "bench/logs", "bench/traces", ".ruff_cache"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ninja-build")
    .pip_install("uv==0.13.4")
    .env({"HF_HOME": "/cache/hf", "UV_LINK_MODE": "copy", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(REPO, REMOTE, copy=True, ignore=IGNORE)
    .workdir(REMOTE)
    .run_commands("scripts/setup_envs.sh all")
)

app = modal.App("minfer-bench", image=image)
weights = modal.Volume.from_name("minfer-hf-cache", create_if_missing=True)


@app.function(gpu="H100", cpu=32, memory=131072, timeout=6 * 60 * 60, volumes={"/cache": weights})
def run(target: str = "bench") -> bytes:
    """Run ``make <target>`` and return ``bench/results`` + ``bench/plots`` as a tarball."""
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(["make", "env"], cwd=REMOTE, check=True)
    subprocess.run(["make", target], cwd=REMOTE, check=True)
    weights.commit()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("bench/results", "bench/plots", "bench/ENV.md"):
            archive.add(Path(REMOTE) / name, arcname=name)
    return buffer.getvalue()


@app.local_entrypoint()
def main(target: str = "bench") -> None:
    """Run the suite remotely and unpack the results into this checkout."""
    payload = run.remote(target)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(REPO, filter="data")
    print(f"results written to {REPO / 'bench' / 'results'}")
