"""B-013 diagnostic: minfer full stack, JCT at 128 sessions, with the staging cap off.

    .venv/bin/python bench/results/b013/diagnostic.py

Runs ``bench.jct`` for ``minfer/session_host`` at 128 sessions with
``--host-staging-mb 0`` added. ``metrics_poll.log`` next to this file is the server's
``/metrics`` (staging and KV lines) and the GPU's used memory, sampled every 20 s meanwhile.
"""

import importlib
import sys

import bench.pressure as pressure


def main() -> None:
    label, flags, hints = pressure.CONFIGS["minfer/session_host"]
    pressure.CONFIGS["minfer/session_host"] = (
        f"{label} (no staging cap)",
        [*flags, "--host-staging-mb", "0"],
        hints,
    )
    jct = importlib.import_module("bench.jct")  # after the patch: it copies CONFIGS
    sys.argv = ["jct", "--configs", "minfer/session_host", "--concurrency", "128"]
    sys.argv += ["--out", "/tmp/b013_nocap.json"]
    jct.main()


if __name__ == "__main__":
    main()
