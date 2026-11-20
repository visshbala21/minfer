"""Load Qwen2 weights straight from ``.safetensors`` files."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from minfer.model.config import ModelConfig
from minfer.model.qwen2 import Qwen2Model

_CHECKPOINT_PREFIX = "model."


def resolve_model_dir(name_or_path: str | Path) -> Path:
    """Return a local directory holding ``config.json`` and the weight files.

    A local path is used as is; anything else is treated as a Hugging Face Hub repo id and
    fetched into ``$HF_HOME`` (or reused from it).
    """
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
    return Path(snapshot_download(str(name_or_path), allow_patterns=patterns))


def read_state_dict(
    model_dir: Path, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Read every tensor of a (possibly sharded) checkpoint onto ``device`` as ``dtype``.

    Tensors are materialized directly on the target device, so host memory never holds a
    second copy of the model. Names are returned without the checkpoint's ``model.`` prefix.
    """
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        with safe_open(file, framework="pt", device=str(device)) as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open handles are not iterable
                state[key.removeprefix(_CHECKPOINT_PREFIX)] = handle.get_tensor(key).to(dtype)
    return state


def load_model(
    name_or_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
) -> Qwen2Model:
    """Build a :class:`Qwen2Model` and load its weights.

    Args:
        name_or_path: Local model directory or Hugging Face Hub repo id.
        device: Device to place the model on.
        dtype: Weight dtype; defaults to the checkpoint's own (bfloat16 for Qwen2.5).

    Raises:
        KeyError: If the checkpoint and the architecture disagree on any tensor name.
    """
    model_dir = resolve_model_dir(name_or_path)
    config = ModelConfig.from_pretrained(model_dir)
    device = torch.device(device)
    dtype = dtype or config.torch_dtype

    model = Qwen2Model(config)
    state = read_state_dict(model_dir, device, dtype)
    if config.tie_word_embeddings:
        state.setdefault("lm_head.weight", state["embed_tokens.weight"])

    expected = set(model.state_dict().keys())
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise KeyError(
            f"checkpoint/architecture mismatch in {model_dir}: "
            f"missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}"
        )
    model.load_state_dict(state, strict=True, assign=True)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    model.to(device=device, dtype=dtype)
    return model.eval()
