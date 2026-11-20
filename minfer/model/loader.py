"""Load Qwen2 weights straight from ``.safetensors`` files."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from minfer.model.config import ModelConfig
from minfer.model.qwen2 import Qwen2Model, Shard

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


def shard_state_dict(
    state: dict[str, torch.Tensor], config: ModelConfig, rank: int, world_size: int
) -> dict[str, torch.Tensor]:
    """Slice a full state dict down to one tensor-parallel rank's share.

    Column-parallel layers (q/k/v, gate/up) keep the *rows* that produce this rank's heads
    or hidden units; row-parallel layers (o_proj, down_proj) keep the matching *columns*.
    Everything else (embeddings, norms, lm_head) is replicated.
    """
    head_dim = config.head_dim

    def part(total: int) -> slice:
        share = total // world_size
        return slice(rank * share, (rank + 1) * share)

    q_rows = part(config.num_heads * head_dim)
    kv_rows = part(config.num_kv_heads * head_dim)
    mlp_units = part(config.intermediate_size)
    sliced = {}
    for name, tensor in state.items():
        leaf = name.rsplit(".", 2)[-2:]
        if leaf[0] == "q_proj":
            tensor = tensor[q_rows]
        elif leaf[0] in ("k_proj", "v_proj"):
            tensor = tensor[kv_rows]
        elif leaf[0] == "o_proj":
            tensor = tensor[:, q_rows]
        elif leaf[0] in ("gate_proj", "up_proj"):
            tensor = tensor[mlp_units]
        elif leaf[0] == "down_proj":
            tensor = tensor[:, mlp_units]
        sliced[name] = tensor.contiguous()
    return sliced


def load_model(
    name_or_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    shard: Shard | None = None,
) -> Qwen2Model:
    """Build a :class:`Qwen2Model` and load its weights.

    Args:
        name_or_path: Local model directory or Hugging Face Hub repo id.
        device: Device to place the model on.
        dtype: Weight dtype; defaults to the checkpoint's own (bfloat16 for Qwen2.5).
        shard: Load only this tensor-parallel rank's slice of the weights.

    Raises:
        KeyError: If the checkpoint and the architecture disagree on any tensor name.
    """
    model_dir = resolve_model_dir(name_or_path)
    config = ModelConfig.from_pretrained(model_dir)
    device = torch.device(device)
    dtype = dtype or config.torch_dtype

    model = Qwen2Model(config, shard)
    state = read_state_dict(model_dir, device, dtype)
    if config.tie_word_embeddings:
        state.setdefault("lm_head.weight", state["embed_tokens.weight"])
    if shard is not None:
        state = shard_state_dict(state, config, shard.rank, shard.world_size)

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
