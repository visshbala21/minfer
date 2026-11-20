"""The interface between the engine and a source of draft tokens."""

from __future__ import annotations

from collections.abc import Sequence as Seq
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from minfer.scheduler.sequence import Sequence


@dataclass
class Draft:
    """Tokens proposed as the continuation of one sequence.

    Attributes:
        tokens: The proposed tokens, in order (possibly empty: nothing to verify).
        probs: ``[len(tokens), vocab]`` — the distribution each token was *sampled* from,
            for stochastic requests drafted by a model. ``None`` means the proposal is
            deterministic (a point mass on each token), as for prompt lookup and for
            greedy requests.
    """

    tokens: list[int]
    probs: Tensor | None = None


class Drafter(Protocol):
    """Proposes continuations for decoding sequences."""

    def propose(self, seqs: Seq[Sequence], limits: Seq[int]) -> list[Draft]:
        """One draft per sequence, at most ``limits[i]`` tokens long."""
        ...

    def release(self, seq_id: int) -> None:
        """Forget a sequence (finished, aborted or preempted)."""
        ...
