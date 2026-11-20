"""Prompt-lookup drafting: propose the continuation of an earlier occurrence of the context.

Agent outputs repeat their inputs — tool names, JSON keys, file paths, code copied from a
tool result into an edit. If the last ``n`` tokens of a sequence occurred before, the
tokens that followed them then are a free guess at what follows now. No draft model, no
extra forward pass: the cost is a dictionary lookup per step.
"""

from __future__ import annotations

from collections.abc import Sequence as Seq

from minfer.scheduler.sequence import Sequence
from minfer.spec.base import Draft


class NgramIndex:
    """Positions of the n-grams of one growing token sequence.

    Maps each n-gram to the position just after its most recent occurrence, excluding the
    occurrence that ends the sequence (which has no continuation yet). The index is
    extended incrementally as tokens are appended, so a lookup is O(1) amortized.
    """

    __slots__ = ("_indexed", "_positions", "n")

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self._positions: dict[tuple[int, ...], int] = {}
        self._indexed = 0  # n-grams ending before this position are in the dictionary

    def propose(self, tokens: Seq[int], max_tokens: int) -> list[int]:
        """Up to ``max_tokens`` tokens that followed the last n-gram of ``tokens`` earlier.

        Returns an empty list when the context has not occurred before.
        """
        n, length = self.n, len(tokens)
        if length < n + 1 or max_tokens < 1:
            return []
        positions = self._positions
        # Index every n-gram that ends strictly before the last token.
        for end in range(max(self._indexed, n), length):
            positions[tuple(tokens[end - n : end])] = end
        self._indexed = length
        start = positions.get(tuple(tokens[length - n :]))
        if start is None:
            return []
        return list(tokens[start : start + max_tokens])

    def truncate(self) -> None:
        """Forget everything (the sequence was rewound or replaced)."""
        self._positions.clear()
        self._indexed = 0


class PromptLookupDrafter:
    """Drafter that proposes prompt-lookup continuations (one :class:`NgramIndex` each)."""

    def __init__(self, ngram_size: int = 3) -> None:
        self.ngram_size = ngram_size
        self._indexes: dict[int, NgramIndex] = {}

    def propose(self, seqs: Seq[Sequence], limits: Seq[int]) -> list[Draft]:
        """See :meth:`minfer.spec.base.Drafter.propose`."""
        drafts = []
        for seq, limit in zip(seqs, limits, strict=True):
            index = self._indexes.get(seq.seq_id)
            if index is None:
                index = self._indexes[seq.seq_id] = NgramIndex(self.ngram_size)
            drafts.append(Draft(index.propose(seq.tokens, limit)))
        return drafts

    def release(self, seq_id: int) -> None:
        """See :meth:`minfer.spec.base.Drafter.release`."""
        self._indexes.pop(seq_id, None)
