"""Speculative decoding: drafters that propose tokens for the target model to verify."""

from minfer.spec.base import Draft, Drafter
from minfer.spec.ngram import NgramIndex, PromptLookupDrafter

__all__ = ["Draft", "Drafter", "NgramIndex", "PromptLookupDrafter"]
