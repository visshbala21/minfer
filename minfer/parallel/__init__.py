"""Tensor parallelism: one model split across several devices."""

from minfer.parallel.group import TensorParallelGroup, all_reduce

__all__ = ["TensorParallelGroup", "all_reduce"]
