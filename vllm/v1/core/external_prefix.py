# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Coordinates for read-only external history and native writable suffix KV.

This metadata does not own storage or establish a cache hit. A caller must hold
an external-storage lease before using its coverage. Model positions remain
absolute; only native storage coordinates are relative to the suffix.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalPrefixCoverage:
    """Whole-page prefix coverage for one full-attention cache group.

    Not yet connected to the scheduler or worker. Hybrid state, local-prefix
    composition and speculative execution require separate integration.

    Args:
        num_tokens: Number of positions backed by external storage.
        block_size: Native cache group's block size in tokens.
    """

    num_tokens: int
    block_size: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_tokens < 0 or self.num_tokens % self.block_size:
            raise ValueError("external prefix must cover complete blocks")

    def native_num_tokens(self, absolute_end: int) -> int:
        """Return native suffix coverage for an exclusive absolute end."""
        if absolute_end < self.num_tokens:
            raise ValueError("absolute end precedes external prefix")
        return absolute_end - self.num_tokens

    def native_slot(self, absolute_position: int, block_ids: list[int]) -> int:
        """Resolve a writable position through a suffix-only native block table.

        Raises:
            ValueError: Position is external or the physical block ID is invalid.
            IndexError: Native suffix capacity has not been allocated.
        """
        offset = self.native_num_tokens(absolute_position)
        block_index, row = divmod(offset, self.block_size)
        if block_index >= len(block_ids):
            raise IndexError("native suffix capacity is not allocated")
        block_id = block_ids[block_index]
        if block_id < 0:
            raise ValueError("native block ID must be non-negative")
        return block_id * self.block_size + row
