# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.agentic_kv.statistics.contracts import StatisticsDestination


@dataclass(frozen=True)
class StatisticsCapacity:
    slots: int
    layers: int
    kv_heads: int
    max_tokens: int

    def __post_init__(self) -> None:
        if min(self.slots, self.layers, self.kv_heads, self.max_tokens) <= 0:
            raise ValueError("statistics capacity must be positive")

    @property
    def tensor_bytes(self) -> int:
        return self.slots * self.layers * self.kv_heads * (4 * self.max_tokens + 2) * 4


class StatisticsBuffers:
    """Fixed storage, allocated once; no growth or normalization outputs.

    Charge tensor_bytes to the extension startup budget before native KV sizing.
    Event/allocator overhead and attention scratch require separate accounting.
    CPU device is supported for contract tests, not serving activation.
    """

    def __init__(self, capacity: StatisticsCapacity, device: torch.device) -> None:
        self.capacity = capacity
        self.sums = torch.empty(
            (
                capacity.slots,
                capacity.layers,
                4,
                capacity.kv_heads,
                capacity.max_tokens,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.counts = torch.empty(
            (capacity.slots, capacity.layers, capacity.kv_heads),
            dtype=torch.int32,
            device=device,
        )
        self.flags = torch.empty_like(self.counts)

    def record_stream(self, stream: object) -> None:
        if self.sums.is_cuda:
            for tensor in (self.sums, self.counts, self.flags):
                tensor.record_stream(stream)

    def reset(self, slot: int, covered_tokens: int) -> StatisticsDestination:
        destination = StatisticsDestination(
            self.sums[slot, :, :, :, :covered_tokens],
            self.counts[slot],
            self.flags[slot],
        )
        destination.sums.zero_()
        destination.counts.zero_()
        destination.flags.zero_()
        return destination
