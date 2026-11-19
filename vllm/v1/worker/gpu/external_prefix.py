# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit worker-side coverage state; serving activation remains unwired."""

import torch

from vllm.v1.core.external_prefix import ExternalPrefixCoverage
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor


class ExternalPrefixState:
    """Stage immutable request coverage into stable slot-mapping input storage.

    Construct only for an enabled external-prefix reader. Callers remove state
    on finish/preemption, add it on admission, then apply writes before preparing
    attention. Updates and readers must use the same ordered GPU stream (or
    equivalent event dependencies); this is not a cross-stream lifetime fence.
    Bank handles, readiness and storage leases belong to the connector/reader.
    """

    def __init__(
        self, block_sizes: list[int], max_num_reqs: int, device: torch.device
    ) -> None:
        if not block_sizes or any(size <= 0 for size in block_sizes):
            raise ValueError("external prefix requires positive group block sizes")
        if max_num_reqs <= 0:
            raise ValueError("max_num_reqs must be positive")
        self.block_sizes = tuple(block_sizes)
        self.max_num_reqs = max_num_reqs
        self._lengths = StagedWriteTensor(
            (len(block_sizes), max_num_reqs), dtype=torch.int64, device=device
        )
        self._requests: dict[str, int] = {}
        self._occupied: set[int] = set()
        self._pending: dict[int, int] = {}

    @property
    def lengths(self) -> torch.Tensor:
        """Persistent [group, request-state slot] tensor for slot mapping."""
        return self._lengths.gpu

    def add_request(self, request_id: str, slot: int, num_tokens: int) -> None:
        """Validate all groups before installing a request's fixed coverage."""
        if not 0 <= slot < self.max_num_reqs:
            raise ValueError("request slot is outside capacity")
        if request_id in self._requests or slot in self._occupied:
            raise ValueError("request or slot is already registered")
        for block_size in self.block_sizes:
            ExternalPrefixCoverage(num_tokens, block_size)
        self._requests[request_id] = slot
        self._occupied.add(slot)
        self._pending[slot] = num_tokens

    def remove_request(self, request_id: str) -> None:
        """Clear coverage without affecting another request reusing the slot."""
        slot = self._requests.pop(request_id, None)
        if slot is not None:
            self._occupied.remove(slot)
            self._pending[slot] = 0

    def apply_staged_writes(self) -> None:
        """Enqueue changed offsets; no device readback or global synchronize.

        Coalesce updates before staging so remove/add in one batch cannot race
        two GPU writes to the same element. Unchanged decode steps do no work.
        """
        if not self._pending:
            return
        for group in range(len(self.block_sizes)):
            for slot, num_tokens in self._pending.items():
                self._lengths.stage_write(group, slot, [num_tokens])
        self._lengths.apply_write()
        self._pending.clear()
