# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-address routes for single-model-stream eager execution and replay."""

from collections.abc import Callable, Sequence

import torch

from vllm.agentic_kv.statistics.contracts import CompletionEvent, StatisticsHandle
from vllm.agentic_kv.statistics.forge import resolve_routes
from vllm.agentic_kv.statistics.pool import StatisticsPool


class StatisticsRoutes:
    """Bounded H2D staging; no request registry or execution scheduling.

    Allocate before profiling/capture. All route copies and readers must use the
    pool's model stream. Graphs capture only readers of gpu, never stage(). CPU
    staging slots retire after their H2D copy, not after the whole model step.
    """

    def __init__(
        self,
        pool: StatisticsPool,
        max_num_reqs: int,
        max_concurrent_copies: int,
        event_factory: Callable[[], CompletionEvent] = torch.cuda.Event,
    ) -> None:
        if min(max_num_reqs, max_concurrent_copies) <= 0:
            raise ValueError("route capacity must be positive")
        self.pool = pool
        self.max_num_reqs = max_num_reqs
        device = pool.buffers.sums.device
        self.gpu = torch.full((max_num_reqs, 2), -1, dtype=torch.int64, device=device)
        self._cpu = [
            torch.empty(
                (max_num_reqs, 2), dtype=torch.int64, pin_memory=device.type == "cuda"
            )
            for _ in range(max_concurrent_copies)
        ]
        self._arrays = [buffer.numpy() for buffer in self._cpu]
        self._events = [event_factory() for _ in self._cpu]
        self._free = list(range(max_concurrent_copies))
        self._pending: list[int] = []
        if self.gpu.is_cuda:
            if torch.cuda.current_stream(device) != pool.writer_stream:
                raise RuntimeError("allocate routes on the bound model stream")
            self.gpu.record_stream(pool.writer_stream)

    @property
    def device_bytes(self) -> int:
        return self.max_num_reqs * 2 * 8

    @property
    def host_bytes(self) -> int:
        return self.device_bytes * len(self._cpu)

    def stage(self, owners: Sequence[StatisticsHandle | None]) -> bool:
        """Validate and enqueue routes immediately before model submission.

        Returns whether any observation was enabled. On staging exhaustion, enqueue
        disabled routes and decline only this observation. Never wait or grow.
        Stale/duplicate handles fail before GPU mutation. Call on every replay,
        including all-unsampled and dummy batches, to erase previous destinations.
        """
        if self.gpu.is_cuda:
            if torch.cuda.current_stream(self.gpu.device) != self.pool.writer_stream:
                raise RuntimeError("stage routes on the bound model stream")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("route staging must run outside graph capture")
        if len(owners) > self.max_num_reqs:
            raise ValueError("batch exceeds reserved route capacity")
        routes = resolve_routes(self.pool, owners)
        if not any(owner is not None for owner in owners):
            self.gpu.fill_(-1)
            return False
        pending = []
        for index in self._pending:
            if self._events[index].query():
                self._free.append(index)
            else:
                pending.append(index)
        self._pending = pending
        if not self._free:
            self.gpu.fill_(-1)
            return False
        index = self._free.pop()
        # A failed copy/event record quarantines this slot rather than reusing it.
        array = self._arrays[index]
        array.fill(-1)
        array[: len(routes)] = routes
        self.gpu.copy_(self._cpu[index], non_blocking=True)
        self._events[index].record(self.pool.writer_stream)
        self._pending.append(index)
        return True
