# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm.agentic_kv.statistics.buffers import StatisticsBuffers
from vllm.agentic_kv.statistics.contracts import (
    CompletionEvent,
    CompletionStream,
    StatisticsDestination,
    StatisticsHandle,
    StatisticsIdentity,
    StatisticsLease,
)


@dataclass
class _Slot:
    ready: CompletionEvent
    done: CompletionEvent
    owner: StatisticsHandle | StatisticsLease | CompletionEvent | None = None
    destination: StatisticsDestination | None = None


class StatisticsPool:
    """Bounded worker-local storage, independent of request execution policy.

    Native request state owns a writable handle. Detachment transfers ownership
    to a save lease, then a completion event retains storage until safe reuse.
    All methods run on one worker control thread. The adapter must prevent new
    writes after detachment, including already prepared batches/graph replays.
    """

    def __init__(
        self,
        buffers: StatisticsBuffers,
        writer_stream: object,
        event_factory: Callable[[], CompletionEvent] = torch.cuda.Event,
    ) -> None:
        self.buffers = buffers
        self.writer_stream = writer_stream
        buffers.record_stream(writer_stream)
        self._slots = [
            _Slot(event_factory(), event_factory())
            for _ in range(buffers.capacity.slots)
        ]
        self._pending: list[tuple[_Slot, CompletionEvent]] = []
        self._serial = 0

    def _check_writer(self) -> None:
        if self.buffers.sums.is_cuda and (
            torch.cuda.current_stream(self.buffers.sums.device) != self.writer_stream
        ):
            raise RuntimeError("statistics producer must use the bound model stream")

    def _owned(self, owner: StatisticsHandle | StatisticsLease) -> _Slot:
        handle = owner.handle if isinstance(owner, StatisticsLease) else owner
        if not 0 <= handle.slot < len(self._slots):
            raise ValueError("foreign statistics owner")
        slot = self._slots[handle.slot]
        if slot.owner is not owner:
            raise ValueError("stale or transferred statistics owner")
        return slot

    def acquire(self, identity: StatisticsIdentity) -> StatisticsHandle | None:
        """Reserve storage; native request state must attach the returned handle.

        No owner registry, sampling decision, readiness query or storage growth.
        Caller prevents duplicate acquisition for an already attached request.
        """
        self._check_writer()
        if identity.covered_tokens > self.buffers.capacity.max_tokens:
            raise ValueError("statistics coverage exceeds reserved capacity")
        for index, slot in enumerate(self._slots):
            if slot.owner is not None:
                continue
            self._serial += 1
            handle = StatisticsHandle(identity, index, self._serial)
            slot.owner = handle
            slot.destination = self.buffers.reset(index, identity.covered_tokens)
            return handle
        return None

    def writable(self, handle: StatisticsHandle) -> StatisticsDestination:
        """Borrow destinations while the native request still owns the handle.

        Re-resolve at submission, not only when preparing/capturing a batch.
        Previously borrowed tensor references cannot be revoked by this pool.
        Sampling, ordinals, GPU counts/flags and model failures belong to callers.
        """
        self._check_writer()
        slot = self._owned(handle)
        assert slot.destination is not None
        return slot.destination

    def detach(self, handle: StatisticsHandle) -> StatisticsLease:
        """Transfer after final permitted writes; caller clears request metadata.

        Seal ownership before recording readiness. If recording fails, keep the
        allocation quarantined; a failed device operation never frees storage.
        """
        self._check_writer()
        slot = self._owned(handle)
        lease = StatisticsLease(handle)
        slot.owner = lease
        slot.ready.record(self.writer_stream)
        return lease

    def consume(
        self,
        lease: StatisticsLease,
        stream: CompletionStream,
        enqueue: Callable[[StatisticsDestination], None],
    ) -> None:
        """Transfer a lease to one GPU reader and retain it through completion.

        Enqueue the entire read lifetime on this stream, without retaining tensor
        views. The save operation owns validity and publication, not the pool.
        Partial submission failures still retire only after the last GPU read.
        """
        slot = self._owned(lease)
        slot.owner = slot.done  # Claim once, including reentrant callbacks.
        context = (
            torch.cuda.stream(stream) if self.buffers.sums.is_cuda else nullcontext()
        )
        with context:
            try:
                stream.wait_event(slot.ready)
                self.buffers.record_stream(stream)
                assert slot.destination is not None
                enqueue(slot.destination)
            finally:
                slot.done.record(stream)
                self._pending.append((slot, slot.done))

    def discard(self, owner: StatisticsHandle | StatisticsLease) -> None:
        """Cancel/decline a save, retaining any already-submitted model writes."""
        if isinstance(owner, StatisticsHandle):
            owner = self.detach(owner)
        slot = self._owned(owner)
        slot.owner = slot.ready
        self._pending.append((slot, slot.ready))

    def reap(self) -> int:
        """Nonblocking completion processing; never called by allocation."""
        pending = []
        released = 0
        for slot, completion in self._pending:
            if completion.query():
                slot.destination = None
                slot.owner = None
                released += 1
            else:
                pending.append((slot, completion))
        self._pending = pending
        return released
