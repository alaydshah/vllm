# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
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
    StatisticsSnapshot,
)


@dataclass
class _Slot:
    ready: CompletionEvent
    done: CompletionEvent
    handle: StatisticsHandle | None = None
    destination: StatisticsDestination | None = None
    phase: str = "free"
    ordinal: int = 0
    samples: int = 0
    retirement: CompletionEvent | None = None
    snapshot: StatisticsSnapshot | None = None


class StatisticsCollector:
    """Single worker-thread owner of bounded producer and snapshot leases.

    All producers use one model stream. Every execution, including graph replay,
    must pass through submit. Consumers enqueue their entire read lifetime on
    one stream through consume. No callback may retain destination views for
    later work. This component does not publish history or validate model output.
    A consumer must gate publication on GPU flags/counts AND model success.
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
        self._serial = 0

    def _check_writer(self) -> None:
        if self.buffers.sums.is_cuda and (
            torch.cuda.current_stream(self.buffers.sums.device) != self.writer_stream
        ):
            raise RuntimeError("statistics producer must use the bound model stream")

    def _get(self, handle: StatisticsHandle) -> _Slot:
        if not 0 <= handle.slot < len(self._slots):
            raise ValueError("foreign statistics handle")
        slot = self._slots[handle.slot]
        if slot.handle is not handle:
            raise ValueError("stale or foreign statistics handle")
        return slot

    def acquire(self, identity: StatisticsIdentity) -> StatisticsHandle | None:
        """Return None on capacity pressure; never poll/wait or grow storage."""
        self._check_writer()
        if identity.covered_tokens > self.buffers.capacity.max_tokens:
            raise ValueError("statistics coverage exceeds reserved capacity")
        if any(
            s.handle
            and (s.handle.identity.request_id, s.handle.identity.generation)
            == (identity.request_id, identity.generation)
            for s in self._slots
        ):
            raise ValueError("statistics generation already held")
        for index, slot in enumerate(self._slots):
            if slot.phase != "free":
                continue
            self._serial += 1
            slot.handle = StatisticsHandle(identity, index, self._serial)
            slot.phase = "failed"
            slot.destination = self.buffers.reset(index, identity.covered_tokens)
            slot.ordinal = slot.samples = 0
            slot.snapshot = None
            slot.phase = "active"
            return slot.handle
        return None

    def submit(
        self,
        handle: StatisticsHandle,
        ordinal: int,
        enqueue: Callable[[StatisticsDestination | None], None],
    ) -> None:
        """Call the backend with a destination only at ordinals 1, 9, 17, ...

        None means plain attention, not skipping model execution. The callback
        owns routing and GPU count/flag updates. This guard runs on every replay.
        """
        self.submit_batch(
            [(handle, ordinal)], lambda destinations: enqueue(destinations[0])
        )

    def submit_batch(
        self,
        observations: Sequence[tuple[StatisticsHandle, int]],
        enqueue: Callable[[tuple[StatisticsDestination | None, ...]], None],
    ) -> None:
        """Guard one batched launch in caller-supplied native owner order.

        No persistent batch-slot mapping is introduced. The backend stages the
        destinations/mask using its existing batch preparation, then enqueues
        model work once. A partial failure invalidates every submitted owner.
        """
        self._check_writer()
        slots = [self._get(handle) for handle, _ in observations]
        if not slots or len({id(slot) for slot in slots}) != len(slots):
            raise ValueError("batch must contain distinct statistics generations")
        if any(slot.phase != "active" for slot in slots):
            raise RuntimeError("closed statistics generation")
        if any(
            ordinal != slot.ordinal + 1
            for slot, (_, ordinal) in zip(slots, observations)
        ):
            for slot in slots:
                slot.phase = "failed"
            raise RuntimeError("out-of-sequence observation")
        sampled = [(ordinal - 1) % 8 == 0 for _, ordinal in observations]
        destinations = tuple(
            slot.destination if due else None for slot, due in zip(slots, sampled)
        )
        for slot in slots:
            slot.phase = "submitting"
        try:
            enqueue(destinations)
        except BaseException:
            for slot in slots:
                slot.phase = "failed"
            raise
        for slot, (_, ordinal), due in zip(slots, observations, sampled):
            slot.ordinal = ordinal
            slot.samples += int(due)
            slot.phase = "active"

    def seal(self, handle: StatisticsHandle) -> StatisticsSnapshot:
        self._check_writer()
        slot = self._get(handle)
        if slot.phase != "active" or slot.ordinal == 0:
            raise RuntimeError("generation is not ready to seal")
        slot.phase = "failed"
        slot.ready.record(self.writer_stream)
        slot.snapshot = StatisticsSnapshot(handle, slot.ordinal, slot.samples)
        slot.phase = "sealed"
        return slot.snapshot

    def consume(
        self,
        snapshot: StatisticsSnapshot,
        stream: CompletionStream,
        enqueue: Callable[[StatisticsDestination], None],
    ) -> None:
        """Enqueue one consumer; keep storage charged until its completion event.

        Even a partially submitted, failing consumer retains its buffers. The
        callback must reject invalid candidates; this method grants read access,
        not publication permission. No normalization/output tensors are allocated.
        """
        slot = self._get(snapshot.handle)
        if slot.phase != "sealed" or slot.snapshot is not snapshot:
            raise RuntimeError("snapshot is not available for consumption")
        slot.phase = "reading"
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
                slot.retirement = slot.done
                slot.phase = "retiring"

    def abandon(self, handle: StatisticsHandle) -> None:
        """Drop an active, failed or unread sealed generation after its writes."""
        self._check_writer()
        slot = self._get(handle)
        if slot.phase not in ("active", "failed", "sealed"):
            raise RuntimeError("generation has an in-flight callback or retirement")
        slot.phase = "failed"
        slot.ready.record(self.writer_stream)
        slot.retirement = slot.ready
        slot.phase = "retiring"

    def reap(self) -> int:
        """Poll at a background completion boundary, never inside acquire/submit."""
        released = 0
        for slot in self._slots:
            if slot.phase != "retiring":
                continue
            assert slot.retirement is not None
            if not slot.retirement.query():
                continue
            slot.handle = None
            slot.destination = None
            slot.snapshot = None
            slot.retirement = None
            slot.phase = "free"
            released += 1
        return released
