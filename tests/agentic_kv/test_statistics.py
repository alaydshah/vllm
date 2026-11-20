# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.agentic_kv.statistics.buffers import StatisticsBuffers, StatisticsCapacity
from vllm.agentic_kv.statistics.contracts import StatisticsIdentity
from vllm.agentic_kv.statistics.pool import StatisticsPool


class Event:
    def __init__(self):
        self.complete = False
        self.recorded = False
        self.polls = 0

    def record(self, stream):
        self.recorded = True
        self.complete = False
        stream.events.append(self)

    def query(self):
        assert self.recorded
        self.polls += 1
        return self.complete


class Stream:
    def __init__(self):
        self.events = []
        self.waits = []

    def wait_event(self, event):
        assert event.recorded
        self.waits.append(event)


@pytest.fixture
def setup():
    buffers = StatisticsBuffers(StatisticsCapacity(2, 2, 2, 8), torch.device("cpu"))
    writer = Stream()
    return StatisticsPool(buffers, writer, Event), buffers, writer


def identity(generation=0, covered_tokens=3):
    return StatisticsIdentity("A", generation, "source7", "bank7", covered_tokens)


def add_sample(destination):
    destination.sums.add_(0.25)
    destination.counts.add_(1)


def test_request_detach_transfers_ownership_without_submission_wrapper(setup):
    pool, buffers, _ = setup
    request = SimpleNamespace(stats=pool.acquire(identity()))
    add_sample(pool.writable(request.stats))
    add_sample(pool.writable(request.stats))
    handle = request.stats
    lease = pool.detach(handle)
    request.stats = None
    with pytest.raises(ValueError, match="transferred"):
        pool.writable(handle)
    reader = Stream()

    def read(destination):
        assert len(reader.waits) == 1
        assert destination.sums.shape == (2, 4, 2, 3)
        assert torch.all(destination.sums == 0.5)
        assert torch.all(destination.counts == 2)

    pool.consume(lease, reader, read)
    assert buffers.capacity.tensor_bytes == sum(
        t.numel() * t.element_size()
        for t in (buffers.sums, buffers.counts, buffers.flags)
    )


def test_early_return_and_reader_completion_bound_reuse(setup):
    pool, buffers, writer = setup
    pointer = buffers.sums.data_ptr()
    first = pool.acquire(identity())
    add_sample(pool.writable(first))
    lease = pool.detach(first)
    second = pool.acquire(identity(1))
    assert second.slot != first.slot
    assert pool.acquire(identity(2)) is None
    assert writer.events[0].polls == 0
    reader = Stream()
    pool.consume(lease, reader, lambda _: None)
    assert pool.reap() == 0
    reader.events[-1].complete = True
    assert pool.reap() == 1
    third = pool.acquire(identity(2))
    assert third.slot == first.slot and third.serial != first.serial
    assert buffers.sums.data_ptr() == pointer
    assert torch.all(pool.writable(third).counts == 0)
    with pytest.raises(ValueError, match="stale"):
        pool.writable(first)


def test_native_failure_discards_partial_writes_after_completion(setup):
    pool, _, writer = setup
    handle = pool.acquire(identity())
    add_sample(pool.writable(handle))
    # Native model failure/cancellation drives discard, not a pool state machine.
    pool.discard(handle)
    assert pool.reap() == 0
    with pytest.raises(ValueError, match="transferred"):
        pool.writable(handle)
    writer.events[-1].complete = True
    assert pool.reap() == 1


def test_consumer_failure_and_reentrancy_cannot_release_early(setup):
    pool, _, _ = setup
    lease = pool.detach(pool.acquire(identity()))
    reader = Stream()

    def failing_reader(destination):
        with pytest.raises(ValueError, match="transferred"):
            pool.consume(lease, reader, lambda _: None)
        assert pool.reap() == 0  # Completion not yet recorded/enqueued.
        raise RuntimeError("encode failed after enqueue")

    with pytest.raises(RuntimeError, match="encode failed"):
        pool.consume(lease, reader, failing_reader)
    assert pool.reap() == 0
    reader.events[-1].complete = True
    assert pool.reap() == 1


def test_foreign_owner_and_duplicate_lease_consumption_rejected(setup):
    pool, _, _ = setup
    handle = pool.acquire(identity())
    with pytest.raises(ValueError, match="stale"):
        pool.writable(replace(handle))
    with pytest.raises(ValueError, match="capacity"):
        pool.acquire(identity(1, 9))
    lease = pool.detach(handle)
    with pytest.raises(ValueError, match="stale"):
        pool.consume(replace(lease), Stream(), lambda _: None)
    pool.consume(lease, Stream(), lambda _: None)
    with pytest.raises(ValueError, match="transferred"):
        pool.discard(lease)


def test_zero_samples_and_error_flags_are_consumer_policy(setup):
    pool, _, _ = setup
    first, second = pool.acquire(identity()), pool.acquire(identity(1))
    pool.writable(second).flags[0, 0] = 1

    def check_empty(destination):
        assert torch.all(destination.counts == 0)

    def check_invalid(destination):
        assert destination.flags[0, 0] == 1

    pool.consume(pool.detach(first), Stream(), check_empty)
    pool.consume(pool.detach(second), Stream(), check_invalid)


def test_declined_save_retires_unread_lease_after_last_write(setup):
    pool, _, writer = setup
    lease = pool.detach(pool.acquire(identity()))
    pool.discard(lease)
    assert pool.reap() == 0
    writer.events[-1].complete = True
    assert pool.reap() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_cuda_stream_handoff_preserves_detached_data():
    writer, reader = torch.cuda.Stream(), torch.cuda.Stream()
    buffers = StatisticsBuffers(StatisticsCapacity(1, 2, 2, 8), torch.device("cuda"))
    pool = StatisticsPool(buffers, writer)
    result = torch.empty((2, 4, 2, 3), device="cuda")
    with torch.cuda.stream(writer):
        handle = pool.acquire(identity())
        add_sample(pool.writable(handle))
        lease = pool.detach(handle)
    pool.consume(lease, reader, lambda destination: result.copy_(destination.sums))
    reader.synchronize()  # Test boundary only.
    assert torch.all(result == 0.25)
    assert pool.reap() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_cuda_graph_adapter_must_resolve_handle_at_submission():
    writer = torch.cuda.Stream()
    buffers = StatisticsBuffers(StatisticsCapacity(1, 2, 2, 8), torch.device("cuda"))
    pool = StatisticsPool(buffers, writer)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(writer):
        handle = pool.acquire(identity())
        destination = pool.writable(handle)
        writer.synchronize()
        with torch.cuda.graph(graph, stream=writer):
            add_sample(destination)

        def native_adapter_submit():
            pool.writable(handle)  # Must run on every submission, not only capture.
            graph.replay()

        native_adapter_submit()
        lease = pool.detach(handle)
        with pytest.raises(ValueError, match="transferred"):
            native_adapter_submit()
    with pytest.raises(RuntimeError, match="model stream"):
        pool.acquire(identity(1))
    pool.consume(lease, writer, lambda _: None)
    writer.synchronize()
    assert pool.reap() == 1
