# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.agentic_kv.statistics.buffers import StatisticsBuffers, StatisticsCapacity
from vllm.agentic_kv.statistics.collector import StatisticsCollector
from vllm.agentic_kv.statistics.contracts import StatisticsIdentity


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
    collector = StatisticsCollector(buffers, writer, Event)
    return collector, buffers, writer


def identity(generation=0, covered_tokens=3):
    return StatisticsIdentity("A", generation, "source7", "bank7", covered_tokens)


def add_sample(destination):
    if destination is not None:
        destination.sums.add_(0.25)
        destination.counts.add_(1)


def test_sampling_keeps_plain_steps_and_direct_sums(setup):
    collector, buffers, _ = setup
    handle = collector.acquire(identity())
    seen = []

    def backend(destination):
        seen.append(destination is not None)
        add_sample(destination)

    for ordinal in range(1, 10):
        collector.submit(handle, ordinal, backend)
    assert seen == [True] + [False] * 7 + [True]
    snapshot = collector.seal(handle)
    assert snapshot.sample_count == 2
    assert snapshot.last_ordinal == 9
    stream = Stream()

    def read(destination):
        assert len(stream.waits) == 1
        assert destination.sums.shape == (2, 4, 2, 3)
        assert torch.all(destination.sums == 0.5)
        assert torch.all(destination.counts == 2)

    collector.consume(snapshot, stream, read)
    assert buffers.capacity.tensor_bytes == sum(
        t.numel() * t.element_size()
        for t in (buffers.sums, buffers.counts, buffers.flags)
    )


def test_early_return_and_reader_completion_bound_reuse(setup):
    collector, buffers, writer = setup
    pointer = buffers.sums.data_ptr()
    first = collector.acquire(identity())
    collector.submit(first, 1, add_sample)
    snapshot = collector.seal(first)
    second = collector.acquire(identity(1))
    assert second.slot != first.slot
    assert collector.acquire(identity(2)) is None
    assert writer.events[0].polls == 0  # Admission does not query GPU readiness.
    with pytest.raises(RuntimeError, match="closed"):
        collector.submit(first, 2, add_sample)
    reader = Stream()
    collector.consume(snapshot, reader, lambda destination: None)
    assert collector.reap() == 0
    assert collector.acquire(identity(2)) is None
    reader.events[-1].complete = True
    assert collector.reap() == 1
    third = collector.acquire(identity(2))
    assert third.slot == first.slot
    assert third.serial != first.serial
    assert buffers.sums.data_ptr() == pointer
    with pytest.raises(ValueError, match="stale"):
        collector.submit(first, 2, add_sample)


def test_failed_partial_submission_cannot_seal_or_reuse_early(setup):
    collector, _, writer = setup
    handle = collector.acquire(identity())

    def failing_backend(destination):
        add_sample(destination)
        raise RuntimeError("model failure after enqueue")

    with pytest.raises(RuntimeError, match="model failure"):
        collector.submit(handle, 1, failing_backend)
    with pytest.raises(RuntimeError, match="not ready"):
        collector.seal(handle)
    collector.abandon(handle)
    assert collector.reap() == 0
    writer.events[-1].complete = True
    assert collector.reap() == 1


def test_failing_consumer_holds_storage_until_its_work_finishes(setup):
    collector, _, _ = setup
    handle = collector.acquire(identity())
    collector.submit(handle, 1, add_sample)
    snapshot = collector.seal(handle)
    reader = Stream()

    def failing_reader(destination):
        raise RuntimeError("encode failed")

    with pytest.raises(RuntimeError, match="encode failed"):
        collector.consume(snapshot, reader, failing_reader)
    assert collector.reap() == 0
    reader.events[-1].complete = True
    assert collector.reap() == 1


def test_foreign_snapshot_and_duplicate_generation(setup):
    collector, _, _ = setup
    handle = collector.acquire(identity())
    with pytest.raises(ValueError, match="already held"):
        collector.acquire(identity())
    with pytest.raises(ValueError, match="capacity"):
        collector.acquire(identity(1, 9))
    with pytest.raises(ValueError, match="already held"):
        collector.acquire(replace(identity(), bank_version="different"))
    collector.submit(handle, 1, add_sample)
    snapshot = collector.seal(handle)
    with pytest.raises(RuntimeError, match="not available"):
        collector.consume(replace(snapshot), Stream(), lambda destination: None)
    collector.consume(snapshot, Stream(), lambda destination: None)
    with pytest.raises(RuntimeError, match="not available"):
        collector.consume(snapshot, Stream(), lambda destination: None)


def test_out_of_order_batch_fails_without_launching(setup):
    collector, _, _ = setup
    a, b = collector.acquire(identity()), collector.acquire(identity(1))
    with pytest.raises(RuntimeError, match="out-of-sequence"):
        collector.submit_batch([(a, 1), (b, 2)], lambda _: pytest.fail("launched"))
    for handle in (a, b):
        with pytest.raises(RuntimeError, match="not ready"):
            collector.seal(handle)


def test_native_owner_permutation_uses_one_batched_dispatch(setup):
    collector, _, _ = setup
    a, b = collector.acquire(identity()), collector.acquire(identity(1))
    collector.submit(a, 1, add_sample)
    calls = []

    def backend(destinations):
        calls.append(destinations)
        assert destinations[0] is not None  # B's first step.
        assert destinations[1] is None  # A's second step.
        for destination in destinations:
            add_sample(destination)

    collector.submit_batch([(b, 1), (a, 2)], backend)
    assert len(calls) == 1
    assert collector.seal(a).sample_count == collector.seal(b).sample_count == 1


def test_error_flags_are_preserved_until_consumer_validates(setup):
    collector, _, _ = setup
    handle = collector.acquire(identity())

    def mark_invalid(destination):
        add_sample(destination)
        destination.flags[0, 0] |= 1

    collector.submit(handle, 1, mark_invalid)
    for ordinal in range(2, 10):
        collector.submit(handle, ordinal, add_sample)

    def consumer(destination):
        # Fake publication gate; the real codec is not installed yet.
        valid = (destination.flags == 0).all() & (destination.counts > 0).all()
        assert not bool(valid)

    collector.consume(collector.seal(handle), Stream(), consumer)


def test_abandoned_snapshot_cannot_be_consumed(setup):
    collector, _, writer = setup
    handle = collector.acquire(identity())
    collector.submit(handle, 1, add_sample)
    snapshot = collector.seal(handle)
    collector.abandon(handle)
    with pytest.raises(RuntimeError, match="not available"):
        collector.consume(snapshot, Stream(), lambda destination: None)
    assert collector.reap() == 0
    writer.events[-1].complete = True
    assert collector.reap() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_cuda_stream_handoff_preserves_snapshot():
    device = torch.device("cuda")
    writer, reader = torch.cuda.Stream(), torch.cuda.Stream()
    buffers = StatisticsBuffers(StatisticsCapacity(1, 2, 2, 8), device)
    collector = StatisticsCollector(buffers, writer)
    result = torch.empty((2, 4, 2, 3), device=device)
    with torch.cuda.stream(writer):
        handle = collector.acquire(identity())
        collector.submit(handle, 1, add_sample)
        snapshot = collector.seal(handle)
    collector.consume(
        snapshot, reader, lambda destination: result.copy_(destination.sums)
    )
    reader.synchronize()  # Test boundary only; collector never synchronizes host.
    assert torch.all(result == 0.25)
    assert collector.reap() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_cuda_graph_replay_is_rejected_after_seal():
    device = torch.device("cuda")
    writer = torch.cuda.Stream()
    buffers = StatisticsBuffers(StatisticsCapacity(1, 2, 2, 8), device)
    collector = StatisticsCollector(buffers, writer)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(writer):
        handle = collector.acquire(identity())
        writer.synchronize()  # Isolate test capture setup.

        def capture_and_replay(destination):
            with torch.cuda.graph(graph, stream=writer):
                add_sample(destination)
            graph.replay()

        collector.submit(handle, 1, capture_and_replay)
        snapshot = collector.seal(handle)
        with pytest.raises(RuntimeError, match="closed"):
            collector.submit(handle, 2, lambda _: graph.replay())
    with pytest.raises(RuntimeError, match="model stream"):
        collector.acquire(identity(1))
    collector.consume(snapshot, writer, lambda _: None)
    writer.synchronize()
    assert collector.reap() == 1
