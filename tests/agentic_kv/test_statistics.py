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
        with torch.cuda.graph(graph):
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


def test_forge_routes_follow_native_order_and_reject_stale_or_duplicate_handles():
    from vllm.agentic_kv.statistics.forge import resolve_routes

    buffers = StatisticsBuffers(StatisticsCapacity(2, 2, 4, 8), torch.device("cpu"))
    pool = StatisticsPool(buffers, Stream(), Event)
    a, b = pool.acquire(identity()), pool.acquire(identity(1, 7))
    assert resolve_routes(pool, [b, None, a]) == ((b.slot, 7), (-1, 0), (a.slot, 3))
    with pytest.raises(ValueError, match="duplicate"):
        resolve_routes(pool, [a, a])
    pool.detach(a)
    with pytest.raises(ValueError, match="transferred"):
        resolve_routes(pool, [a])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_forge_reduction_routes_coverage_and_flags_across_graph_replay():
    from vllm.agentic_kv.statistics.forge import (
        accumulate_forge_statistics,
        load_forge_reduction,
    )
    from vllm.agentic_kv.statistics.routes import StatisticsRoutes

    device = torch.device("cuda")
    writer = torch.cuda.current_stream()
    buffers = StatisticsBuffers(StatisticsCapacity(2, 2, 4, 37), device)
    buffers.sums.fill_(-123)
    pool = StatisticsPool(buffers, writer)
    a, b = pool.acquire(identity(0, 17)), pool.acquire(identity(1, 9))
    meta = torch.zeros((8, 10), dtype=torch.int64, device=device)
    meta[:, 1] = 32
    scores = torch.zeros((8, 32, 8), device=device)
    scores[:, :, 3] = scores[:, :, 7] = float("nan")
    summary = torch.zeros((48, 18), device=device)
    summary[:, 1], summary[:, 2], summary[:, 3] = 1 / 32, 0.5, 2.0
    staging = StatisticsRoutes(pool, 2, 2)
    assert staging.stage([b, None])
    routes = staging.gpu

    kernel = load_forge_reduction()

    def reduce():
        accumulate_forge_statistics(meta, scores, summary, routes, buffers, 1, kernel)

    reduce()
    writer.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reduce()
    assert staging.stage([b, None])
    graph.replay()  # B's second observation.
    assert staging.stage([None, a])
    graph.replay()  # Same graph now writes A, not B.
    expected = torch.tensor([1 / 32, 1 / 1024, 0.5 / 1024, 1 / 1024], device=device)
    for handle, count, length in ((a, 1, 17), (b, 2, 9)):
        destination = pool.writable(handle)
        torch.testing.assert_close(
            destination.sums[1],
            (expected[:, None, None] * count).expand(4, 4, length),
        )
        assert torch.all(destination.counts[1] == count)
        assert torch.all(destination.counts[0] == 0)
        assert torch.all(destination.flags == 0)
        assert torch.all(buffers.sums[handle.slot, :, :, :, length:] == -123)

    scores[4, 0, 0] = float("nan")
    assert staging.stage([None, a])
    graph.replay()
    assert pool.writable(a).flags[1, 0] & 1
    scores[4, 0, 0] = 0
    assert staging.stage([None, a])
    graph.replay()
    assert pool.writable(a).flags[1, 0] & 1
    assert torch.all(pool.writable(b).flags == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_forge_reduction_flags_bad_coverage_and_count_overflow():
    from vllm.agentic_kv.statistics.forge import (
        accumulate_forge_statistics,
        load_forge_reduction,
        resolve_routes,
    )

    device = torch.device("cuda")
    buffers = StatisticsBuffers(StatisticsCapacity(1, 1, 4, 16), device)
    pool = StatisticsPool(buffers, torch.cuda.current_stream())
    handle = pool.acquire(identity(0, 16))
    pool.writable(handle).counts.fill_(2**31 - 1)
    meta = torch.zeros((4, 10), dtype=torch.int64, device=device)
    meta[:, 1] = 8
    scores = torch.zeros((4, 16, 8), device=device)
    summary = torch.zeros((24, 18), device=device)
    summary[:, 1] = 1 / 8
    routes = torch.tensor(resolve_routes(pool, [handle]), device=device)
    kernel = load_forge_reduction()
    accumulate_forge_statistics(meta, scores, summary, routes, buffers, 0, kernel)
    assert torch.all((pool.writable(handle).flags & 6) == 6)


def test_native_slot_attachment_reorder_and_stale_batch(setup):
    """Batch position, native slot and statistics slot are independent."""
    from vllm.agentic_kv.statistics.worker import attach, discard, prepare_owners

    pool, _, writer = setup
    states = SimpleNamespace(
        req_id_to_index={"A": 2, "B": 0, "C": 1},
        max_num_reqs=3,
        statistics_pool=None,
        statistics_handles=None,
    )
    a = attach(states, pool, identity())
    b = attach(states, pool, replace(identity(), request_id="B"))
    assert attach(states, pool, replace(identity(), request_id="C")) is None
    with pytest.raises(ValueError, match="already owns"):
        attach(states, pool, identity())
    batch = SimpleNamespace(
        num_reqs=3,
        req_ids=["B", "C", "A"],
        idx_mapping_np=[0, 1, 2],
        is_prefilling_np=[False] * 3,
        num_scheduled_tokens=[1] * 3,
        num_computed_tokens_np=[8] * 3,
    )
    assert prepare_owners(states, batch, [True] * 3) == (b, None, a)
    assert prepare_owners(states, batch, [False, True, True]) == (None, None, a)
    old_batch = prepare_owners(states, batch, [True] * 3)
    discard(states, "A")
    assert states.statistics_handles[2] is None
    with pytest.raises(ValueError, match="transferred"):
        pool.writable(old_batch[2])
    assert pool.reap() == 0
    writer.events[-1].complete = True
    assert pool.reap() == 1
    replacement = attach(states, pool, identity(1))
    assert replacement.slot == a.slot and replacement.serial != a.serial
    with pytest.raises(ValueError, match="transferred"):
        pool.writable(a)
    assert prepare_owners(states, batch, [True] * 3)[2] is replacement


def test_worker_detach_clears_native_slot_and_preserves_consumer_hold(setup):
    from vllm.agentic_kv.statistics.worker import attach, detach, discard

    pool, _, _ = setup
    states = SimpleNamespace(
        req_id_to_index={"A": 0},
        max_num_reqs=1,
        statistics_pool=None,
        statistics_handles=None,
    )
    handle = attach(states, pool, identity())
    lease = detach(states, "A")
    assert lease.handle is handle
    assert states.statistics_handles == [None]
    discard(states, "A")  # Native removal must not discard the save's lease.
    assert pool.reap() == 0
    reader = Stream()
    pool.consume(lease, reader, lambda _: None)
    assert pool.reap() == 0
    reader.events[-1].complete = True
    assert pool.reap() == 1


def test_worker_routes_reject_stale_mapping_and_unsupported_observations(setup):
    from vllm.agentic_kv.statistics.worker import attach, prepare_owners

    pool, _, _ = setup
    states = SimpleNamespace(
        req_id_to_index={"A": 0},
        max_num_reqs=1,
        statistics_pool=None,
        statistics_handles=None,
    )
    attach(states, pool, identity())
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["A"],
        idx_mapping_np=[1],
        is_prefilling_np=[False],
        num_scheduled_tokens=[1],
        num_computed_tokens_np=[8],
    )
    with pytest.raises(ValueError, match="mask"):
        prepare_owners(states, batch, [])
    with pytest.raises(ValueError, match="stale native"):
        prepare_owners(states, batch, [True])
    batch.idx_mapping_np = [0]
    batch.num_computed_tokens_np = [2]
    with pytest.raises(ValueError, match="coverage"):
        prepare_owners(states, batch, [True])
    batch.num_computed_tokens_np = [8]
    batch.is_prefilling_np = [True]
    with pytest.raises(ValueError, match="single-query"):
        prepare_owners(states, batch, [True])
    assert prepare_owners(states, batch, [False]) == (None,)
    batch.is_prefilling_np = [False]
    batch.num_scheduled_tokens = [2]
    with pytest.raises(ValueError, match="single-query"):
        prepare_owners(states, batch, [True])


def test_route_staging_declines_busy_copy_without_reusing_host_storage():
    from vllm.agentic_kv.statistics.routes import StatisticsRoutes

    buffers = StatisticsBuffers(StatisticsCapacity(2, 1, 4, 8), torch.device("cpu"))
    writer = Stream()
    pool = StatisticsPool(buffers, writer, Event)
    a, b = pool.acquire(identity()), pool.acquire(identity(1))
    routes = StatisticsRoutes(pool, 3, 2, Event)
    pointer = routes.gpu.data_ptr()
    assert routes.device_bytes == 48 and routes.host_bytes == 96
    assert routes.stage([a, None, b])
    first = routes.gpu.clone()
    assert routes.stage([b, a])
    second = routes.gpu.clone()
    assert not routes.stage([a])  # Both copy-completion events still pending.
    assert torch.all(routes.gpu == -1)
    assert torch.equal(first, torch.tensor([[a.slot, 3], [-1, 0], [b.slot, 3]]))
    assert torch.equal(second, torch.tensor([[b.slot, 3], [a.slot, 3], [-1, -1]]))
    writer.events[0].complete = True
    assert routes.stage([None, b])
    assert routes.gpu.data_ptr() == pointer
    assert not routes.stage([])  # A dummy/unsampled replay clears old routes.
    assert torch.all(routes.gpu == -1)


def test_route_submission_rejects_detached_duplicate_and_oversized_owners():
    from vllm.agentic_kv.statistics.routes import StatisticsRoutes

    buffers = StatisticsBuffers(StatisticsCapacity(1, 1, 4, 8), torch.device("cpu"))
    pool = StatisticsPool(buffers, Stream(), Event)
    a = pool.acquire(identity())
    routes = StatisticsRoutes(pool, 2, 1, Event)
    with pytest.raises(ValueError, match="duplicate"):
        routes.stage([a, a])
    with pytest.raises(ValueError, match="capacity"):
        routes.stage([None] * 3)
    pool.detach(a)
    with pytest.raises(ValueError, match="transferred"):
        routes.stage([a])
    assert torch.all(routes.gpu == -1)


def test_failed_route_event_record_quarantines_host_buffer():
    from vllm.agentic_kv.statistics.routes import StatisticsRoutes

    class FailedEvent(Event):
        def record(self, stream):
            raise RuntimeError("event recording failed")

    pool = StatisticsPool(
        StatisticsBuffers(StatisticsCapacity(1, 1, 4, 8), torch.device("cpu")),
        Stream(),
        Event,
    )
    a = pool.acquire(identity())
    routes = StatisticsRoutes(pool, 1, 1, FailedEvent)
    with pytest.raises(RuntimeError, match="event recording"):
        routes.stage([a])
    assert not routes.stage([a])
    assert torch.all(routes.gpu == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_gpu_routes_preserve_queued_graph_readers_and_reject_stale_replay():
    from vllm.agentic_kv.statistics.routes import StatisticsRoutes

    class HeldEvent:
        """Delay CPU slot reclamation deterministically, without blocking the GPU."""

        def __init__(self):
            self.event = torch.cuda.Event()
            self.release = False

        def record(self, stream):
            self.event.record(stream)

        def query(self):
            return self.release and self.event.query()

    device = torch.device("cuda")
    writer = torch.cuda.current_stream()
    pool = StatisticsPool(
        StatisticsBuffers(StatisticsCapacity(2, 1, 4, 8), device), writer
    )
    a, b = pool.acquire(identity()), pool.acquire(identity(1))
    events = []

    def event_factory():
        event = HeldEvent()
        events.append(event)
        return event

    routes = StatisticsRoutes(pool, 3, 2, event_factory)
    observed = torch.empty_like(routes.gpu)
    results = [torch.empty_like(observed) for _ in range(4)]
    # Capture only the reader. Every real replay revalidates/stages outside capture.
    observed.copy_(routes.gpu)
    writer.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observed.copy_(routes.gpu)
    assert routes.stage([a, None, b])
    graph.replay()
    results[0].copy_(observed)
    assert routes.stage([b, a])
    graph.replay()
    results[1].copy_(observed)
    assert not routes.stage([a])  # Exhaustion disables; no host wait or overwrite.
    graph.replay()
    results[2].copy_(observed)
    pool.discard(a)
    with pytest.raises(ValueError, match="transferred"):
        routes.stage([a])
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(RuntimeError, match="bound model stream"),
    ):
        routes.stage([b])
    writer.synchronize()
    assert pool.reap() == 1
    replacement = pool.acquire(identity(2))
    assert replacement.slot == a.slot
    for event in events:
        event.release = True
    assert routes.stage([replacement])
    graph.replay()
    results[3].copy_(observed)
    expected = [
        [[a.slot, 3], [-1, 0], [b.slot, 3]],
        [[b.slot, 3], [a.slot, 3], [-1, -1]],
        [[-1, -1]] * 3,
        [[replacement.slot, 3], [-1, -1], [-1, -1]],
    ]
    for result, rows in zip(results, expected):
        torch.testing.assert_close(result.cpu(), torch.tensor(rows))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_native_request_slot_reuse_does_not_inherit_statistics():
    from vllm.agentic_kv.statistics.worker import attach
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    states = RequestState(2, 8, 8, 0, 16, device)
    pool = StatisticsPool(
        StatisticsBuffers(StatisticsCapacity(2, 1, 4, 8), device),
        torch.cuda.current_stream(),
    )
    states.add_request("A", 3, [1, 2, 3], 3, 5)
    handle = attach(states, pool, identity())
    native_slot = states.req_id_to_index["A"]
    assert states.remove_request("A") == native_slot
    assert states.statistics_handles[native_slot] is None
    with pytest.raises(ValueError, match="transferred"):
        pool.writable(handle)
    states.add_request("A", 3, [1, 2, 3], 3, 5)
    assert states.req_id_to_index["A"] == native_slot
    assert states.statistics_handles[native_slot] is None
    replacement = attach(states, pool, identity(1))
    assert replacement is not handle
    assert states.remove_request("A") == native_slot
    assert states.remove_request("A") is None
    torch.cuda.current_stream().synchronize()
    assert pool.reap() == 2
