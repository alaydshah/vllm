# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Statistics attachment to native v2 slots; no separate request registry."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.agentic_kv.statistics.contracts import (
    StatisticsHandle,
    StatisticsIdentity,
    StatisticsLease,
)
from vllm.agentic_kv.statistics.pool import StatisticsPool

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


def attach(
    states: "RequestState", pool: StatisticsPool, identity: StatisticsIdentity
) -> StatisticsHandle | None:
    """Attach explicit provenance; exhaustion only declines observation."""
    index = states.req_id_to_index[identity.request_id]
    if states.statistics_pool is not None and states.statistics_pool is not pool:
        raise ValueError("request state already uses another statistics pool")
    if states.statistics_handles is None:
        states.statistics_handles = [None] * states.max_num_reqs
    if states.statistics_handles[index] is not None:
        raise ValueError("request already owns statistics")
    states.statistics_pool = pool
    handle = pool.acquire(identity)
    states.statistics_handles[index] = handle
    return handle


def detach(states: "RequestState", req_id: str) -> StatisticsLease | None:
    """Transfer to a save before native removal; clear even on event failure."""
    index = states.req_id_to_index[req_id]
    if states.statistics_handles is None:
        return None
    handle = states.statistics_handles[index]
    if handle is None:
        return None
    assert states.statistics_pool is not None
    states.statistics_handles[index] = None
    return states.statistics_pool.detach(handle)


def discard(states: "RequestState", req_id: str) -> None:
    lease = detach(states, req_id)
    if lease is not None:
        assert states.statistics_pool is not None
        states.statistics_pool.discard(lease)


def prepare_owners(
    states: "RequestState", batch: "InputBatch", sampled: Sequence[bool]
) -> tuple[StatisticsHandle | None, ...]:
    """Resolve final native batch order; selection policy stays with the caller.

    CPU handles only, not GPU write permission. The backend must resolve_routes
    again immediately before submission/replay and safely stage its GPU routes.
    This first adapter only accepts single-query decode observations.
    """
    if len(sampled) != batch.num_reqs:
        raise ValueError("statistics mask must match native batch size")
    owners: list[StatisticsHandle | None] = []
    for row, req_id in enumerate(batch.req_ids):
        if not sampled[row]:
            owners.append(None)
            continue
        if batch.is_prefilling_np[row] or batch.num_scheduled_tokens[row] != 1:
            raise ValueError("statistics adapter only supports single-query decode")
        index = states.req_id_to_index[req_id]
        if index != batch.idx_mapping_np[row]:
            raise ValueError("stale native statistics batch mapping")
        handle = (
            states.statistics_handles[index]
            if states.statistics_handles is not None
            else None
        )
        if handle is not None:
            if handle.identity.covered_tokens > batch.num_computed_tokens_np[row]:
                raise ValueError("statistics coverage exceeds computed history")
            assert states.statistics_pool is not None
            states.statistics_pool.writable(handle)
        owners.append(handle)
    return tuple(owners)
