# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Forge decode statistics adapter; not a native attention backend registration.

Consumes the source kernel's log2 scores and combine summaries (4 KV heads,
6 query heads per KV head, 8 score lanes, 18 summary fields). No attention math,
sampling policy, buffer allocation or request lifecycle is owned here.
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import torch

from vllm.agentic_kv.statistics.buffers import StatisticsBuffers
from vllm.agentic_kv.statistics.contracts import StatisticsHandle
from vllm.agentic_kv.statistics.pool import StatisticsPool


def resolve_routes(
    pool: StatisticsPool, owners: Sequence[StatisticsHandle | None]
) -> tuple[tuple[int, int], ...]:
    """Resolve native batch order at submission; None means unsampled.

    Caller stages these CPU (slot, coverage) pairs in its reusable int64 GPU
    metadata. The same selection must gate Forge score/summary exports. Resolve
    on every submission, including graph replay. Do not overwrite in-flight
    route storage. Duplicate destinations are rejected to prevent write races.
    """
    if pool.buffers.capacity.kv_heads != 4:
        raise ValueError("Forge statistics require four KV heads")
    result = []
    seen = set()
    for owner in owners:
        if owner is None:
            result.append((-1, 0))
            continue
        pool.writable(owner)
        if owner.slot in seen:
            raise ValueError("duplicate statistics destination in batch")
        seen.add(owner.slot)
        result.append((owner.slot, owner.identity.covered_tokens))
    return tuple(result)


class ForgeReductionKernel(Protocol):
    def accumulate(
        self,
        meta: torch.Tensor,
        scores: torch.Tensor,
        summary: torch.Tensor,
        routes: torch.Tensor,
        sums: torch.Tensor,
        counts: torch.Tensor,
        flags: torch.Tensor,
        layer: int,
    ) -> None: ...


@lru_cache(maxsize=1)
def load_forge_reduction() -> ForgeReductionKernel:
    """Explicit startup build; never invoke from request dispatch or graph capture.

    This opt-in source port requires a working CUDA/ROCm compiler. Packaging it
    with the future native backend remains separate from this adapter milestone.
    """
    from torch.utils.cpp_extension import load

    return load(
        name="agentic_kv_forge_statistics",
        sources=[str(Path(__file__).with_name("forge_kernel.cu"))],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def accumulate_forge_statistics(
    meta: torch.Tensor,
    scores: torch.Tensor,
    summary: torch.Tensor,
    routes: torch.Tensor,
    buffers: StatisticsBuffers,
    layer: int,
    kernel: ForgeReductionKernel,
) -> None:
    """Reduce Forge summaries directly into one layer of the fixed pool.

    Routes are pre-staged int64 [batch, 2] from resolve_routes. Caller selects
    plain attention and skips this call for an all-unsampled batch. No tensor
    allocation, host result read or metadata transfer occurs here. Same-stream
    producer ordering and matching sampled score exports are caller contracts.
    """
    capacity = buffers.capacity
    if capacity.kv_heads != 4 or not 0 <= layer < capacity.layers:
        raise ValueError("invalid Forge statistics head/layer geometry")
    if routes.ndim != 2 or routes.shape[1] != 2 or routes.shape[0] == 0:
        raise ValueError("routes must have shape [batch, 2]")
    batch = routes.shape[0]
    if (
        meta.shape != (batch * 4, 10)
        or scores.ndim != 3
        or scores.shape[0] != batch * 4
        or scores.shape[2] != 8
        or scores.shape[1] <= 0
        or summary.shape != (batch * 24, 18)
    ):
        raise ValueError("invalid Forge score/summary geometry")
    for tensor, dtype in (
        (meta, torch.int64),
        (routes, torch.int64),
        (scores, torch.float32),
        (summary, torch.float32),
    ):
        if (
            tensor.dtype != dtype
            or tensor.device != buffers.sums.device
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Forge inputs must be contiguous GPU tensors on pool device"
            )
    if scores.data_ptr() % 16:
        raise ValueError("Forge vector score loads require 16-byte alignment")
    kernel.accumulate(
        meta,
        scores,
        summary,
        routes,
        buffers.sums,
        buffers.counts,
        buffers.flags,
        layer,
    )
