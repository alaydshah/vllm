# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Forge prefix reconstruction with direct reads from native BF16 suffix pages.

The native adapter lives in backend.py. Payload leases, batch/job preparation,
RoPE tables and page allocation remain caller-owned. Only gfx950, four KV heads,
six query heads per KV head and head dimension 256 are currently supported.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from vllm.agentic_kv.statistics.buffers import StatisticsBuffers
from vllm.agentic_kv.statistics.forge import (
    ForgeReductionKernel,
    accumulate_forge_statistics,
)


@lru_cache(maxsize=1)
def load_paged_forge() -> Any:
    """Build explicitly during startup, outside model execution and capture."""
    if not torch.version.hip:
        raise RuntimeError("paged Forge requires ROCm gfx950")
    if not torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).gcnArchName.startswith("gfx950"):
        raise RuntimeError("paged Forge requires ROCm gfx950")
    from torch.utils.cpp_extension import load

    return load(
        name="agentic_kv_paged_forge",
        sources=[str(Path(__file__).with_name("forge_paged.cu"))],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        with_cuda=True,
        verbose=False,
    )


@dataclass(frozen=True)
class PagedDecodePlan:
    """Trusted codec/job-builder output, retained through every queued reader.

    owner_ptrs preserves Forge's eight-pointer record (last two entries unused).
    Payload references must retain all allocations addressed by the first six.
    Jobs cover the full absolute context; suffix block tables start at prefix end.
    Native preparation must populate consistent, in-bounds jobs/meta before each
    submission. This type is not a validator for arbitrary raw device pointers.
    """

    meta: torch.Tensor
    jobs: torch.Tensor
    owner_ptrs: torch.Tensor
    rotary: torch.Tensor
    lookup: torch.Tensor
    payloads: tuple[torch.Tensor, ...]
    batch: int
    max_seq_len: int
    tile: int = 72
    chunk: int = 32

    def __post_init__(self) -> None:
        if self.batch <= 0 or self.max_seq_len <= 0:
            raise ValueError("invalid decode capacity")
        if self.tile not in (72, 112) or self.chunk not in (32, 128):
            raise ValueError("unsupported Forge launch geometry")
        if (
            self.meta.shape != (self.batch * 4, 10)
            or self.owner_ptrs.shape != (self.batch * 4, 8)
            or self.jobs.ndim != 2
            or self.jobs.shape[1] != 16
            or self.jobs.shape[0] <= 0
            or self.rotary.ndim != 2
            or self.rotary.shape[1] != 68
            or self.lookup.shape != (512, 4)
            or not self.payloads
        ):
            raise ValueError("invalid packed Forge metadata")
        for tensor, dtype in (
            (self.meta, torch.int64),
            (self.jobs, torch.int32),
            (self.owner_ptrs, torch.int64),
            (self.rotary, torch.float32),
            (self.lookup, torch.float32),
        ):
            if tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError("invalid Forge metadata dtype or layout")
            if tensor.device != self.meta.device:
                raise ValueError("Forge metadata must share one device")
        if any(t.device != self.meta.device for t in self.payloads):
            raise ValueError("payloads must share the metadata device")


class DecodeWorkspace:
    """Reserve per-execution scratch before native KV pool sizing; never grow.

    One workspace may be reused by sequential layers on the same stream, not by
    concurrent executions on different streams. No full reconstructed KV is held.
    """

    def __init__(self, plan: PagedDecodePlan, *, collect_statistics: bool) -> None:
        self.plan = plan
        self.collect_statistics = collect_statistics
        self.partial = torch.empty(
            (plan.jobs.shape[0], 6, 256), dtype=torch.float16, device=plan.meta.device
        )
        self.ml = torch.empty(
            (plan.jobs.shape[0], 6, 2), dtype=torch.float32, device=plan.meta.device
        )
        self.summary = torch.empty(
            (plan.batch * 24, 18), dtype=torch.float32, device=plan.meta.device
        )
        stride = ((plan.max_seq_len + 255) // 256 * 256) if collect_statistics else 1
        self.scores = torch.empty(
            (plan.batch * 4, stride, 8), dtype=torch.float32, device=plan.meta.device
        )

    @property
    def tensor_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.partial, self.ml, self.summary, self.scores)
        )


def paged_decode(
    kernel: Any,
    workspace: DecodeWorkspace,
    query: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    routes: torch.Tensor,
    *,
    statistics: StatisticsBuffers | None = None,
    reduction: ForgeReductionKernel | None = None,
    layer: int = 0,
) -> torch.Tensor:
    """Read compressed prefix + paged suffix without gathering or dense expansion.

    Native KV update must already have written this step's K/V. Cache views are
    [blocks, page tokens, four KV heads, 256], with matching strides and contiguous
    head dimensions (including vLLM's interleaved K/V-content views). Routes are
    the same staged int64 [batch, 2] used by exports and reduction. Jobs and page
    IDs are trusted native metadata; publication validates compressed payloads.
    """
    plan = workspace.plan
    if (
        query.shape != (plan.batch, 4, 6, 256)
        or output.shape != query.shape
        or query.dtype != torch.bfloat16
        or output.dtype != query.dtype
        or not query.is_contiguous()
        or not output.is_contiguous()
        or not query.is_cuda
    ):
        raise ValueError("paged Forge requires contiguous BF16 [batch, 4, 6, 256]")
    if (
        key_cache.ndim != 4
        or key_cache.shape[2:] != (4, 256)
        or key_cache.shape[1] <= 0
        or value_cache.shape != key_cache.shape
        or key_cache.dtype != query.dtype
        or value_cache.dtype != query.dtype
        or key_cache.stride() != value_cache.stride()
        or key_cache.stride(-1) != 1
        or any(s <= 0 for s in key_cache.stride())
        or any(s % 4 for s in key_cache.stride()[:3])
        or key_cache.data_ptr() % 8
        or value_cache.data_ptr() % 8
        or block_table.ndim != 2
        or block_table.shape[0] != plan.batch
        or block_table.dtype != torch.int32
        or block_table.stride(1) != 1
        or routes.shape != (plan.batch, 2)
        or routes.dtype != torch.int64
        or not routes.is_contiguous()
    ):
        raise ValueError("invalid native cache, block table or routes")
    if any(
        t.device != query.device
        for t in (
            output,
            key_cache,
            value_cache,
            block_table,
            routes,
            plan.meta,
        )
    ):
        raise ValueError("paged Forge inputs must share one GPU")
    if workspace.collect_statistics != (statistics is not None):
        raise ValueError("statistics selection must match reserved workspace")
    if statistics is not None and (
        reduction is None
        or not 0 <= layer < statistics.capacity.layers
        or statistics.capacity.kv_heads != 4
        or statistics.sums.device != query.device
    ):
        raise ValueError("invalid statistics destination or reducer")
    kernel.decode_split(
        plan.meta,
        plan.jobs,
        plan.rotary,
        query,
        plan.lookup,
        workspace.partial,
        workspace.ml,
        workspace.scores,
        plan.owner_ptrs,
        key_cache,
        value_cache,
        block_table,
        routes,
        plan.tile,
        workspace.collect_statistics,
    )
    kernel.decode_combine(
        plan.meta,
        query,
        workspace.partial,
        workspace.ml,
        output,
        workspace.summary,
        routes,
        plan.chunk,
        workspace.collect_statistics,
    )
    if statistics is not None:
        assert reduction is not None
        accumulate_forge_statistics(
            plan.meta,
            workspace.scores,
            workspace.summary,
            routes,
            statistics,
            layer,
            reduction,
        )
    return output
