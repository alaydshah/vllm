# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicitly registered, decode-only Forge backend; serving remains gated."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.agentic_kv.attention.forge import DecodeWorkspace, paged_decode
from vllm.agentic_kv.statistics.buffers import StatisticsBuffers
from vllm.agentic_kv.statistics.forge import ForgeReductionKernel
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass
class ForgeMetadata:
    num_actual_tokens: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class ForgeMetadataBuilder(AttentionMetadataBuilder[ForgeMetadata]):
    # Kernel replay is validated separately. Native capture/job binding is pending.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: "KVCacheSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ForgeMetadata:
        common = common_attn_metadata
        if (
            common_prefix_len
            or not common.causal
            or common.max_query_len != 1
            or common.num_reqs != common.num_actual_tokens
        ):
            raise NotImplementedError("Forge backend currently supports decode only")
        return ForgeMetadata(
            common.num_actual_tokens, common.block_table_tensor, common.slot_mapping
        )


@dataclass(frozen=True)
class DecodeBinding:
    """Per-layer kernel inputs, not request state or ownership of a saved history.

    The worker must hold payload leases, stage consistent jobs and routes, and
    reserve workspace before binding. It must keep all buffers alive through GPU
    completion. These requirements do not activate the connector's worker role.
    """

    kernel: Any
    workspace: DecodeWorkspace
    routes: torch.Tensor
    statistics: StatisticsBuffers | None = None
    reduction: ForgeReductionKernel | None = None
    layer_index: int = 0


class ForgeAttentionImpl(TritonAttentionImpl):
    """Use the native Triton cache writer; substitute only the attention reader."""

    supports_dense_mha_prefill: ClassVar[bool] = False

    binding: DecodeBinding | None = None

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ForgeMetadata | TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)
        binding = self.binding
        if binding is None:
            raise RuntimeError("Forge worker must bind a validated decode payload")
        if (
            self.num_heads != 24
            or self.num_kv_heads != 4
            or self.head_size != 256
            or self.scale != 0.0625
            or self.kv_cache_dtype not in ("auto", "bfloat16")
            or self.alibi_slopes is not None
            or self.sliding_window != (-1, -1)
            or self.logits_soft_cap != 0
            or self.kv_sharing_target_layer_name is not None
            or self.use_alibi_sqrt
            or self.chunk_lookback != -1
            or self.sinks is not None
            or self.attn_type != AttentionType.DECODER
            or output_scale is not None
            or output_block_scale is not None
        ):
            raise NotImplementedError("unsupported Forge attention configuration")
        batch = binding.workspace.plan.batch
        if attn_metadata.num_actual_tokens != batch:
            raise ValueError("native batch does not match the bound decode plan")
        # Views only; native do_kv_cache_update already wrote the current tokens.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(256, dim=-1)
        paged_decode(
            binding.kernel,
            binding.workspace,
            query[:batch].view(batch, 4, 6, 256),
            output[:batch].view(batch, 4, 6, 256),
            key_cache,
            value_cache,
            attn_metadata.block_table,
            binding.routes,
            statistics=binding.statistics,
            reduction=binding.reduction,
            layer=binding.layer_index,
        )
        return output


class ForgeAttentionBackend(AttentionBackend):
    forward_includes_kv_cache_update = False
    supports_external_prefix = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "AGENTIC_FORGE"

    @staticmethod
    def get_impl_cls() -> type[ForgeAttentionImpl]:
        return ForgeAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[ForgeMetadataBuilder]:
        return ForgeMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 256

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


def register_forge_backend() -> None:
    """Explicit opt-in through the native CUSTOM registry; no import side effects."""
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm.agentic_kv.attention.backend.ForgeAttentionBackend",
    )
