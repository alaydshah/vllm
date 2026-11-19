# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in allocation adapter; not registered with the engine's manager factory."""

from collections.abc import Sequence

from vllm.v1.core.external_prefix import ExternalPrefixCoverage
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request


class ExternalPrefixFullAttentionManager(FullAttentionManager):
    """Use native blocks only for the suffix of externally backed requests.

    The caller owns external storage leases and registers coverage before
    admission. Coverage stays fixed until native request bookkeeping is popped;
    GPU reader completion and external lease release remain caller duties.
    Local-hit composition, context parallelism and speculative allocation are
    deliberately unsupported for externally backed requests.
    """

    def __init__(self, kv_cache_spec: FullAttentionSpec, **kwargs) -> None:
        if type(kv_cache_spec) is not FullAttentionSpec:
            raise ValueError("external prefix requires a full-attention spec")
        super().__init__(kv_cache_spec, **kwargs)
        self.external_prefixes: dict[str, ExternalPrefixCoverage] = {}

    def register_external_prefix(
        self, request_id: str, coverage: ExternalPrefixCoverage
    ) -> None:
        """Register leased, immutable coverage before native admission."""
        if self.dcp_world_size != 1 or self.pcp_world_size != 1:
            raise ValueError("external prefix does not support context parallelism")
        if coverage.block_size != self.block_size or coverage.num_tokens == 0:
            raise ValueError("external prefix requires positive group-aligned coverage")
        if (
            request_id in self.external_prefixes
            or request_id in self.req_to_blocks
            or request_id in self.num_cached_block
        ):
            raise ValueError("external prefix must be registered before admission")
        self.external_prefixes[request_id] = coverage

    def _validate_external_hit(
        self, request_id: str, local_tokens: int, external_tokens: int
    ) -> None:
        if (
            local_tokens
            or external_tokens != self.external_prefixes[request_id].num_tokens
        ):
            raise ValueError("external prefix cannot compose with another cache hit")

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        coverage = self.external_prefixes.get(request_id)
        if coverage is not None:
            if new_computed_blocks or num_local_computed_tokens:
                raise ValueError(
                    "external prefix cannot compose with a local cache hit"
                )
            if num_tokens != num_tokens_main_model:
                raise ValueError(
                    "external prefix does not support speculative allocation"
                )
            num_tokens = coverage.native_num_tokens(num_tokens)
            num_tokens_main_model = num_tokens
            total_computed_tokens = coverage.native_num_tokens(total_computed_tokens)
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if request_id in self.external_prefixes:
            if new_computed_blocks:
                raise ValueError(
                    "external prefix cannot compose with a local cache hit"
                )
            self._validate_external_hit(
                request_id, num_local_computed_tokens, num_external_computed_tokens
            )
            num_external_computed_tokens = 0
        super().add_local_computed_blocks(
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if request_id in self.external_prefixes:
            self._validate_external_hit(
                request_id, num_local_computed_tokens, num_external_computed_tokens
            )
            return
        super().allocate_external_computed_blocks(
            request_id, num_local_computed_tokens, num_external_computed_tokens
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        coverage = self.external_prefixes.get(request_id)
        if coverage is not None:
            if num_tokens != num_tokens_main_model:
                raise ValueError(
                    "external prefix does not support speculative allocation"
                )
            num_tokens = coverage.native_num_tokens(num_tokens)
            num_tokens_main_model = num_tokens
        return super().allocate_new_blocks(
            request_id, num_tokens, num_tokens_main_model
        )

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        if request.request_id in self.external_prefixes:
            # Suffix KV depends on the external representation, not just token
            # hashes. Never publish it into the ordinary exact prefix cache.
            return
        super().cache_blocks(
            request,
            num_tokens,
            retention_interval,
            replay_boundaries=replay_boundaries,
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        if self.external_prefixes:
            return 0
        return super().get_num_common_prefix_blocks(running_request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        blocks = super().pop_blocks_for_free(request_id)
        self.external_prefixes.pop(request_id, None)
        return blocks
