# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Admission-side connector for ready compressed histories.

No compression, device work or readiness waits run in lookup. Scheduler-side
histories must contain descriptors, not GPU storage. Worker publication and the
compressed attention reader are not installed yet: worker construction fails
closed so selecting this connector cannot silently enable incomplete serving.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.agentic_kv.history import HistoryLease
from vllm.agentic_kv.store import HistoryStore
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.external_prefix_manager import (
    ExternalPrefixFullAttentionManager,
    register_external_prefix_coverage,
    withdraw_external_prefix_coverage,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


@dataclass
class AgenticKVMetadata(KVConnectorMetadata):
    prefixes: dict[str, int]

    def get_external_prefix_lengths(self) -> dict[str, int]:
        return self.prefixes.copy()


@dataclass
class _Admission:
    lease: HistoryLease
    allocated: bool = False
    submitted: bool = False
    use_external: bool = True


class AgenticKVConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            raise NotImplementedError("Agentic KV worker reader is not installed")
        if not vllm_config.use_v2_model_runner or vllm_config.speculative_config:
            raise ValueError("Agentic KV requires v2 runner without speculation")
        parallel = vllm_config.parallel_config
        if any(
            getattr(parallel, name) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            raise ValueError("Agentic KV currently requires single-worker execution")
        extra = self._kv_transfer_config.kv_connector_extra_config
        self.namespace = extra["history_namespace"]
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError(
                "history_namespace must identify this engine/configuration"
            )
        self.history_store = HistoryStore(
            extra["history_capacity_bytes"], extra["history_max_entries"]
        )
        self._admissions: dict[str, _Admission] = {}
        self._awaiting_completion: set[str] = set()

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        groups = kv_cache_manager.coordinator.single_type_managers
        if (
            not groups
            or any(
                not isinstance(g, ExternalPrefixFullAttentionManager) for g in groups
            )
            or kv_cache_manager.use_eagle
            or kv_cache_manager.coordinator.num_reprefillable_tokens
            or any(g.dcp_world_size != 1 or g.pcp_world_size != 1 for g in groups)
        ):
            raise ValueError("Agentic KV requires configured full-attention groups")
        super().bind_kv_cache_manager(kv_cache_manager)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        manager = self._kv_cache_manager
        assert manager is not None
        req_id = request.request_id
        if req_id in self._awaiting_completion:
            raise ValueError("finished request is awaiting reader completion")
        # Resume conservatively through native recomputation. A previously
        # submitted lease stays held until finish, including in-flight readers.
        if (
            num_computed_tokens
            or request.num_preemptions
            or request.sampling_params is None
            or request.sampling_params.prompt_logprobs is not None
            or request.mm_features
            or request.prompt_embeds is not None
            or request.lora_request is not None
        ):
            withdraw_external_prefix_coverage(manager, req_id)
            state = self._admissions.get(req_id)
            if state is not None:
                state.use_external = False
                if not state.submitted:
                    self.history_store.release(state.lease)
                    del self._admissions[req_id]
            return 0, False
        lease = self.history_store.acquire_matching(
            req_id,
            self.namespace,
            request.block_hashes,
            manager.block_pool.hash_block_size,
            request.num_prompt_tokens,
        )
        if lease is None:
            return 0, False
        try:
            register_external_prefix_coverage(
                manager, request, lease.history.prefix_tokens
            )
        except Exception:
            if req_id not in self._admissions:
                self.history_store.release(lease)
            raise
        self._admissions.setdefault(req_id, _Admission(lease))
        return lease.history.prefix_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        state = self._admissions.get(request.request_id)
        if num_external_tokens:
            if (
                state is None
                or not state.use_external
                or (num_external_tokens != state.lease.history.prefix_tokens)
            ):
                raise ValueError("allocation does not match the leased history")
            state.allocated = True

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> AgenticKVMetadata:
        prefixes = {}
        for req in scheduler_output.scheduled_new_reqs:
            state = self._admissions.get(req.req_id)
            if state is not None and state.use_external:
                if not state.allocated:
                    raise ValueError("external history scheduled before allocation")
                prefixes[req.req_id] = state.lease.history.prefix_tokens
                state.submitted = True
        return AgenticKVMetadata(prefixes)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        state = self._admissions.get(request.request_id)
        if state is None:
            return False, None
        if not state.submitted:
            self.history_store.release(state.lease)
            del self._admissions[request.request_id]
            return False, None
        self._awaiting_completion.add(request.request_id)
        return True, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [])

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        for req_id in connector_output.finished_sending or ():
            if req_id not in self._awaiting_completion:
                raise ValueError("unexpected history reader completion")
            self.history_store.release(self._admissions.pop(req_id).lease)
            self._awaiting_completion.remove(req_id)

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        raise NotImplementedError("Agentic KV worker reader is not installed")

    def wait_for_layer_load(self, layer_name: str) -> None:
        raise NotImplementedError("Agentic KV worker reader is not installed")

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs: Any) -> None:
        raise NotImplementedError("Agentic KV asynchronous save is not installed")

    def wait_for_save(self) -> None:
        raise NotImplementedError("Agentic KV asynchronous save is not installed")
