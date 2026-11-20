# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.agentic_kv.attention.forge import DecodeWorkspace, PagedDecodePlan


def test_workspace_reserves_scores_only_when_collection_is_enabled():
    """Unsampled operation must not reserve a full context-sized score export."""
    plan = PagedDecodePlan(
        meta=torch.empty(8, 10, dtype=torch.int64),
        jobs=torch.empty(16, 16, dtype=torch.int32),
        owner_ptrs=torch.empty(8, 8, dtype=torch.int64),
        rotary=torch.empty(32, 68),
        lookup=torch.empty(512, 4),
        payloads=(torch.empty(1),),
        batch=2,
        max_seq_len=4097,
    )
    plain = DecodeWorkspace(plan, collect_statistics=False)
    sampled = DecodeWorkspace(plan, collect_statistics=True)
    assert plain.scores.shape == (8, 1, 8)
    assert sampled.scores.shape == (8, 4352, 8)
    assert sampled.tensor_bytes - plain.tensor_bytes == 8 * (4352 - 1) * 8 * 4
    with pytest.raises(ValueError, match="launch geometry"):
        PagedDecodePlan(
            plan.meta,
            plan.jobs,
            plan.owner_ptrs,
            plan.rotary,
            plan.lookup,
            plan.payloads,
            2,
            4097,
            tile=17,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native backend imports")
def test_native_metadata_rejects_prefill_and_preserves_page_table():
    from vllm.agentic_kv.attention.backend import ForgeMetadataBuilder

    builder = ForgeMetadataBuilder(None, [], None, torch.device("cpu"))
    common = SimpleNamespace(
        causal=True,
        max_query_len=1,
        num_reqs=2,
        num_actual_tokens=2,
        block_table_tensor=torch.zeros(2, 3, dtype=torch.int32),
        slot_mapping=torch.zeros(2, dtype=torch.int64),
    )
    metadata = builder.build(0, common)
    assert metadata.block_table is common.block_table_tensor
    assert metadata.slot_mapping is common.slot_mapping
    common.max_query_len = 2
    with pytest.raises(NotImplementedError, match="decode only"):
        builder.build(0, common)
    common.max_query_len = 1
    with pytest.raises(NotImplementedError, match="decode only"):
        builder.build(16, common)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native backend imports")
def test_custom_backend_registration_and_unbound_reader_fail_closed(monkeypatch):
    from vllm.agentic_kv.attention.backend import (
        ForgeAttentionBackend,
        ForgeAttentionImpl,
        ForgeMetadata,
        register_forge_backend,
    )
    from vllm.v1.attention.backends import registry

    monkeypatch.setattr(registry, "_ATTN_OVERRIDES", {})
    register_forge_backend()
    assert registry.AttentionBackendEnum.CUSTOM.get_class() is ForgeAttentionBackend
    assert not ForgeAttentionBackend.supports_dtype(torch.float16)
    assert not ForgeAttentionBackend.supports_kv_cache_dtype("fp8")
    assert not ForgeAttentionBackend.supports_head_size(128)
    impl = ForgeAttentionImpl.__new__(ForgeAttentionImpl)
    with pytest.raises(RuntimeError, match="bind a validated"):
        impl.forward(None, None, None, None, None, ForgeMetadata(1, None, None), None)
