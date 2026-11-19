# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch

from vllm.agentic_kv.connector import AgenticKVConnector, AgenticKVMetadata
from vllm.agentic_kv.history import HistoryKey, SavedHistory
from vllm.config import KVTransferConfig, ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.external_prefix_manager import (
    configure_external_prefix_groups,
    withdraw_external_prefix_coverage,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


@pytest.fixture
def setup():
    init_none_hash(sha256)

    def create(num_blocks=12):
        cache = configure_external_prefix_groups(
            KVCacheConfig(
                num_blocks,
                [],
                [
                    KVCacheGroupSpec(
                        ["layer"],
                        FullAttentionSpec(
                            16, num_kv_heads=1, head_size=1, dtype=torch.float32
                        ),
                    )
                ],
            )
        )
        config = SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="AgenticKVConnector",
                kv_role="kv_both",
                kv_connector_module_path=(
                    "vllm.distributed.kv_transfer.kv_connector.v1.agentic_kv_connector"
                ),
                kv_connector_extra_config={
                    "history_namespace": "test-engine",
                    "history_capacity_bytes": 300,
                    "history_max_entries": 3,
                },
            ),
            use_v2_model_runner=True,
            speculative_config=None,
            parallel_config=ParallelConfig(),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        )
        connector = KVConnectorFactory.create_connector(
            config, KVConnectorRole.SCHEDULER, cache
        )
        assert isinstance(connector, AgenticKVConnector)
        manager = KVCacheManager(cache, 256, 16, 16, enable_caching=True)
        connector.bind_kv_cache_manager(manager)
        return connector, manager, config, cache

    return create


def request(name="A", tokens=None):
    return Request(
        name,
        list(range(81)) if tokens is None else tokens,
        SamplingParams(),
        None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def publish(connector, req, name="saved"):
    history = SavedHistory(
        name,
        64,
        16,
        "bank7",
        100,
        object(),
        HistoryKey("test-engine", 16, 64, req.block_hashes[3]),
    )
    assert connector.history_store.reserve(name, 100)
    connector.history_store.publish(name, history)
    return history


def metadata(connector, manager, req):
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData.from_request(req, manager.get_block_ids(req.request_id))
    ]
    return connector.build_connector_meta(output)


def test_real_factory_keeps_worker_serving_gated(setup):
    _, _, config, cache = setup()
    with pytest.raises(NotImplementedError, match="reader"):
        KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, cache)


def test_pending_save_is_a_miss_without_waiting(setup):
    connector, _, _, _ = setup()
    assert connector.history_store.reserve("pending", 100)
    assert connector.get_num_new_matched_tokens(request(), 0) == (0, False)


def test_failed_native_admission_retries_same_lease_and_defers_release(setup):
    connector, manager, _, _ = setup(3)
    req = request()
    history = publish(connector, req)
    blocker = request("blocker", [999] * 16)
    assert manager.allocate_slots(blocker, 16) is not None
    assert connector.get_num_new_matched_tokens(req, 0) == (64, False)
    assert manager.allocate_slots(req, 17, num_external_computed_tokens=64) is None
    assert connector.build_connector_meta(SchedulerOutput.make_empty()).histories == {}
    publish(connector, req, "newer")
    assert connector.get_num_new_matched_tokens(req, 0) == (64, False)
    assert connector.history_store.evict("saved") is None
    assert connector.history_store.evict("newer") is not None
    manager.free(blocker)
    blocks = manager.allocate_slots(req, 17, num_external_computed_tokens=64)
    assert blocks is not None and len(blocks.blocks[0]) == 2
    connector.update_state_after_alloc(req, blocks, 64)
    with pytest.raises(ValueError, match="admitted"):
        withdraw_external_prefix_coverage(manager, req.request_id)
    assert metadata(connector, manager, req).get_external_prefix_lengths() == {"A": 64}
    with pytest.raises(ValueError, match="unexpected"):
        connector.update_connector_output(KVConnectorOutput(finished_sending={"A"}))
    assert connector.history_store.evict("saved") is None
    assert connector.request_finished(req, blocks.get_block_ids()[0]) == (True, None)
    assert connector.history_store.evict("saved") is None
    connector.update_connector_output(KVConnectorOutput(finished_sending={"A"}))
    manager.free(req)  # Native scheduler does this after consuming completion.
    assert connector.history_store.evict("saved") is history
    assert manager.block_pool.get_num_free_blocks() == 2


@pytest.mark.parametrize("allocated", [False, True])
def test_pending_cancellation_releases_without_worker_ack(setup, allocated):
    connector, manager, _, _ = setup()
    req = request()
    history = publish(connector, req)
    connector.get_num_new_matched_tokens(req, 0)
    if allocated:
        blocks = manager.allocate_slots(req, 17, num_external_computed_tokens=64)
        connector.update_state_after_alloc(req, blocks, 64)
    assert connector.request_finished(req, []) == (False, None)
    manager.free(req)
    assert connector.history_store.evict("saved") is history
    assert not manager.coordinator.single_type_managers[0].external_prefixes


def test_dense_hit_appearing_during_retry_withdraws_only_pending_coverage(setup):
    connector, manager, _, _ = setup()
    req = request()
    history = publish(connector, req)
    assert connector.get_num_new_matched_tokens(req, 0) == (64, False)
    donor = request("donor")
    assert manager.allocate_slots(donor, 64) is not None
    dense, num_tokens, _ = manager.get_computed_blocks(req)
    assert num_tokens == 64
    assert connector.get_num_new_matched_tokens(req, num_tokens) == (0, False)
    assert connector.history_store.evict("saved") is history
    assert (
        manager.allocate_slots(
            req, 17, num_new_computed_tokens=num_tokens, new_computed_blocks=dense
        )
        is not None
    )
    assert len(manager.get_blocks("A").blocks[0]) == 6
    assert metadata(connector, manager, req).histories == {}
    manager.free(req)
    manager.free(donor)


def test_preemption_recomputes_natively_but_keeps_inflight_history_alive(setup):
    connector, manager, _, _ = setup()
    req = request()
    history = publish(connector, req)
    connector.get_num_new_matched_tokens(req, 0)
    blocks = manager.allocate_slots(req, 17, num_external_computed_tokens=64)
    connector.update_state_after_alloc(req, blocks, 64)
    metadata(connector, manager, req)
    manager.free(req)
    req.num_preemptions += 1
    assert connector.get_num_new_matched_tokens(req, 0) == (0, False)
    assert manager.allocate_slots(req, 81) is not None
    assert metadata(connector, manager, req).histories == {}
    assert connector.history_store.evict("saved") is None
    assert connector.request_finished(req, []) == (True, None)
    connector.update_connector_output(KVConnectorOutput(finished_sending={"A"}))
    manager.free(req)
    assert connector.history_store.evict("saved") is history


def test_metadata_transports_immutable_identity_without_payload(setup):
    class LocalPayload:
        pass

    connector, manager, _, _ = setup()
    req = request()
    history = SavedHistory(
        "original",
        64,
        16,
        "bank7",
        100,
        LocalPayload(),
        HistoryKey("test-engine", 16, 64, req.block_hashes[3]),
    )
    connector.history_store.reserve("original", 100)
    connector.history_store.publish("original", history)
    connector.get_num_new_matched_tokens(req, 0)
    newer = replace(history, entry_id="replacement", bank_version="bank8")
    connector.history_store.reserve("replacement", 100)
    connector.history_store.publish("replacement", newer)
    blocks = manager.allocate_slots(req, 17, num_external_computed_tokens=64)
    connector.update_state_after_alloc(req, blocks, 64)
    encoded = MsgpackEncoder().encode(metadata(connector, manager, req))
    assert len(encoded) == 1  # No out-of-band tensor buffers.
    transported = MsgpackDecoder(AgenticKVMetadata).decode(encoded)
    descriptor = transported.histories["A"]
    assert descriptor.entry_id == "original" and descriptor.bank_version == "bank7"
    assert descriptor.key == history.key
    assert not hasattr(descriptor, "payload")
    with pytest.raises(FrozenInstanceError):
        descriptor.entry_id = "replacement"
    offsets = transported.get_external_prefix_lengths()
    offsets["A"] = 0
    assert transported.get_external_prefix_lengths() == {"A": 64}


def test_native_scheduler_admits_suffix_and_owns_delayed_free(setup):
    from tests.v1.core.utils import create_scheduler

    connector, manager, _, cache = setup(3)
    scheduler = create_scheduler(
        num_blocks=3,
        max_num_batched_tokens=128,
        max_model_len=256,
        skip_tokenizer_init=True,
        use_v2_model_runner=True,
    )
    # Explicit experimental wiring; production startup remains gated.
    scheduler.kv_cache_config = cache
    scheduler.kv_cache_manager = manager
    scheduler.connector = connector
    req = request()
    history = publish(connector, req)
    blocker = request("blocker", [999] * 16)
    manager.allocate_slots(blocker, 16)
    scheduler.add_request(req)
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    manager.free(blocker)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"A": 17}
    assert output.scheduled_new_reqs[0].num_computed_tokens == 64
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == 2
    assert output.kv_connector_metadata.get_external_prefix_lengths() == {"A": 64}
    scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
    assert "A" in scheduler.requests
    assert connector.history_store.evict("saved") is None
    scheduler._update_from_kv_xfer_finished(KVConnectorOutput(finished_sending={"A"}))
    assert "A" not in scheduler.requests
    assert connector.history_store.evict("saved") is history
