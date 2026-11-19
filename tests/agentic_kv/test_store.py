# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from vllm.agentic_kv.history import HistoryKey, SavedHistory
from vllm.agentic_kv.store import HistoryStore


def entry(name="v1", size=100):
    return SavedHistory(name, 64, 16, "bank7", size, object())


def test_retry_holds_once_and_cancel_allows_eviction():
    store = HistoryStore(200, 2)
    assert store.reserve("save", 100)
    assert store.acquire("A", "v1") is None  # Reserved does not mean ready.
    history = entry()
    store.publish("save", history)
    lease = store.acquire("A", "v1")
    assert lease is not None
    assert store.acquire("A", "v1") is lease
    assert store.evict("v1") is None
    store.release(lease)
    store.release(lease)
    assert store.evict("v1") is history
    assert store.resident_bytes == 100
    store.confirm_disposed(history)
    store.confirm_disposed(history)
    assert store.resident_bytes == store.reserved_bytes == 0


def test_multiple_readers_and_stale_completion():
    store = HistoryStore(200, 2)
    store.reserve("save", 100)
    store.publish("save", entry())
    a = store.acquire("A", "v1")
    b = store.acquire("B", "v1")
    assert a is not None and b is not None
    store.release(a)
    newer_a = store.acquire("A", "v1")
    assert newer_a is not None and newer_a is not a
    store.release(a)
    store.release(b)
    assert store.evict("v1") is None
    store.release(newer_a)
    assert store.evict("v1") is not None


def test_reservations_charge_publication_overlap_and_preserve_old_entry():
    store = HistoryStore(200, 3)
    assert store.reserve("old", 120)
    store.publish("old", entry("old", 120))
    assert not store.reserve("new", 100)
    assert store.resident_bytes == 120
    assert store.reserve("new", 80)
    with pytest.raises(ValueError, match="reservation"):
        store.publish("new", entry("new", 81))
    assert store.reserved_bytes == 80
    store.abort("new")
    store.abort("new")
    assert store.reserved_bytes == 0
    assert store.acquire("A", "old") is not None


def test_entry_limit_and_generation_are_immutable():
    store = HistoryStore(1000, 1)
    assert store.reserve("save", 100)
    assert store.reserve("save", 100)
    assert not store.reserve("second", 1)
    with pytest.raises(ValueError, match="change"):
        store.reserve("save", 101)
    store.publish("save", entry())
    assert not store.reserve("second", 1)
    with pytest.raises(ValueError, match="immutable"):
        store.publish("save", entry())
    assert store.acquire("A", "v1") is not None
    with pytest.raises(ValueError, match="different"):
        store.acquire("A", "v2")


@pytest.mark.parametrize(
    "tokens,block_size,size", [(0, 16, 10), (65, 16, 10), (64, 0, 10), (64, 16, 0)]
)
def test_invalid_history_is_rejected(tokens, block_size, size):
    with pytest.raises(ValueError):
        SavedHistory("v1", tokens, block_size, "bank7", size, object())


def test_eviction_does_not_return_capacity_before_disposal():
    store = HistoryStore(100, 1)
    store.reserve("save", 100)
    history = entry()
    store.publish("save", history)
    assert store.evict("v1") is history
    assert store.acquire("A", "v1") is None
    assert not store.reserve("next", 100)
    store.confirm_disposed(history)
    assert store.reserve("next", 100)


def publish_keyed(store, name, tokens=64, prefix_hash=b"h8"):
    key = HistoryKey("engine1", 8, tokens, prefix_hash)
    history = replace(entry(name), prefix_tokens=tokens, key=key)
    assert store.reserve(name, 100)
    store.publish(name, history)
    return history


HASHES = [f"h{i}".encode() for i in range(1, 9)]


def test_matching_uses_longest_ready_prefix_and_keeps_prompt_token():
    store = HistoryStore(300, 3)
    publish_keyed(store, "short", 32, b"h4")
    publish_keyed(store, "long")
    store.reserve("pending", 100)
    # Native hash size (8) need not equal the saved page size (16).
    lease = store.acquire_matching("A", "engine1", HASHES, 8, 65)
    assert lease is not None and lease.history.entry_id == "long"
    lease = store.acquire_matching("B", "engine1", HASHES, 8, 64)
    assert lease is not None and lease.history.entry_id == "short"
    assert store.acquire_matching("C", "engine1", HASHES, 8, 32) is None


def test_matching_rejects_different_namespace_hash_or_hash_size():
    store = HistoryStore(100, 1)
    publish_keyed(store, "saved")
    assert store.acquire_matching("A", "engine2", HASHES, 8, 65) is None
    assert store.acquire_matching("A", "engine1", HASHES, 16, 65) is None
    changed = [*HASHES[:-1], b"different-salt-or-prefix"]
    assert store.acquire_matching("A", "engine1", changed, 8, 65) is None
    assert store.acquire_matching("A", "engine1", [], 8, 65) is None
    assert store.acquire_matching("A", "engine1", HASHES, 8, 65) is not None
    with pytest.raises(ValueError, match="identity"):
        store.acquire_matching("A", "engine1", changed, 8, 65)


def test_publication_during_retry_preserves_leased_generation():
    store = HistoryStore(300, 3)
    publish_keyed(store, "old", 32, b"h4")
    lease = store.acquire_matching("A", "engine1", HASHES, 8, 65)
    assert lease is not None
    publish_keyed(store, "replacement", 32, b"h4")
    publish_keyed(store, "longer")
    assert store.acquire_matching("A", "engine1", HASHES, 8, 65) is lease
    newer = store.acquire_matching("B", "engine1", HASHES, 8, 65)
    assert newer is not None and newer.history.entry_id == "longer"
    store.release(lease)
    assert store.evict("old") is lease.history
    replacement = store.acquire_matching("C", "engine1", HASHES, 8, 64)
    assert replacement is not None
    assert replacement.history.entry_id == "replacement"


def test_eviction_removes_lookup_before_disposal_credit():
    store = HistoryStore(100, 1)
    history = publish_keyed(store, "saved")
    assert store.evict("saved") is history
    assert store.acquire_matching("A", "engine1", HASHES, 8, 65) is None
    assert not store.reserve("next", 100)
    store.confirm_disposed(history)
    assert store.reserve("next", 100)


def test_saved_identity_must_cover_same_prefix():
    with pytest.raises(ValueError, match="saved prefix"):
        replace(entry(), key=HistoryKey("engine1", 8, 32, b"h4"))
