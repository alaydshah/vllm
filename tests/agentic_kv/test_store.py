# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.agentic_kv.history import SavedHistory
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
