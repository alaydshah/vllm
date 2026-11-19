# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.agentic_kv.history import HistoryKey, SavedHistory
from vllm.agentic_kv.store import HistoryStore
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request


def test_native_chained_hash_preserves_salt_and_earlier_token_identity():
    """Match real Request hashes, not a second connector-owned token hasher."""
    init_none_hash(sha256)
    hasher = get_request_block_hasher(8, sha256)

    def request(name, tokens, salt="tenant1"):
        return Request(
            name,
            tokens,
            SamplingParams(),
            None,
            cache_salt=salt,
            block_hasher=hasher,
        )

    source = request("source", list(range(65)))
    key = HistoryKey("engine1", 8, 64, source.block_hashes[7])
    history = SavedHistory("generation1", 64, 16, "bank1", 100, object(), key)
    store = HistoryStore(100, 1)
    assert store.reserve("save", 100)
    store.publish("save", history)

    def match(candidate):
        return store.acquire_matching(
            candidate.request_id,
            "engine1",
            candidate.block_hashes,
            8,
            candidate.num_prompt_tokens,
        )

    # A different suffix is fine; changing an earlier block or salt is not.
    returning = request("return", list(range(64)) + [999, 1000])
    lease = match(returning)
    assert lease is not None and lease.history is history
    assert match(request("other-salt", list(range(65)), "tenant2")) is None
    assert match(request("other-prefix", [999] + list(range(1, 65)))) is None
    assert match(returning) is lease
