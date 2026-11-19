# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded ready-history storage, serialized by its connector-side owner.

Reserve before allocating a candidate. Workers report readiness to this owner;
they must not mutate this store from background threads. These byte counts are
admission accounting, not a GPU allocator or an automatic memory profiler.
Attention scratch, compression scratch and native source blocks need separate
budgets. They are not charged as saved resident payloads here.
"""

from collections.abc import Sequence

from vllm.agentic_kv.history import HistoryKey, HistoryLease, SavedHistory


class HistoryStore:
    def __init__(self, capacity_bytes: int, max_entries: int) -> None:
        if capacity_bytes <= 0 or max_entries <= 0:
            raise ValueError("store limits must be positive")
        self.capacity_bytes = capacity_bytes
        self.max_entries = max_entries
        self.resident_bytes = 0
        self.reserved_bytes = 0
        self._reservations: dict[str, int] = {}
        self._entries: dict[str, SavedHistory] = {}
        self._retired: dict[str, SavedHistory] = {}
        self._leases: dict[str, HistoryLease] = {}
        self._readers: dict[str, int] = {}
        self._index: dict[HistoryKey, str] = {}

    def reserve(self, job_id: str, num_bytes: int) -> bool:
        """Reserve candidate capacity without changing native request admission."""
        if not job_id or num_bytes <= 0:
            raise ValueError("reservation requires an identity and positive bytes")
        if job_id in self._reservations:
            if self._reservations[job_id] != num_bytes:
                raise ValueError("cannot change an existing reservation")
            return True
        if (
            self.resident_bytes + self.reserved_bytes + num_bytes > self.capacity_bytes
            or len(self._entries) + len(self._retired) + len(self._reservations)
            >= self.max_entries
        ):
            return False
        self._reservations[job_id] = num_bytes
        self.reserved_bytes += num_bytes
        return True

    def abort(self, job_id: str) -> None:
        """Release a reservation after candidate storage is safely disposed of."""
        self.reserved_bytes -= self._reservations.pop(job_id, 0)

    def publish(self, job_id: str, history: SavedHistory) -> None:
        """Publish only ready storage; failure leaves the reservation intact."""
        if history.entry_id in self._entries or history.entry_id in self._retired:
            raise ValueError("cannot replace an immutable history generation")
        reserved = self._reservations.get(job_id)
        if reserved is None or history.resident_bytes > reserved:
            raise ValueError("history exceeds its reservation")
        self._entries[history.entry_id] = history
        self._readers[history.entry_id] = 0
        self.resident_bytes += history.resident_bytes
        if history.key is not None:
            self._index[history.key] = history.entry_id
        self.abort(job_id)

    def acquire_matching(
        self,
        request_id: str,
        namespace: str,
        block_hashes: Sequence[bytes],
        hash_block_size: int,
        num_prompt_tokens: int,
    ) -> HistoryLease | None:
        """Lease the longest ready prefix, leaving a prompt token to compute.

        Only precomputed native hashes are inspected; this does no device work,
        hashing or readiness waits. A retry retains its original generation,
        even if a longer or newer history has since been published. The caller
        handles local-cache preference and unsupported execution modes first.
        """
        if not request_id or not namespace or hash_block_size <= 0:
            raise ValueError("matching requires identities and a positive hash size")
        if num_prompt_tokens <= 0:
            raise ValueError("matching requires a positive prompt length")
        limit = min(len(block_hashes), (num_prompt_tokens - 1) // hash_block_size)
        existing = self._leases.get(request_id)
        if existing is not None:
            key = existing.history.key
            if (
                key is None
                or key.namespace != namespace
                or key.hash_block_size != hash_block_size
                or key.prefix_tokens // hash_block_size > limit
                or block_hashes[key.prefix_tokens // hash_block_size - 1]
                != key.prefix_hash
            ):
                raise ValueError("retry changed the leased history identity")
            return existing
        for count in range(limit, 0, -1):
            key = HistoryKey(
                namespace,
                hash_block_size,
                count * hash_block_size,
                block_hashes[count - 1],
            )
            entry_id = self._index.get(key)
            if entry_id is not None:
                return self.acquire(request_id, entry_id)
        return None

    def acquire(self, request_id: str, entry_id: str) -> HistoryLease | None:
        """Acquire once per request; identical retries retain the original hold."""
        if not request_id:
            raise ValueError("request identity must be nonempty")
        existing = self._leases.get(request_id)
        if existing is not None:
            if existing.history.entry_id != entry_id:
                raise ValueError("request already holds a different history generation")
            return existing
        history = self._entries.get(entry_id)
        if history is None:
            return None
        lease = HistoryLease(request_id, history)
        self._leases[request_id] = lease
        self._readers[entry_id] += 1
        return lease

    def release(self, lease: HistoryLease) -> None:
        """Release after GPU readers finish, or immediately if never submitted.

        Object identity prevents a stale completion from releasing a newer hold
        if the caller reuses a request ID. Repeated completion is a no-op.
        """
        if self._leases.get(lease.request_id) is lease:
            del self._leases[lease.request_id]
            self._readers[lease.history.entry_id] -= 1

    def evict(self, entry_id: str) -> SavedHistory | None:
        """Retire an unleased entry; its bytes remain charged until disposal."""
        if entry_id not in self._entries or self._readers[entry_id]:
            return None
        history = self._entries.pop(entry_id)
        if history.key is not None and self._index.get(history.key) == entry_id:
            # Conservatively miss rather than resurrect an older generation.
            del self._index[history.key]
        del self._readers[entry_id]
        self._retired[entry_id] = history
        return history

    def confirm_disposed(self, history: SavedHistory) -> None:
        """Return byte credit only after the caller safely releases GPU storage.

        The caller must drop storage references/complete retirement events first.
        This acknowledgement is not itself a device free or synchronization.
        """
        if self._retired.get(history.entry_id) is history:
            del self._retired[history.entry_id]
            self.resident_bytes -= history.resident_bytes
