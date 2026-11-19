# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HistoryKey:
    """An engine-scoped native chained hash at a complete prefix boundary.

    The connector supplies a namespace tied to the model, execution/codec
    configuration and hash seed. Use native request hashes, including their
    salt, adapter and multimodal extra keys; never hash token IDs separately.
    This identifies approximate saved history, not an exact native cache hit.
    """

    namespace: str
    hash_block_size: int
    prefix_tokens: int
    prefix_hash: bytes

    def __post_init__(self) -> None:
        if (
            not self.namespace
            or not self.prefix_hash
            or self.hash_block_size <= 0
            or self.prefix_tokens <= 0
            or self.prefix_tokens % self.hash_block_size
        ):
            raise ValueError("history key requires a namespace and aligned native hash")


@dataclass(frozen=True)
class SavedHistory:
    """Ready, immutable history owned by the store until explicit eviction.

    The producer must finish GPU writes before publication and never mutate the
    payload afterwards. resident_bytes conservatively charges the entire owned
    bundle, including its bank. Shared-bank deduplication is not implemented.
    The opaque payload holds actual storage references; entry_id identifies one
    immutable generation, not an agent name or a token-only prefix hash.
    """

    entry_id: str
    prefix_tokens: int
    block_size: int
    bank_version: str
    resident_bytes: int
    payload: object = field(repr=False, compare=False)
    key: HistoryKey | None = None

    def __post_init__(self) -> None:
        if not self.entry_id or not self.bank_version:
            raise ValueError("history and bank identities must be nonempty")
        if (
            self.block_size <= 0
            or self.prefix_tokens <= 0
            or self.prefix_tokens % self.block_size
        ):
            raise ValueError("history must cover complete positive pages")
        if self.resident_bytes <= 0:
            raise ValueError("resident_bytes must be positive")
        if self.key is not None and self.key.prefix_tokens != self.prefix_tokens:
            raise ValueError("history key must cover the saved prefix")


@dataclass(frozen=True)
class HistoryDescriptor:
    """CPU-only identity sent to the worker; payload resolution stays local.

    A descriptor does not transfer storage or establish worker readiness. The
    reader must resolve this exact generation and bank in its resident catalog.
    """

    entry_id: str
    bank_version: str
    block_size: int
    key: HistoryKey

    @classmethod
    def from_history(cls, history: SavedHistory) -> "HistoryDescriptor":
        if history.key is None:
            raise ValueError("worker descriptors require a native history key")
        return cls(
            history.entry_id, history.bank_version, history.block_size, history.key
        )


@dataclass(frozen=True)
class HistoryLease:
    """One connector-owned hold, retained across native admission retries."""

    request_id: str
    history: SavedHistory
