# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class HistoryLease:
    """One connector-owned hold, retained across native admission retries."""

    request_id: str
    history: SavedHistory
