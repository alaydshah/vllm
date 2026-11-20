# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Protocol

import torch


class CompletionEvent(Protocol):
    def record(self, stream: object) -> None: ...

    def query(self) -> bool: ...


class CompletionStream(Protocol):
    def wait_event(self, event: CompletionEvent) -> None: ...


@dataclass(frozen=True)
class StatisticsIdentity:
    """Immutable provenance, independent of recyclable native batch slots."""

    request_id: str
    generation: int
    source_version: str
    bank_version: str
    covered_tokens: int

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or self.generation < 0
            or not self.source_version
            or not self.bank_version
            or self.covered_tokens <= 0
        ):
            raise ValueError("invalid statistics identity or coverage")


@dataclass(frozen=True)
class StatisticsHandle:
    identity: StatisticsIdentity
    slot: int
    serial: int


@dataclass(frozen=True)
class StatisticsDestination:
    """Borrowed views, valid only inside the guarded producer/consumer callback.

    sums: FP32 [layers, four fields, KV heads, covered positions].
    counts/flags: int32 [layers, KV heads]. Producer reduces directly into sums,
    increments counts once per sampled layer/head, and ORs sticky error flags.
    The caller must stage owner routing through native worker metadata. These
    tensors must not be cached for later unguarded graph replay or writes.
    """

    sums: torch.Tensor
    counts: torch.Tensor
    flags: torch.Tensor


@dataclass(frozen=True)
class StatisticsSnapshot:
    handle: StatisticsHandle
    last_ordinal: int
    sample_count: int
