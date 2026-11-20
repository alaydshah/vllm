# Worker attention statistics

This opt-in component owns fixed-capacity turn statistics and asynchronous read
leases. It does not activate Agentic KV serving. The worker reader and mixed
attention backend are still unavailable; the connector continues to fail closed.

## Boundaries

| Component | Responsibility |
| --- | --- |
| `contracts.py` | Generation/source/bank identity, borrowed tensor destinations, snapshot identity |
| `buffers.py` | Startup allocation of FP32 sums and int32 counts/flags; explicit tensor byte charge |
| `collector.py` | Sampling cadence, guarded batched submission, sealing, event ordering, retirement |
| Future backend adapter | Map native batch order to destinations, stage masks, directly reduce statistics |
| Future compression consumer | Validate every layer/head, normalize, encode and gate publication on model success |

No scheduler, native KV manager, request-state table or global registry is added.
No device values are read during acquisition or submission. `reap()` performs
nonblocking event queries at the worker's completion boundary. All methods run
on one worker control thread; background GPU work uses streams, not concurrent
mutation of the collector from Python threads.

## Producer contract

Reserve `StatisticsBuffers` before sizing the native KV pool. Its `tensor_bytes`
is exact tensor storage, not a complete startup HBM estimate: attention scratch,
output candidates, events and allocator overhead must also be reserved.

Acquire a handle for one immutable owner generation and covered old prefix.
Capacity exhaustion returns `None`; it never grows storage or polls completion.
An early return requires another slot while its previous snapshot remains held.

Call `submit_batch([(handle, ordinal), ...], enqueue)` once per model invocation
in native owner order. The callback receives a tuple of optional destinations:
`None` means this owner is unsampled, not that its attention should be skipped.
Ordinals 1, 9, 17, ... preserve the current cadence. No tensor mask or second
batch-slot mapping is owned here: native worker staging belongs in the adapter.
The callback must enqueue on the bound model stream. It increments GPU counts
once per sampled layer/head and preserves sticky error flags.

Destinations cover `[layers, 4 fields, KV heads, covered_tokens]`. Coverage can
be shorter than pool capacity, so kernels must honor tensor strides. This is
not yet a drop-in destination for the fixed-shape experimental HIP kernel.

Every replay must pass through the submission guard. Do not retain destination
views for unguarded future writes. A callback failure invalidates all owners in
that submitted batch; a sequence mismatch also closes the affected generations.
Asynchronous whole-model failure must be propagated by the future worker adapter.

## Consumer contract

`seal(handle)` rejects future submissions and records readiness after writes.
`consume(snapshot, stream, enqueue)` waits on that event on the consumer stream,
then records completion after the callback's final read. Storage stays charged
until `reap()` sees completion, including when the callback raises after partial
submission. One consumer is supported. Its callback must enqueue its entire read
lifetime on that stream and must not retain views for later use.

Readiness grants read access, not publication permission. The consumer must
validate all flags/counts and model success before replacing committed history.
The collector neither normalizes nor allocates codec output. `abandon(handle)`
discards active, failed or unread snapshots after queued model writes finish.
Native source KV holds remain the connector's separate responsibility.

## Activation gates

- Install the mixed-attention worker backend and its capability check.
- Stage owner masks/destinations using native worker metadata and graph lifetimes.
- Wire whole-model failure, preemption/recompute and cancellation to abandonment.
- Add prefill observations and retain the existing turn/global merging rule.
- Connect validity-gated codec publication and source-page holds.
- Reserve startup HBM, then measure wrapper and end-to-end serving overhead.

Contract tests use CPU tensors and deterministic events. CUDA/ROCm tests exercise
real cross-stream reads and submission-time graph-replay rejection. Neither is
a full-model attention correctness or serving-performance evaluation.
