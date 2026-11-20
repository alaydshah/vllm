# Worker statistics storage

This opt-in pool owns fixed-capacity storage and GPU completion holds. It does
not manage request execution, sampling or model success. The mixed-attention
worker is still unavailable, so native serving remains disabled.

## Ownership instead of request phases

| Owner | Meaning |
| --- | --- |
| Pool (available) | Storage may be reserved and reset |
| Native request state (handle) | Backend may borrow writable destinations |
| Save operation / completion queue (lease) | Detached storage stays held through its last GPU read |

`contracts.py` defines identities, handles and detached leases. `buffers.py`
allocates fixed tensor storage. `pool.py` transfers ownership and polls completion.
There is no request registry, ordinal counter, sample counter on CPU, model
submission wrapper, or seven-phase request-like state machine.

GPU sample counts and error flags remain in the buffers. The producer updates
those; the consumer interprets them. Zero samples means absence of evidence.
Native request progress supplies the unchanged sampling policy in the future
worker adapter. Removing policy from the pool does not change that policy.

## Native v2 attachment and CPU batch metadata

`RequestState` owns an optional statistics pool reference and a lazily allocated
handle list indexed by its existing native request slots. `worker.attach()` uses
the native request-ID lookup, rejects duplicate attachment and returns `None` on
pool exhaustion. No second request registry or admission decision is introduced.
The backend supplies immutable provenance explicitly; attachment is not automatic.

`RequestState.remove_request()` discards any attached handle before freeing its
slot. The v2 runner already uses this path for completion and preemption, so
statistics follow that lifecycle. A temporarily unscheduled request keeps its
native slot and handle. Preemption discards observations conservatively; a resumed
request starts with no handle. This does not implement cross-preemption merging.

A future save calls `worker.detach()` before removal to transfer ownership and
clear the native slot. Subsequent native removal leaves that consumer lease alone.
The pool still requires `reap()` at worker completion processing; automatic native
completion polling and save callbacks are not installed in this milestone.

The v2 runner has a disabled-by-default `statistics_sampler` callback. When
installed by a compatible backend, its existing input preparation calls
`worker.prepare_owners()` after final batch ordering and places a CPU handle tuple
on `InputBatch.statistics_owners`. The callback supplies the per-owner mask; the
adapter introduces no sampling policy. Only selected single-query decode rows
are supported. Unselected rows and exhausted slots carry `None`.

This tuple is an ownership snapshot, **not GPU write permission**. The backend
must call `forge.resolve_routes()` immediately before every actual submission or
graph replay, then stage routes with correct in-flight buffer lifetimes. Detaching
or recycling a handle invalidates old snapshots. The pool cannot revoke already
borrowed GPU pointers or intercept arbitrary graph replays.

CPU attachment/routing contract tests cover reorder, exhaustion, duplicate
attachment, stale handles, consumer holds and unsupported observations. These
use lightweight native-state fixtures; full runner and GPU staging validation
remain pending. No statistics backend, sampler or pool is activated by default.

## Save and cancellation

`detach(handle)` transfers ownership and records readiness after queued writes.
`consume(lease, stream, enqueue)` inserts a GPU wait, invokes one consumer, then
holds storage until a completion event passes. The callback must enqueue its
entire read lifetime on that stream and must not retain tensor views. A partially
submitted failing consumer still holds storage until GPU completion.

`discard(handle_or_unread_lease)` retires after queued model writes without a
consumer; the caller clears request metadata on cancellation/failure. Invalid
model output and invalid statistics are publication decisions owned by the native
execution/save path, not additional pool phases.

`reap()` processes nonblocking completion queries at a worker completion boundary.
Allocation and writable resolution never poll completion or grow storage. All
methods run on one worker control thread. A failed event recording quarantines
storage rather than making it reusable without proof of completion.

## Memory and activation gates

Tensor storage is charged by `StatisticsCapacity.tensor_bytes`. Startup must
subtract it before sizing native KV pages, and separately reserve scratch, codec
outputs, allocator and event overhead. An early return uses a different reserved
slot while the previous turn is held. Capacity exhaustion declines optional
statistics/save work, not native request admission.

Destinations have shape `[layers, 4 fields, KV heads, covered_tokens]`; kernels
must honor strides when coverage is shorter than capacity. Native KV page holds
remain the connector's separate responsibility.

Before activation: connect backend attachment and completion callbacks; stage masks
and destinations through native batch metadata; enforce submission-time graph
lifetimes; connect model failure/preemption/cancellation; port prefill/turn-global
merging and connect the implemented stride-aware reduction; wire validity-gated codec publication;
reserve startup HBM and measure end-to-end quality and overhead.

## Forge statistics producer adapter

`forge.resolve_routes(pool, owners)` resolves sampled handles in native batch
order into CPU `(slot, covered_tokens)` pairs. Unsampled owners are represented
by `None` and become slot `-1`. Sampling remains a caller decision. Duplicate
write destinations and stale/detached handles are rejected. The caller stages
these pairs into reusable int64 GPU metadata and uses the same selection to
control Forge score/summary exports.

`forge.accumulate_forge_statistics(...)` consumes the real Forge split/combine
outputs and reduces directly into one layer of the pool with tensor strides.
It preserves the four statistics and six-query-head mean, increments counts once
per sampled layer/head, and ORs sticky GPU flags (1: invalid/overflowed values,
2: inconsistent coverage, 4: count overflow). Inputs are log2 score lanes
`0,1,2,4,5,6` and combine metadata containing normalization and energy partials.
This adapter deliberately supports the source kernel's four-KV-head geometry;
it is not a general attention-observation ABI.

The reduction has no explicit tensor allocation, host result read or metadata
transfer. Call `load_forge_reduction()` explicitly at startup, then warm the kernel before
timed/captured execution. This milestone uses an opt-in source build requiring a
working ROCm/CUDA toolchain; native backend build packaging remains pending.
An all-unsampled batch should use plain attention and skip reduction entirely.
The caller must resolve handles at each submission, stage routes without
modifying in-flight metadata, and order writes on the pool's model stream.

The adapter is tested with the actual research attention kernels; it does not
install those kernels as a native vLLM attention backend. GPU route staging,
model failure handling and real codec publication remain pending.

## Fixed-address GPU route staging

`routes.StatisticsRoutes` reserves one int64 GPU table `[max_num_reqs, 2]`
and a bounded number of pinned CPU tables at startup. Configure the CPU copy
capacity from the native maximum concurrent batches. Tensor costs are
`16 * max_num_reqs` GPU bytes and that amount times the copy capacity on the host;
allocator/event overhead is additional. No tensor grows during submission.

The v2 runner's opt-in `statistics_routes` stages the current owner snapshot
before eager execution or FULL/PIECEWISE replay. Staging revalidates live handles
and duplicate destinations, fills unused/padded rows with disabled slots, and
copies once per batch. All layers share the fixed GPU pointer. Dummy and
all-unsampled batches explicitly disable previous routes.

Copies and every reader must use the pool's one model stream. Consequently the
next GPU copy is ordered after the previous model's readers, even while CPU
preparation overlaps GPU execution. Each pinned CPU table has a copy-completion
event: CPU mutation is forbidden until that copy completes. Queries are
nonblocking; if no table is reusable, staging disables this batch's optional
observations. It does not wait, grow storage or change request admission. This
capacity fallback must gate both score exports and reduction through the same
GPU routes; zero counts remain absence of evidence. Invalid handles raise rather
than being treated as an ordinary capacity miss.

Graphs capture route readers, never `stage()`. Every replay must pass through the
runner's staging hook. Capture starts with disabled routes; backend registration
must bind that same fixed pointer to its capture metadata. A copy/event failure
quarantines its CPU table. Microbatch overlap, context parallelism and speculative
execution are rejected when routing is enabled; their stream/layout contracts
have not been implemented. Direct calls bypassing the runner are unsupported.

The constructor and runner hook remain opt-in, not automatically activated.
Backend capture binding, startup budget enforcement, producer export masking,
completion polling and valid-save publication still gate real model serving.
