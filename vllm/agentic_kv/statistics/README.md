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

## Proposed native attachment

The following is integration pseudocode, not installed runner fields or hooks:

```python
request_state.stats = pool.acquire(identity)  # None means no capacity

# Native batch preparation chooses sampled owners and stages routing.
# Revalidate at actual submission, including every graph replay.
destination = pool.writable(request_state.stats)
attention(..., stats=destination)

# Native lifecycle: after final permitted writes have been submitted.
lease = pool.detach(request_state.stats)
request_state.stats = None
save_operation.attach(lease)
```

Handles and events stay worker-local, not in scheduler metadata. The native
request state must prevent duplicate acquisition and clear its handle when
ownership transfers. `acquire()` does not maintain a second request registry.

Destinations are borrowed tensor views. Native metadata and graph staging must
prevent old prepared work from writing after detachment. `writable()` rejects
transferred/stale handles, but cannot revoke previously borrowed pointers.
The pool does not intercept arbitrary `graph.replay()` calls.

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

Before activation: attach handles to native worker request state; stage masks
and destinations through native batch metadata; enforce submission-time graph
lifetimes; connect model failure/preemption/cancellation; port prefill/turn-global
merging and the stride-aware reduction; wire validity-gated codec publication;
reserve startup HBM and measure end-to-end quality and overhead.
