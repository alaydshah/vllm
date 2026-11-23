// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Migrated from serving/source_stack/kernels/forge_kernel.py (_HIP).
// Prefix reconstruction and split/combine arithmetic are preserved.
// Suffix reads follow native page tables; statistics use per-owner routes.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>

using BF16 = __hip_bfloat16;
typedef float f2 __attribute__((ext_vector_type(2)));  // -> v_pk_*_f32
// Native 4-wide vector: a row load stays ONE <4 x float> load through ISel
// (HIP's float4 is a struct that SROA splits into scalars, which the pre-RA
// scheduler then places / re-merges by use proximity).
typedef float f4 __attribute__((ext_vector_type(4)));
typedef uint32_t u2 __attribute__((ext_vector_type(2)));
typedef uint32_t u16 __attribute__((ext_vector_type(16)));
struct alignas(16) Record {
  uint16_t pk, pv;
  __half gk, gv;
  int32_t ks, vs;
};
static_assert(sizeof(Record) == 16);
// Scores are kept in log2 units: scale (1/16) * log2(e) folded into the score
// so all softmax exponentials are a single native v_exp_f32.
constexpr float SCORE_SCALE = 0.0625f * 1.4426950408889634f;
__device__ __forceinline__ float fexp2(float x) {
  return __builtin_amdgcn_exp2f(x);
}
// Split partials: *normalised* y rows (y / l, a convex combination of v rows,
// so FP16 is range-safe) stored as FP16 [row][6][256] (512-B pitch) and a
// separate contiguous FP32 (m, l) stream [row][6][2] that the combine's pass 1
// reads coalesced; the combine weights each row by exp2(m_i - M) * l_i.
constexpr int STRIDE = 256;
constexpr int META_STRIDE = 10;
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  h2 r = {(_Float16)a, (_Float16)b};
  return __builtin_bit_cast(uint32_t, r);
}
__device__ __forceinline__ f2 unpack_h2(uint32_t u) {
  h2 r = __builtin_bit_cast(h2, u);
  return f2{(float)r.x, (float)r.y};
}

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
  for (int d = 1; d < 64; d *= 2) x += __shfl_xor(x, d, 64);
  return x;
}
__device__ __forceinline__ float readlane(float v, int l) {
  return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), l));
}
// Cross-lane exchanges on the VALU (no LDS round trip): xor 1/2 = quad_perm,
// xor 8 = row_ror:8, xor 4 = row_shl:4 (banks 0,2) + row_shr:4 (banks 1,3); xor
// 16/32 use the gfx950 permlane swaps on a register pair (each lane ends up
// with both halves of one register).
template <int CTRL, int BANK>
__device__ __forceinline__ float dppv(float v) {
  return __int_as_float(
      __builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, BANK, true));
}
template <int CTRL, int BANK>
__device__ __forceinline__ float dppi(float old, float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(
      __float_as_int(old), __float_as_int(v), CTRL, 0xf, BANK, false));
}
__device__ __forceinline__ float xor1(float v) { return dppv<0xB1, 0xf>(v); }
__device__ __forceinline__ float xor2(float v) { return dppv<0x4E, 0xf>(v); }
__device__ __forceinline__ float xor8(float v) { return dppv<0x128, 0xf>(v); }
// v + xor4(v) as two arithmetic DPP adds (banks 0,2 read lane+4 via row_shl,
// banks 1,3 read lane-4 via row_shr); the second op fills the banks the first
// left untouched, so no zero-init. The leading s_nop covers the VALU-write ->
// DPP-read hazard the compiler cannot see inside the asm.
__device__ __forceinline__ float add_xor4(float v) {
  float d;
  asm("s_nop 1\n\t"
      "v_add_f32_dpp %0, %1, %1 row_shl:4 row_mask:0xf bank_mask:0x5 "
      "bound_ctrl:1\n\t"
      "v_add_f32_dpp %0, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xa "
      "bound_ctrl:1"
      : "=&v"(d)
      : "v"(v));
  return d;
}
__device__ __forceinline__ void swap32(float& a, float& b) {
  auto r = __builtin_amdgcn_permlane32_swap(__float_as_int(a),
                                            __float_as_int(b), false, false);
  a = __int_as_float(r[0]);
  b = __int_as_float(r[1]);
}
__device__ __forceinline__ void swap16(float& a, float& b) {
  auto r = __builtin_amdgcn_permlane16_swap(__float_as_int(a),
                                            __float_as_int(b), false, false);
  a = __int_as_float(r[0]);
  b = __int_as_float(r[1]);
}
// Hadamard butterfly stage over lane bit 5 / bit 4 for a register pair: after
// the swap each lane holds (own, partner) of one register and computes the sum
// (lower lane's result) and the difference (upper lane's result). The results
// are NOT swapped back: the lower lanes keep both results of register a and the
// upper lanes both results of register b, i.e. the stage exchanges the lane bit
// with slot bit 0. The K side of the whole kernel works in that permuted dim
// order (see KPERM in prepare), so nothing has to be undone.
__device__ __forceinline__ void bfly32(float& a, float& b) {
  swap32(a, b);
  float s = a + b, d = a - b;
  a = s;
  b = d;
}
__device__ __forceinline__ void bfly16(float& a, float& b) {
  swap16(a, b);
  float s = a + b, d = a - b;
  a = s;
  b = d;
}

// Six simultaneous 64-lane sums with one transposed tree: each stage halves the
// live values where possible; the six totals land in lanes 0/8/16/32/40/48 of
// the returned register (head g <-> lane HEAD_LANE[g]; lanes 24/56 hold
// duplicates of heads 2/5 and are never read). Stage order is lane bit 5
// (permlane32 swaps), bit 3 (bank-masked DPP adds: DPP banks are lane bits 2-3,
// so bank_mask 0x3 / 0xc address the bit-3-clear / bit-3-set lanes and the two
// adds merge a0 and a1 into one register with no select), bit 4 (one permlane16
// swap), then bits 2, 1, 0.
__device__ __forceinline__ float reduce6(const float (&x)[6], int lane) {
  float a0 = x[0], s0 = x[3], a1 = x[1], s1 = x[4], a2 = x[2], s2 = x[5];
  swap32(a0, s0);
  a0 += s0;
  swap32(a1, s1);
  a1 += s1;
  swap32(a2, s2);
  a2 += s2;  // lo: x0,x1,x2  hi: x3,x4,x5
  float b0, b1;
  asm("s_nop 1\n\t"
      "v_add_f32_dpp %0, %2, %2 row_ror:8 row_mask:0xf bank_mask:0x3 "
      "bound_ctrl:1\n\t"  // bit3=0: x0 | x3
      "v_add_f32_dpp %0, %3, %3 row_ror:8 row_mask:0xf bank_mask:0xc "
      "bound_ctrl:1\n\t"  // bit3=1: x1 | x4
      "v_add_f32_dpp %1, %4, %4 row_ror:8 row_mask:0xf bank_mask:0xf "
      "bound_ctrl:1"  // x2 | x5
      : "=&v"(b0), "=&v"(b1)
      : "v"(a0), "v"(a1), "v"(a2));
  swap16(b0, b1);
  float c0 = b0 + b1;  // rows: (x0|x1) x2 (x3|x4) x5
  c0 = add_xor4(c0);
  c0 += xor2(c0);
  c0 += xor1(c0);
  return c0;
}
__device__ __forceinline__ constexpr int head_lane(int g) {
  return g < 3 ? g * 8 : (g + 1) * 8;
}
// Lazy online-softmax reference: the per-head reference m_ref only moves when a
// score exceeds it by more than TAU (log2 units), so the 24-accumulator rescale
// is skipped on almost every token. Weights stay <= 2^TAU per token;
// combine/moments normalise against the published (m_ref, denom), so p is
// unchanged. Non-head lanes carry a large finite reference so exp2(s - m_ref) =
// 0 and alpha = 1 there.
constexpr float LAZY_TAU = 16.f;
constexpr float NONHEAD_REF = 3.0e38f;
// K residual: byte of four 2-bit codes -> LDS table (codebook values with the
// two in-lane Hadamard stages pre-applied) -> 4 DPP stages + 2 permlane-swap
// stages complete the unnormalised 256-point Walsh-Hadamard; the caller applies
// the per-row kscale = norm / ||c|| * 1/16. Stages over lane bits 0, 1, 3 are
// fused DPP FMAs (partner * sg + self): on lanes with the bit set this yields
// self - partner = -(partner - self), i.e. the exact butterfly up to a per-lane
// sign
// (-1)^popcount(lane & 0b1011) that every later stage carries through unchanged
// (both inputs of each later butterfly agree on bits 0, 1, 3); that lane sign
// is folded into the stored K-side operands (banks_k, tail_k) and the query by
// prepare / the prologue, so no per-token sign multiply is needed. The four
// in-register stages are one asm block (the compiler cannot fuse DPP into
// v_fmac because the FMA is still VOP3 when its DPP combiner runs): stage-major
// / register-minor order puts >= 3 VALU between every write of a register and
// its next DPP read, so only the leading s_nop is needed. Stage over lane bit 2
// has a DPP-bank-granular sign pattern, so it is two arithmetic DPP ops (banks
// 0,2: partner + v; banks 1,3: partner - v), same pairing as add_xor4 above,
// into fresh registers. Only two fresh registers are needed: once a0 / a1 have
// been consumed by their own xor-4 pair they take the results of a2 / a3 (every
// DPP read stays >= 3 VALU behind the last write of its source), which keeps
// the loop's register peak (this stage) two VGPRs lower.
#define FWHT_FMAC(R, SG, CTRL)                \
  "v_fmac_f32_dpp " R ", " R ", " SG " " CTRL \
  " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
#define FWHT_X4(D, R)                                      \
  "v_add_f32_dpp " D ", " R ", " R                         \
  " row_shl:4 row_mask:0xf bank_mask:0x5 bound_ctrl:1\n\t" \
  "v_sub_f32_dpp " D ", " R ", " R                         \
  " row_shr:4 row_mask:0xf bank_mask:0xa bound_ctrl:1\n\t"
__device__ __forceinline__ void residual_k(const f4 t, const float (&sg)[4],
                                           float (&v)[4]) {
  float a0 = t.x, a1 = t.y, a2 = t.z, a3 = t.w, d0, d1;
  asm("s_nop 1\n\t" FWHT_FMAC("%0", "%6", "quad_perm:[1,0,3,2]")
          FWHT_FMAC("%1", "%6", "quad_perm:[1,0,3,2]")
              FWHT_FMAC("%2", "%6", "quad_perm:[1,0,3,2]") FWHT_FMAC(
                  "%3", "%6", "quad_perm:[1,0,3,2]")
                  FWHT_FMAC("%0", "%7", "quad_perm:[2,3,0,1]") FWHT_FMAC(
                      "%1", "%7", "quad_perm:[2,3,0,1]")
                      FWHT_FMAC("%2", "%7", "quad_perm:[2,3,0,1]") FWHT_FMAC(
                          "%3", "%7", "quad_perm:[2,3,0,1]") FWHT_X4("%4", "%0")
                          FWHT_X4("%5", "%1") FWHT_X4("%0", "%2") FWHT_X4(
                              "%1", "%3") FWHT_FMAC("%4", "%8", "row_ror:8")
                              FWHT_FMAC("%5", "%8", "row_ror:8")
                                  FWHT_FMAC("%0", "%8", "row_ror:8")
                                      FWHT_FMAC("%1", "%8", "row_ror:8")
      : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "=&v"(d0), "=&v"(d1)
      : "v"(sg[0]), "v"(sg[1]), "v"(sg[3]));
  v[0] = d0;
  v[1] = d1;
  v[2] = a0;
  v[3] = a1;
  bfly16(v[0], v[1]);
  bfly16(v[2], v[3]);
  bfly32(v[0], v[1]);
  bfly32(v[2], v[3]);
}

// K-side physical dim order (KPERM in prepare): lane l reads K-residual byte
// rho(l) (lane bits 3 and 5 swapped) and, after the swap-back-free permlane
// stages, holds the logical dims {D, D+32, D+2, D+34} with D(l) = l4 | (l&7)<<2
// | l5<<6 | l3<<7 in its four slots. The rotary partner (dim bit 5) is thus
// in-lane: slot pairs (0,1) and (2,3) rotate with (cos, sin) of dims D and D+2.
// Rotary table row (prepare-built, ROT_STRIDE floats per prefix position, score
// scale folded in):
//   16 quads (cos[D], cos[D+2], sin[D], sin[D+2]) for the rotary lanes
//   {0..7, 16..23} in rank order rank = (l&7) | l4<<3, then one identity quad
//   (1, 1, 0, 0) at float 64 for all other lanes.
constexpr int ROT_STRIDE = 68;

// Per-token row loads go through buffer resources (SGPR quads built once per
// block): the address is resource base + soffset (the wave-uniform row byte
// offset, 32-bit SALU arithmetic) + voffset (the lane's loop-invariant byte
// offset), so a token costs no VALU address instructions at all. A global load
// would need either a 64-bit VALU add or a per-token zero-extension of the lane
// offset. Raw buffer (stride 0), num_records = 0xFFFFFFFF (no clamping), word 3
// = 0x00020000 (gfx9-family 32-bit data format, the value CK uses for gfx90a /
// gfx94x / gfx950).
constexpr int BUF_WORD3 = 0x00020000;
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, -1,
                                           BUF_WORD3);
}
__device__ __forceinline__ f4 buffer_load16(__amdgpu_buffer_rsrc_t r,
                                            uint32_t voff, uint32_t soff) {
  return __builtin_bit_cast(
      f4, __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0));
}
// Structured (stride 16 / 8) resources for the anchor and tail rows: the lane
// index goes through vindex (idxen), so the row loads share the plain `lane`
// VGPR with the residual byte loads and no per-lane byte offset has to be held.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t
make_rsrc_rows(const void* base, int stride) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base),
                                           (short)stride, -1, BUF_WORD3);
}
// (clang exposes no __builtin for the structured form; the LLVM intrinsics are
// bound by name, as CK does)
__device__ f4 llvm_struct_buffer_load_f32x4(
    __amdgpu_buffer_rsrc_t rsrc, int vindex, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.struct.ptr.buffer.load.v4f32");
__device__ u2 llvm_struct_buffer_load_i32x2(
    __amdgpu_buffer_rsrc_t rsrc, int vindex, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.struct.ptr.buffer.load.v2i32");
__device__ __forceinline__ f4 buffer_load16_lane(__amdgpu_buffer_rsrc_t r,
                                                 int lane, uint32_t soff) {
  return llvm_struct_buffer_load_f32x4(r, lane, 0, (int)soff, 0);
}
__device__ __forceinline__ u2 buffer_load8_lane(__amdgpu_buffer_rsrc_t r,
                                                int lane, uint32_t soff) {
  return llvm_struct_buffer_load_i32x2(r, lane, 0, (int)soff, 0);
}
struct NativePages {
  const BF16 *key, *value;
  const int32_t* blocks;
  int block_size, table_stride;
  int64_t block_stride, token_stride, head_stride;
};
struct Owner {  // wave-uniform per-(owner, head) state hoisted out of the token
                // loop
  const Record* __restrict__ records;
  const uint8_t* __restrict__ vres;
  const float* __restrict__ kscale;
  __amdgpu_buffer_rsrc_t rk, rv, rbk, rbv,
      rrot;  // K/V residual streams, this bank's K/V anchors, rotary table,
             // tails
  uint32_t koff,
      voff;  // 32-bit offsets of this (owner, head) inside the shared streams
  int n, head;
  const int32_t* blocks;
  NativePages pages;
};
// Opaque-value barrier: forces the operands to be materialised here (keeps the
// residual lookups ahead of the wave-uniform branches so the compiler cannot
// sink the loads back into them). The quad is a native 4-wide vector (one
// ds_read_b128 / global_load_dwordx4), so the whole register quad is pinned at
// once.
__device__ __forceinline__ void pin4(f4& t) { asm volatile("" : "+v"(t)); }

// Every lane in a wave consumes the same16byte token record. Explicitly
// scalarize its four bit-packed words after the ordinary memory load.
__device__ __forceinline__ Record uniform_record(const Record& r) {
  const uint32_t a = (uint32_t)r.pk | ((uint32_t)r.pv << 16);
  const uint32_t b = (uint32_t)__half_as_ushort(r.gk) |
                     ((uint32_t)__half_as_ushort(r.gv) << 16);
  const uint32_t c = (uint32_t)r.ks, d = (uint32_t)r.vs;
  const uint32_t ua = __builtin_amdgcn_readfirstlane(a),
                 ub = __builtin_amdgcn_readfirstlane(b);
  const uint32_t uc = __builtin_amdgcn_readfirstlane(c),
                 ud = __builtin_amdgcn_readfirstlane(d);
#ifdef ANCHOR_UNIFORM_AUDIT
  if (__ballot(a == ua && b == ub && c == uc && d == ud) != ~0ull)
    __builtin_trap();
#endif
  return Record{(uint16_t)ua,
                (uint16_t)(ua >> 16),
                __ushort_as_half((uint16_t)ub),
                __ushort_as_half((uint16_t)(ub >> 16)),
                (int32_t)uc,
                (int32_t)ud};
}

// Owner payload is immutable for the duration of a launch. The descriptor's
// integer-to-pointer cast otherwise loses LLVM's read-only address-space fact.
// Let the compiler schedule its scalar loads and waits, rather than embedding
// asynchronous SMEM instructions in divergent control flow.
typedef uint32_t __attribute__((address_space(4))) ConstantU32;
typedef uint32_t u4 __attribute__((ext_vector_type(4)));
typedef u4 __attribute__((address_space(4))) ConstantU4;
__device__ __forceinline__ const ConstantU32* constant_words(const void* ptr) {
  const uint64_t raw = (uint64_t)ptr;
  const uint64_t uniform =
      ((uint64_t)__builtin_amdgcn_readfirstlane((uint32_t)(raw >> 32)) << 32) |
      (uint32_t)__builtin_amdgcn_readfirstlane((uint32_t)raw);
  return reinterpret_cast<const ConstantU32*>(uniform);
}
__device__ __forceinline__ Record load_record_readonly(const Record* base,
                                                       int pos) {
  const u4 words =
      *reinterpret_cast<const ConstantU4*>(constant_words(base + pos));
  const uint32_t a = words[0], b = words[1], c = words[2], d = words[3];
#ifdef ANCHOR_UNIFORM_AUDIT
  const Record raw = base[pos];
  if (__ballot(a == ((uint32_t)raw.pk | ((uint32_t)raw.pv << 16)) &&
               b == ((uint32_t)__half_as_ushort(raw.gk) |
                     ((uint32_t)__half_as_ushort(raw.gv) << 16)) &&
               c == (uint32_t)raw.ks && d == (uint32_t)raw.vs) != ~0ull)
    __builtin_trap();
#endif
  return Record{(uint16_t)a,
                (uint16_t)(a >> 16),
                __ushort_as_half((uint16_t)b),
                __ushort_as_half((uint16_t)(b >> 16)),
                (int32_t)c,
                (int32_t)d};
}

// One memory round trip per token: every per-token load (residual code bytes,
// scalar kscale / V header, rotary quads, anchor rows) is issued before the
// first wait. Residual rows are clamped to row 0 for tokens without a residual
// (always in-bounds: prepare appends a dummy row to each stream); only the
// Hadamard / apply arithmetic stays behind the wave-uniform branches.
// PAGE_SIZE == 0 keeps runtime sizing for other supported native page sizes.
template <int PAGE_SIZE>
__device__ __forceinline__ void token(const Owner& o, const Record& r,
                                      const f4* __restrict__ ktab,
                                      const f4* __restrict__ vtab, int pos,
                                      int lane, uint32_t offr,
                                      const float (&sg)[4], float (&k)[4],
                                      float (&v)[4]) {
  if (pos < o.n) {
    const uint32_t kb = (uint32_t)max(r.ks, 0), vb = (uint32_t)max(r.vs, 0);
    const uint32_t krow = o.koff + (kb << 6),
                   vrow = o.voff + vb * 68u;  // wave-uniform byte offsets
    const uint8_t kbyte =
        __builtin_amdgcn_raw_buffer_load_b8(o.rk, (uint32_t)lane, krow, 0);
    const uint8_t vbyte =
        __builtin_amdgcn_raw_buffer_load_b8(o.rv, (uint32_t)lane, vrow, 0);
    float ksc = __uint_as_float(*constant_words(o.kscale + kb));
    uint32_t hdr = *constant_words(o.vres + vrow + 64u);  // s_load
    const f4 rq =
        buffer_load16(o.rrot, offr, (uint32_t)pos * (uint32_t)(ROT_STRIDE * 4));
    const f4 ak = buffer_load16_lane(o.rbk, lane, (uint32_t)r.pk * 1024u);
    const f4 av = buffer_load16_lane(o.rbv, lane, (uint32_t)r.pv * 1024u);
    f4 tk = ktab[kbyte];
    f4 tv = vtab[vbyte];
    // The pins are scheduling boundaries: they sit after every load so all of
    // them are in flight before the first wait, and keep the scalar loads /
    // lookups ahead of the branches. Read-only address-space loads already
    // yield the original scalar values.
#ifdef ANCHOR_UNIFORM_AUDIT
    if (__ballot(__float_as_uint(o.kscale[kb]) == __float_as_uint(ksc) &&
                 *reinterpret_cast<const uint32_t*>(o.vres + vrow + 64u) ==
                     hdr) != ~0ull)
      __builtin_trap();
#endif
    asm volatile("" : "+s"(ksc), "+s"(hdr));
    pin4(tk);
    pin4(tv);
    const float gamma = __half2float(r.gk), g = __half2float(r.gv);
    k[0] = gamma * ak.x;
    k[1] = gamma * ak.y;
    k[2] = gamma * ak.z;
    k[3] = gamma * ak.w;
    // the anchor V row is applied here, in the same block as its load, so
    // MachineSink cannot push the load below the K-residual branch (which would
    // serialise it behind the LDS lookups)
    v[0] = g * av.x;
    v[1] = g * av.y;
    v[2] = g * av.z;
    v[3] = g * av.w;
    if (r.ks >= 0) {
      float res[4];
      residual_k(tk, sg, res);
#pragma unroll
      for (int d = 0; d < 4; ++d) k[d] = fmaf(ksc, res[d], k[d]);
    }
    // in-lane rotary on the adjacent slot pairs (0,1) = dims (D, D+32) and
    // (2,3) = dims (D+2, D+34): (k0,k1)' = (c_a*k0 - s_a*k1, c_a*k1 + s_a*k0),
    // same for (k2,k3) with (c_b, s_b). Written as VOP3P with op_sel (half swap
    // / broadcast) and neg_lo so no register shuffles are needed: 4 VALU total.
    {
      f2 ka = {k[0], k[1]}, kb = {k[2], k[3]}, ta, tb;
      const f2 cp = {rq.x, rq.y}, sp = {rq.z, rq.w};
      asm("v_pk_mul_f32 %2, %5, %0 op_sel:[0,1] op_sel_hi:[0,0]\n\t"  // (s_a*k1,
                                                                      // s_a*k0)
          "v_pk_mul_f32 %3, %5, %1 op_sel:[1,1] op_sel_hi:[1,0]\n\t"  // (s_b*k3,
                                                                      // s_b*k2)
          "v_pk_fma_f32 %0, %4, %0, %2 op_sel_hi:[0,1,1] neg_lo:[0,0,1]\n\t"  // (c_a*k0 - ., c_a*k1 + .)
          "v_pk_fma_f32 %1, %4, %1, %3 op_sel:[1,0,0] op_sel_hi:[1,1,1] "
          "neg_lo:[0,0,1]"
          : "+v"(ka), "+v"(kb), "=&v"(ta), "=&v"(tb)
          : "v"(cp), "v"(sp));
      k[0] = ka.x;
      k[1] = ka.y;
      k[2] = kb.x;
      k[3] = kb.y;
    }
    if (r.vs >= 0) {
      const float scale =
          __half2float(__ushort_as_half((unsigned short)(hdr & 0xffffu)));
      const float minimum =
          __half2float(__ushort_as_half((unsigned short)(hdr >> 16)));
      v[0] = fmaf(scale, tv.x, v[0] + minimum);
      v[1] = fmaf(scale, tv.y, v[1] + minimum);
      v[2] = fmaf(scale, tv.z, v[2] + minimum);
      v[3] = fmaf(scale, tv.w, v[3] + minimum);
    }
  } else {
    // Native suffix pages contain natural-order, post-RoPE BF16 K/V.
    // Only the compressed prefix uses Forge's packed physical K order.
    const int suffix_pos = pos - o.n;
    constexpr int fixed_page_size = PAGE_SIZE;
    const int page_size =
        fixed_page_size == 0 ? o.pages.block_size : fixed_page_size;
    const int block = o.blocks[suffix_pos / page_size];
    const int offset = suffix_pos % page_size;
    const int64_t row = (int64_t)block * o.pages.block_stride +
                        offset * o.pages.token_stride +
                        o.head * o.pages.head_stride;
    const BF16* kr = o.pages.key + row;
    const BF16* vr = o.pages.value + row;
    // Two aligned vector loads gather the four physical K coordinates, as
    // in the source query prologue. V remains four consecutive coordinates.
    const int dq =
        ((lane & 7) << 2) | (((lane >> 5) & 1) << 6) | (((lane >> 3) & 1) << 7);
    const uint32_t shift = (lane & 16) ? 0u : 16u;
    const u2* kp = reinterpret_cast<const u2*>(kr + dq);
    const u2 lo = kp[0], hi = kp[8];
    const u2 vv = reinterpret_cast<const u2*>(vr)[lane];
    const float sign = (__builtin_popcount(lane & 0xB) & 1) ? -1.f : 1.f;
    k[0] = __uint_as_float((lo.x << shift) & 0xffff0000u) * sign * SCORE_SCALE;
    k[1] = __uint_as_float((hi.x << shift) & 0xffff0000u) * sign * SCORE_SCALE;
    k[2] = __uint_as_float((lo.y << shift) & 0xffff0000u) * sign * SCORE_SCALE;
    k[3] = __uint_as_float((hi.y << shift) & 0xffff0000u) * sign * SCORE_SCALE;
    v[0] = __uint_as_float(vv.x << 16);
    v[1] = __uint_as_float(vv.x & 0xffff0000u);
    v[2] = __uint_as_float(vv.y << 16);
    v[3] = __uint_as_float(vv.y & 0xffff0000u);
  }
}

// Online-softmax step for one token: six dot products, transposed reduction,
// (sampled) raw score store, lazy rescale, exponential and the 24 accumulator
// FMAs.
template <bool SAMPLE>
__device__ __forceinline__ void token_accumulate(
    const float (&k)[4], const float (&v)[4], const f2 (&qp)[3][4],
    float (&value)[6][4], float& m_ref, float& denom, int lane, bool store_lane,
    uint32_t sc_base, int pos, float* __restrict__ scores, bool sample_owner) {
  // Three packed chains (head pair j): k[d] is broadcast straight from its half
  // of the (k0,k1) / (k2,k3) register pair through op_sel / op_sel_hi (the
  // compiler only forms the low-half broadcast and copies k1 / k3 into a fresh
  // register otherwise).
  float dot[6];
  {
    const f2 ka = {k[0], k[1]}, kb = {k[2], k[3]};
    f2 acc0, acc1, acc2;
    asm("v_pk_mul_f32 %0, %3, %15 op_sel_hi:[1,0]\n\t"
        "v_pk_mul_f32 %1, %7, %15 op_sel_hi:[1,0]\n\t"
        "v_pk_mul_f32 %2, %11, %15 op_sel_hi:[1,0]\n\t"
        "v_pk_fma_f32 %0, %4, %15, %0 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\t"
        "v_pk_fma_f32 %1, %8, %15, %1 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\t"
        "v_pk_fma_f32 %2, %12, %15, %2 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\t"
        "v_pk_fma_f32 %0, %5, %16, %0 op_sel_hi:[1,0,1]\n\t"
        "v_pk_fma_f32 %1, %9, %16, %1 op_sel_hi:[1,0,1]\n\t"
        "v_pk_fma_f32 %2, %13, %16, %2 op_sel_hi:[1,0,1]\n\t"
        "v_pk_fma_f32 %0, %6, %16, %0 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\t"
        "v_pk_fma_f32 %1, %10, %16, %1 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\t"
        "v_pk_fma_f32 %2, %14, %16, %2 op_sel:[0,1,0] op_sel_hi:[1,1,1]"
        : "=&v"(acc0), "=&v"(acc1), "=&v"(acc2)
        : "v"(qp[0][0]), "v"(qp[0][1]), "v"(qp[0][2]), "v"(qp[0][3]),
          "v"(qp[1][0]), "v"(qp[1][1]), "v"(qp[1][2]), "v"(qp[1][3]),
          "v"(qp[2][0]), "v"(qp[2][1]), "v"(qp[2][2]), "v"(qp[2][3]), "v"(ka),
          "v"(kb));
    dot[0] = acc0.x;
    dot[1] = acc0.y;
    dot[2] = acc1.x;
    dot[3] = acc1.y;
    dot[4] = acc2.x;
    dot[5] = acc2.y;
  }
  const float s = reduce6(dot, lane);
  // Lane 8g stores slot g: its byte offset lane/2 inside the 32-B row is
  // recomputed per token (volatile, so it is not hoisted back into a
  // loop-carried VGPR: the sampled variant must stay within the 80-VGPR
  // budget).
  if (SAMPLE && sample_owner) {
    if (store_lane) {
      uint32_t off;
      asm volatile("v_lshrrev_b32 %0, 1, %1" : "=v"(off) : "v"(lane));
      *reinterpret_cast<float*>(reinterpret_cast<char*>(scores) +
                                (sc_base + (uint32_t)pos * 32u) + off) = s;
    }
  }
  float e = s - m_ref;
  if (__builtin_amdgcn_ballot_w64(e > LAZY_TAU) !=
      0) {  // wave-uniform, rare: move the reference
    const float m_new = fmaxf(m_ref, s);
    const float alpha = fexp2(m_ref - m_new);
    denom *= alpha;
    m_ref = m_new;
    e = s - m_ref;
#pragma unroll
    for (int g = 0; g < 6; ++g) {
      const float a = readlane(alpha, head_lane(g));
#pragma unroll
      for (int d = 0; d < 4; ++d) value[g][d] *= a;
    }
  }
  const float w = fexp2(e);
  denom += w;
#pragma unroll
  for (int g = 0; g < 6; ++g) {
    const float wg = readlane(w, head_lane(g));
#pragma unroll
    for (int d = 0; d < 4; ++d) value[g][d] = fmaf(wg, v[d], value[g][d]);
  }
}

__global__ void pack_kernel(Record* dst, const int32_t* pk, const int32_t* pv,
                            const __half* gk, const __half* gv,
                            const int32_t* ks, const int32_t* vs,
                            int64_t offset, int count) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < count)
    dst[offset + j] =
        Record{(uint16_t)pk[j], (uint16_t)pv[j], gk[j], gv[j], ks[j], vs[j]};
}

// One token per wave-iteration, with lane-indexed structured prefix row loads,
// a pre-permuted K byte stream and a two-register FWHT xor-4 stage. Flat 64-B
// job record (prepare-built, one s_load_dwordx16, no dependent meta chain):
//   0 bh  1 split  2 begin  3 end  4 n  5 length  6 record_base  7 koff  8 voff
//   9 kscale_base 10 bank_off (floats)  11 toff  12 prow  13..15 unused
constexpr int JOB_STRIDE = 16;
template <bool SAMPLE, int PAGE_SIZE>
__device__ __forceinline__ void split_body(
    const int64_t* __restrict__ meta, const int32_t* __restrict__ jobs,
    const float* __restrict__ rot, const BF16* __restrict__ query,
    const float4* __restrict__ lut, uint32_t* __restrict__ partial,
    float* __restrict__ ml, float* __restrict__ scores, int64_t score_stride,
    int tile_tokens, const int64_t* __restrict__ owner_ptrs, NativePages pages,
    const int64_t* __restrict__ routes) {
  const int32_t* jb = jobs + blockIdx.x * JOB_STRIDE;
  const int bh = jb[0], split = jb[1], begin = jb[2], n = jb[4], length = jb[5];
  const int end = min(jb[3], length);
  const bool sample_owner = routes[2 * (bh / 4)] >= 0;
  const int lane = threadIdx.x & 63,
            wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6),
            d0 = lane * 4;
  // partial rows are indexed by the logical (owner, head, split) row, not the
  // launch slot, so the job list in `jobs` is free to permute launch order
  // (head-affine ordering, see prepare).
  const int64_t prow = jb[12];
  if (begin >= length) {
    for (int i = threadIdx.x; i < 6 * STRIDE / 2; i += 256)
      partial[prow * 6 * STRIDE / 2 + i] = 0u;  // FP16 zeros (pairs)
    if (threadIdx.x < 12)
      ml[prow * 12 + threadIdx.x] = (threadIdx.x & 1) ? 0.f : -INFINITY;
    return;
  }
  // Per-byte residual decode tables (prepare-built, 8 KB): ktab = K codebook
  // values after the two in-lane Hadamard stages, vtab = the four raw 2-bit V
  // codes as floats. They alias the epilogue's state_y buffer (the tables are
  // dead once every wave has left the token loop; a barrier separates the two
  // uses) so the block stays within 1/5 of the CU's LDS and 5 workgroups can be
  // resident alongside the 96-VGPR budget.
  __shared__ __align__(16) float state_y[24 * 256];
  __shared__ float state_m[24], state_l[24];
  float4* ktab = reinterpret_cast<float4*>(state_y);
  float4* vtab = ktab + 256;
  ktab[threadIdx.x] = lut[threadIdx.x];
  vtab[threadIdx.x] = lut[256 + threadIdx.x];
  // Query held as (head 2j, head 2j+1) pairs so the six dot products run as
  // packed FP32 FMAs (SCORE_SCALE is folded into the rotary table / the tail
  // branch, so q is loaded unscaled), gathered in the K-side physical dim order
  // {D, D+32, D+2, D+34} with the fused-butterfly lane sign applied.
  f2 qp[3][4];
  float value[6][4];
  // Dims D and D+2 (and the other lane's D+1, D+3) are the 4 consecutive bf16
  // at the 8-B aligned D & ~1, so per head row two 64-bit loads (at D&~1 and
  // D&~1 + 32) cover the lane's four dims; the twelve loads are pinned together
  // so the prologue is a single memory round trip.
  const uint32_t dq = ((lane & 7) << 2) | (((lane >> 5) & 1) << 6) |
                      (((lane >> 3) & 1) << 7);  // = D & ~1
  const uint32_t qsh =
      (lane & 16) ? 0u
                  : 16u;  // odd lane-bit-4 half: dims D'+1 / D'+3 (high halves)
  const float ksgn = (__builtin_popcount(lane & 0xB) & 1) ? -1.f : 1.f;
  {
    const u2* qrow = reinterpret_cast<const u2*>(
        query + (bh * 6) * 256 + dq);  // 64 u2 per head row, +8 = +32 dims
    u2 ql[6], qh[6];
#pragma unroll
    for (int g = 0; g < 6; ++g) {
      ql[g] = qrow[g * 64];
      qh[g] = qrow[g * 64 + 8];
    }
    asm volatile(""
                 : "+v"(ql[0]), "+v"(ql[1]), "+v"(ql[2]), "+v"(ql[3]),
                   "+v"(ql[4]), "+v"(ql[5]), "+v"(qh[0]), "+v"(qh[1]),
                   "+v"(qh[2]), "+v"(qh[3]), "+v"(qh[4]), "+v"(qh[5]));
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int ga = 2 * j, gb = 2 * j + 1;
      // slot 0: D (ql.x), slot 1: D+32 (qh.x), slot 2: D+2 (ql.y), slot 3: D+34
      // (qh.y)
      const uint32_t sa[4] = {ql[ga].x, qh[ga].x, ql[ga].y, qh[ga].y},
                     sb[4] = {ql[gb].x, qh[gb].x, ql[gb].y, qh[gb].y};
#pragma unroll
      for (int d = 0; d < 4; ++d)
        qp[j][d] = f2{__uint_as_float((sa[d] << qsh) & 0xffff0000u) * ksgn,
                      __uint_as_float((sb[d] << qsh) & 0xffff0000u) * ksgn};
    }
  }
  __syncthreads();
  // One job/owner descriptor is wave-uniform. Make that property explicit
  // before constructing scalar buffer resources or scalar kscale/header loads.
  const int owner_bh = __builtin_amdgcn_readfirstlane(bh);
  const int64_t* ptr = owner_ptrs + owner_bh * 8;
  uint64_t scalar_ptr[8];
  // Read the64byte owner descriptor directly through SMEM into SGPRs.
  // The descriptor address is constant within a wave, as is job bh.
  const uint64_t descriptor_address = (uint64_t)ptr;
  u16 descriptor;
  asm volatile("s_load_dwordx16 %0, %1, 0\n\ts_waitcnt lgkmcnt(0)"
               : "=s"(descriptor)
               : "s"(descriptor_address)
               : "memory");
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const uint32_t lo = __builtin_amdgcn_readfirstlane(descriptor[2 * c]);
    const uint32_t hi = __builtin_amdgcn_readfirstlane(descriptor[2 * c + 1]);
    scalar_ptr[c] = ((uint64_t)hi << 32) | lo;
#ifdef ANCHOR_UNIFORM_AUDIT
    if (__ballot((uint64_t)ptr[c] == scalar_ptr[c]) != ~0ull) __builtin_trap();
#endif
  }
  const Record* records = reinterpret_cast<const Record*>(scalar_ptr[0]);
  const uint8_t* kres_all = reinterpret_cast<const uint8_t*>(scalar_ptr[1]);
  const uint8_t* vres_all = reinterpret_cast<const uint8_t*>(scalar_ptr[2]);
  const float* kscale_all = reinterpret_cast<const float*>(scalar_ptr[3]);
  const float* bk = reinterpret_cast<const float*>(scalar_ptr[4]);
  const float* bv = reinterpret_cast<const float*>(scalar_ptr[5]);

  Owner o;
  o.records = records + jb[6];
  o.n = n;
  o.vres = vres_all;
  o.koff = (uint32_t)jb[7];
  o.voff = (uint32_t)jb[8];
  o.kscale = kscale_all + jb[9];
  o.rk = make_rsrc(kres_all);
  o.rv = make_rsrc(vres_all);
  o.rbk = make_rsrc_rows(bk + jb[10], 16);
  o.rbv = make_rsrc_rows(bv + jb[10], 16);
  o.rrot = make_rsrc(rot);
  o.pages = pages;
  o.head = bh % 4;
  o.blocks = pages.blocks + (bh / 4) * pages.table_stride;
  // rotary lanes {0..7, 16..23} (dims < 64) read their (cos, sin) quad at float
  // 4*rank; the others the identity quad at float 64 (byte offsets)
  const uint32_t offr =
      (lane & 0x28) ? 256u
                    : (uint32_t)(16 * ((lane & 7) | ((lane >> 4) & 1) << 3));
  const float sg[4] = {(lane & 1) ? -1.f : 1.f, (lane & 2) ? -1.f : 1.f,
                       (lane & 4) ? -1.f : 1.f, (lane & 8) ? -1.f : 1.f};
#pragma unroll
  for (int g = 0; g < 6; ++g) {
#pragma unroll
    for (int d = 0; d < 4; ++d) value[g][d] = 0.f;
  }
  // Softmax state lives in the six head lanes of one register each (see
  // reduce6).
  const bool is_head = ((lane & 7) == 0) && lane != 24 && lane != 56;
  float m_ref = is_head ? -INFINITY : NONHEAD_REF, denom = 0.f;
  // Sampled score store: all eight (lane&7)==0 lanes store (slots 3/7 of the
  // [pos][8] row are dead padding the moments kernel never reads); wave-uniform
  // row base, per-lane slot offset derived per token.
  const bool store_lane = (lane & 7) == 0;
  const uint32_t sc_base = (uint32_t)(bh * score_stride * 32);
  int pos = begin + wave;
  Record r = Record{0, 0, __half(), __half(), -1, -1};
  if (pos < n) r = load_record_readonly(o.records, pos);
  for (; pos < end; pos += 4) {
    Record rn = r;
    const int pn = pos + 4;
    if (pn < n)
      rn = load_record_readonly(
          o.records, pn);  // wave-uniform prefetch of the next record
    float k[4], v[4];
    token<PAGE_SIZE>(o, r, reinterpret_cast<const f4*>(ktab),
                     reinterpret_cast<const f4*>(vtab), pos, lane, offr, sg, k,
                     v);
    r = rn;
    token_accumulate<SAMPLE>(k, v, qp, value, m_ref, denom, lane, store_lane,
                             sc_base, pos, scores, sample_owner);
  }
  __syncthreads();  // all waves are done with the decode tables before state_y
                    // overwrites them
#pragma unroll
  for (int g = 0; g < 6; ++g) {
    const float mg = readlane(m_ref, head_lane(g)),
                lg = readlane(denom, head_lane(g));
    if (lane == 0) {
      state_m[wave * 6 + g] = mg;
      state_l[wave * 6 + g] = lg;
    }
#pragma unroll
    for (int d = 0; d < 4; ++d)
      state_y[(wave * 6 + g) * 256 + d0 + d] = value[g][d];
  }
  __syncthreads();
  for (int g = wave; g < 6;
       g +=
       4) {  // waves 0,1 merge heads {0,4},{1,5}; waves 2,3 merge heads {2},{3}
    {
      float mm = -INFINITY, dd = 0.f, yy[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int w = 0; w < 4; ++w) mm = fmaxf(mm, state_m[w * 6 + g]);
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        float weight = (state_m[w * 6 + g] == -INFINITY)
                           ? 0.f
                           : fexp2(state_m[w * 6 + g] - mm);
        dd += weight * state_l[w * 6 + g];
#pragma unroll
        for (int d = 0; d < 4; ++d)
          yy[d] += weight * state_y[(w * 6 + g) * 256 + d0 + d];
      }
      // normalised FP16 partial row: 4 halves (one uint2) per lane at dims
      // d0..d0+3
      const float inv = (dd > 0.f) ? 1.f / dd : 0.f;
      uint32_t* result = partial + ((prow * 6 + g) * STRIDE + d0) / 2;
      *reinterpret_cast<uint2*>(result) = make_uint2(
          pack_h2(yy[0] * inv, yy[1] * inv), pack_h2(yy[2] * inv, yy[3] * inv));
      if (lane == 0)
        *reinterpret_cast<float2*>(ml + (prow * 6 + g) * 2) =
            make_float2(mm, dd);
    }
  }
}

#define SPLIT_ARGS                                                    \
  const int64_t *__restrict__ meta, const int32_t *__restrict__ jobs, \
      const float *__restrict__ rot, const BF16 *__restrict__ query,  \
      const float4 *__restrict__ lut, uint32_t *__restrict__ partial, \
      float *__restrict__ ml, float *__restrict__ scores,             \
      int64_t score_stride, int tile_tokens,                          \
      const int64_t *__restrict__ owner_ptrs, NativePages pages,      \
      const int64_t *__restrict__ routes
#define SPLIT_PASS                                                             \
  meta, jobs, rot, query, lut, partial, ml, scores, score_stride, tile_tokens, \
      owner_ptrs, pages, routes
// Let the compiler balance occupancy against native-page addressing state.
// Forcing six waves per SIMD spills values used by both token paths.
template <bool SAMPLE, int PAGE_SIZE>
__global__ void __launch_bounds__(256) split_kernel(SPLIT_ARGS) {
  split_body<SAMPLE, PAGE_SIZE>(SPLIT_PASS);
}

constexpr int MAX_SPLITS =
    1024;  // per (owner, head); prepare asserts count <= MAX_SPLITS
constexpr int MAX_CCHUNKS =
    8;  // statistics layout sized for the narrowest chunk (32 dims)
constexpr int STAT_STRIDE =
    2 + 2 * MAX_CCHUNKS;  // per query head: max, 1/denom, (qn, yn) per chunk
// LDS-only barrier: the compiler's __syncthreads() also drains vmcnt, which
// would serialise the global loads issued ahead of pass 1 behind the (m, l)
// reduction.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Block = (query head, CH-dim chunk), 4 waves; grid B*24*(256/CH). CH is picked
// per prep from the batch size: narrow chunks give the B=1 combine (24 heads)
// enough blocks, wide chunks keep the B=8 combine from paying for 1536 tiny
// blocks. Lane layout for the FP16 partial rows: CH/8 lanes per row (8 halves,
// one uint4 each), 2048/CH rows per load slot, UNROLL slots in flight. All
// loads are unconditional (row index clamped) and the (m, l) loads go first, so
// pass 1 waits with vmcnt(UNROLL) while the first batch of y rows stays in
// flight: the kernel is ~one dependent memory round trip for <= UNROLL*2048/CH
// splits. Pass 1 (thread-over-split, 4 per thread): block max, weights exp2(m_i
// - M) * l_i into LDS (rows are normalised), L = sum of the weights. Pass 2:
// weighted FMAs, then lanes sharing dims (xor multiples of CH/8) shuffle-reduce
// + 4-wave LDS sum.
template <bool SAMPLE, int CH>
__global__ void __launch_bounds__(256)
    combine_kernel(const int64_t* meta, const BF16* query, const uint4* partial,
                   const float* ml, BF16* output, float* statistics,
                   const int64_t* routes) {
  constexpr int LPR = CH / 8, CROWS = 256 / LPR, NCH = 256 / CH,
                UNROLL = (CH == 256) ? 24 : 16;
  const int qh = blockIdx.x / NCH, c = blockIdx.x % NCH, bh = qh / 6,
            g = qh % 6, t = threadIdx.x;
  const bool sample_owner = routes[2 * (qh / 24)] >= 0;
  const int lane = t & 63, wave = t >> 6, row = t / LPR, sub = t % LPR;
  const int64_t* m = meta + bh * META_STRIDE;
  const int start = (int)m[7], count = (int)m[8];
  const uint4* base =
      partial + (((int64_t)start * 6 + g) * STRIDE + c * CH + sub * 8) /
                    8;  // uint4 = 8 halves
  const float* mlb = ml + ((int64_t)start * 6 + g) * 2;
  __shared__ float weights[MAX_SPLITS];
  __shared__ float red[16];
  __shared__ __align__(16) float cy[4 * CH];
  float2 mlv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    mlv[j] = *reinterpret_cast<const float2*>(
        mlb + (int64_t)min(t + j * 256, count - 1) * 12);
  uint4 vv[UNROLL];
#pragma unroll
  for (int u = 0; u < UNROLL; ++u)
    vv[u] = base[(int64_t)min(row + u * CROWS, count - 1) * 6 * STRIDE / 8];
  float mi[4], li[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const bool ok = t + j * 256 < count;
    mi[j] = ok ? mlv[j].x : -INFINITY;
    li[j] = ok ? mlv[j].y : 0.f;
  }
  float wm = fmaxf(fmaxf(mi[0], mi[1]), fmaxf(mi[2], mi[3]));
#pragma unroll
  for (int s = 32; s > 0; s >>= 1) wm = fmaxf(wm, __shfl_xor(wm, s, 64));
  if (lane == 0) red[wave] = wm;
  lds_barrier();
  const float maximum = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  float wl = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float wi =
        (mi[j] == -INFINITY) ? 0.f : fexp2(mi[j] - maximum) * li[j];
    weights[t + j * 256] = wi;
    wl += wi;
  }
  wl = wave_sum(wl);
  if (lane == 0) red[4 + wave] = wl;
  lds_barrier();
  f2 wy[4] = {f2{0.f, 0.f}, f2{0.f, 0.f}, f2{0.f, 0.f}, f2{0.f, 0.f}};
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    const int i = row + u * CROWS;
    const float w = (i < count) ? weights[i] : 0.f;
    const f2 w2 = {w, w};
    wy[0] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].x), wy[0]);
    wy[1] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].y), wy[1]);
    wy[2] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].z), wy[2]);
    wy[3] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].w), wy[3]);
  }
  for (int i0 = UNROLL * CROWS; i0 < count; i0 += UNROLL * CROWS) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      vv[u] =
          base[(int64_t)min(i0 + row + u * CROWS, count - 1) * 6 * STRIDE / 8];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int i = i0 + row + u * CROWS;
      const float w = (i < count) ? weights[i] : 0.f;
      const f2 w2 = {w, w};
      wy[0] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].x), wy[0]);
      wy[1] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].y), wy[1]);
      wy[2] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].z), wy[2]);
      wy[3] = __builtin_elementwise_fma(w2, unpack_h2(vv[u].w), wy[3]);
    }
  }
#pragma unroll
  for (int s = LPR; s < 64; s <<= 1) {
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      wy[d].x += __shfl_xor(wy[d].x, s, 64);
      wy[d].y += __shfl_xor(wy[d].y, s, 64);
    }
  }
  if (lane < LPR) {
    *reinterpret_cast<float4*>(cy + wave * CH + sub * 8) =
        make_float4(wy[0].x, wy[0].y, wy[1].x, wy[1].y);
    *reinterpret_cast<float4*>(cy + wave * CH + sub * 8 + 4) =
        make_float4(wy[2].x, wy[2].y, wy[3].x, wy[3].y);
  }
  lds_barrier();
  const float denom = red[4] + red[5] + red[6] + red[7];
  float y = 0.f, qq = 0.f;
  if (t < CH) {  // threads 0..CH-1 produce the CH output dims of this chunk
    y = (cy[t] + cy[CH + t] + cy[2 * CH + t] + cy[3 * CH + t]) / denom;
    output[(int64_t)qh * 256 + c * CH + t] = BF16(y);
    if (SAMPLE && sample_owner)
      qq = (float)query[(int64_t)qh * 256 + c * CH + t];
  }
  if (SAMPLE && sample_owner) {  // per-chunk (qn, yn) partials; moments_kernel
                                 // sums the chunks
    const float qn = wave_sum(qq * qq), yn = wave_sum(y * y);
    if (lane == 0) {
      red[8 + wave * 2] = qn;
      red[9 + wave * 2] = yn;
    }
    lds_barrier();
    if (t == 0) {
      float* st = statistics + qh * STAT_STRIDE;
      if (c == 0) {
        st[0] = maximum;
        st[1] = 1.f / denom;
#pragma unroll
        for (int cc = NCH; cc < MAX_CCHUNKS; ++cc) {
          st[2 + 2 * cc] = 0.f;
          st[3 + 2 * cc] = 0.f;
        }
      }
      st[2 + 2 * c] =
          (red[8] + red[10] + red[12] + red[14]) * 0.0000152587890625f;
      st[3 + 2 * c] = red[9] + red[11] + red[13] + red[15];
    }
  }
}

void decode_split(torch::Tensor meta, torch::Tensor jobs, torch::Tensor rot,
                  torch::Tensor q, torch::Tensor lut, torch::Tensor partial,
                  torch::Tensor ml, torch::Tensor scores,
                  torch::Tensor owner_ptrs, torch::Tensor key,
                  torch::Tensor value, torch::Tensor blocks,
                  torch::Tensor routes, int64_t tile, bool sample) {
  const c10::cuda::CUDAGuard guard(q.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  NativePages pages{reinterpret_cast<const BF16*>(key.data_ptr()),
                    reinterpret_cast<const BF16*>(value.data_ptr()),
                    blocks.data_ptr<int32_t>(),
                    (int)key.size(1),
                    (int)blocks.stride(0),
                    key.stride(0),
                    key.stride(1),
                    key.stride(2)};
#define LAUNCH(SAMPLE, PAGE_SIZE)                                            \
  hipLaunchKernelGGL(                                                        \
      (split_kernel<SAMPLE, PAGE_SIZE>), dim3(jobs.size(0)), dim3(256), 0,   \
      stream, meta.data_ptr<int64_t>(), jobs.data_ptr<int32_t>(),            \
      rot.data_ptr<float>(), reinterpret_cast<const BF16*>(q.data_ptr()),    \
      reinterpret_cast<const float4*>(lut.data_ptr()),                       \
      reinterpret_cast<uint32_t*>(partial.data_ptr()), ml.data_ptr<float>(), \
      scores.data_ptr<float>(), scores.size(1), tile,                        \
      owner_ptrs.data_ptr<int64_t>(), pages, routes.data_ptr<int64_t>())
  if (pages.block_size == 16) {
    if (sample) {
      LAUNCH(true, 16);
    } else {
      LAUNCH(false, 16);
    }
  } else if (pages.block_size == 32) {
    if (sample) {
      LAUNCH(true, 32);
    } else {
      LAUNCH(false, 32);
    }
  } else {
    if (sample) {
      LAUNCH(true, 0);
    } else {
      LAUNCH(false, 0);
    }
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int CH>
void launch_combine(torch::Tensor meta, torch::Tensor q, torch::Tensor partial,
                    torch::Tensor ml, torch::Tensor out, torch::Tensor summary,
                    torch::Tensor routes, bool sample, hipStream_t stream) {
#define LAUNCH(SAMPLE)                                                 \
  hipLaunchKernelGGL(                                                  \
      (combine_kernel<SAMPLE, CH>), dim3(q.size(0) * 24 * (256 / CH)), \
      dim3(256), 0, stream, meta.data_ptr<int64_t>(),                  \
      reinterpret_cast<const BF16*>(q.data_ptr()),                     \
      reinterpret_cast<const uint4*>(partial.data_ptr()),              \
      ml.data_ptr<float>(), reinterpret_cast<BF16*>(out.data_ptr()),   \
      summary.data_ptr<float>(), routes.data_ptr<int64_t>())
  if (sample) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
}

void decode_combine(torch::Tensor meta, torch::Tensor q, torch::Tensor partial,
                    torch::Tensor ml, torch::Tensor out, torch::Tensor summary,
                    torch::Tensor routes, int64_t chunk, bool sample) {
  const c10::cuda::CUDAGuard guard(q.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  switch (chunk) {
    case 32:
      launch_combine<32>(meta, q, partial, ml, out, summary, routes, sample,
                         stream);
      break;
    case 128:
      launch_combine<128>(meta, q, partial, ml, out, summary, routes, sample,
                          stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported combine chunk");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_split", &decode_split);
  m.def("decode_combine", &decode_combine);
}
