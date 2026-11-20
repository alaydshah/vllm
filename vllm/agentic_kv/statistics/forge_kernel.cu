// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// Source Forge moments reduction: preserve token-per-thread vector loads and
// six-head sequential arithmetic; change only routing, strides and validity.
__device__ __forceinline__ float score_exp2(float x) {
#if defined(__HIP_PLATFORM_AMD__)
  return __builtin_amdgcn_exp2f(x);
#else
  return exp2f(x);
#endif
}

__global__ void __launch_bounds__(256) accumulate_kernel(
    const int64_t* __restrict__ meta, const float* __restrict__ scores,
    const float* __restrict__ summary, const int64_t* __restrict__ routes,
    float* __restrict__ sums, int32_t* counts, int32_t* flags,
    int64_t score_length, int64_t capacity, int slots, int64_t s0, int64_t s2,
    int64_t s3, int64_t s4, int64_t c0, int64_t c2, int64_t f0, int64_t f2) {
  const int bh = blockIdx.x, h = bh % 4, b = bh / 4;
  const int64_t slot = routes[b * 2], coverage = routes[b * 2 + 1];
  if (slot < 0 || slot >= slots) return;
  const int64_t length = meta[bh * 10 + 1] + meta[bh * 10 + 2];
  int32_t* flag = flags + slot * f0 + h * f2;
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    int32_t* count = counts + slot * c0 + h * c2;
    *count = static_cast<int32_t>(static_cast<uint32_t>(*count) + 1u);
    int code = (*count <= 0) ? 4 : 0;
    if (coverage <= 0 || coverage > capacity || coverage > length ||
        length > score_length)
      code |= 2;
    if (code) atomicOr(flag, code);
  }
  const int64_t first = static_cast<int64_t>(blockIdx.y) * 256;
  if (first >= length || first >= coverage || first >= capacity) return;
  const int64_t pos = first + threadIdx.x;
  const bool live =
      pos < length && pos < coverage && pos < capacity && pos < score_length;
  __shared__ float st[6][4];
  if (threadIdx.x < 6) {
    const float* sg = summary + (bh * 6 + threadIdx.x) * 18;
    float qn = 0.f, yn = 0.f;
#pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
      qn += sg[2 + 2 * chunk];
      yn += sg[3 + 2 * chunk];
    }
    st[threadIdx.x][0] = sg[0];
    st[threadIdx.x][1] = sg[1];
    st[threadIdx.x][2] = qn;
    st[threadIdx.x][3] = qn * yn;
  }
  const float4* sc = reinterpret_cast<const float4*>(
      scores + (bh * score_length + (live ? pos : 0)) * 8);
  const float4 a = sc[0], z = sc[1];
  const float score[6] = {a.x, a.y, a.z, z.x, z.y, z.z};
  float* dst = sums + slot * s0 + h * s3 + (live ? pos : 0) * s4;
  float cur[4];
#pragma unroll
  for (int field = 0; field < 4; ++field) cur[field] = dst[field * s2];
  __syncthreads();
  if (!live) return;
  float mass = 0.f, w2 = 0.f, qw2 = 0.f, qoutw2 = 0.f;
#pragma unroll
  for (int g = 0; g < 6; ++g) {
    const float p = score_exp2(score[g] - st[g][0]) * st[g][1], p2 = p * p;
    mass += p;
    w2 += p2;
    qw2 += p2 * st[g][2];
    qoutw2 += p2 * st[g][3];
  }
  const float value[4] = {mass * (1.f / 6.f), w2 * (1.f / 6.f),
                          qw2 * (1.f / 6.f), qoutw2 * (1.f / 6.f)};
  bool bad = false;
#pragma unroll
  for (int field = 0; field < 4; ++field) {
    const float total = cur[field] + value[field];
    bad |= !isfinite(total) || value[field] < 0.f;
    dst[field * s2] = total;
  }
  if (bad) atomicOr(flag, 1);
}

void accumulate(torch::Tensor meta, torch::Tensor scores, torch::Tensor summary,
                torch::Tensor routes, torch::Tensor sums, torch::Tensor counts,
                torch::Tensor flags, int64_t layer) {
  const c10::cuda::CUDAGuard guard(scores.device());
  const auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
  const int64_t capacity = sums.size(4), score_length = scores.size(1);
  const int64_t extent = std::min(capacity, score_length);
  accumulate_kernel<<<dim3(meta.size(0), (extent + 255) / 256), 256, 0,
                      stream>>>(
      meta.data_ptr<int64_t>(), scores.data_ptr<float>(),
      summary.data_ptr<float>(), routes.data_ptr<int64_t>(),
      sums.data_ptr<float>() + layer * sums.stride(1),
      counts.data_ptr<int32_t>() + layer * counts.stride(1),
      flags.data_ptr<int32_t>() + layer * flags.stride(1), score_length,
      capacity, sums.size(0), sums.stride(0), sums.stride(2), sums.stride(3),
      sums.stride(4), counts.stride(0), counts.stride(2), flags.stride(0),
      flags.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("accumulate", &accumulate); }
