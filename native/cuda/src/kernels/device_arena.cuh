#if defined(CUBBS_BBS_ENABLE_SPPARK_MSM)

#ifndef CUBBS_BBS_DEVICE_ARENA_CUH
#define CUBBS_BBS_DEVICE_ARENA_CUH

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Persistent Thread-Local GPU Bump-Allocator Arena (Zero Per-Batch Malloc/Free)
// ============================================================================
// Eliminates 12-18 cudaMallocAsync / cudaFreeAsync calls per batch by serving
// all transient device buffers out of a single grow-only, 256-byte-aligned
// persistent cudaMalloc slab. Rewind is O(1) (resetting a single size_t).
struct CuBbsDeviceArena {
  uint8_t *base{nullptr};
  size_t capacity{0u};
  size_t offset{0u};
  int device{-1};

  CuBbsDeviceArena() { (void)cudaGetDevice(&device); }

  ~CuBbsDeviceArena() {
    if (base != nullptr) {
      int prev = -1;
      (void)cudaGetDevice(&prev);
      if (device >= 0 && device != prev) (void)cudaSetDevice(device);
      (void)cudaFree(base);
      if (prev >= 0 && prev != device) (void)cudaSetDevice(prev);
      base = nullptr;
      capacity = 0u;
      offset = 0u;
    }
  }

  CuBbsDeviceArena(const CuBbsDeviceArena &) = delete;
  CuBbsDeviceArena &operator=(const CuBbsDeviceArena &) = delete;

  static size_t align_up(size_t n, size_t align = 256u) {
    return (n + align - 1u) & ~(align - 1u);
  }

  cudaError_t reserve_and_reset(size_t total_bytes) {
    offset = 0u;
    if (total_bytes == 0u) return cudaSuccess;
    if (total_bytes <= capacity && base != nullptr) return cudaSuccess;
    if (base != nullptr) {
      (void)cudaFree(base);
      base = nullptr;
      capacity = 0u;
    }
    // Grow with 1.25x headroom (aligned to 1 MiB) so subsequent shapes reuse cleanly.
    const size_t min_bytes = align_up(total_bytes + (total_bytes >> 2u), 1u << 20u);
    const size_t alloc_bytes = std::max<size_t>(min_bytes, 4u << 20u);
    cudaError_t err = cudaMalloc(&base, alloc_bytes);
    if (err != cudaSuccess) {
      // Retry exact size if 1.25x headroom exceeded available VRAM.
      err = cudaMalloc(&base, align_up(total_bytes, 256u));
      if (err == cudaSuccess) capacity = align_up(total_bytes, 256u);
      return err;
    }
    capacity = alloc_bytes;
    return cudaSuccess;
  }

  template <typename T>
  T *alloc(size_t count) {
    if (count == 0u) return nullptr;
    const size_t bytes = count * sizeof(T);
    const size_t aligned_off = align_up(offset, 256u);
    offset = aligned_off + bytes;
    return reinterpret_cast<T *>(base + aligned_off);
  }
};

static CuBbsDeviceArena &cubbs_bbs_thread_device_arena() {
  static thread_local CuBbsDeviceArena arena;
  return arena;
}

// Separate sub-arena for nested helper kernels (e.g., Lim-Lee masks/partials,
// Pippenger large MSM scratch) so callers using cubbs_bbs_thread_device_arena()
// never alias with MSM transient buffers.
static CuBbsDeviceArena &cubbs_bbs_thread_msm_arena() {
  static thread_local CuBbsDeviceArena arena;
  return arena;
}

#endif // CUBBS_BBS_DEVICE_ARENA_CUH
#endif // CUBBS_BBS_ENABLE_SPPARK_MSM
