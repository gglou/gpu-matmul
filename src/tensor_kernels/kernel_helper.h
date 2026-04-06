#ifndef PTX_INSTRUCTION_HELPER_H
#define PTX_INSTRUCTION_HELPER_H

__device__ __forceinline__
void wgmma_m64n16k8(float (&d)[8], uint64_t desc_a, uint64_t desc_b) {
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 0, -1, -1;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
      "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    : "l"(desc_a), "l"(desc_b)
  );
}

// Helper for creating tensor maps (input to the kernel function)
// In the fp32 we were passing the actual pointer to A & B.
// Now we are mapping a tensor map which describes how to move
// data from the global memory to shared memory.
template <uint32_t BLOCK_MAJOR, uint32_t BLOCK_MINOR>
__host__ void create_tma_descriptor_2d(
    CUtensorMap& tma_desc,
    void* base_ptr,
    uint64_t dimMajor,
    uint64_t dimMinor
) {
  uint64_t globalDim[2]    = {dimMinor, dimMajor}; // Minor dimension first
  uint64_t globalStride[1] = {dimMinor * sizeof(__nv_bfloat16)};
  uint32_t boxDim[2]       = {BLOCK_MINOR, BLOCK_MAJOR}; // Minor dimension first
  uint32_t elemStride[2]   = {1, 1};

  // Swizzling must be enabled for optimal shared memory access patterns.
  // Assert that the inner dimension fits within 128 bytes (the swizzle boundary).
  static_assert(BLOCK_MINOR * sizeof(__nv_bfloat16) <= 128, 
                "BLOCK_MINOR too large for 128B swizzling");

  cuTensorMapEncodeTiled(
      &tma_desc,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      base_ptr,
      globalDim,
      globalStride,
      boxDim,
      elemStride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor  // Make smem descriptor.
// Create a shared memory descriptor for wgmma operations.
// SMEM_ROWS: Number of rows in the shared memory tile
// SMEM_COLS: Number of columns in the shared memory tile
template <uint32_t SMEM_ROWS, uint32_t SMEM_COLS>
__device__ uint64_t make_smem_desc(__nv_bfloat16* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  
  uint64_t desc = 0x0000000000000000;
  desc |= matrix_descriptor_encode(addr);
  desc |= matrix_descriptor_encode((uint64_t)16) << 16; // It's always 16 when it's swizzled.
  desc |= matrix_descriptor_encode((uint64_t)(8 * SMEM_COLS * sizeof(__nv_bfloat16))) << 32;
  desc |= 1llu << 62; // 128B swizzle
  
  return desc;
}

  
}




#endif // PTX_INSTRUCTION_HELPER_H
