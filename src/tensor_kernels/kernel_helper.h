#ifndef PTX_INSTRUCTION_HELPER_H
#define PTX_INSTRUCTION_HELPER_H

// TMA descriptor for 2D tiles of bf16 data.
// BLOCK_MAJOR = number of rows in the tile (outer dimension).
// BLOCK_MINOR = number of columns in the tile (inner/contiguous dimension).
template <uint32_t BLOCK_MAJOR, uint32_t BLOCK_MINOR>
__host__ void create_tma_descriptor_2d(
    CUtensorMap& tma_desc,
    void* base_ptr,
    uint64_t dimMajor,
    uint64_t dimMinor
) {
  uint64_t globalDim[2]    = {dimMinor, dimMajor};
  uint64_t globalStride[1] = {dimMinor * sizeof(__nv_bfloat16)};
  uint32_t boxDim[2]       = {BLOCK_MINOR, BLOCK_MAJOR};
  uint32_t elemStride[2]   = {1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      &tma_desc,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      base_ptr,
      globalDim,
      globalStride,
      boxDim,
      elemStride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  if (err != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(err, &errStr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s (code %d)\n",
            errStr ? errStr : "unknown", (int)err);
  }
}

#endif // PTX_INSTRUCTION_HELPER_H
