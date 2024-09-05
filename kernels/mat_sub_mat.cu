extern "C" __global__ void
mat_sub_mat(float *A, float *B, float *C) {
    C[threadIdx.x] = A[threadIdx.x] - B[threadIdx.x];
}
