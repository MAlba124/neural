extern "C" __global__ void
dsigmoid(float *A, float *B) {
    B[threadIdx.x] = A[threadIdx.x] * (1.0 - A[threadIdx.x]);
}
