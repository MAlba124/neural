extern "C" __global__ void
mat_add_scalar(float *A, float s) {
    A[threadIdx.x] += s;
}