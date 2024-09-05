extern "C" __global__ void
sigmoid(float *A) {
    A[threadIdx.x] = 1.0 / (1.0 + expf(-A[threadIdx.x]));
}
