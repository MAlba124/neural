__kernel void
gemm(const int M, const int N, const int K,
  __global float* A, __global float* B, __global float* C) {
  int row = get_global_id(1);
  int col = get_global_id(0);
  float sum = 0.0;

  for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
  }

  C[row * N + col] = sum;
}

__kernel void
mult_scalar(__global float* A, const float scalar) {
  A[get_global_id(0)] *= scalar;
}
