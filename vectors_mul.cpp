#include <iostream>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#define N 5  // Vector size

int A[N] = { 1, 2, 3, 4, 5 };
int B[N] = { 10, 20, 30, 40, 50 };
int C[N] = { 0 };

#ifdef __CUDACC__
__global__ void vectorAddKernel(int* A, int* B, int* C, int N) {
    int idx = threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
#endif

void vectorAddCPU() {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
#ifdef __CUDACC__
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    vectorAddKernel << <1, N >> > (d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
#else
    vectorAddCPU();
#endif

    std::cout << "Vector A: ";
    for (int i = 0; i < N; i++) std::cout << A[i] << " ";
    std::cout << "\nVector B: ";
    for (int i = 0; i < N; i++) std::cout << B[i] << " ";
    std::cout << "\nVector C = A + B: ";
    for (int i = 0; i < N; i++) std::cout << C[i] << " ";
    std::cout << "\n";

    return 0;
}
